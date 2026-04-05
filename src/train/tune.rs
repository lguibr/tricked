use crate::cli::TuneConfig;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub fn run_tuning_pipeline(tune_cfg: TuneConfig) {
    println!("⚙️  Starting Native Rust Holistic Tuning Pipeline...");

    let workspace_db = tune_cfg
        .workspace_db
        .unwrap_or_else(|| "tricked_workspace.db".to_string());

    for trial_idx in 0..tune_cfg.trials {
        println!(
            "\n[Native Tune] Requesting hyperparameters for Trial {}...",
            trial_idx
        );

        let output = Command::new("python")
            .arg("scripts/optuna_ask.py")
            .arg("--config")
            .arg(&tune_cfg.config_path)
            .arg("--bounds")
            .arg(&tune_cfg.bounds)
            .arg("--resnet-blocks")
            .arg(tune_cfg.resnet_blocks.to_string())
            .arg("--resnet-channels")
            .arg(tune_cfg.resnet_channels.to_string())
            .output()
            .expect("Failed to execute optuna_ask.py");

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            eprintln!("optuna_ask.py failed:\n{}", err);
            continue;
        }

        let out_str = String::from_utf8_lossy(&output.stdout);

        let json_line = out_str
            .lines()
            .find(|l| l.starts_with('{'))
            .expect("No JSON payload from optuna_ask.py");
        let trial_data: serde_json::Value =
            serde_json::from_str(json_line).expect("Invalid JSON from optuna_ask");

        let trial_number = trial_data["trial_number"].as_u64().unwrap();
        let config_json = trial_data["config"].to_string();

        let experiment_name = format!("unified_tune_trial_{:03}", trial_number);

        let conn = rusqlite::Connection::open(&workspace_db).unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                config JSON,
                tags JSON,
                artifacts_dir TEXT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME
            )",
            [],
        )
        .unwrap();

        let artifacts_dir = format!("artifacts/{}", experiment_name);
        conn.execute(
            "INSERT OR REPLACE INTO runs (id, name, type, status, config, tags, artifacts_dir) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                &experiment_name,
                &experiment_name,
                "TUNING_TRIAL",
                "RUNNING",
                &config_json,
                serde_json::json!(["unified_tune"]).to_string(),
                &artifacts_dir
            ],
        ).unwrap();

        println!("\n[Native Tune] Trial {} Started.", trial_number);
        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("train")
            .arg("--run-id")
            .arg(&experiment_name)
            .arg("--workspace-db")
            .arg(&workspace_db)
            .arg("--max-steps")
            .arg(tune_cfg.max_steps.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Failed to spawn engine");

        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);

        let mut final_loss = f64::MAX;
        let mut mcts_time_avg = f64::MAX;
        let start_time = Instant::now();
        let mut pruned = false;

        let (tx, rx) = std::sync::mpsc::channel();

        let exp_name_clone = experiment_name.clone();
        // Reading thread
        std::thread::spawn(move || {
            for line in reader.lines() {
                if let Ok(line) = line {
                    println!("[CHILD:{}] {}", exp_name_clone, line);

                    if line.contains("FINAL_EVAL_SCORE:") {
                        if let Some(val_str) = line.split("FINAL_EVAL_SCORE:").nth(1) {
                            if let Ok(val) = val_str.trim().parse::<f64>() {
                                let _ = tx.send(("loss", val));
                            }
                        }
                    }
                    if line.contains("search::mcts_search") && line.contains('|') {
                        let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
                        if parts.len() > 4 {
                            let avg_str = parts[3];
                            let parsed = if avg_str.contains("ms") {
                                avg_str
                                    .replace("ms", "")
                                    .trim()
                                    .parse::<f64>()
                                    .unwrap_or(f64::MAX)
                                    / 1000.0
                            } else if avg_str.contains("µs") {
                                avg_str
                                    .replace("µs", "")
                                    .trim()
                                    .parse::<f64>()
                                    .unwrap_or(f64::MAX)
                                    / 1_000_000.0
                            } else {
                                f64::MAX
                            };
                            let _ = tx.send(("time", parsed));
                        }
                    }
                }
            }
        });

        let mut last_db_check = Instant::now();

        loop {
            if let Ok(status) = child.try_wait() {
                if status.is_some() {
                    break;
                }
            }

            while let Ok((key, val)) = rx.try_recv() {
                if key == "loss" {
                    final_loss = val;
                }
                if key == "time" {
                    mcts_time_avg = val;
                }
            }

            if last_db_check.elapsed() > Duration::from_secs(2) {
                last_db_check = Instant::now();
                if let Ok(last_vram) = conn.query_row(
                    "SELECT vram_usage_mb FROM metrics WHERE run_id = ?1 ORDER BY step DESC LIMIT 1",
                    rusqlite::params![&experiment_name],
                    |row| row.get::<_, f64>(0),
                ) {
                    if last_vram > 11500.0 {
                        println!("[Native Tune] Trial {} PRUNED: VRAM limit.", trial_number);
                        let _ = child.kill();
                        pruned = true;
                        break;
                    }
                }
            }

            if start_time.elapsed() > Duration::from_secs(tune_cfg.timeout) {
                println!("[Native Tune] Trial {} TIMEOUT.", trial_number);
                let _ = child.kill();
                pruned = true;
                break;
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        let _ = child.wait();

        while let Ok((key, val)) = rx.try_recv() {
            if key == "loss" {
                final_loss = val;
            }
            if key == "time" {
                mcts_time_avg = val;
            }
        }

        let wall_clock = start_time.elapsed().as_secs_f64();
        let hardware_penalty = if mcts_time_avg < f64::MAX {
            wall_clock + (mcts_time_avg * 100.0)
        } else {
            wall_clock
        };

        let mut tell_cmd = Command::new("python");
        tell_cmd
            .arg("scripts/optuna_tell.py")
            .arg("--trial")
            .arg(trial_number.to_string())
            .arg("--loss")
            .arg(final_loss.to_string())
            .arg("--hardware")
            .arg(hardware_penalty.to_string());

        if pruned {
            tell_cmd.arg("--pruned");
        }

        let tell_out = tell_cmd.output().expect("Failed to execute optuna_tell.py");
        if !tell_out.status.success() {
            eprintln!(
                "optuna_tell failed: {}",
                String::from_utf8_lossy(&tell_out.stderr)
            );
        }
    }

    println!("✅ Native Tuning Complete!");
}

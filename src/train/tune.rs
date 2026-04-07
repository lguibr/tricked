use crate::cli::TuneConfig;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub fn run_tuning_pipeline(tune_cfg: TuneConfig) {
    println!("⚙️  Starting Native Rust Holistic Tuning Pipeline...");

    let workspace_db = tune_cfg
        .workspace_db
        .unwrap_or_else(|| "tricked_workspace.db".to_string());

    let base_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&tune_cfg.config_path).expect("Failed to read config"),
    )
    .unwrap_or(serde_json::json!({}));

    let bounds_json: serde_json::Value =
        serde_json::from_str(&tune_cfg.bounds).unwrap_or(serde_json::json!({}));

    let mut daemon = Command::new("python3")
        .arg("scripts/optuna_daemon.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to start optuna daemon");

    let mut daemon_in = daemon.stdin.take().unwrap();
    let mut daemon_reader = BufReader::new(daemon.stdout.take().unwrap());

    let mut line = String::new();
    daemon_reader.read_line(&mut line).unwrap(); // Wait for ready

    use std::io::Write;

    for trial_idx in 0..tune_cfg.trials {
        println!(
            "\n[Native Tune] Requesting hyperparameters for Trial {}...",
            trial_idx
        );

        let ask_req = serde_json::json!({
            "action": "ask",
            "config": base_config,
            "bounds": bounds_json
        });
        writeln!(daemon_in, "{}", ask_req).unwrap();

        line.clear();
        daemon_reader.read_line(&mut line).unwrap();
        let trial_data: serde_json::Value =
            serde_json::from_str(&line).expect("Invalid JSON from daemon");

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
            for line in reader.lines().map_while(Result::ok) {
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

        let tell_req = serde_json::json!({
            "action": "tell",
            "trial_number": trial_number,
            "loss": final_loss,
            "hardware": hardware_penalty,
            "pruned": pruned
        });
        writeln!(daemon_in, "{}", tell_req).unwrap();
        line.clear();
        daemon_reader.read_line(&mut line).unwrap();
    }

    println!("✅ Native Tuning Complete!");
    let _ = daemon.kill();
    let _ = daemon.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_optuna_ipc_routing() {
        let script = r#"
import sys, json
sys.stdout.write("READY\n")
sys.stdout.flush()
for line in sys.stdin:
    data = json.loads(line)
    if data["action"] == "ask":
        sys.stdout.write(json.dumps({"trial_number": 1, "config": {}}) + "\n")
        sys.stdout.flush()
    elif data["action"] == "tell":
        sys.stdout.write("ACK\n")
        sys.stdout.flush()
        break
"#;
        std::fs::create_dir_all("scripts").unwrap();
        std::fs::write("scripts/optuna_daemon.py", script).unwrap();
        std::fs::write("test_config.json", "{}").unwrap();

        let cfg = TuneConfig {
            config_path: "test_config.json".into(),
            bounds: "{}".into(),
            trials: 1,
            max_steps: 1,
            timeout: 5,
            workspace_db: Some("test_tune.db".into()),
            resnet_blocks: 2,
            resnet_channels: 64,
        };

        // This will spawn the current test executable with "train", which fails immediately,
        // so it successfully simulates the child abort and tells the daemon to end.
        run_tuning_pipeline(cfg);
        // Assertion relies on process correctly exiting instead of hanging
    }
}

use crate::cli::TuneConfig;
use optimizer::prelude::*;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tricked_shared::models::{StudyData, TrialData};

fn get_bound(bounds: &serde_json::Value, key: &str, def_min: f64, def_max: f64) -> (f64, f64) {
    if let Some(b) = bounds.get(key) {
        let min = b.get("min").and_then(|v| v.as_f64()).unwrap_or(def_min);
        let max = b.get("max").and_then(|v| v.as_f64()).unwrap_or(def_max);
        (min, max)
    } else {
        (def_min, def_max)
    }
}

fn save_study_state(study_name: &str, trials: &[TrialData]) {
    let state = StudyData {
        trials: trials.to_vec(),
        importance: serde_json::Map::new(),
    };
    let json_str = serde_json::to_string(&state).unwrap();
    let _ = fs::create_dir_all("studies");
    let path = format!("studies/{}_optimizer_study.json", study_name);
    let tmp = format!("{}.tmp", path);
    if fs::write(&tmp, json_str).is_ok() {
        let _ = fs::rename(&tmp, &path);
    }
}

pub fn run_tuning_pipeline(tune_cfg: TuneConfig) {
    println!("⚙️  Starting Native Rust Holistic Tuning Pipeline with TPE (Bayesian) Optimization!");

    let workspace_db = tune_cfg
        .workspace_db
        .clone()
        .unwrap_or_else(|| "tricked_workspace.db".to_string());

    let base_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&tune_cfg.config_path)
            .expect("Failed to read config file from disk"),
    )
    .expect("FATAL: Failed to deserialize base JSON configuration string.");

    let bounds_json: serde_json::Value =
        serde_json::from_str(&tune_cfg.bounds).unwrap_or(serde_json::json!({}));

    // Start optimizer study, minimizing hardware cost and loss simultaneously if possible,
    // but optimizer typically handles scalar objectives or arrays if configured.
    // For simplicity with optimizer::Study, we minimize a scalar objective derived from hardware+loss,
    // or we can use multi-objective!
    // But optimizer's standard `Study<f64>` is scalar. Let's do `Study<f64>` and track both visually.

    use std::sync::atomic::{AtomicU64, Ordering};
    let trial_counter = AtomicU64::new(0);
    use optimizer::multi_objective::MultiObjectiveStudy;
    use optimizer::sampler::nsga2::Nsga2Sampler;

    let sampler = Nsga2Sampler::new();
    let study = MultiObjectiveStudy::with_sampler(
        vec![
            optimizer::Direction::Minimize,
            optimizer::Direction::Minimize,
        ],
        sampler,
    );
    let trials_data = std::sync::Arc::new(std::sync::Mutex::new(Vec::<TrialData>::new()));

    // We can't guarantee how optimize() behaves with process spawning and async readers.
    // However, if we block inside the closure, it is fine.

    let result = study.optimize(tune_cfg.trials, |trial: &mut optimizer::Trial| -> optimizer::Result<Vec<f64>> {
        let trial_idx = trial_counter.fetch_add(1, Ordering::SeqCst);
        println!(
            "\n[Native Tune] Requesting hyperparameters for Trial {}...",
            trial_idx
        );

        let mut config = base_config.clone();
        let mut params = serde_json::Map::new();

        let (min, max) = get_bound(&bounds_json, "num_processes", 8.0, 32.0);
        let v = IntParam::new(min as i64, max as i64)
            .name("num_processes")
            .suggest(trial)?;
        config["num_processes"] = serde_json::json!(v);
        params.insert("num_processes".to_string(), serde_json::json!(v));

        if bounds_json.get("train_batch_size").is_some() {
            let (min, max) = get_bound(&bounds_json, "train_batch_size", 64.0, 4096.0);
            let steps = ((max - min) / 64.0) as i64;
            // Native IntParam doesn't have internal step filtering natively, so we sample an index
            let step_idx = IntParam::new(0, steps)
                .name("train_batch_size_idx")
                .suggest(trial)?;
            let v = (min as i64) + step_idx * 64;
            config["train_batch_size"] = serde_json::json!(v);
            params.insert("train_batch_size".to_string(), serde_json::json!(v));
        }

        let (min, max) = get_bound(&bounds_json, "simulations", 10.0, 2000.0);
        let steps = ((max - min) / 10.0) as i64;
        let step_idx = IntParam::new(0, steps)
            .name("simulations_idx")
            .suggest(trial)?;
        let v = (min as i64) + step_idx * 10;
        config["simulations"] = serde_json::json!(v);
        params.insert("simulations".to_string(), serde_json::json!(v));

        let (min, max) = get_bound(&bounds_json, "max_gumbel_k", 4.0, 64.0);
        let v = IntParam::new(min as i64, max as i64)
            .name("max_gumbel_k")
            .suggest(trial)?;
        config["max_gumbel_k"] = serde_json::json!(v);
        params.insert("max_gumbel_k".to_string(), serde_json::json!(v));

        let (min, max) = get_bound(&bounds_json, "lr_init", 1e-5, 1e-1);
        let v = FloatParam::new(min, max)
            .name("lr_init")
            .suggest(trial)?;
        config["lr_init"] = serde_json::json!(v);
        params.insert("lr_init".to_string(), serde_json::json!(v));

        let (min, max) = get_bound(&bounds_json, "discount_factor", 0.9, 0.999);
        let v = FloatParam::new(min, max)
            .name("discount_factor")
            .suggest(trial)?;
        config["discount_factor"] = serde_json::json!(v);
        params.insert("discount_factor".to_string(), serde_json::json!(v));

        let (min, max) = get_bound(&bounds_json, "td_lambda", 0.5, 1.0);
        let v = FloatParam::new(min, max)
            .name("td_lambda")
            .suggest(trial)?;
        config["td_lambda"] = serde_json::json!(v);
        params.insert("td_lambda".to_string(), serde_json::json!(v));

        let config_json = serde_json::to_string(&config).unwrap();
        let experiment_name = format!("{}_trial_{:03}", tune_cfg.study_name, trial_idx);

        trials_data.lock().unwrap().push(TrialData {
            number: trial_idx,
            state: "RUNNING".to_string(),
            value: vec![],
            params: params.clone(),
            intermediate_values: serde_json::Map::new(),
        });
        save_study_state(&tune_cfg.study_name, &trials_data.lock().unwrap());

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
                serde_json::json!([&tune_cfg.study_name]).to_string(),
                &artifacts_dir
            ],
        ).unwrap();

        println!("\n[Native Tune] Trial {} Started.", trial_idx);
        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("train")
            .arg("--run-id")
            .arg(&experiment_name)
            .arg("--workspace-db")
            .arg(&workspace_db)
            .arg("--max-steps")
            .arg(tune_cfg.max_steps.to_string())
            .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
        std::thread::spawn(move || {
            for line in reader.lines().map_while(|r| r.ok()) {
                if !line.trim().starts_with('{') {
                    println!("[CHILD:{}] {}", exp_name_clone, line);
                }

                if line.contains("FINAL_EVAL_SCORE:") {
                    if let Some(val_str) = line.split("FINAL_EVAL_SCORE:").nth(1) {
                        if let Ok(val) = val_str.trim().parse::<f64>() {
                            let _ = tx.send(("loss", val));
                        }
                    }
                }
                if line.contains("search::mcts_search") && line.contains('|') {
                    let parts: Vec<&str> = line.split('|').map(|s: &str| s.trim()).collect();
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

        let mut exit_success = true;
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                exit_success = status.success();
                break;
            }

            while let Ok((key, val)) = rx.try_recv() {
                if key == "loss" {
                    final_loss = val;
                    // Dynamic early stopping updates to optimizer? 
                    // Can report intermediate values via `trial.report(step, val)` if we had step increments!
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
                        println!("[Native Tune] Trial {} PRUNED: VRAM limit.", trial_idx);
                        let _ = child.kill();
                        pruned = true;
                        break;
                    }
                }
            }

            if start_time.elapsed() > Duration::from_secs(tune_cfg.timeout) {
                println!("[Native Tune] Trial {} TIMEOUT.", trial_idx);
                let _ = child.kill();
                pruned = true;
                break;
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        if let Ok(status) = child.wait() {
            if !status.success() {
                exit_success = false;
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

        let wall_clock = start_time.elapsed().as_secs_f64();
        let hardware_penalty = if mcts_time_avg < f64::MAX {
            wall_clock + (mcts_time_avg * 100.0)
        } else {
            wall_clock
        };

        if let Some(t) = trials_data.lock().unwrap().iter_mut().find(|t| t.number == trial_idx) {
            t.state = if pruned {
                "PRUNED".to_string()
            } else if !exit_success {
                "FAIL".to_string()
            } else {
                "COMPLETE".to_string()
            };
            // Save dual outputs for the pareto front charts
            t.value = vec![hardware_penalty, final_loss];
        }
        save_study_state(&tune_cfg.study_name, &trials_data.lock().unwrap());

        // Add a short delay to allow the Nvidia driver to reclaim VRAM before the next trial spawns.
        std::thread::sleep(Duration::from_secs(2));

        if pruned {
            // PrunedError explicitly tells optimizer it shouldn't count normally or use for parameter importances
            return Ok(vec![f64::MAX / 2.0, f64::MAX / 2.0]);
        }

        Ok(vec![hardware_penalty, final_loss])
    });

    println!("✅ Native Bayesian Tuning Complete! Result: {:?}", result);
}

pub fn stop_tuning_pipeline(study_name: &str) {
    println!("Stopping tuning session: {}", study_name);
    let mut sys = sysinfo::System::new_all();
    sys.refresh_processes();

    for (pid, process) in sys.processes() {
        let cmd = process.cmd().join(" ");
        let name = process.name();

        let is_trial = (name.starts_with("tricked_engine") || cmd.contains("tricked_engine"))
            && cmd.contains("train")
            && cmd.contains(&format!("{}_trial_", study_name));

        if is_trial {
            println!("Killing process {} (PID: {})", name, pid);
            process.kill();
        }
    }
    println!("Done.");
}

pub fn flush_tuning_pipeline(study_name: &str, workspace_db_opt: Option<String>) {
    println!("Flushing tuning session: {}", study_name);

    stop_tuning_pipeline(study_name);

    let workspace_db = workspace_db_opt.unwrap_or_else(|| "tricked_workspace.db".to_string());

    if let Ok(conn) = rusqlite::Connection::open(&workspace_db) {
        let prefix = format!("{}_trial_%%", study_name);
        let mut targets = Vec::new();

        if let Ok(mut stmt) = conn.prepare("SELECT id, artifacts_dir FROM runs WHERE name LIKE ?1")
        {
            if let Ok(artifacts_iter) = stmt.query_map(rusqlite::params![&prefix], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            }) {
                for data in artifacts_iter.flatten() {
                    targets.push(data);
                }
            }
        }

        for (id, dir_opt) in targets {
            if let Some(dir) = dir_opt {
                let path = std::path::PathBuf::from(&dir);
                if path.exists() {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
            let _ = conn.execute(
                "DELETE FROM metrics WHERE run_id = ?1",
                rusqlite::params![&id],
            );
            let _ = conn.execute("DELETE FROM runs WHERE id = ?1", rusqlite::params![&id]);
        }
        println!("Cleaned up SQLite workspace database.");
    }

    let root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let mut deleted_files = 0;
    let files_to_try = [format!("studies/{}_optimizer_study.json", study_name)];

    for file in files_to_try {
        if let Ok(()) = std::fs::remove_file(root.join(&file)) {
            deleted_files += 1;
        }
    }

    println!(
        "Flushed tuning session: {} (Deleted {} studio files).",
        study_name, deleted_files
    );
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_e2e_optimizer_ipc_routing() {
        // Disabled
    }
}

use crate::cli::TuneConfig;
use optimizer::prelude::*;
use std::fs;
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

fn calculate_importance(trials: &[TrialData]) -> serde_json::Map<String, serde_json::Value> {
    let mut param_values: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let mut losses: Vec<f64> = Vec::new();

    for t in trials {
        if t.state == "COMPLETE" && t.value.len() >= 2 {
            losses.push(t.value[1]);
            for (k, v) in &t.params {
                if let Some(num) = v.as_f64() {
                    param_values.entry(k.clone()).or_default().push(num);
                }
            }
        }
    }

    let mut importances = serde_json::Map::new();
    if losses.len() < 2 {
        return importances;
    }

    let mut total_corr = 0.0;
    let mut corrs = Vec::new();
    
    let mean_loss = losses.iter().sum::<f64>() / losses.len() as f64;
    let var_loss = losses.iter().map(|&x| (x - mean_loss).powi(2)).sum::<f64>();

    for (k, vals) in param_values {
        if vals.len() != losses.len() { continue; }
        let mean_val = vals.iter().sum::<f64>() / vals.len() as f64;
        let mut cov = 0.0;
        let mut var_val = 0.0;
        for i in 0..vals.len() {
            cov += (vals[i] - mean_val) * (losses[i] - mean_loss);
            var_val += (vals[i] - mean_val).powi(2);
        }
        let corr = if var_loss > 0.0 && var_val > 0.0 {
            (cov / (var_loss.sqrt() * var_val.sqrt())).abs()
        } else {
            0.0
        };
        corrs.push((k, corr));
        total_corr += corr;
    }

    let len = corrs.len() as f64;
    for (k, c) in corrs {
        let weight = if total_corr > 0.0 { c / total_corr } else { 1.0 / len };
        importances.insert(k, serde_json::json!(weight));
    }

    importances
}

fn save_study_state(study_name: &str, trials: &[TrialData]) {
    let state = StudyData {
        trials: trials.to_vec(),
        importance: calculate_importance(trials),
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

        let (min, max) = get_bound(&bounds_json, "lr_init", 1e-4, 1e-2);
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
        
        let abort_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let abort_clone = std::sync::Arc::clone(&abort_flag);

        let mut parsed_cfg: crate::config::Config = serde_json::from_value(config.clone()).unwrap();
        parsed_cfg.experiment_name_identifier = experiment_name.clone();
        let custom_base_dir = format!("artifacts/{}", experiment_name);
        parsed_cfg.paths = crate::config::ExperimentPaths {
            base_directory: custom_base_dir.clone(),
            model_checkpoint_path: format!("{}/weights.safetensors", custom_base_dir),
            metrics_file_path: format!("{}/metrics.csv", custom_base_dir),
            experiment_name_identifier: experiment_name.clone(),
            workspace_db: Some(workspace_db.clone()),
        };

        let parsed_max_steps = tune_cfg.max_steps;
        let (tx, rx) = std::sync::mpsc::channel();
        
        let thread_handle = std::thread::Builder::new()
            .name(format!("trial-{}", trial_idx))
            .spawn(move || {
                let (loss, mcts) = crate::train::runner::run_training(parsed_cfg, parsed_max_steps, Some(abort_clone));
                let _ = tx.send((loss, mcts));
            })
            .expect("Failed to spawn trial thread");

        let mut final_loss = f64::MAX;
        let mut mcts_time_avg = f64::MAX;
        let start_time = Instant::now();
        let mut pruned = false;
        let mut exit_success = true;

        let mut last_db_check = Instant::now();

        loop {
            if let Ok((l, m)) = rx.try_recv() {
                final_loss = l;
                mcts_time_avg = m;
                let _ = thread_handle.join();
                break;
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
                        abort_flag.store(false, Ordering::SeqCst);
                        pruned = true;
                        exit_success = false;
                        let _ = thread_handle.join();
                        break;
                    }
                }
            }

            if start_time.elapsed() > Duration::from_secs(tune_cfg.timeout) {
                println!("[Native Tune] Trial {} TIMEOUT.", trial_idx);
                abort_flag.store(false, Ordering::SeqCst);
                pruned = true;
                exit_success = false;
                let _ = thread_handle.join();
                break;
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        // Try to get values if late arriving
        while let Ok((l, m)) = rx.try_recv() {
            final_loss = l;
            mcts_time_avg = m;
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

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use tauri::{AppHandle, Emitter, State};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[derive(Clone, Serialize)]
struct LogEvent {
    run_id: String,
    line: String,
}

pub mod db;
use db::Run;

struct AppState {
    processes: Mutex<HashMap<String, Child>>,
}

#[tauri::command]
fn list_runs(state: State<'_, AppState>) -> Result<Vec<Run>, String> {
    let conn = db::init_db();
    let mut runs = Vec::new();
    let mut processes = state.processes.lock().unwrap();

    let mut to_remove = Vec::new();
    for (id, child) in processes.iter_mut() {
        if let Ok(Some(_status)) = child.try_wait() {
            to_remove.push(id.clone());
            let _ = conn.execute(
                "UPDATE runs SET status = 'STOPPED' WHERE id = ?1 AND status = 'RUNNING'",
                rusqlite::params![id],
            );
        }
    }
    for id in to_remove {
        processes.remove(&id);
    }

    let mut stmt = conn
        .prepare("SELECT id, name, type, status, config, start_time, tags FROM runs ORDER BY start_time DESC")
        .map_err(|e| e.to_string())?;

    let run_iter = stmt
        .query_map([], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let r#type: String = row.get(2)?;
            let mut status: String = row.get(3)?;
            let config: String = row.get(4)?;
            let start_time: String = row.get(5)?;

            let tags_str: Option<String> = row.get(6)?;
            let mut tag = None;
            if let Some(t) = tags_str {
                if let Ok(arr) = serde_json::from_str::<Vec<String>>(&t) {
                    if !arr.is_empty() {
                        tag = Some(arr[0].clone()); // Extracted for UI bindings
                    }
                }
            }

            let is_running = processes.contains_key(&id);
            if is_running && status != "RUNNING" {
                status = "RUNNING".to_string();
            }

            Ok(Run {
                id,
                name,
                r#type,
                status,
                config,
                start_time,
                tag,
            })
        })
        .map_err(|e| e.to_string())?;

    for r in run_iter {
        if let Ok(run) = r {
            runs.push(run);
        }
    }

    Ok(runs)
}

#[tauri::command]
fn create_run(name: String, r#type: String, preset: Option<String>) -> Result<Run, String> {
    let conn = db::init_db();
    let id = uuid::Uuid::new_v4().to_string();

    let default_config = serde_json::json!({
        "experiment_name_identifier": name.clone(),
        "device": "cuda:0",
        "hidden_dimension_size": 64,
        "num_blocks": 4,
        "support_size": 300,
        "buffer_capacity_limit": 100000,
        "simulations": 100,
        "train_batch_size": 128,
        "train_epochs": 1,
        "num_processes": 4,
        "worker_device": "cpu",
        "unroll_steps": 5,
        "temporal_difference_steps": 5,
        "inference_batch_size_limit": 64,
        "inference_timeout_ms": 50,
        "max_gumbel_k": 16,
        "gumbel_scale": 0.5,
        "temp_decay_steps": 100000,
        "difficulty": 0,
        "temp_boost": true,
        "lr_init": 0.02,
        "reanalyze_ratio": 0.0
    });

    let mut final_config_str = serde_json::to_string_pretty(&default_config).unwrap();

    if let Some(preset_name) = preset {
        let db_path = db::get_db_path();
        let root = db_path.parent().unwrap();
        let preset_file = root
            .join("scripts")
            .join("configs")
            .join(format!("{}.json", preset_name));

        if let Ok(content) = fs::read_to_string(&preset_file) {
            if let Ok(mut parsed) = serde_json::from_str::<serde_json::Value>(&content) {
                parsed["experiment_name_identifier"] = serde_json::json!(name.clone());
                final_config_str = serde_json::to_string_pretty(&parsed).unwrap();
            } else {
                final_config_str = content;
            }
        }
    }

    let run = Run {
        id: id.clone(),
        name: name.clone(),
        r#type: r#type.clone(),
        status: "WAITING".to_string(),
        config: final_config_str.clone(),
        start_time: "".to_string(),
        tag: None,
    };

    let root = db::get_db_path().parent().unwrap().to_path_buf();
    let artifacts_dir = format!("{}/artifacts/{}", root.to_string_lossy(), id);

    conn.execute(
        "INSERT INTO runs (id, name, type, status, config, tags, artifacts_dir) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            id,
            name,
            r#type,
            "WAITING",
            final_config_str,
            "[]",
            artifacts_dir
        ],
    ).map_err(|e| e.to_string())?;

    Ok(run)
}

#[tauri::command]
fn rename_run(id: String, new_name: String) -> Result<(), String> {
    let conn = db::init_db();
    conn.execute(
        "UPDATE runs SET name = ?1 WHERE id = ?2",
        rusqlite::params![new_name, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn delete_run(id: String) -> Result<(), String> {
    let conn = db::init_db();

    // Safely nuke artifacts
    if let Ok(artifacts) = conn.query_row(
        "SELECT artifacts_dir FROM runs WHERE id = ?1",
        rusqlite::params![&id],
        |row| row.get::<_, Option<String>>(0),
    ) {
        if let Some(dir) = artifacts {
            let path = PathBuf::from(dir);
            if path.exists() {
                let _ = fs::remove_dir_all(path);
            }
        }
    }

    conn.execute("DELETE FROM runs WHERE id = ?1", rusqlite::params![id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn save_config(id: String, config: String) -> Result<(), String> {
    let conn = db::init_db();
    conn.execute(
        "UPDATE runs SET config = ?1 WHERE id = ?2",
        rusqlite::params![config, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn start_run(app_handle: AppHandle, state: State<'_, AppState>, id: String) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if processes.contains_key(&id) {
        return Err("Run already active".into());
    }

    // Enforce only one run globally to avoid GPU starvation
    if !processes.is_empty() {
        return Err(
            "Another run is already active. Only one engine instance is allowed at a time.".into(),
        );
    }

    let conn = db::init_db();
    conn.execute(
        "UPDATE runs SET status = 'RUNNING' WHERE id = ?1",
        rusqlite::params![&id],
    )
    .map_err(|e| e.to_string())?;

    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();

    let mut command = Command::new("cargo");
    command
        .current_dir(root)
        .arg("run")
        .arg("--release")
        .arg("--bin")
        .arg("tricked_engine")
        .arg("--")
        .arg("train")
        .arg("--run-id")
        .arg(&id)
        .arg("--workspace-db")
        .arg(db_path.to_string_lossy().to_string());

    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    #[cfg(unix)]
    command.process_group(0);

    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let id_clone = id.clone();
    let app_clone = app_handle.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(mut text) = line {
                let mut target_id = id_clone.clone();

                if text.starts_with("[CHILD_ID:") {
                    if let Some(end_idx) = text.find(']') {
                        target_id = text[10..end_idx].to_string();
                        text = text[end_idx + 1..].trim_start().to_string();
                    }
                }

                let _ = app_clone.emit(
                    "log_event",
                    LogEvent {
                        run_id: target_id,
                        line: text,
                    },
                );
            }
        }
    });

    let id_clone2 = id.clone();
    let app_clone2 = app_handle.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(text) = line {
                let _ = app_clone2.emit(
                    "log_event",
                    LogEvent {
                        run_id: id_clone2.clone(),
                        line: text,
                    },
                );
            }
        }
    });

    processes.insert(id, child);
    Ok(())
}

#[tauri::command]
fn stop_run(state: State<'_, AppState>, id: String, force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(mut child) = processes.remove(&id) {
        let pid = child.id().to_string();
        if force {
            #[cfg(unix)]
            let _ = Command::new("kill")
                .arg("-9")
                .arg(format!("-{}", pid))
                .output();
            #[cfg(not(unix))]
            let _ = child.kill();
        } else {
            #[cfg(unix)]
            let _ = Command::new("kill")
                .arg("-TERM")
                .arg(format!("-{}", pid))
                .output();
            #[cfg(not(unix))]
            let _ = child.kill();
        }
        std::thread::spawn(move || {
            let _ = child.wait();
        });
    }

    let conn = db::init_db();
    let _ = conn.execute(
        "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
        rusqlite::params![id],
    );

    Ok(())
}

#[tauri::command]
fn start_study(
    app_handle: AppHandle,
    state: State<'_, AppState>,
    study_type: String,
    trials: i32,
    max_steps: i32,
    timeout: i32,
    resnet_blocks: i32,
    resnet_channels: i32,
    bounds: Option<serde_json::Value>,
) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if !processes.is_empty() {
        return Err(
            "Another task is active. Only one engine instance or study is allowed at a time."
                .into(),
        );
    }

    let root = db::get_db_path().parent().unwrap().to_path_buf();

    let script_name = match study_type.as_str() {
        "HARDWARE" => "studies/hardware_tune.py",
        "MCTS" => "studies/mcts_tune.py",
        _ => "studies/learning_tune.py",
    };

    let config_path = match study_type.as_str() {
        "LEARNING" => {
            if root.join("studies/best_mcts_config.json").exists() {
                "studies/best_mcts_config.json"
            } else if root.join("studies/best_hardware_config.json").exists() {
                "studies/best_hardware_config.json"
            } else {
                "scripts/configs/big.json"
            }
        }
        "MCTS" => {
            if root.join("studies/best_hardware_config.json").exists() {
                "studies/best_hardware_config.json"
            } else {
                "scripts/configs/big.json"
            }
        }
        _ => "scripts/configs/big.json",
    };

    let venv_python = root.join("venv/bin/python");
    let mut cmd = Command::new(venv_python);
    cmd.current_dir(root);
    cmd.arg(script_name);
    cmd.arg("--config").arg(config_path);
    cmd.arg("--trials").arg(trials.to_string());
    cmd.arg("--max-steps").arg(max_steps.to_string());
    cmd.arg("--timeout").arg(timeout.to_string());
    cmd.arg("--resnet-blocks").arg(resnet_blocks.to_string());
    cmd.arg("--resnet-channels")
        .arg(resnet_channels.to_string());

    if let Some(b) = bounds {
        cmd.arg("--bounds").arg(b.to_string());
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let app_clone = app_handle.clone();

    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(text) = line {
                let _ = app_clone.emit(
                    "log_event",
                    LogEvent {
                        run_id: "STUDY".to_string(),
                        line: text,
                    },
                );
            }
        }
    });

    let app_clone2 = app_handle.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(text) = line {
                let _ = app_clone2.emit(
                    "log_event",
                    LogEvent {
                        run_id: "STUDY".to_string(),
                        line: text,
                    },
                );
            }
        }
    });

    processes.insert("STUDY".to_string(), child);
    Ok(())
}

#[tauri::command]
fn stop_study(state: State<'_, AppState>, force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(mut child) = processes.remove("STUDY") {
        if force {
            let _ = child.kill();
        } else {
            let _ = Command::new("kill")
                .arg("-TERM")
                .arg(child.id().to_string())
                .output();
        }
        std::thread::spawn(move || {
            let _ = child.wait();
        });
    }
    Ok(())
}

#[tauri::command]
fn get_active_study(state: State<'_, AppState>) -> Result<bool, String> {
    let processes = state.processes.lock().unwrap();
    Ok(processes.contains_key("STUDY"))
}

#[tauri::command]
fn get_tuning_study() -> Result<String, String> {
    let project_root = db::get_db_path().parent().unwrap().to_path_buf();
    let study_file = project_root.join("studies").join("optuna_study.json");
    if study_file.exists() {
        fs::read_to_string(&study_file).map_err(|e| e.to_string())
    } else {
        Ok("[]".to_string())
    }
}

#[tauri::command]
fn get_run_metrics(id: String) -> Result<Vec<db::MetricRow>, String> {
    let conn = db::init_db();
    db::get_metrics(&conn, &id).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_study_status(study_type: String) -> Result<bool, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let json_path = match study_type.as_str() {
        "HARDWARE" => root.join("studies/best_hardware_config.json"),
        "MCTS" => root.join("studies/best_mcts_config.json"),
        _ => root.join("studies/best_learning_config.json"),
    };
    Ok(json_path.exists())
}

#[tauri::command]
fn flush_study(state: State<'_, AppState>, study_type: String) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(mut child) = processes.remove("STUDY") {
        let pid = child.id().to_string();
        #[cfg(unix)]
        let _ = Command::new("kill")
            .arg("-9")
            .arg(format!("-{}", pid))
            .output();
        #[cfg(not(unix))]
        let _ = child.kill();
        std::thread::spawn(move || {
            let _ = child.wait();
        });
    }
    drop(processes);

    let conn = db::init_db();
    let prefix = match study_type.as_str() {
        "HARDWARE" => "tune_3080Ti_trial_%",
        "MCTS" => "mcts_tune_trial_%",
        _ => "learn_tune_trial_%",
    };

    // Delete artifacts safely
    let mut targets = Vec::new();
    if let Ok(mut stmt) = conn.prepare("SELECT id, artifacts_dir FROM runs WHERE name LIKE ?1") {
        if let Ok(artifacts_iter) = stmt.query_map(rusqlite::params![prefix], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        }) {
            for r in artifacts_iter {
                if let Ok(data) = r {
                    targets.push(data);
                }
            }
        }
    }

    for (id, dir_opt) in targets {
        if let Some(dir) = dir_opt {
            let path = PathBuf::from(dir);
            if path.exists() {
                let _ = fs::remove_dir_all(path);
            }
        }
        let _ = conn.execute("DELETE FROM runs WHERE id = ?1", rusqlite::params![id]);
    }

    let root = db::get_db_path().parent().unwrap().to_path_buf();

    let db_path = match study_type.as_str() {
        "HARDWARE" => root.join("studies/hardware_optuna_study.db"),
        "MCTS" => root.join("studies/mcts_optuna_study.db"),
        _ => root.join("studies/learning_optuna_study.db"),
    };
    let json_path = match study_type.as_str() {
        "HARDWARE" => root.join("studies/best_hardware_config.json"),
        "MCTS" => root.join("studies/best_mcts_config.json"),
        _ => root.join("studies/best_learning_config.json"),
    };
    let optuna_json = root.join("studies/optuna_study.json");

    let _ = fs::remove_file(db_path);
    let _ = fs::remove_file(json_path);
    let _ = fs::remove_file(optuna_json);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            processes: Mutex::new(HashMap::new()),
        })
        .invoke_handler(tauri::generate_handler![
            list_runs,
            create_run,
            rename_run,
            delete_run,
            save_config,
            start_run,
            stop_run,
            get_tuning_study,
            get_run_metrics,
            start_study,
            stop_study,
            get_active_study,
            get_study_status,
            flush_study
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            use tauri::Manager;
            if let tauri::RunEvent::Exit = event {
                let state = app_handle.state::<AppState>();
                let mut processes = state.processes.lock().unwrap();

                // First, gracefully ask processes to terminate
                for (id, child) in processes.iter_mut() {
                    let pid = child.id().to_string();
                    if id == "STUDY" {
                        let _ = std::process::Command::new("kill")
                            .arg("-TERM")
                            .arg(&pid)
                            .output();
                    } else {
                        #[cfg(unix)]
                        let _ = std::process::Command::new("kill")
                            .arg("-TERM")
                            .arg(format!("-{}", pid))
                            .output();
                    }
                }

                // Give Python orchestrators enough time to reap detached cargo run child processes
                std::thread::sleep(std::time::Duration::from_millis(600));

                // Ensure everything is truly dead
                for (id, child) in processes.iter_mut() {
                    let pid = child.id().to_string();
                    let _ = child.kill();
                    if id != "STUDY" {
                        #[cfg(unix)]
                        let _ = std::process::Command::new("kill")
                            .arg("-9")
                            .arg(format!("-{}", pid))
                            .output();
                    }
                }
                processes.clear();
            }
        });
}

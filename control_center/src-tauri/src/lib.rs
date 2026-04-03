use serde::{Deserialize, Serialize};
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

#[derive(Serialize, Deserialize, Clone)]
pub struct Run {
    pub id: String,
    pub name: String,
    pub r#type: String,
    pub status: String,
    pub config: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

struct AppState {
    processes: Mutex<HashMap<String, Child>>,
}

fn get_runs_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap();
    let root = if cwd.ends_with("src-tauri") {
        cwd.parent().unwrap().parent().unwrap().to_path_buf()
    } else if cwd.ends_with("control_center") {
        cwd.parent().unwrap().to_path_buf()
    } else {
        cwd
    };
    let runs_dir = root.join("runs");
    if !runs_dir.exists() {
        fs::create_dir_all(&runs_dir).unwrap();
    }
    runs_dir
}

#[tauri::command]
fn list_runs(state: State<'_, AppState>) -> Result<Vec<Run>, String> {
    let runs_dir = get_runs_dir();
    let mut runs = Vec::new();
    let mut processes = state.processes.lock().unwrap();

    // Clean up dead processes
    let mut to_remove = Vec::new();
    for (id, child) in processes.iter_mut() {
        if let Ok(Some(_status)) = child.try_wait() {
            to_remove.push(id.clone());
        }
    }
    for id in to_remove {
        processes.remove(&id);
    }

    if let Ok(entries) = fs::read_dir(runs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let config_file = path.join("config.json");
                let run_file = path.join("run_info.json");

                if let Ok(content) = fs::read_to_string(&run_file) {
                    if let Ok(mut run) = serde_json::from_str::<Run>(&content) {
                        let is_running = processes.contains_key(&run.id);
                        if is_running {
                            run.status = "RUNNING".to_string();
                        } else if run.status == "RUNNING" {
                            run.status = "STOPPED".to_string();
                            let _ = fs::write(&run_file, serde_json::to_string(&run).unwrap());
                        }

                        if let Ok(config_str) = fs::read_to_string(&config_file) {
                            run.config = config_str;
                        }
                        runs.push(run);
                    }
                }
            }
        }
    }
    Ok(runs)
}

#[tauri::command]
fn create_run(name: String, r#type: String, preset: Option<String>) -> Result<Run, String> {
    let id = uuid::Uuid::new_v4().to_string();
    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;

    let default_config = serde_json::json!({
        "experiment_name_identifier": name.clone(),
        "device": "cuda:0",
        "hidden_dimension_size": 256,
        "num_blocks": 10,
        "support_size": 300,
        "buffer_capacity_limit": 100000,
        "simulations": 800,
        "train_batch_size": 1024,
        "train_epochs": 1,
        "num_processes": 8,
        "worker_device": "cpu",
        "unroll_steps": 5,
        "temporal_difference_steps": 5,
        "inference_batch_size_limit": 256,
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
        let root = runs_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let preset_file = root
            .join("scripts")
            .join("configs")
            .join(format!("{}.json", preset_name));
        if let Ok(content) = fs::read_to_string(&preset_file) {
            // Apply experiment name override back
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
        name,
        r#type,
        status: "WAITING".to_string(),
        config: final_config_str,
        tag: None,
    };

    let run_info = serde_json::to_string(&run).unwrap();
    fs::write(run_dir.join("run_info.json"), run_info).map_err(|e| e.to_string())?;
    fs::write(run_dir.join("config.json"), &run.config).map_err(|e| e.to_string())?;

    Ok(run)
}

#[tauri::command]
fn rename_run(id: String, new_name: String) -> Result<(), String> {
    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    let run_file = run_dir.join("run_info.json");

    if let Ok(content) = fs::read_to_string(&run_file) {
        if let Ok(mut run) = serde_json::from_str::<Run>(&content) {
            run.name = new_name;
            let _ = fs::write(&run_file, serde_json::to_string(&run).unwrap());
        }
    }
    Ok(())
}

#[tauri::command]
fn delete_run(id: String) -> Result<(), String> {
    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    if run_dir.exists() {
        fs::remove_dir_all(run_dir).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn save_config(id: String, config: String) -> Result<(), String> {
    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    fs::write(run_dir.join("config.json"), config).map_err(|e| e.to_string())?;
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

    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    let config_path = run_dir.join("config.json");

    let mut experiment_name = id.clone();
    let run_file = run_dir.join("run_info.json");
    if let Ok(content) = fs::read_to_string(&run_file) {
        if let Ok(mut run) = serde_json::from_str::<Run>(&content) {
            experiment_name = run.name.clone();
            run.status = "RUNNING".to_string();
            let _ = fs::write(&run_file, serde_json::to_string(&run).unwrap());
        }
    }

    let root = runs_dir.parent().unwrap();
    let mut command = Command::new("cargo");
    command
        .current_dir(root)
        .arg("run")
        .arg("--release")
        .arg("--bin")
        .arg("tricked_engine")
        .arg("--")
        .arg("train")
        .arg("--experiment-name")
        .arg(&experiment_name)
        .arg("--config")
        .arg(config_path.to_string_lossy().to_string());

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

    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    let run_file = run_dir.join("run_info.json");
    if let Ok(content) = fs::read_to_string(&run_file) {
        if let Ok(mut run) = serde_json::from_str::<Run>(&content) {
            run.status = "STOPPED".to_string();
            let _ = fs::write(&run_file, serde_json::to_string(&run).unwrap());
        }
    }

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
) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if !processes.is_empty() {
        return Err(
            "Another task is active. Only one engine instance or study is allowed at a time."
                .into(),
        );
    }

    let runs_dir = get_runs_dir();
    let root = runs_dir.parent().unwrap();

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
    let runs_dir = get_runs_dir();
    let project_root = runs_dir.parent().unwrap();
    let study_file = project_root.join("studies").join("optuna_study.json");
    if study_file.exists() {
        fs::read_to_string(&study_file).map_err(|e| e.to_string())
    } else {
        Ok("[]".to_string())
    }
}

#[tauri::command]
fn get_run_metrics(id: String) -> Result<Vec<HashMap<String, String>>, String> {
    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    let run_file = run_dir.join("run_info.json");

    let mut experiment_name = id.clone();
    if let Ok(content) = fs::read_to_string(&run_file) {
        if let Ok(run) = serde_json::from_str::<Run>(&content) {
            experiment_name = run.name;
        }
    }

    let metrics_file = run_dir.join(format!("{}_metrics.csv", experiment_name));

    if !metrics_file.exists() {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(&metrics_file).map_err(|e| e.to_string())?;
    let mut lines = content.lines();

    let headers: Vec<&str> = match lines.next() {
        Some(line) => line.split(',').collect(),
        None => return Ok(Vec::new()),
    };

    let mut metrics = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let values: Vec<&str> = line.split(',').collect();
        let mut map = HashMap::new();
        for (i, value) in values.iter().enumerate() {
            if i < headers.len() {
                map.insert(headers[i].to_string(), value.to_string());
            }
        }
        metrics.push(map);
    }

    Ok(metrics)
}

#[tauri::command]
fn get_study_status(study_type: String) -> Result<bool, String> {
    let runs_dir = get_runs_dir();
    let root = runs_dir.parent().unwrap();
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

    let runs_dir = get_runs_dir();
    let root = runs_dir.parent().unwrap();

    let prefix = match study_type.as_str() {
        "HARDWARE" => "tune_3080Ti_trial_",
        "MCTS" => "mcts_tune_trial_",
        _ => "learn_tune_trial_",
    };

    if let Ok(entries) = fs::read_dir(&runs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with(prefix) {
                        let _ = fs::remove_dir_all(path);
                    }
                }
            }
        }
    }

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

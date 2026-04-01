use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use tauri::{AppHandle, Emitter, Manager, State};

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
}

struct AppState {
    processes: Mutex<HashMap<String, Child>>,
}

fn get_runs_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap();
    let root = if cwd.ends_with("control_center") {
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
fn create_run(
    name: String,
    r#type: String,
    _base_config_id: Option<String>,
) -> Result<Run, String> {
    let id = uuid::Uuid::new_v4().to_string();
    let runs_dir = get_runs_dir();
    let run_dir = runs_dir.join(&id);
    fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;

    let run = Run {
        id: id.clone(),
        name,
        r#type,
        status: "WAITING".to_string(),
        config: "{\n  \"message\": \"Empty configuration\"\n}".to_string(),
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

    let mut run_type = String::new();
    let run_file = run_dir.join("run_info.json");
    if let Ok(content) = fs::read_to_string(&run_file) {
        if let Ok(mut run) = serde_json::from_str::<Run>(&content) {
            run_type = run.r#type.clone();
            run.status = "RUNNING".to_string();
            let _ = fs::write(&run_file, serde_json::to_string(&run).unwrap());
        }
    }

    let root = runs_dir.parent().unwrap();
    let mut command = if run_type == "TUNING" {
        let mut cmd = Command::new("python3");
        cmd.current_dir(root);
        cmd.arg("studies/auto_tune_sota.py");
        cmd
    } else {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(root);
        cmd.arg("run")
            .arg("--release")
            .arg("--bin")
            .arg("tricked_engine");
        cmd.arg("--")
            .arg("--config")
            .arg(config_path.to_string_lossy().to_string());
        cmd
    };

    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let id_clone = id.clone();
    let app_clone = app_handle.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(text) = line {
                let _ = app_clone.emit(
                    "log_event",
                    LogEvent {
                        run_id: id_clone.clone(),
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
        if force {
            let _ = child.kill();
        } else {
            let _ = Command::new("kill")
                .arg("-TERM")
                .arg(child.id().to_string())
                .output();
        }
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
            stop_run
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

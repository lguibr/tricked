use crate::{db, AppState};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::thread;
use tauri::{AppHandle, Emitter, State};
use tricked_shared::models::LogEvent;

pub fn start_run_impl(
    app_handle: Option<AppHandle>,
    processes: &mut HashMap<String, Child>,
    id: String,
) -> Result<(), String> {
    if processes.contains_key(&id) {
        return Err("Run already active".into());
    }
    if !processes.is_empty() {
        return Err("Another run is already active.".into());
    }

    let conn = db::init_db();
    conn.execute(
        "UPDATE runs SET status = 'RUNNING' WHERE id = ?1",
        rusqlite::params![&id],
    )
    .map_err(|e| e.to_string())?;

    let root = db::get_db_path().parent().unwrap().to_path_buf();
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
        .arg(db::get_db_path().to_string_lossy().to_string());
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

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
                if let Some(app) = &app_clone {
                    let _ = app.emit(
                        "log_event",
                        LogEvent {
                            run_id: target_id,
                            line: text,
                        },
                    );
                }
            }
        }
    });

    let id_clone2 = id.clone();
    let app_clone2 = app_handle.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(text) = line {
                if let Some(app) = &app_clone2 {
                    let _ = app.emit(
                        "log_event",
                        LogEvent {
                            run_id: id_clone2.clone(),
                            line: text,
                        },
                    );
                }
            }
        }
    });

    processes.insert(id, child);
    Ok(())
}

#[tauri::command]
pub fn start_run(
    app_handle: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    start_run_impl(Some(app_handle), &mut state.processes.lock().unwrap(), id)
}

#[tauri::command]
pub fn stop_run(state: State<'_, AppState>, id: String, force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(mut child) = processes.remove(&id) {
        let pid = child.id().to_string();
        let signal = if force { "-9" } else { "-TERM" };
        #[cfg(unix)]
        let _ = Command::new("kill")
            .arg(signal)
            .arg(format!("-{}", pid))
            .output();
        #[cfg(not(unix))]
        let _ = child.kill();
        thread::spawn(move || {
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
pub fn start_study(
    app_handle: AppHandle,
    state: State<'_, AppState>,
    trials: i32,
    max_steps: i32,
    timeout: i32,
    resnet_blocks: i32,
    resnet_channels: i32,
    bounds: Option<serde_json::Value>,
) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if !processes.is_empty() {
        return Err("Another task is active.".into());
    }

    let root = db::get_db_path().parent().unwrap().to_path_buf();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root)
        .arg("run")
        .arg("--release")
        .arg("--bin")
        .arg("tricked_engine")
        .arg("--")
        .arg("tune")
        .arg("--config")
        .arg("scripts/configs/big.json")
        .arg("--trials")
        .arg(trials.to_string())
        .arg("--max-steps")
        .arg(max_steps.to_string())
        .arg("--timeout")
        .arg(timeout.to_string())
        .arg("--resnet-blocks")
        .arg(resnet_blocks.to_string())
        .arg("--resnet-channels")
        .arg(resnet_channels.to_string());
    if let Some(b) = bounds {
        cmd.arg("--bounds").arg(b.to_string());
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let app_clone = Some(app_handle.clone());

    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(text) = line {
                if let Some(app) = &app_clone {
                    let _ = app.emit(
                        "log_event",
                        LogEvent {
                            run_id: "STUDY".to_string(),
                            line: text,
                        },
                    );
                }
            }
        }
    });

    let app_clone2 = Some(app_handle.clone());
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(text) = line {
                if let Some(app) = &app_clone2 {
                    let _ = app.emit(
                        "log_event",
                        LogEvent {
                            run_id: "STUDY".to_string(),
                            line: text,
                        },
                    );
                }
            }
        }
    });

    processes.insert("STUDY".to_string(), child);
    Ok(())
}

#[tauri::command]
pub fn stop_study(state: State<'_, AppState>, force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(mut child) = processes.remove("STUDY") {
        let pid = child.id().to_string();
        let signal = if force { "-9" } else { "-TERM" };
        #[cfg(unix)]
        let _ = Command::new("kill")
            .arg(signal)
            .arg(format!("-{}", pid))
            .output();
        #[cfg(not(unix))]
        let _ = child.kill();
        thread::spawn(move || {
            let _ = child.wait();
        });
    }
    Ok(())
}

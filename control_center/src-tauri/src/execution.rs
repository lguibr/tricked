use crate::{db, AppState};
use std::collections::HashMap;
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;
use tricked_shared::models::LogEvent;

pub fn start_run_impl(
    app_handle: AppHandle,
    processes: &mut HashMap<String, CommandChild>,
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

    let sidecar_command = app_handle
        .shell()
        .sidecar("tricked_engine")
        .map_err(|e| e.to_string())?
        .arg("train")
        .arg("--run-id")
        .arg(&id)
        .arg("--workspace-db")
        .arg(db::get_db_path().to_string_lossy().to_string());

    let (mut rx, child) = sidecar_command.spawn().map_err(|e| e.to_string())?;

    let id_clone = id.clone();
    let app_clone = app_handle.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                tauri_plugin_shell::process::CommandEvent::Stdout(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    for line in text.lines() {
                        if let Ok(parsed_json) = serde_json::from_str::<serde_json::Value>(line) {
                            let _ = app_clone.emit("live_metric", parsed_json);
                        } else {
                            let _ = app_clone.emit(
                                "log_event",
                                LogEvent {
                                    run_id: id_clone.clone(),
                                    line: line.to_string(),
                                },
                            );
                        }
                    }
                }
                tauri_plugin_shell::process::CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    for line in text.lines() {
                        let _ = app_clone.emit(
                            "log_event",
                            LogEvent {
                                run_id: id_clone.clone(),
                                line: line.to_string(),
                            },
                        );
                    }
                }
                _ => {}
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
    start_run_impl(app_handle, &mut state.processes.lock().unwrap(), id)
}

#[tauri::command]
pub fn stop_run(state: State<'_, AppState>, id: String, force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(child) = processes.remove(&id) {
        let pid = child.pid();
        let signal = if force { "-9" } else { "-TERM" };
        #[cfg(unix)]
        let _ = std::process::Command::new("kill")
            .arg(signal)
            .arg(format!("-{}", pid))
            .output();
        #[cfg(not(unix))]
        let _ = child.kill();
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

    let mut sidecar_command = app_handle
        .shell()
        .sidecar("tricked_engine")
        .map_err(|e| e.to_string())?
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
        sidecar_command = sidecar_command.arg("--bounds").arg(b.to_string());
    }

    let (mut rx, child) = sidecar_command.spawn().map_err(|e| e.to_string())?;

    let app_clone = app_handle.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                tauri_plugin_shell::process::CommandEvent::Stdout(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    for line in text.lines() {
                        if let Ok(parsed_json) = serde_json::from_str::<serde_json::Value>(line) {
                            let _ = app_clone.emit("live_metric", parsed_json);
                        } else {
                            let _ = app_clone.emit(
                                "log_event",
                                LogEvent {
                                    run_id: "STUDY".to_string(),
                                    line: line.to_string(),
                                },
                            );
                        }
                    }
                }
                tauri_plugin_shell::process::CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    for line in text.lines() {
                        let _ = app_clone.emit(
                            "log_event",
                            LogEvent {
                                run_id: "STUDY".to_string(),
                                line: line.to_string(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
    });

    processes.insert("STUDY".to_string(), child);
    Ok(())
}

#[tauri::command]
pub fn stop_study(state: State<'_, AppState>, force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(child) = processes.remove("STUDY") {
        let pid = child.pid();
        let signal = if force { "-9" } else { "-TERM" };
        #[cfg(unix)]
        let _ = std::process::Command::new("kill")
            .arg(signal)
            .arg(format!("-{}", pid))
            .output();
        #[cfg(not(unix))]
        let _ = child.kill();
    }
    Ok(())
}

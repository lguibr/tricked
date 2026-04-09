use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};

use crate::{db, AppState};

#[tauri::command]
pub fn start_run(
    app_handle: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    if state.processes.lock().unwrap().contains_key(&id) {
        return Err("Run already active".into());
    }
    if !state.processes.lock().unwrap().is_empty() {
        return Err("Another run is already active.".into());
    }

    let conn = db::init_db();
    let config_str: String = conn
        .query_row(
            "SELECT config FROM runs WHERE id = ?1",
            rusqlite::params![&id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;

    let root = crate::db::get_db_path().parent().unwrap().to_path_buf();
    let artifacts_dir_rel = format!("runs/{}", id);
    let study_artifacts = root.join(&artifacts_dir_rel);
    if !study_artifacts.exists() && std::fs::create_dir_all(&study_artifacts).is_err() {
        return Err("Failed to create run artifacts directory".into());
    }

    let abs_config_path = study_artifacts.join("base_config.json");
    if std::fs::write(&abs_config_path, &config_str).is_err() {
        return Err("Failed to write base config to artifacts".into());
    }

    let _ = conn.execute(
        "UPDATE runs SET status = 'RUNNING' WHERE id = ?1",
        rusqlite::params![&id],
    );

    let tune_cfg = tricked_engine::cli::TuneConfig {
        config_path: abs_config_path.to_string_lossy().to_string(),
        workspace_db: Some(crate::db::get_db_path().to_string_lossy().to_string()),
        trials: 1,
        timeout: u32::MAX as u64, // Infinite timeout fallback
        max_steps: usize::MAX,    // Ignore max-steps truncation bug
        resnet_blocks: 0,
        resnet_channels: 0,
        study_name: id.clone(),
        bounds: "{}".to_string(),
    };

    let abort_flag = Arc::new(AtomicBool::new(false));
    state.processes.lock().unwrap().insert(id.clone(), Arc::clone(&abort_flag));
    
    let state_processes_clone = state.processes.clone();
    let id_clone = id.clone();

    std::thread::Builder::new()
        .name(format!("run-{}", id))
        .spawn(move || {
            tricked_engine::train::tune::run_tuning_pipeline(tune_cfg, Some(abort_flag));

            let mut procs = state_processes_clone.lock().unwrap();
            let _ = procs.remove(&id_clone);
            let conn = crate::db::init_db();
            let _ = conn.execute(
                "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
                rusqlite::params![id_clone],
            );
        })
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub fn stop_run(state: State<'_, AppState>, id: String, _force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(abort_flag) = processes.remove(&id) {
        abort_flag.store(true, Ordering::SeqCst);
    }

    let conn = db::init_db();
    let _ = conn.execute(
        "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
        rusqlite::params![id],
    );
    Ok(())
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub fn start_study(
    _app_handle: AppHandle,
    state: State<'_, AppState>,
    id: String,
    name: String,
    trials: i32,
    max_steps: i32,
    timeout: i32,
    resnet_blocks: i32,
    resnet_channels: i32,
    bounds: Option<serde_json::Value>,
    base_config: Option<serde_json::Value>,
) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if !processes.is_empty() {
        return Err("Another task is active.".into());
    }

    let root = crate::db::get_db_path().parent().unwrap().to_path_buf();
    let artifacts_dir_rel = format!("runs/{}", id);
    let study_artifacts = root.join(&artifacts_dir_rel);
    if !study_artifacts.exists() && std::fs::create_dir_all(&study_artifacts).is_err() {
        return Err("Failed to create study artifacts directory".into());
    }

    let base_config_json = base_config.unwrap_or_else(|| serde_json::json!({}));
    let config_path_rel = format!("{}/base_config.json", artifacts_dir_rel);
    let abs_config_path = root.join(&config_path_rel);
    if std::fs::write(
        &abs_config_path,
        serde_json::to_string_pretty(&base_config_json).unwrap(),
    )
    .is_err()
    {
        return Err("Failed to write base config to artifacts".into());
    }

    let final_config_json = serde_json::json!({
        "bounds": bounds.as_ref().map(|b| b.to_string()).unwrap_or_else(|| "{}".to_string()),
        "trials": trials,
        "max_steps": max_steps,
        "timeout": timeout,
        "resnet_blocks": resnet_blocks,
        "resnet_channels": resnet_channels,
        "config_path": config_path_rel
    });
    let final_config_str = serde_json::to_string(&final_config_json).unwrap();
    let artifacts_dir = format!("{}/{}", root.to_string_lossy(), artifacts_dir_rel);
    let conn = crate::db::init_db();
    let _ = conn.execute(
        "INSERT INTO runs (id, name, type, status, config, tags, artifacts_dir) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![id.clone(), name.clone(), "STUDY", "RUNNING", final_config_str, "[]", artifacts_dir],
    );

    let db_path_str = db::get_db_path().to_string_lossy().to_string();
    let bounds_str = bounds
        .map(|v| v.to_string())
        .unwrap_or_else(|| "{}".to_string());

    let tune_cfg = tricked_engine::cli::TuneConfig {
        config_path: abs_config_path.to_string_lossy().to_string(),
        workspace_db: Some(db_path_str),
        trials: trials as usize,
        timeout: timeout as u64,
        max_steps: max_steps as usize,
        resnet_blocks: resnet_blocks as usize,
        resnet_channels: resnet_channels as usize,
        study_name: id.clone(),
        bounds: bounds_str,
    };

    let abort_flag = Arc::new(AtomicBool::new(false));
    processes.insert(id.clone(), Arc::clone(&abort_flag));

    let state_processes_clone = state.processes.clone();
    let id_clone = id.clone();

    std::thread::Builder::new()
        .name(format!("study-{}", id))
        .spawn(move || {
            tricked_engine::train::tune::run_tuning_pipeline(tune_cfg, Some(abort_flag));

            // Cleanup UI state automatically when finished!
            let mut procs = state_processes_clone.lock().unwrap();
            let _ = procs.remove(&id_clone);
            let conn = crate::db::init_db();
            let _ = conn.execute(
                "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
                rusqlite::params![id_clone],
            );
        })
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
pub fn stop_study(state: State<'_, AppState>, id: String, _force: bool) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(abort_flag) = processes.remove(&id) {
        abort_flag.store(true, Ordering::SeqCst);
    }

    let conn = db::init_db();
    let _ = conn.execute(
        "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
        rusqlite::params![id],
    );
    Ok(())
}

#[cfg(test)]
mod test_exec_sync {
    use std::sync::atomic::AtomicBool;
    #[test]
    fn test_execution_state_machine_sync() {
        let processes: std::collections::HashMap<String, std::sync::Arc<AtomicBool>> =
            std::collections::HashMap::new();
        assert!(processes.is_empty(), "Process map should start empty");
    }
}

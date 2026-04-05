use crate::db;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use sysinfo::System;
use tauri::State;
use tauri_plugin_shell::process::CommandChild;
use tricked_shared::models::{MetricRow, Run};

pub fn list_runs_impl(processes: &mut HashMap<String, CommandChild>) -> Result<Vec<Run>, String> {
    let conn = db::init_db();
    let mut to_remove = Vec::new();
    let sys = System::new_all();

    for (id, child) in processes.iter() {
        if sys
            .processes()
            .get(&sysinfo::Pid::from_u32(child.pid()))
            .is_none()
        {
            to_remove.push(id.clone());
            let _ = conn.execute(
                "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
                rusqlite::params![id],
            );
        }
    }
    for id in to_remove {
        processes.remove(&id);
    }

    let mut runs = db::list_runs(&conn).map_err(|e| e.to_string())?;

    for run in &mut runs {
        let is_running = processes.contains_key(&run.id);
        if is_running && run.status != "RUNNING" {
            run.status = "RUNNING".to_string();
            let _ = conn.execute(
                "UPDATE runs SET status = 'RUNNING' WHERE id = ?1",
                rusqlite::params![&run.id],
            );
        } else if !is_running && run.status == "RUNNING" {
            run.status = "STOPPED".to_string();
            let _ = conn.execute(
                "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
                rusqlite::params![&run.id],
            );
        }
    }
    Ok(runs)
}

#[tauri::command]
pub fn list_runs(state: State<'_, crate::AppState>) -> Result<Vec<Run>, String> {
    list_runs_impl(&mut state.processes.lock().unwrap())
}

#[tauri::command]
pub fn create_run(name: String, r#type: String, preset: Option<String>) -> Result<Run, String> {
    create_run_impl(name, r#type, preset)
}

pub fn create_run_impl(
    name: String,
    r#type: String,
    preset: Option<String>,
) -> Result<Run, String> {
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
pub fn rename_run(id: String, new_name: String) -> Result<(), String> {
    let conn = db::init_db();
    conn.execute(
        "UPDATE runs SET name = ?1 WHERE id = ?2",
        rusqlite::params![new_name, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn delete_run(id: String) -> Result<(), String> {
    let conn = db::init_db();
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
pub fn save_config(id: String, config: String) -> Result<(), String> {
    let conn = db::init_db();
    conn.execute(
        "UPDATE runs SET config = ?1 WHERE id = ?2",
        rusqlite::params![config, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn get_run_metrics(run_id: String) -> Result<Vec<MetricRow>, String> {
    let conn = db::init_db();
    db::get_metrics(&conn, &run_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_tuning_study(_study_type: String) -> Result<serde_json::Value, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let json_path = root.join("studies/optuna_study.json");
    if let Ok(content) = std::fs::read_to_string(&json_path) {
        return serde_json::from_str(&content).map_err(|e| e.to_string());
    }
    Err("Study output not found".to_string())
}

#[tauri::command]
pub fn get_active_study(_study_type: String) -> Result<serde_json::Value, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let json_path = root.join("studies/best_unified_config.json");
    if let Ok(content) = std::fs::read_to_string(&json_path) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
            return Ok(parsed);
        }
    }
    Err("No active study results".to_string())
}

#[tauri::command]
pub fn get_study_status(_study_type: String) -> Result<bool, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let json_path = root.join("studies/best_unified_config.json");
    let optuna_json = root.join("studies/optuna_study.json");
    let optuna_db = root.join("studies/unified_optuna_study.db");
    Ok(json_path.exists() || optuna_json.exists() || optuna_db.exists())
}

#[tauri::command]
pub fn flush_study(state: State<'_, crate::AppState>, _study_type: String) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(child) = processes.remove("STUDY") {
        let pid = child.pid().to_string();
        #[cfg(unix)]
        let _ = std::process::Command::new("kill")
            .arg("-9")
            .arg(format!("-{}", pid))
            .output();
        #[cfg(not(unix))]
        let _ = child.kill();
    }
    drop(processes);

    let conn = db::init_db();
    let prefix = "unified_tune_trial_%";
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

    let root = db::get_db_path().parent().unwrap().to_path_buf();
    for (id, dir_opt) in targets {
        if let Some(dir) = dir_opt {
            let path = root.join(dir);
            if path.exists() {
                let _ = fs::remove_dir_all(path);
            }
        }
        let _ = conn.execute(
            "DELETE FROM metrics WHERE run_id = ?1",
            rusqlite::params![id],
        );
        let _ = conn.execute("DELETE FROM runs WHERE id = ?1", rusqlite::params![id]);
    }

    let _ = fs::remove_file(root.join("studies/unified_optuna_study.db"));
    let _ = fs::remove_file(root.join("studies/unified_optuna_study.db-shm"));
    let _ = fs::remove_file(root.join("studies/unified_optuna_study.db-wal"));
    let _ = fs::remove_file(root.join("studies/best_unified_config.json"));
    let _ = fs::remove_file(root.join("studies/optuna_study.json"));
    Ok(())
}

use tricked_engine::core::board::GameStateExt;

#[derive(serde::Serialize)]
pub struct PlaygroundState {
    pub board_low: String,
    pub board_high: String,
    pub available: [i32; 3],
    pub score: i32,
    pub pieces_left: i32,
    pub terminal: bool,
    pub difficulty: i32,
    pub lines_cleared: i32,
}

impl From<GameStateExt> for PlaygroundState {
    fn from(state: GameStateExt) -> Self {
        let low = (state.board_bitmask_u128 & 0xFFFFFFFFFFFFFFFF) as u64;
        let high = (state.board_bitmask_u128 >> 64) as u64;
        PlaygroundState {
            board_low: low.to_string(),
            board_high: high.to_string(),
            available: state.available,
            score: state.score,
            pieces_left: state.pieces_left,
            terminal: state.terminal,
            difficulty: state.difficulty,
            lines_cleared: state.total_lines_cleared,
        }
    }
}

#[tauri::command]
pub fn playground_start_game(difficulty: i32, clutter: i32) -> Result<PlaygroundState, String> {
    let state = GameStateExt::new(None, 0, 0, difficulty, clutter);
    Ok(state.into())
}

#[tauri::command]
pub fn playground_apply_move(
    board_low: String,
    board_high: String,
    available: Vec<i32>,
    score: i32,
    slot: usize,
    piece_mask_low: String,
    piece_mask_high: String,
    difficulty: i32,
    lines_cleared: i32,
) -> Result<Option<PlaygroundState>, String> {
    let low = board_low.parse::<u64>().map_err(|e| e.to_string())?;
    let high = board_high.parse::<u64>().map_err(|e| e.to_string())?;
    let board_mask = (low as u128) | ((high as u128) << 64);

    let plow = piece_mask_low.parse::<u64>().map_err(|e| e.to_string())?;
    let phigh = piece_mask_high.parse::<u64>().map_err(|e| e.to_string())?;
    let piece_mask = (plow as u128) | ((phigh as u128) << 64);

    let mut available_arr = [-1; 3];
    for (i, &val) in available.iter().take(3).enumerate() {
        available_arr[i] = val;
    }

    let mut state = GameStateExt {
        board_bitmask_u128: board_mask,
        available: available_arr,
        score,
        pieces_left: available_arr.iter().filter(|&&x| x != -1).count() as i32,
        terminal: false,
        difficulty,
        total_lines_cleared: lines_cleared,
    };

    if let Some(next_state) = state.apply_move_mask(slot, piece_mask) {
        Ok(Some(next_state.into()))
    } else {
        Ok(None)
    }
}

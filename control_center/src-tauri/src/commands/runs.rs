#![allow(clippy::collapsible_match)]
use crate::db;
use std::collections::HashMap;
use std::fs;
use tauri::State;
use tricked_shared::models::{MetricRow, Run};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub fn sync_run_states(
    active_runs: &[String],
    conn: &rusqlite::Connection,
) -> Result<Vec<Run>, String> {
    let mut runs = crate::db::list_runs(conn).map_err(|e| e.to_string())?;

    for run in &mut runs {
        let is_running = active_runs.contains(&run.id);
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

pub fn list_runs_impl(
    processes: &mut HashMap<String, Arc<AtomicBool>>,
) -> Result<Vec<Run>, String> {
    let conn = db::init_db();
    let active_runs: Vec<String> = processes.keys().cloned().collect();
    sync_run_states(&active_runs, &conn)
}

#[tauri::command]
pub fn list_runs(state: State<'_, crate::AppState>) -> Result<Vec<Run>, String> {
    list_runs_impl(&mut state.processes.lock().unwrap())
}

#[tauri::command]
pub fn get_active_jobs(
    state: State<'_, crate::AppState>,
) -> Result<Vec<tricked_shared::models::ActiveJob>, String> {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();
    let jobs = crate::process::build_process_tree(&sys, &state.processes);
    Ok(jobs)
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
        "checkpoint_interval": 100,
        "hardware": {
            "device": "cuda:0",
            "num_processes": 4,
            "worker_device": "cpu",
            "inference_batch_size_limit": 64,
            "inference_timeout_ms": 50
        },
        "architecture": {
            "hidden_dimension_size": 64,
            "num_blocks": 4,
            "value_support_size": 300,
            "reward_support_size": 300,
            "spatial_channel_count": 64,
            "hole_predictor_dim": 64
        },
        "optimizer": {
            "buffer_capacity_limit": 100000,
            "train_batch_size": 128,
            "discount_factor": 0.99,
            "td_lambda": 0.9,
            "weight_decay": 1e-4,
            "lr_init": 0.02,
            "unroll_steps": 5,
            "temporal_difference_steps": 5,
            "reanalyze_ratio": 0.0
        },
        "mcts": {
            "simulations": 100,
            "max_gumbel_k": 16,
            "gumbel_scale": 0.5
        },
        "environment": {
            "difficulty": 0,
            "temp_decay_steps": 100000,
            "temp_boost": true
        }
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

    // Determine if it's a study
    let is_study: bool = conn
        .query_row(
            "SELECT type FROM runs WHERE id = ?1",
            rusqlite::params![&id],
            |row| {
                let t: String = row.get(0)?;
                Ok(t == "STUDY")
            },
        )
        .unwrap_or(false);

    let prefix = format!("{}_trial_%", id);
    if is_study {
        let mut stmt = conn
            .prepare("SELECT artifacts_dir FROM runs WHERE id LIKE ?1 OR id = ?2")
            .unwrap();
        let artifact_dirs = stmt
            .query_map(rusqlite::params![prefix, id], |row| {
                row.get::<_, Option<String>>(0)
            })
            .unwrap();
        for dir_res in artifact_dirs {
            if let Ok(Some(dir)) = dir_res {
                let path = std::path::PathBuf::from(dir);
                if path.exists() {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
        }

        let db_path = db::get_db_path();
        let root = db_path.parent().unwrap();
        let files = [
            format!("studies/{}_optimizer_study.db", id),
            format!("studies/{}_optimizer_study.db-shm", id),
            format!("studies/{}_optimizer_study.db-wal", id),
            format!("studies/best_{}_config.json", id),
            format!("studies/{}_optimizer_study.json", id),
        ];
        for f in files {
            let _ = std::fs::remove_file(root.join(f));
        }

        conn.execute(
            "DELETE FROM runs WHERE id LIKE ?1 OR id = ?2",
            rusqlite::params![prefix, id],
        )
        .map_err(|e| e.to_string())?;
    } else {
        if let Ok(artifacts) = conn.query_row(
            "SELECT artifacts_dir FROM runs WHERE id = ?1",
            rusqlite::params![&id],
            |row| row.get::<_, Option<String>>(0),
        ) {
            if let Some(dir) = artifacts {
                let path = std::path::PathBuf::from(dir);
                if path.exists() {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
        }
        conn.execute("DELETE FROM runs WHERE id = ?1", rusqlite::params![id])
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[tauri::command]
pub fn flush_run(id: String) -> Result<(), String> {
    let conn = db::init_db();

    let is_study: bool = conn
        .query_row(
            "SELECT type FROM runs WHERE id = ?1",
            rusqlite::params![&id],
            |row| {
                let t: String = row.get(0)?;
                Ok(t == "STUDY")
            },
        )
        .unwrap_or(false);

    let prefix = format!("{}_trial_%", id);
    if is_study {
        let mut stmt = conn
            .prepare("SELECT artifacts_dir FROM runs WHERE id LIKE ?1")
            .unwrap();
        let artifact_dirs = stmt
            .query_map(rusqlite::params![&prefix], |row| {
                row.get::<_, Option<String>>(0)
            })
            .unwrap();
        for dir_res in artifact_dirs {
            if let Ok(Some(dir)) = dir_res {
                let path = std::path::PathBuf::from(dir);
                if path.exists() {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
        }

        let db_path = db::get_db_path();
        let root = db_path.parent().unwrap();
        let files = [
            format!("studies/{}_optimizer_study.db", id),
            format!("studies/{}_optimizer_study.db-shm", id),
            format!("studies/{}_optimizer_study.db-wal", id),
            format!("studies/best_{}_config.json", id),
            format!("studies/{}_optimizer_study.json", id),
        ];
        for f in files {
            let _ = std::fs::remove_file(root.join(f));
        }

        conn.execute(
            "DELETE FROM runs WHERE id LIKE ?1",
            rusqlite::params![&prefix],
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM metrics WHERE run_id LIKE ?1",
            rusqlite::params![&prefix],
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM log_events WHERE run_id LIKE ?1",
            rusqlite::params![&prefix],
        )
        .map_err(|e| e.to_string())?;
    } else {
        if let Ok(artifacts) = conn.query_row(
            "SELECT artifacts_dir FROM runs WHERE id = ?1",
            rusqlite::params![&id],
            |row| row.get::<_, Option<String>>(0),
        ) {
            if let Some(dir) = artifacts {
                let path = std::path::PathBuf::from(dir);
                if path.exists() {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
        }
        conn.execute("DELETE FROM runs WHERE id = ?1", rusqlite::params![&id])
            .map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM metrics WHERE run_id = ?1",
            rusqlite::params![&id],
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM log_events WHERE run_id = ?1",
            rusqlite::params![&id],
        )
        .map_err(|e| e.to_string())?;
    }

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

#[derive(serde::Serialize)]
pub struct FrontendGameStep {
    pub board_low: String,
    pub board_high: String,
    pub available: Vec<i32>,
    pub action_taken: i64,
    pub piece_identifier: i64,
}

#[derive(serde::Serialize)]
pub struct FrontendVaultGame {
    pub source_run_id: String,
    pub source_run_name: String,
    pub run_type: String,
    pub difficulty_setting: i32,
    pub episode_score: f32,
    pub steps: Vec<FrontendGameStep>,
    pub lines_cleared: u32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
}

#[tauri::command]
pub fn get_vault_games() -> Result<Vec<FrontendVaultGame>, String> {
    let conn = db::init_db();
    let runs = db::list_runs(&conn).map_err(|e| e.to_string())?;

    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();

    let mut all_games = Vec::new();

    for run in runs {
        let vault_file = root.join("runs").join(&run.id).join("vault.bincode");
        if !vault_file.exists() {
            continue;
        }

        if let Ok(file) = std::fs::File::open(vault_file) {
            let reader = std::io::BufReader::new(file);
            if let Ok(games) = bincode::deserialize_from::<
                _,
                Vec<tricked_engine::train::buffer::OwnedGameData>,
            >(reader)
            {
                for g in games {
                    let mut frontend_steps = Vec::with_capacity(g.steps.len());
                    for step in g.steps {
                        frontend_steps.push(FrontendGameStep {
                            board_low: step.board_state[0].to_string(),
                            board_high: step.board_state[1].to_string(),
                            available: step.available_pieces.to_vec(),
                            action_taken: step.action_taken,
                            piece_identifier: step.piece_identifier,
                        });
                    }

                    all_games.push(FrontendVaultGame {
                        source_run_id: run.id.clone(),
                        source_run_name: run.name.clone(),
                        run_type: run.r#type.clone(),
                        difficulty_setting: g.difficulty_setting,
                        episode_score: g.episode_score,
                        steps: frontend_steps,
                        lines_cleared: g.lines_cleared,
                        mcts_depth_mean: g.mcts_depth_mean,
                        mcts_search_time_mean: g.mcts_search_time_mean,
                    });
                }
            }
        }
    }

    all_games.sort_by(|a, b| {
        b.episode_score
            .partial_cmp(&a.episode_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_games.truncate(100);

    Ok(all_games)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::collections::HashMap;
    use std::process::Command;

    #[test]
    fn test_sync_runs_crash_simulation() {
        let conn = Connection::open_in_memory().unwrap();
        // Setup initial schema
        conn.execute(
            "CREATE TABLE runs (
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

        // Insert a 'RUNNING' run WITH valid JSON for config
        let run_id = "test-run-123".to_string();
        conn.execute(
            "INSERT INTO runs (id, name, type, status, config) VALUES (?1, 'test', 'PPO', 'RUNNING', '{}')",
            rusqlite::params![&run_id],
        )
        .unwrap();

        // Spawn a dummy child process
        let mut child = Command::new("sleep").arg("0").spawn().unwrap();
        let pid = child.id();

        let mut tracked_pids = HashMap::new();
        tracked_pids.insert(run_id.clone(), pid);

        // Wait for the dummy process to exit
        let _ = child.wait();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let runs = sync_run_states(&[], &conn).unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, "STOPPED");
    }
}

#[cfg(test)]
mod tests_phase4_tauri {
    use super::*;

    #[test]
    fn test_tauri_run_lifecycle_and_artifact_eradication() {
        std::env::set_var("TEST_DB", "test_eradication.db");
        let _ = std::fs::remove_file("test_eradication.db");
        let conn = db::init_db();
        let run = create_run_impl("test_erad_run".to_string(), "PPO".to_string(), None).unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE id = ?1",
                rusqlite::params![run.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "Run was not created properly!");

        delete_run(run.id.clone()).unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE id = ?1",
                rusqlite::params![run.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            count, 0,
            "Artifact eradication failed to wipe SQLite records!"
        );
        let _ = std::fs::remove_file("test_eradication.db");
    }
}

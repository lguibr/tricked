#![allow(clippy::collapsible_match)]
use crate::db;
use std::collections::HashMap;
use std::fs;
use sysinfo::System;
use tauri::State;
use tauri_plugin_shell::process::CommandChild;
use tricked_shared::models::{MetricRow, Run};

pub fn sync_run_states(
    tracked_pids: &HashMap<String, u32>,
    conn: &rusqlite::Connection,
    sys: &mut System,
) -> Result<(Vec<String>, Vec<Run>), String> {
    let mut to_remove = Vec::new();
    sys.refresh_processes();

    for (id, &pid) in tracked_pids.iter() {
        if sys.processes().get(&sysinfo::Pid::from_u32(pid)).is_none() {
            to_remove.push(id.clone());
            let _ = conn.execute(
                "UPDATE runs SET status = 'STOPPED' WHERE id = ?1",
                rusqlite::params![id],
            );
        }
    }

    let mut runs = crate::db::list_runs(conn).map_err(|e| e.to_string())?;

    for run in &mut runs {
        let is_running = tracked_pids.contains_key(&run.id) && !to_remove.contains(&run.id);
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
    Ok((to_remove, runs))
}

pub fn list_runs_impl(processes: &mut HashMap<String, CommandChild>) -> Result<Vec<Run>, String> {
    let conn = db::init_db();
    let mut sys = System::new_all();

    let tracked_pids: HashMap<String, u32> = processes
        .iter()
        .map(|(k, v)| (k.clone(), v.pid()))
        .collect();

    let (to_remove, runs) = sync_run_states(&tracked_pids, &conn, &mut sys)?;

    for id in to_remove {
        processes.remove(&id);
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
        "value_support_size": 300,
        "reward_support_size": 300,
        "spatial_channel_count": 64,
        "hole_predictor_dim": 64,
        "buffer_capacity_limit": 100000,
        "simulations": 100,
        "train_batch_size": 128,
        "discount_factor": 0.99,
        "td_lambda": 0.9,
        "weight_decay": 1e-4,
        "checkpoint_interval": 100,
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

#[tauri::command]
pub fn get_vault_games(
    run_id: String,
) -> Result<Vec<tricked_engine::train::buffer::OwnedGameData>, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let vault_file = root.join("artifacts").join(run_id).join("vault.bincode");

    if !vault_file.exists() {
        return Err("No vault found".to_string());
    }

    let file = std::fs::File::open(vault_file).map_err(|e| e.to_string())?;
    let reader = std::io::BufReader::new(file);
    let games: Vec<tricked_engine::train::buffer::OwnedGameData> =
        bincode::deserialize_from(reader).map_err(|e| e.to_string())?;
    Ok(games)
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

        let mut sys = sysinfo::System::new_all();

        // Wait for the dummy process to exit
        let _ = child.wait();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let (to_remove, runs) = sync_run_states(&tracked_pids, &conn, &mut sys).unwrap();

        assert_eq!(to_remove.len(), 1);
        assert_eq!(to_remove[0], run_id);

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

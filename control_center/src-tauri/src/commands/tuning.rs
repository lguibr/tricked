use crate::db;
use std::fs;
use tauri::State;

#[tauri::command]
pub fn get_tuning_study(study_id: String) -> Result<serde_json::Value, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let json_path = root.join(format!("studies/{}_optimizer_study.json", study_id));
    if let Ok(content) = std::fs::read_to_string(&json_path) {
        return serde_json::from_str(&content).map_err(|e| e.to_string());
    }
    Err("Study output not found".to_string())
}

#[tauri::command]
pub fn get_active_study(study_id: String) -> Result<serde_json::Value, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let json_path = root.join(format!("studies/best_{}_config.json", study_id));
    if let Ok(content) = std::fs::read_to_string(&json_path) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
            return Ok(parsed);
        }
    }
    Err("No active study results".to_string())
}

#[tauri::command]
pub fn get_study_status(study_id: String) -> Result<bool, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let json_path = root.join(format!("studies/best_{}_config.json", study_id));
    let optimizer_json = root.join(format!("studies/{}_optimizer_study.json", study_id));
    let optimizer_db = root.join(format!("studies/{}_optimizer_study.db", study_id));
    Ok(json_path.exists() || optimizer_json.exists() || optimizer_db.exists())
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
            for data in artifacts_iter.flatten() {
                targets.push(data);
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

    let _ = fs::remove_file(root.join("studies/unified_tune_optimizer_study.db"));
    let _ = fs::remove_file(root.join("studies/unified_tune_optimizer_study.db-shm"));
    let _ = fs::remove_file(root.join("studies/unified_tune_optimizer_study.db-wal"));
    let _ = fs::remove_file(root.join("studies/best_unified_tune_config.json"));
    let _ = fs::remove_file(root.join("studies/unified_tune_optimizer_study.json"));
    Ok(())
}

#[cfg(test)]
mod tuning_tests {

    #[test]
    fn test_optimizer_study_total_flush() {}
}

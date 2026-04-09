use crate::db;
use std::fs;
use tauri::State;

#[tauri::command]
pub fn get_tuning_study(study_id: String) -> Result<serde_json::Value, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let new_path = root.join(format!("runs/{}/optimizer_study.json", study_id));
    let legacy_path = root.join(format!("studies/{}_optimizer_study.json", study_id));

    match std::fs::read_to_string(&new_path).or_else(|_| std::fs::read_to_string(&legacy_path)) {
        Ok(content) => serde_json::from_str(&content).map_err(|e| e.to_string()),
        Err(e) => Err(format!(
            "Study output not found at {:?} or fallback {:?}. Error: {}",
            new_path, legacy_path, e
        )),
    }
}

#[tauri::command]
pub fn get_active_study(study_id: String) -> Result<serde_json::Value, String> {
    let db_path = db::get_db_path();
    let root = db_path.parent().unwrap();
    let new_path = root.join(format!("runs/{}/best_config.json", study_id));
    let legacy_path = root.join(format!("studies/best_{}_config.json", study_id));

    if let Ok(content) =
        std::fs::read_to_string(&new_path).or_else(|_| std::fs::read_to_string(&legacy_path))
    {
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
    let new_path1 = root.join(format!("runs/{}/best_config.json", study_id));
    let new_path2 = root.join(format!("runs/{}/optimizer_study.json", study_id));

    let json_path = root.join(format!("studies/best_{}_config.json", study_id));
    let optimizer_json = root.join(format!("studies/{}_optimizer_study.json", study_id));
    let optimizer_db = root.join(format!("studies/{}_optimizer_study.db", study_id));
    Ok(new_path1.exists()
        || new_path2.exists()
        || json_path.exists()
        || optimizer_json.exists()
        || optimizer_db.exists())
}

use std::sync::atomic::Ordering;

#[tauri::command]
pub fn flush_study(state: State<'_, crate::AppState>, _study_type: String) -> Result<(), String> {
    let mut processes = state.processes.lock().unwrap();
    if let Some(abort_flag) = processes.remove("STUDY") {
        abort_flag.store(true, Ordering::SeqCst);
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
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;

    #[test]
    fn test_tuning_status_check() {
        let db_path = crate::db::get_db_path();
        let root = db_path.parent().unwrap();
        let study_id = "test_status_chk";

        // Ensure starting false
        let _ = fs::remove_file(root.join(format!("runs/{}/optimizer_study.json", study_id)));
        let _ = fs::remove_file(root.join(format!("studies/best_{}_config.json", study_id)));
        let _ = fs::remove_file(root.join(format!("studies/{}_optimizer_study.json", study_id)));
        let _ = fs::remove_file(root.join(format!("studies/{}_optimizer_study.db", study_id)));
        assert!(!get_study_status(study_id.to_string()).unwrap_or(true));

        let run_dir = root.join(format!("runs/{}", study_id));
        let _ = fs::create_dir_all(&run_dir);
        {
            let mut file = File::create(run_dir.join("best_config.json")).unwrap();
            file.write_all(b"{}").unwrap();
        }

        assert!(get_study_status(study_id.to_string()).unwrap());
        let _ = fs::remove_file(run_dir.join("best_config.json"));
    }

    #[test]
    fn test_tuning_recovery_from_json() {
        let db_path = crate::db::get_db_path();
        let root = db_path.parent().unwrap();
        let study_id = "test_json_recov";

        // Cleanup before asserting Err!
        let _ = fs::remove_file(root.join(format!("runs/{}/optimizer_study.json", study_id)));
        let _ = fs::remove_file(root.join(format!("studies/{}_optimizer_study.json", study_id)));

        // Should err initially
        assert!(get_tuning_study(study_id.to_string()).is_err());

        // Write mock JSON to new path
        let run_dir = root.join(format!("runs/{}", study_id));
        let _ = fs::create_dir_all(&run_dir);
        let content = r#"{"trials": [{"number": 1, "state": "COMPLETE", "value": [0.5], "params": {}, "intermediate_values": {}}], "importance": {}}"#;
        {
            let mut file = File::create(run_dir.join("optimizer_study.json")).unwrap();
            file.write_all(content.as_bytes()).unwrap();
            file.sync_all().unwrap();
        }

        let val = get_tuning_study(study_id.to_string()).unwrap();
        assert_eq!(val["trials"][0]["state"], "COMPLETE");

        let _ = fs::remove_file(run_dir.join("optimizer_study.json"));
    }

    #[test]
    fn test_flush_tuning_pipeline_artifacts_flow() {
        // Can't run `flush_study` easily because it takes `tauri::State`,
        // but we can verify the DB schema doesn't crash us locally on initial init.
        let conn = crate::db::init_db();
        assert!(crate::db::list_runs(&conn).is_ok());
    }
}

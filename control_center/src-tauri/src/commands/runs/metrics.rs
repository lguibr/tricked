use crate::db;
use tricked_shared::models::MetricRow;

#[tauri::command]
pub fn get_run_metrics(run_id: String) -> Result<Vec<MetricRow>, String> {
    let conn = db::init_db();
    db::get_metrics(&conn, &run_id).map_err(|e| e.to_string())
}

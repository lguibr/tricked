pub mod commands;
pub mod db;
pub mod execution;
pub mod log_interceptor;
pub mod process;
pub mod telemetry;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::Mutex;

pub struct AppState {
    pub processes: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let processes_telemetry = std::sync::Arc::new(Mutex::new(HashMap::new()));
    let processes_state = processes_telemetry.clone();

    tauri::Builder::default()
        .manage(AppState {
            processes: processes_state,
        })
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::list_runs,
            commands::create_run,
            commands::rename_run,
            commands::delete_run,
            commands::save_config,
            commands::get_run_metrics,
            commands::get_tuning_study,
            commands::get_active_study,
            commands::get_study_status,
            commands::flush_study,
            commands::get_vault_games,
            commands::playground_start_game,
            commands::playground_apply_move,
            commands::playground_commit_to_vault,
            execution::start_run,
            execution::stop_run,
            execution::start_study,
            execution::stop_study,
            execution::start_evaluation,
            execution::stop_evaluation,
            execution::list_checkpoints,
        ])
        .setup(move |app| {
            log_interceptor::spawn_interceptor(app.handle().clone());
            telemetry::spawn_telemetry_loop(app.handle().clone(), processes_telemetry);
            telemetry::spawn_udp_listener(app.handle().clone());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

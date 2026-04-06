use crate::db;
use std::collections::HashMap;
use std::sync::Mutex;
use sysinfo::Pid;
use tauri_plugin_shell::process::CommandChild;
use tricked_shared::models::{ActiveJob, ProcessInfo};

pub fn build_process_info_recursive(sys: &sysinfo::System, pid: u32) -> Option<ProcessInfo> {
    let p = sys.process(Pid::from_u32(pid))?;

    let mut children = Vec::new();
    for (child_pid, child_proc) in sys.processes() {
        if child_proc.thread_kind().is_some() {
            continue;
        }
        if child_proc.parent() == Some(Pid::from_u32(pid)) {
            if let Some(child_info) = build_process_info_recursive(sys, child_pid.as_u32()) {
                children.push(child_info);
            }
        }
    }

    let status_str = match p.status() {
        sysinfo::ProcessStatus::Run => "Running",
        sysinfo::ProcessStatus::Sleep => "Sleeping",
        sysinfo::ProcessStatus::Zombie => "Zombie",
        sysinfo::ProcessStatus::Stop => "Stopped",
        sysinfo::ProcessStatus::Idle => "Idle",
        sysinfo::ProcessStatus::Tracing => "Tracing",
        sysinfo::ProcessStatus::Dead => "Dead",
        sysinfo::ProcessStatus::Wakekill => "Wakekill",
        sysinfo::ProcessStatus::Waking => "Waking",
        sysinfo::ProcessStatus::Parked => "Parked",
        sysinfo::ProcessStatus::LockBlocked => "LockBlocked",
        sysinfo::ProcessStatus::UninterruptibleDiskSleep => "Disk Wait",
        sysinfo::ProcessStatus::Unknown(_) => "Unknown",
    }
    .to_string();

    Some(ProcessInfo {
        pid,
        name: p.name().to_string(),
        status: status_str,
        cpu_usage: p.cpu_usage(),
        memory_mb: p.memory() as f64 / 1024.0 / 1024.0,
        cmd: p.cmd().to_vec(),
        children,
    })
}

pub fn build_process_tree(
    sys: &sysinfo::System,
    processes: &std::sync::Arc<Mutex<HashMap<String, CommandChild>>>,
) -> Vec<ActiveJob> {
    let mut jobs = Vec::new();

    let active_pids: Vec<(String, u32)> = {
        let guard = processes.lock().unwrap();
        guard
            .iter()
            .map(|(id, child)| (id.clone(), child.pid()))
            .collect()
    };

    let db_path = db::get_db_path();
    let conn = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    );

    for (id, root_pid) in active_pids {
        let root_info = build_process_info_recursive(sys, root_pid);

        let job_name = if id == "STUDY" {
            "Optuna Tuning Study".to_string()
        } else if let Ok(ref c) = conn {
            c.query_row(
                "SELECT name FROM runs WHERE id = ?1",
                rusqlite::params![&id],
                |row| row.get::<_, String>(0),
            )
            .unwrap_or_else(|_| "Experiment".to_string())
        } else {
            "Experiment".to_string()
        };

        let job_type = if id == "STUDY" {
            "TUNING".to_string()
        } else {
            "EXPERIMENT".to_string()
        };

        jobs.push(ActiveJob {
            id,
            name: job_name,
            job_type,
            root_process: root_info,
        });
    }

    jobs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::time::Duration;
    use sysinfo::System;

    #[test]
    fn test_process_info_for_invalid_pid() {
        let mut sys = System::new_all();
        sys.refresh_all();
        // A massively high PID that shouldn't exist
        let info = build_process_info_recursive(&sys, 9999999);
        assert!(info.is_none(), "Expected None for non-existent PID");
    }

    #[test]
    fn test_process_info_for_running_process() {
        let mut cmd = Command::new("sleep")
            .arg("2")
            .stdout(Stdio::null())
            .spawn()
            .expect("Failed to start sleep process");

        std::thread::sleep(Duration::from_millis(100));
        let mut sys = System::new_all();
        sys.refresh_all();

        let info = build_process_info_recursive(&sys, cmd.id()).expect("Process should exist");

        assert_eq!(info.pid, cmd.id());
        assert!(info.name.contains("sleep") || info.cmd.join(" ").contains("sleep"));

        // Given it's a real OS process, it could honestly be anything, but usually sleeping/running/idle
        assert!(
            info.status == "Sleeping" || info.status == "Running" || info.status == "Idle",
            "Unrecognized active status: {}",
            info.status
        );

        cmd.kill().unwrap();
    }

    #[test]
    fn test_process_memory_and_cpu_metrics() {
        let mut cmd = Command::new("sleep")
            .arg("2")
            .spawn()
            .expect("Failed to start sleep process");

        std::thread::sleep(Duration::from_millis(50));
        let mut sys = System::new_all();
        sys.refresh_all();

        let info = build_process_info_recursive(&sys, cmd.id()).unwrap();

        // Assert that memory metrics are calculated and exist
        assert!(info.memory_mb >= 0.0);
        assert!(info.cpu_usage >= 0.0);

        cmd.kill().unwrap();
    }

    #[test]
    fn test_active_job_generation_without_db() {
        let mut sys = System::new_all();
        sys.refresh_all();

        let map = std::sync::Arc::new(Mutex::new(HashMap::new()));
        let jobs = build_process_tree(&sys, &map);

        assert_eq!(jobs.len(), 0, "No jobs should be returned for empty map");
    }

    #[test]
    fn test_completed_process_not_found() {
        let mut cmd = Command::new("echo")
            .arg("done")
            .spawn()
            .expect("Failed to start echo process");

        cmd.wait().unwrap(); // wait for it to finish immediately

        let mut sys = System::new_all();
        sys.refresh_all();

        let info = build_process_info_recursive(&sys, cmd.id());
        assert!(
            info.is_none(),
            "Completed process should not be found in active tree"
        );
    }
}

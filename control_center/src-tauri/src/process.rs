use crate::db;
use std::collections::HashMap;
use std::process::Child;
use std::sync::Mutex;
use sysinfo::Pid;
use tricked_shared::models::{ActiveJob, ProcessInfo};

pub fn build_process_info_recursive(sys: &sysinfo::System, pid: u32) -> Option<ProcessInfo> {
    let p = sys.process(Pid::from_u32(pid))?;

    let mut children = Vec::new();
    for (child_pid, child_proc) in sys.processes() {
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
    processes: &std::sync::Arc<Mutex<HashMap<String, Child>>>,
) -> Vec<ActiveJob> {
    let mut jobs = Vec::new();

    let active_pids: Vec<(String, u32)> = {
        let guard = processes.lock().unwrap();
        guard
            .iter()
            .map(|(id, child)| (id.clone(), child.id()))
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

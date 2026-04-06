use std::collections::HashMap;
use std::sync::Mutex;
use sysinfo::{Disks, Networks, System};
use tauri::AppHandle;
use tauri::Emitter;
use tauri_plugin_shell::process::CommandChild;

use crate::process::build_process_tree;

fn get_gpu_metrics() -> (f32, f32) {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=utilization.gpu,memory.used")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if let Ok(out_str) = String::from_utf8(output.stdout) {
            if let Some(first_line) = out_str.trim().lines().next() {
                let parts: Vec<&str> = first_line.split(", ").collect();
                if parts.len() == 2 {
                    let gpu_util = parts[0].parse::<f32>().unwrap_or(0.0);
                    let vram_used = parts[1].parse::<f32>().unwrap_or(0.0);
                    return (gpu_util, vram_used);
                }
            }
        }
    }
    (0.0, 0.0)
}

fn get_disk_io_bytes() -> (u64, u64) {
    if let Ok(content) = std::fs::read_to_string("/proc/diskstats") {
        let mut read_bytes = 0;
        let mut write_bytes = 0;
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 10 {
                let name = parts[2];
                if !name.starts_with("loop") && !name.starts_with("ram") {
                    if let (Ok(r), Ok(w)) = (parts[5].parse::<u64>(), parts[9].parse::<u64>()) {
                        read_bytes += r * 512;
                        write_bytes += w * 512;
                    }
                }
            }
        }
        return (read_bytes, write_bytes);
    }
    (0, 0)
}

pub fn spawn_telemetry_loop(
    app_handle: AppHandle,
    processes_telemetry: std::sync::Arc<Mutex<HashMap<String, CommandChild>>>,
) {
    std::thread::spawn(move || {
        let mut sys = System::new_all();
        let mut networks = Networks::new_with_refreshed_list();

        let (mut last_disk_read, mut last_disk_write) = get_disk_io_bytes();
        let mut last_time = std::time::Instant::now();

        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            sys.refresh_all();

            let now = std::time::Instant::now();
            let dt = now.duration_since(last_time).as_secs_f64().max(0.1);
            last_time = now;

            networks.refresh_list();
            let mut rx_bytes = 0;
            let mut tx_bytes = 0;
            for (_, net) in &networks {
                rx_bytes += net.received();
                tx_bytes += net.transmitted();
            }
            let network_rx_mbps = (rx_bytes as f64 / dt) / 1024.0 / 1024.0;
            let network_tx_mbps = (tx_bytes as f64 / dt) / 1024.0 / 1024.0;

            let (cur_read, cur_write) = get_disk_io_bytes();
            let disk_read_mbps =
                (cur_read.saturating_sub(last_disk_read) as f64 / dt) / 1024.0 / 1024.0;
            let disk_write_mbps =
                (cur_write.saturating_sub(last_disk_write) as f64 / dt) / 1024.0 / 1024.0;
            last_disk_read = cur_read;
            last_disk_write = cur_write;

            let cpu_usage = sys.global_cpu_info().cpu_usage();
            let cpu_cores_usage: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
            let ram_used = sys.used_memory() as f64 / 1024.0 / 1024.0;
            let ram_total = sys.total_memory() as f64 / 1024.0 / 1024.0;
            let ram_usage_pct = if ram_total > 0.0 {
                (ram_used / ram_total) * 100.0
            } else {
                0.0
            };

            let disks = Disks::new_with_refreshed_list();
            let mut total_disk = 0;
            let mut used_disk = 0;
            for disk in &disks {
                total_disk += disk.total_space();
                used_disk += disk.total_space() - disk.available_space();
            }
            let disk_usage_pct = if total_disk > 0 {
                (used_disk as f64 / total_disk as f64) * 100.0
            } else {
                0.0
            };

            let (gpu_util, vram_used) = get_gpu_metrics();

            let metrics = serde_json::json!({
                "cpu_usage": cpu_usage,
                "cpu_cores_usage": cpu_cores_usage,
                "ram_usage_pct": ram_usage_pct,
                "ram_used_mb": ram_used,
                "gpu_util": gpu_util,
                "vram_used_mb": vram_used,
                "disk_usage_pct": disk_usage_pct,
                "network_rx_mbps": network_rx_mbps,
                "network_tx_mbps": network_tx_mbps,
                "disk_read_mbps": disk_read_mbps,
                "disk_write_mbps": disk_write_mbps
            });

            let _ = app_handle.emit("hardware_telemetry", metrics);

            let process_tree = build_process_tree(&sys, &processes_telemetry);
            let _ = app_handle.emit("process_telemetry", process_tree);
        }
    });
}

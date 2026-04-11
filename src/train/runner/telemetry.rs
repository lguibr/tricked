use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub fn spawn_telemetry_thread(
    telemetry_active: Arc<AtomicBool>,
    t_cpu: Arc<AtomicU32>,
    t_ram: Arc<AtomicU32>,
    t_disk_r: Arc<AtomicU32>,
    t_disk_w: Arc<AtomicU32>,
    t_net_rx: Arc<AtomicU32>,
    t_net_tx: Arc<AtomicU32>,
    t_disk_pct: Arc<AtomicU32>,
) {
    let _ = std::thread::Builder::new()
        .name("telemetry".into())
        .spawn(move || {
            let mut sys = sysinfo::System::new_all();
            let mut networks = sysinfo::Networks::new_with_refreshed_list();
            let mut last_time = std::time::Instant::now();
            let mut last_disk_read = 0;
            let mut last_disk_write = 0;

            while telemetry_active.load(Ordering::Relaxed) {
                sys.refresh_cpu_usage();
                sys.refresh_memory();
                networks.refresh_list();

                let now = std::time::Instant::now();
                let dt = now.duration_since(last_time).as_secs_f64().max(0.1);
                last_time = now;

                let mut rx_bytes = 0;
                let mut tx_bytes = 0;
                for (_, net) in &networks {
                    rx_bytes += net.received();
                    tx_bytes += net.transmitted();
                }
                let network_rx_mbps = (rx_bytes as f64 / dt) / 1024.0 / 1024.0;
                let network_tx_mbps = (tx_bytes as f64 / dt) / 1024.0 / 1024.0;

                let mut cur_read = 0;
                let mut cur_write = 0;
                if let Ok(content) = std::fs::read_to_string("/proc/diskstats") {
                    let mut read_bytes = 0;
                    let mut write_bytes = 0;
                    for line in content.lines() {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 10 {
                            let name = parts[2];
                            if !name.starts_with("loop") && !name.starts_with("ram") {
                                if let (Ok(r), Ok(w)) =
                                    (parts[5].parse::<u64>(), parts[9].parse::<u64>())
                                {
                                    read_bytes += r * 512;
                                    write_bytes += w * 512;
                                }
                            }
                        }
                    }
                    cur_read = read_bytes;
                    cur_write = write_bytes;
                }
                let disk_read_mbps =
                    (cur_read.saturating_sub(last_disk_read) as f64 / dt) / 1024.0 / 1024.0;
                let disk_write_mbps =
                    (cur_write.saturating_sub(last_disk_write) as f64 / dt) / 1024.0 / 1024.0;
                last_disk_read = cur_read;
                last_disk_write = cur_write;

                let cpu = sys.global_cpu_info().cpu_usage();
                let ram = sys.used_memory() as f32 / 1024.0 / 1024.0;

                let disks = sysinfo::Disks::new_with_refreshed_list();
                let mut total_disk = 0;
                let mut used_disk = 0;
                for disk in &disks {
                    total_disk += disk.total_space();
                    used_disk += disk.total_space() - disk.available_space();
                }
                let disk_pct = if total_disk > 0 {
                    (used_disk as f32 / total_disk as f32) * 100.0
                } else {
                    0.0
                };

                t_cpu.store(cpu.to_bits(), Ordering::Relaxed);
                t_ram.store(ram.to_bits(), Ordering::Relaxed);
                t_disk_r.store(
                    (disk_read_mbps as f32).to_bits(),
                    Ordering::Relaxed,
                );
                t_disk_w.store(
                    (disk_write_mbps as f32).to_bits(),
                    Ordering::Relaxed,
                );
                t_net_rx.store(
                    (network_rx_mbps as f32).to_bits(),
                    Ordering::Relaxed,
                );
                t_net_tx.store(
                    (network_tx_mbps as f32).to_bits(),
                    Ordering::Relaxed,
                );
                t_disk_pct.store(disk_pct.to_bits(), Ordering::Relaxed);

                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
}

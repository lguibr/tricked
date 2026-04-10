use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use rusqlite::{params, Connection};
use std::net::UdpSocket;
use std::thread;
use std::time::{Duration, Instant};
pub use tricked_shared::models::TelemetryData;

pub struct TelemetryLogger {
    tx: Sender<TelemetryMessage>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

pub enum TelemetryMessage {
    LogMetric(Box<TelemetryData>),
    LogStdout(String),
    Flush,
}

impl TelemetryLogger {
    pub fn new(db_path: String) -> Self {
        let (tx, rx): (Sender<TelemetryMessage>, Receiver<TelemetryMessage>) = bounded(5000);

        let handle = thread::spawn(move || {
            let mut conn = Connection::open(&db_path).unwrap_or_else(|e| {
                panic!("Failed to open workspace DB at {}: {}", db_path, e);
            });

            // Ensure schema is created in case it was missed, and set high performance pragmas
            let _ = conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 
                 CREATE TABLE IF NOT EXISTS metrics (
                     run_id TEXT NOT NULL,
                     step INTEGER NOT NULL,
                     total_loss REAL,
                     policy_loss REAL,
                     value_loss REAL,
                     reward_loss REAL,
                     lr REAL,
                     game_score_min REAL,
                     game_score_max REAL,
                     game_score_med REAL,
                     game_score_mean REAL,
                     win_rate REAL,
                     game_lines_cleared INTEGER,
                     game_count INTEGER,
                     ram_usage_mb REAL,
                     gpu_usage_pct REAL,
                     cpu_usage_pct REAL,
                     disk_usage_pct REAL,
                     vram_usage_mb REAL,
                     mcts_depth_mean REAL,
                     mcts_search_time_mean REAL,
                     elapsed_time REAL,
                     network_tx_mbps REAL DEFAULT 0.0,
                     network_rx_mbps REAL DEFAULT 0.0,
                     disk_read_mbps REAL DEFAULT 0.0,
                     disk_write_mbps REAL DEFAULT 0.0,
                     policy_entropy REAL DEFAULT 0.0,
                     gradient_norm REAL DEFAULT 0.0,
                     representation_drift REAL DEFAULT 0.0,
                     mean_td_error REAL DEFAULT 0.0,
                     queue_saturation_ratio REAL DEFAULT 0.0,
                     sps_vs_tps REAL DEFAULT 0.0,
                     queue_latency_us REAL DEFAULT 0.0,
                     sumtree_contention_us REAL DEFAULT 0.0,
                     action_space_entropy REAL DEFAULT 0.0,
                     layer_gradient_norms TEXT,
                     spatial_heatmap TEXT,
                     difficulty REAL DEFAULT 0.0,
                     FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                 );
                ",
            );

            // Auto-migrate legacy DBs lacking elapsed_time and new I/O cols
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN elapsed_time REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN network_tx_mbps REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN network_rx_mbps REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN disk_read_mbps REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN disk_write_mbps REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN action_space_entropy REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN queue_latency_us REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN sumtree_contention_us REAL DEFAULT 0.0",
                [],
            );
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN layer_gradient_norms TEXT",
                [],
            );
            let _ = conn.execute("ALTER TABLE metrics ADD COLUMN spatial_heatmap TEXT", []);
            let _ = conn.execute(
                "ALTER TABLE metrics ADD COLUMN difficulty REAL DEFAULT 0.0",
                [],
            );

            let udp_socket = UdpSocket::bind("127.0.0.1:0").ok();
            if let Some(ref sock) = udp_socket {
                let _ = sock.set_nonblocking(true);
            }

            let mut batch = Vec::new();
            let mut last_flush = Instant::now();

            loop {
                let msg = match rx.recv_timeout(Duration::from_secs(1)) {
                    Ok(m) => m,
                    Err(RecvTimeoutError::Timeout) => TelemetryMessage::Flush,
                    Err(_) => break, // Channel disconnected
                };

                match msg {
                    TelemetryMessage::LogMetric(data) => {
                        if let Some(ref sock) = udp_socket {
                            if let Ok(bytes) = bincode::serialize(&*data) {
                                let _ = sock.send_to(&bytes, "127.0.0.1:5555");
                            }
                        }
                        batch.push(*data);
                    }
                    TelemetryMessage::LogStdout(text) => {
                        println!("{}", text);
                    }
                    TelemetryMessage::Flush => {}
                }

                if batch.len() >= 100 || last_flush.elapsed() >= Duration::from_secs(5) {
                    if !batch.is_empty() {
                        let tx = conn.transaction();
                        if let Ok(transaction) = tx {
                            for data in batch.drain(..) {
                                if let Err(e) = transaction.execute(
                                    "INSERT INTO metrics (
                                        run_id, step, total_loss, policy_loss, value_loss, reward_loss,
                                        lr, game_score_min, game_score_max, game_score_med, game_score_mean,
                                        win_rate, game_lines_cleared, game_count, ram_usage_mb, gpu_usage_pct,
                                        cpu_usage_pct, disk_usage_pct, vram_usage_mb, mcts_depth_mean,
                                        mcts_search_time_mean, elapsed_time, network_tx_mbps, network_rx_mbps,
                                        disk_read_mbps, disk_write_mbps, policy_entropy, gradient_norm, representation_drift, mean_td_error, queue_saturation_ratio, sps_vs_tps, queue_latency_us, sumtree_contention_us, action_space_entropy, layer_gradient_norms, spatial_heatmap, difficulty
                                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38)",
                                    params![
                                        data.run_id, data.step as i64, data.total_loss as f64, data.policy_loss as f64, data.value_loss as f64, data.reward_loss as f64,
                                        data.lr, data.game_score_min as f64, data.game_score_max as f64, data.game_score_med as f64, data.game_score_mean as f64,
                                        data.winrate_mean as f64, data.game_lines_cleared as i64, data.game_count as i64, data.ram_usage_mb as f64, data.gpu_usage_pct as f64,
                                        data.cpu_usage_pct as f64, data.disk_usage_pct, data.vram_usage_mb as f64, data.mcts_depth_mean as f64,
                                        data.mcts_search_time_mean as f64, data.elapsed_time, data.network_tx_mbps, data.network_rx_mbps,
                                        data.disk_read_mbps, data.disk_write_mbps, data.policy_entropy as f64, data.gradient_norm as f64,
                                        data.representation_drift as f64, data.mean_td_error as f64, data.queue_saturation_ratio as f64,
                                        data.sps_vs_tps as f64, data.queue_latency_us as f64, data.sumtree_contention_us as f64, data.action_space_entropy as f64, data.layer_gradient_norms.clone(),
                                        serde_json::to_string(&data.spatial_heatmap).unwrap_or_default(), data.difficulty as f64
                                    ],
                                ) {
                                    println!("SQL INSERT ERROR: {}", e);
                                }
                            }
                            let _ = transaction.commit();
                        } else {
                            batch.clear(); // drop if tx fails
                        }
                    }
                    last_flush = Instant::now();
                }
            } // end loop

            // Final flush if channel disconnected
            if !batch.is_empty() {
                let tx = conn.transaction();
                if let Ok(transaction) = tx {
                    for data in batch.drain(..) {
                        if let Err(e) = transaction.execute(
                                    "INSERT INTO metrics (
                                        run_id, step, total_loss, policy_loss, value_loss, reward_loss,
                                        lr, game_score_min, game_score_max, game_score_med, game_score_mean,
                                        win_rate, game_lines_cleared, game_count, ram_usage_mb, gpu_usage_pct,
                                        cpu_usage_pct, disk_usage_pct, vram_usage_mb, mcts_depth_mean,
                                        mcts_search_time_mean, elapsed_time, network_tx_mbps, network_rx_mbps,
                                        disk_read_mbps, disk_write_mbps, policy_entropy, gradient_norm, representation_drift, mean_td_error, queue_saturation_ratio, sps_vs_tps, queue_latency_us, sumtree_contention_us, action_space_entropy, layer_gradient_norms, spatial_heatmap, difficulty
                                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38)",
                                    params![
                                        data.run_id, data.step as i64, data.total_loss as f64, data.policy_loss as f64, data.value_loss as f64, data.reward_loss as f64,
                                        data.lr, data.game_score_min as f64, data.game_score_max as f64, data.game_score_med as f64, data.game_score_mean as f64,
                                        data.winrate_mean as f64, data.game_lines_cleared as i64, data.game_count as i64, data.ram_usage_mb as f64, data.gpu_usage_pct as f64,
                                        data.cpu_usage_pct as f64, data.disk_usage_pct, data.vram_usage_mb as f64, data.mcts_depth_mean as f64,
                                        data.mcts_search_time_mean as f64, data.elapsed_time, data.network_tx_mbps, data.network_rx_mbps,
                                        data.disk_read_mbps, data.disk_write_mbps, data.policy_entropy as f64, data.gradient_norm as f64,
                                        data.representation_drift as f64, data.mean_td_error as f64, data.queue_saturation_ratio as f64,
                                        data.sps_vs_tps as f64, data.queue_latency_us as f64, data.sumtree_contention_us as f64, data.action_space_entropy as f64, data.layer_gradient_norms.clone(),
                                        serde_json::to_string(&data.spatial_heatmap).unwrap_or_default(), data.difficulty as f64
                                    ],
                                ) {
                                    println!("SQL FINAL INSERT ERROR: {}", e);
                                }
                    }
                    let _ = transaction.commit();
                }
            }
        });

        Self {
            tx,
            thread_handle: Some(handle),
        }
    }

    pub fn send_metric(&self, data: TelemetryData) {
        let _ = self
            .tx
            .try_send(TelemetryMessage::LogMetric(Box::new(data)));
    }

    pub fn send_stdout(&self, msg: String) {
        let _ = self.tx.try_send(TelemetryMessage::LogStdout(msg));
    }

    pub fn close(mut self) {
        let _ = self.tx.send(TelemetryMessage::Flush);
        drop(self.tx); // Drop the sender first so the channel disconnects
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_telemetry_udp_stream() {
        let db_path = std::env::temp_dir()
            .join(format!(
                "test_telemetry_{}.db",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
            .to_string_lossy()
            .to_string();
        let logger = TelemetryLogger::new(db_path);

        let receptor = UdpSocket::bind("127.0.0.1:5555").unwrap();
        receptor
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();

        for i in 0..100 {
            let data = TelemetryData {
                run_id: "test".to_string(),
                step: i as usize,
                total_loss: 0.1,
                policy_loss: 0.1,
                value_loss: 0.1,
                reward_loss: 0.1,
                lr: 0.001,
                game_score_min: -1.0,
                game_score_max: 1.0,
                game_score_med: 0.0,
                game_score_mean: 0.0,
                winrate_mean: 0.5,
                game_lines_cleared: 10,
                game_count: 1,
                ram_usage_mb: 1000.0,
                gpu_usage_pct: 50.0,
                cpu_usage_pct: 10.0,
                io_usage: 5.0,
                disk_usage_pct: 5.0,
                vram_usage_mb: 4000.0,
                mcts_depth_mean: 10.0,
                mcts_search_time_mean: 100.0,
                elapsed_time: 10.0,
                network_tx_mbps: 1.0,
                network_rx_mbps: 1.0,
                disk_read_mbps: 1.0,
                disk_write_mbps: 1.0,
                policy_entropy: 2.0,
                gradient_norm: 1.0,
                representation_drift: 0.1,
                mean_td_error: 0.5,
                queue_saturation_ratio: 0.8,
                sps_vs_tps: 1.0,
                queue_latency_us: 0.0,
                sumtree_contention_us: 0.0,
                action_space_entropy: 2.0,
                layer_gradient_norms: "Conv1: 0.5".to_string(),
                spatial_heatmap: vec![0.0; 96],
                difficulty: 0.0,
            };
            logger.send_metric(data);
        }

        let mut received = 0;
        let mut buf = [0u8; 8192];
        for _ in 0..500 {
            // Check until we get 100 or timeout
            if let Ok((size, _)) = receptor.recv_from(&mut buf) {
                if let Ok(decoded) = bincode::deserialize::<TelemetryData>(&buf[..size]) {
                    if decoded.run_id == "test" {
                        received += 1;
                    }
                }
            }
            if received == 100 {
                break;
            }
        }

        assert!(
            received >= 95,
            "Failed to receive at least 95% of UDP packets! (Got {})",
            received
        );
    }
}

#[cfg(test)]
mod test_telemetry_flushing {
    use super::*;
    use rusqlite::Connection;
    #[test]
    fn test_telemetry_sqlite_batch_flushing() {
        let db_path = "test_batch_flush.db".to_string();
        let _ = std::fs::remove_file(&db_path);
        {
            let conn = Connection::open(&db_path).unwrap();
            let _ = conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY)", []);
            let _ = conn.execute("INSERT INTO runs (id) VALUES ('test_run')", []);
        }
        let logger = TelemetryLogger::new(db_path.clone());
        for i in 0..150 {
            logger.send_metric(TelemetryData {
                run_id: "test_run".to_string(),
                step: i,
                total_loss: 0.1,
                policy_loss: 0.1,
                value_loss: 0.1,
                reward_loss: 0.1,
                lr: 0.01,
                game_score_min: 0.0,
                game_score_max: 0.0,
                game_score_med: 0.0,
                game_score_mean: 0.0,
                winrate_mean: 0.0,
                game_lines_cleared: 0,
                game_count: 0,
                ram_usage_mb: 0.0,
                gpu_usage_pct: 0.0,
                cpu_usage_pct: 0.0,
                io_usage: 0.0,
                disk_usage_pct: 0.0,
                vram_usage_mb: 0.0,
                mcts_depth_mean: 0.0,
                mcts_search_time_mean: 0.0,
                elapsed_time: 0.0,
                network_tx_mbps: 0.0,
                network_rx_mbps: 0.0,
                disk_read_mbps: 0.0,
                disk_write_mbps: 0.0,
                policy_entropy: 0.0,
                gradient_norm: 0.0,
                representation_drift: 0.0,
                mean_td_error: 0.0,
                queue_saturation_ratio: 0.0,
                sps_vs_tps: 0.0,
                queue_latency_us: 0.0,
                sumtree_contention_us: 0.0,
                action_space_entropy: 0.0,
                layer_gradient_norms: "".to_string(),
                spatial_heatmap: vec![],
                difficulty: 0.0,
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        let conn = Connection::open(&db_path).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM metrics", [], |row| row.get(0))
            .unwrap();
        assert!(
            count >= 100,
            "Batch flush must trigger at exactly 100 items!"
        );
    }
}

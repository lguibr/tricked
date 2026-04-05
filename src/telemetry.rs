use crossbeam_channel::{bounded, Receiver, Sender};
use rusqlite::{params, Connection};
use std::thread;

pub struct TelemetryData {
    pub run_id: String,
    pub step: usize,
    pub total_loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub reward_loss: f32,
    pub lr: f64,
    pub game_score_min: f32,
    pub game_score_max: f32,
    pub game_score_med: f32,
    pub game_score_mean: f32,
    pub winrate_mean: f32,
    pub game_lines_cleared: u32,
    pub game_count: usize,
    pub ram_usage_mb: f32,
    pub gpu_usage_pct: f32,
    pub cpu_usage_pct: f32,
    pub io_usage: f32,
    pub disk_usage_pct: f64,
    pub vram_usage_mb: f32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
    pub network_tx_mbps: f64,
    pub network_rx_mbps: f64,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
}

pub struct TelemetryLogger {
    tx: Sender<TelemetryMessage>,
}

pub enum TelemetryMessage {
    LogMetric(TelemetryData),
    LogStdout(String),
}

impl TelemetryLogger {
    pub fn new(db_path: String) -> Self {
        let (tx, rx): (Sender<TelemetryMessage>, Receiver<TelemetryMessage>) = bounded(5000);

        thread::spawn(move || {
            let conn = Connection::open(&db_path).unwrap_or_else(|e| {
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

            while let Ok(msg) = rx.recv() {
                match msg {
                    TelemetryMessage::LogMetric(data) => {
                        if let Err(e) = conn.execute(
                            "INSERT INTO metrics (
                                run_id, step, total_loss, policy_loss, value_loss, reward_loss,
                                lr, game_score_min, game_score_max, game_score_med, game_score_mean,
                                win_rate, game_lines_cleared, game_count, ram_usage_mb, gpu_usage_pct,
                                cpu_usage_pct, disk_usage_pct, vram_usage_mb, mcts_depth_mean,
                                mcts_search_time_mean, elapsed_time, network_tx_mbps, network_rx_mbps,
                                disk_read_mbps, disk_write_mbps
                            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
                            params![
                                data.run_id,
                                data.step as i64,
                                data.total_loss as f64,
                                data.policy_loss as f64,
                                data.value_loss as f64,
                                data.reward_loss as f64,
                                { data.lr },
                                data.game_score_min as f64,
                                data.game_score_max as f64,
                                data.game_score_med as f64,
                                data.game_score_mean as f64,
                                data.winrate_mean as f64,
                                data.game_lines_cleared as i64,
                                data.game_count as i64,
                                data.ram_usage_mb as f64,
                                data.gpu_usage_pct as f64,
                                data.cpu_usage_pct as f64,
                                { data.disk_usage_pct },
                                data.vram_usage_mb as f64,
                                data.mcts_depth_mean as f64,
                                data.mcts_search_time_mean as f64,
                                0.0, // elapsed_time placeholder
                                data.network_tx_mbps,
                                data.network_rx_mbps,
                                data.disk_read_mbps,
                                data.disk_write_mbps
                            ],
                        ) {
                            eprintln!("[Telemetry] Error inserting metric row: {}", e);
                        }
                    }
                    TelemetryMessage::LogStdout(text) => {
                        println!("{}", text);
                    }
                }
            }
        });

        Self { tx }
    }

    pub fn send_metric(&self, data: TelemetryData) {
        let _ = self.tx.try_send(TelemetryMessage::LogMetric(data));
    }

    pub fn send_stdout(&self, msg: String) {
        let _ = self.tx.try_send(TelemetryMessage::LogStdout(msg));
    }
}

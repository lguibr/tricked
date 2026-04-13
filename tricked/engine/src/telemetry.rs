use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::Duration;
use prost::Message;
use redis::Commands;
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
    pub fn new(_db_path: String) -> Self {
        let (tx, rx): (Sender<TelemetryMessage>, Receiver<TelemetryMessage>) = bounded(5000);

        let handle = thread::spawn(move || {
            let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
            let client = redis::Client::open(redis_url.as_str()).unwrap_or_else(|e| {
                panic!("Failed to connect to Redis at {}: {}", redis_url, e);
            });
            let mut con = client.get_connection().unwrap_or_else(|e| {
                panic!("Failed to get connection to Redis at {}: {}", redis_url, e);
            });

            loop {
                let msg = match rx.recv_timeout(Duration::from_secs(1)) {
                    Ok(m) => m,
                    Err(RecvTimeoutError::Timeout) => TelemetryMessage::Flush,
                    Err(_) => break, // Channel disconnected
                };

                match msg {
                    TelemetryMessage::LogMetric(data) => {
                        let row = crate::config::pb::MetricRow {
                            run_id: data.run_id.clone(),
                            step: data.step as i32,
                            total_loss: Some(data.total_loss),
                            policy_loss: Some(data.policy_loss),
                            value_loss: Some(data.value_loss),
                            reward_loss: Some(data.reward_loss),
                            lr: Some(data.lr as f32),
                            game_score_min: Some(data.game_score_min),
                            game_score_max: Some(data.game_score_max),
                            game_score_med: Some(data.game_score_med),
                            game_score_mean: Some(data.game_score_mean),
                            win_rate: Some(data.winrate_mean),
                            game_lines_cleared: Some(data.game_lines_cleared as f32),
                            game_count: Some(data.game_count as f32),
                            ram_usage_mb: Some(data.ram_usage_mb),
                            gpu_usage_pct: Some(data.gpu_usage_pct),
                            cpu_usage_pct: Some(data.cpu_usage_pct),
                            disk_usage_pct: Some(data.disk_usage_pct as f32),
                            vram_usage_mb: Some(data.vram_usage_mb),
                            mcts_depth_mean: Some(data.mcts_depth_mean),
                            mcts_search_time_mean: Some(data.mcts_search_time_mean),
                            elapsed_time: Some(data.elapsed_time as f32),
                            network_tx_mbps: Some(data.network_tx_mbps as f32),
                            network_rx_mbps: Some(data.network_rx_mbps as f32),
                            disk_read_mbps: Some(data.disk_read_mbps as f32),
                            disk_write_mbps: Some(data.disk_write_mbps as f32),
                            policy_entropy: Some(data.policy_entropy),
                            gradient_norm: Some(data.gradient_norm),
                            representation_drift: Some(data.representation_drift),
                            mean_td_error: Some(data.mean_td_error),
                            queue_saturation_ratio: Some(data.queue_saturation_ratio),
                            sps_vs_tps: Some(data.sps_vs_tps),
                            queue_latency_us: Some(data.queue_latency_us),
                            sumtree_contention_us: Some(data.sumtree_contention_us),
                            action_space_entropy: Some(data.action_space_entropy),
                            layer_gradient_norms: Some(data.layer_gradient_norms.clone()),
                            spatial_heatmap: data.spatial_heatmap.clone(),
                            difficulty: Some(data.difficulty),
                        };

                        let mut buf = Vec::new();
                        if let Ok(_) = row.encode(&mut buf) {
                            let channel = format!("telemetry:metrics:{}", row.run_id);
                            if let Err(e) = con.publish::<_, _, ()>(&channel, buf) {
                                println!("Redis publish error: {}", e);
                            }
                        }
                    }
                    TelemetryMessage::LogStdout(text) => {
                        println!("{}", text);
                    }
                    TelemetryMessage::Flush => {}
                }
            } // end loop
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

}

#[cfg(test)]
mod test_telemetry_flushing {
    use super::*;
    #[test]
    #[ignore]
    fn test_telemetry_sqlite_batch_flushing() {
        // Obsolete test as we use REDIS now.
    }
}

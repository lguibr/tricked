use redis::Commands;
use std::sync::atomic::AtomicU64;

#[derive(Default)]
pub struct PerformanceCounters {
    pub total_simulations: AtomicU64,
    pub total_steps: AtomicU64,
    pub total_games: AtomicU64,
}

pub trait GameLogger: Send + Sync {
    fn log_game_end(&self, difficulty: i32, final_score: f32, steps: i32);
    fn log_training_step(
        &self,
        total_loss: f32,
        policy_loss: f32,
        value_loss: f32,
        reward_loss: f32,
    );
    fn log_metric(&self, name: &str, value: f32);
    fn log_config(&self, config_json: &str);
    fn log_trajectory(
        &self,
        game_id: usize,
        boards: &[u128],
        available: &[[i32; 3]],
        actions: &[i64],
        piece_ids: &[i64],
    );
}

use crossbeam_channel::{unbounded, Sender};

pub enum LogEvent {
    GameEnd {
        difficulty: i32,
        final_score: f32,
        steps: i32,
    },
    TrainingStep {
        total_loss: f32,
        policy_loss: f32,
        value_loss: f32,
        reward_loss: f32,
    },
    Metric {
        name: String,
        value: f32,
    },
    Trajectory {
        game_id: usize,
        boards: Vec<u128>,
        available: Vec<[i32; 3]>,
        actions: Vec<i64>,
        piece_ids: Vec<i64>,
    },
    Config(String),
}

pub struct RedisLogger {
    evaluation_request_transmitter: Sender<LogEvent>,
}

impl RedisLogger {
    pub fn new(url: &str) -> Self {
        let (evaluation_request_transmitter, evaluation_response_receiver) = unbounded();
        let u = url.to_string();
        std::thread::spawn(move || {
            let client = redis::Client::open(u.as_str()).expect("Invalid Redis URL");
            if let Ok(mut con) = client.get_connection() {
                for evt in evaluation_response_receiver {
                    match evt {
                        LogEvent::GameEnd {
                            difficulty,
                            final_score,
                            steps,
                        } => {
                            let payload = serde_json::json!({
                                "difficulty": difficulty,
                                "score": final_score,
                                "steps": steps,
                            });
                            let payload_str = payload.to_string();
                            let _: () = con.lpush("tricked_games", &payload_str).unwrap_or(());
                            let _: () = con.publish("tricked_games", &payload_str).unwrap_or(());
                        }
                        LogEvent::TrainingStep {
                            total_loss,
                            policy_loss,
                            value_loss,
                            reward_loss,
                        } => {
                            let evt = serde_json::json!({
                                "type": "training_step",
                                "loss": total_loss,
                                "policy_loss": policy_loss,
                                "value_loss": value_loss,
                                "reward_loss": reward_loss
                            });
                            let _: () = con
                                .publish("tricked_training", evt.to_string())
                                .unwrap_or(());
                        }
                        LogEvent::Metric { name, value } => {
                            let evt =
                                serde_json::json!({"type": "metric", "name": name, "value": value});
                            let _: () = con
                                .publish("tricked_metrics", evt.to_string())
                                .unwrap_or(());
                        }
                        LogEvent::Trajectory {
                            game_id,
                            boards,
                            available,
                            actions,
                            piece_ids,
                        } => {
                            let mut payload_buffer = Vec::new();
                            let boards_bytes: &[u8] = bytemuck::cast_slice(&boards);
                            let available_bytes: &[u8] = bytemuck::cast_slice(&available);
                            let actions_bytes: &[u8] = bytemuck::cast_slice(&actions);
                            let piece_ids_bytes: &[u8] = bytemuck::cast_slice(&piece_ids);

                            payload_buffer
                                .extend_from_slice(&(boards_bytes.len() as u64).to_le_bytes());
                            payload_buffer
                                .extend_from_slice(&(available_bytes.len() as u64).to_le_bytes());
                            payload_buffer
                                .extend_from_slice(&(actions_bytes.len() as u64).to_le_bytes());
                            payload_buffer
                                .extend_from_slice(&(piece_ids_bytes.len() as u64).to_le_bytes());

                            payload_buffer.extend_from_slice(boards_bytes);
                            payload_buffer.extend_from_slice(available_bytes);
                            payload_buffer.extend_from_slice(actions_bytes);
                            payload_buffer.extend_from_slice(piece_ids_bytes);

                            let compressed_payload =
                                lz4_flex::compress_prepend_size(&payload_buffer);

                            let _: () = redis::cmd("HSET")
                                .arg("tricked_replays")
                                .arg(game_id.to_string())
                                .arg(compressed_payload)
                                .query(&mut con)
                                .unwrap_or(());
                        }
                        LogEvent::Config(config_json) => {
                            let _: () = con.publish("tricked_config", config_json).unwrap_or(());
                        }
                    }
                }
            }
        });

        Self {
            evaluation_request_transmitter,
        }
    }
}

impl GameLogger for RedisLogger {
    fn log_game_end(&self, difficulty: i32, final_score: f32, steps: i32) {
        let _ = self.evaluation_request_transmitter.send(LogEvent::GameEnd {
            difficulty,
            final_score,
            steps,
        });
    }

    fn log_training_step(
        &self,
        total_loss: f32,
        policy_loss: f32,
        value_loss: f32,
        reward_loss: f32,
    ) {
        let _ = self
            .evaluation_request_transmitter
            .send(LogEvent::TrainingStep {
                total_loss,
                policy_loss,
                value_loss,
                reward_loss,
            });
    }

    fn log_metric(&self, name: &str, value: f32) {
        let _ = self.evaluation_request_transmitter.send(LogEvent::Metric {
            name: name.to_string(),
            value,
        });
    }

    fn log_trajectory(
        &self,
        game_id: usize,
        boards: &[u128],
        available: &[[i32; 3]],
        actions: &[i64],
        piece_ids: &[i64],
    ) {
        let _ = self
            .evaluation_request_transmitter
            .send(LogEvent::Trajectory {
                game_id,
                boards: boards.to_vec(),
                available: available.to_vec(),
                actions: actions.to_vec(),
                piece_ids: piece_ids.to_vec(),
            });
    }

    fn log_config(&self, config_json: &str) {
        let _ = self
            .evaluation_request_transmitter
            .send(LogEvent::Config(config_json.to_string()));
    }
}

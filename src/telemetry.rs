use redis::Commands;

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
    fn log_trajectory(&self, game_id: usize, features: &[Vec<f32>]);
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
        features: Vec<Vec<f32>>,
    },
    Config(String),
}

pub struct RedisLogger {
    tx: Sender<LogEvent>,
}

impl RedisLogger {
    pub fn new(url: &str) -> Self {
        let (tx, rx) = unbounded();
        let u = url.to_string();
        std::thread::spawn(move || {
            let client = redis::Client::open(u.as_str()).expect("Invalid Redis URL");
            if let Ok(mut con) = client.get_connection() {
                for evt in rx {
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
                        LogEvent::Trajectory { game_id, features } => {
                            let steps: Vec<_> = features
                                .iter()
                                .map(|feat| {
                                    // CHANGED: The vector is already sliced to 256 by the worker!
                                    serde_json::json!({
                                        "features": feat,
                                    })
                                })
                                .collect();
                            let payload = serde_json::json!({ "steps": steps });
                            let _: () = redis::cmd("HSET")
                                .arg("tricked_replays")
                                .arg(game_id.to_string())
                                .arg(payload.to_string())
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

        Self { tx }
    }
}

impl GameLogger for RedisLogger {
    fn log_game_end(&self, difficulty: i32, final_score: f32, steps: i32) {
        let _ = self.tx.send(LogEvent::GameEnd {
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
        let _ = self.tx.send(LogEvent::TrainingStep {
            total_loss,
            policy_loss,
            value_loss,
            reward_loss,
        });
    }

    fn log_metric(&self, name: &str, value: f32) {
        let _ = self.tx.send(LogEvent::Metric {
            name: name.to_string(),
            value,
        });
    }

    fn log_trajectory(&self, game_id: usize, features: &[Vec<f32>]) {
        let _ = self.tx.send(LogEvent::Trajectory {
            game_id,
            features: features.to_vec(),
        });
    }

    fn log_config(&self, config_json: &str) {
        let _ = self.tx.send(LogEvent::Config(config_json.to_string()));
    }
}

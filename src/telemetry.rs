use redis::Commands;

pub trait GameLogger: Send + Sync {
    fn log_game_end(&self, difficulty: i32, final_score: f32, steps: i32);
    fn log_training_step(&self, loss: f32);
    fn log_metric(&self, name: &str, value: f32);
    fn log_trajectory(&self, game_id: usize, boards: &[[u64; 2]]);
}

pub struct RedisLogger {
    client: redis::Client,
}

impl RedisLogger {
    pub fn new(url: &str) -> Self {
        let client = redis::Client::open(url).expect("Invalid Redis URL");
        Self { client }
    }
}

impl GameLogger for RedisLogger {
    fn log_game_end(&self, difficulty: i32, final_score: f32, steps: i32) {
        if let Ok(mut con) = self.client.get_connection() {
            let payload = serde_json::json!({
                "difficulty": difficulty,
                "score": final_score,
                "steps": steps,
            });
            let payload_str = payload.to_string();
            let _: () = con.lpush("tricked_games", &payload_str).unwrap_or(());
            let _: () = con.publish("tricked_games", &payload_str).unwrap_or(());
        }
    }

    fn log_training_step(&self, loss: f32) {
        if let Ok(mut con) = self.client.get_connection() {
            let evt = serde_json::json!({"type": "training_step", "loss": loss});
            let _: () = con
                .publish("tricked_training", evt.to_string())
                .unwrap_or(());
        }
    }

    fn log_metric(&self, name: &str, value: f32) {
        if let Ok(mut con) = self.client.get_connection() {
            let evt = serde_json::json!({"type": "metric", "name": name, "value": value});
            let _: () = con
                .publish("tricked_metrics", evt.to_string())
                .unwrap_or(());
        }
    }

    fn log_trajectory(&self, game_id: usize, boards: &[[u64; 2]]) {
        if let Ok(mut con) = self.client.get_connection() {
            let steps: Vec<_> = boards
                .iter()
                .map(|b| {
                    let board_u128 = (b[0] as u128) | ((b[1] as u128) << 64);
                    serde_json::json!({ "board": board_u128.to_string() })
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
    }
}

use std::sync::Arc;
use serde::Serialize;
use redis::Commands;

#[derive(Serialize)]
pub struct SpectatorMetrics {
    pub worker: i32,
    pub board: String,
    pub score: i32,
    pub pieces_left: i32,
    pub terminal: bool,
    pub available: Vec<i32>,
    pub hole_logits: Vec<f32>,
}

pub trait GameLogger: Send + Sync {
    fn log_spectator_update(&self, metrics: &SpectatorMetrics);
    fn log_game_end(&self, difficulty: i32, final_score: f32, steps: i32);
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
    fn log_spectator_update(&self, metrics: &SpectatorMetrics) {
        if let Ok(mut con) = self.client.get_connection() {
            let payload = serde_json::to_string(metrics).unwrap();
            let _: () = con.hset("tricked_spectator", metrics.worker.to_string(), &payload).unwrap_or(());
            let evt = serde_json::json!({"type": "spectator", "worker": metrics.worker});
            let _: () = con.publish("tricked_events", evt.to_string()).unwrap_or(());
        }
    }

    fn log_game_end(&self, difficulty: i32, final_score: f32, steps: i32) {
        if let Ok(mut con) = self.client.get_connection() {
            let payload = serde_json::json!({
                "difficulty": difficulty,
                "score": final_score,
                "steps": steps,
            });
            let _: () = con.lpush("tricked_games", payload.to_string()).unwrap_or(());
        }
    }
}

pub struct MockLogger;

impl MockLogger {
    pub fn new() -> Self {
        Self {}
    }
}

impl GameLogger for MockLogger {
    fn log_spectator_update(&self, _metrics: &SpectatorMetrics) {}
    fn log_game_end(&self, _difficulty: i32, _final_score: f32, _steps: i32) {}
}

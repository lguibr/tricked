use crate::board::GameStateExt;
use crate::buffer::EpisodeMeta;
use crate::config::Config;
use crate::mcts::EvalReq;

use std::sync::{Arc, RwLock};

#[derive(Clone, Default)]
pub struct TrainingStatus {
    pub running: bool,
    pub exp_name: String,
    pub loss_total: f32,
    pub loss_value: f32,
    pub loss_policy: f32,
    pub loss_reward: f32,
    pub games_per_second: f32,
    pub games_played: u64,
    pub training_steps: u64,
}

#[derive(Default)]
pub struct TelemetryStore {
    pub spectator_state: Option<GameStateExt>,
    pub top_games: Vec<EpisodeMeta>,
    pub status: TrainingStatus,
}

pub enum EngineCommand {
    StartTraining(Box<Config>),
    StopTraining,
}

#[derive(Clone)]
pub struct AppState {
    pub current_game: Arc<RwLock<GameStateExt>>,
    pub current_difficulty: Arc<RwLock<i32>>,
    pub telemetry: Arc<RwLock<TelemetryStore>>,
    pub cmd_sender: crossbeam_channel::Sender<EngineCommand>,
    #[allow(dead_code)]
    pub eval_tx: Arc<RwLock<Option<crossbeam_channel::Sender<EvalReq>>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_store_initialization() {
        let tel = TelemetryStore::default();
        assert!(!tel.status.running);
        assert_eq!(tel.top_games.len(), 0);
    }
}

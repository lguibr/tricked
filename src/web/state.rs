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
}

pub struct TelemetryStore {
    pub spectator_state: Option<GameStateExt>,
    #[allow(dead_code)]
    pub top_games: Vec<EpisodeMeta>,
    pub status: TrainingStatus,
}

impl Default for TelemetryStore {
    fn default() -> Self {
        Self {
            spectator_state: None,
            top_games: Vec::new(),
            status: TrainingStatus::default(),
        }
    }
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

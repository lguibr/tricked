use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::Arc;
use tch::Tensor;

use crate::train::buffer::state::SharedState;

pub struct ReplayBuffer {
    pub state: Arc<SharedState>,
    pub background_sender: crossbeam_channel::Sender<OwnedGameData>,
    pub arena_pool: (
        crossbeam_channel::Sender<SampleArena>,
        crossbeam_channel::Receiver<SampleArena>,
    ),
}

pub struct SampleArena {
    pub state_features: Tensor,
    pub actions: Tensor,
    pub piece_identifiers: Tensor,
    pub value_prefixs: Tensor,
    pub target_policies: Tensor,
    pub target_values: Tensor,
    pub model_values: Tensor,
    pub transition_boards: Tensor,
    pub transition_actions: Tensor,
    pub transition_metadata: Tensor,
    pub loss_masks: Tensor,
    pub importance_weights: Tensor,
}

impl SampleArena {
    pub fn new(batch_size_limit: usize, unroll_limit: usize) -> Self {
        let pin = |size: &[i64], kind: tch::Kind| {
            let t = Tensor::zeros(size, (kind, tch::Device::Cpu));
            if tch::Cuda::is_available() {
                t.pin_memory(tch::Device::Cuda(0))
            } else {
                t
            }
        };
        Self {
            state_features: pin(&[batch_size_limit as i64, 20, 8, 16], tch::Kind::Float),
            actions: pin(
                &[batch_size_limit as i64, unroll_limit as i64],
                tch::Kind::Int64,
            ),
            piece_identifiers: pin(
                &[batch_size_limit as i64, unroll_limit as i64],
                tch::Kind::Int64,
            ),
            value_prefixs: pin(
                &[batch_size_limit as i64, unroll_limit as i64],
                tch::Kind::Float,
            ),
            target_policies: pin(
                &[batch_size_limit as i64, (unroll_limit + 1) as i64, 288],
                tch::Kind::Float,
            ),
            target_values: pin(
                &[batch_size_limit as i64, (unroll_limit + 1) as i64],
                tch::Kind::Float,
            ),
            model_values: pin(
                &[batch_size_limit as i64, (unroll_limit + 1) as i64],
                tch::Kind::Float,
            ),
            transition_boards: pin(
                &[batch_size_limit as i64, unroll_limit as i64, 8, 2],
                tch::Kind::Int64,
            ),
            transition_actions: pin(
                &[batch_size_limit as i64, unroll_limit as i64, 3],
                tch::Kind::Int,
            ),
            transition_metadata: pin(
                &[batch_size_limit as i64, unroll_limit as i64, 4],
                tch::Kind::Int,
            ),
            loss_masks: pin(
                &[batch_size_limit as i64, (unroll_limit + 1) as i64],
                tch::Kind::Float,
            ),
            importance_weights: pin(&[batch_size_limit as i64], tch::Kind::Float),
        }
    }
}

pub struct BatchTensors {
    pub state_features_batch: Tensor,
    pub actions_batch: Tensor,
    pub piece_identifiers_batch: Tensor,
    pub value_prefixs_batch: Tensor,
    pub target_policies_batch: Tensor,
    pub target_values_batch: Tensor,
    #[allow(dead_code)]
    pub model_values_batch: Tensor,

    // GPU Feature Expansion Targets
    pub transition_boards_batch: Tensor,
    pub transition_actions_batch: Tensor,
    pub transition_metadata_batch: Tensor,

    pub loss_masks_batch: Tensor,
    pub importance_weights_batch: Tensor,
    pub global_indices_sampled: Vec<usize>,
    pub arena: Option<SampleArena>,
}

#[derive(Clone, Debug)]
pub struct GameStep {
    pub board_state: [u64; 2],
    pub available_pieces: [i32; 3],
    pub action_taken: i64,
    pub piece_identifier: i64,
    pub value_prefix_received: f32,
    pub policy_target: [f32; 288],
    pub value_target: f32,
}

pub struct OwnedGameData {
    pub difficulty_setting: i32,
    pub episode_score: f32,
    pub steps: Vec<GameStep>,
    pub lines_cleared: u32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
}

impl ReplayBuffer {
    pub fn new(
        total_buffer_capacity_limit: usize,
        unroll_steps: usize,
        temporal_difference_steps: usize,
    ) -> Self {
        let shared_state = SharedState {
            buffer_capacity_limit: total_buffer_capacity_limit,
            unroll_steps,
            temporal_difference_steps,
            current_diff: AtomicI32::new(1),
            global_write_storage_index: AtomicUsize::new(0),
            global_write_active_storage_index: AtomicUsize::new(0),
            num_states: AtomicUsize::new(0),

            arrays: crate::train::buffer::state::ShardedStorageArrays::new(
                total_buffer_capacity_limit,
                64,
            ),
            per: crate::sumtree::ShardedPrioritizedReplay::new(
                total_buffer_capacity_limit,
                0.6,
                0.4,
                64,
            ),

            episodes: std::sync::RwLock::new(Vec::new()),
            recent_scores: crossbeam_queue::SegQueue::new(),
            completed_games: AtomicUsize::new(0),
        };

        let state_arc = Arc::new(shared_state);
        let (tx, rx) = crossbeam_channel::bounded::<OwnedGameData>(1024);
        let background_state = state_arc.clone();

        std::thread::Builder::new()
            .name("replay_buffer_writer".into())
            .spawn(move || {
                while let Ok(data) = rx.recv() {
                    Self::process_add_game(&background_state, data);
                }
            })
            .unwrap();

        Self {
            state: state_arc,
            background_sender: tx,
            arena_pool: crossbeam_channel::bounded(32),
        }
    }

    pub fn get_length(&self) -> usize {
        self.state.num_states.load(Ordering::Relaxed)
    }

    pub fn return_arena(&self, arena: SampleArena) {
        let _ = self.arena_pool.0.try_send(arena);
    }

    #[allow(dead_code)]
    pub fn get_global_write_storage_index(&self) -> usize {
        self.state
            .global_write_storage_index
            .load(Ordering::Acquire)
    }

    #[allow(dead_code)]
    pub fn get_and_clear_metrics(&self) -> (Vec<f32>, f32, f32, f32) {
        let mut cloned_scores = Vec::new();
        while let Some(score) = self.state.recent_scores.pop() {
            cloned_scores.push(score);
        }
        if cloned_scores.is_empty() {
            return (vec![], 0.0, 0.0, 0.0);
        }

        let mut sorted_scores = cloned_scores.clone();
        sorted_scores
            .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

        let median_score = sorted_scores[sorted_scores.len() / 2];
        let maximum_score = *sorted_scores.last().unwrap_or(&0.0);
        let sum_scores: f32 = cloned_scores.iter().sum();
        let average_score = sum_scores / cloned_scores.len() as f32;

        (cloned_scores, median_score, maximum_score, average_score)
    }
}

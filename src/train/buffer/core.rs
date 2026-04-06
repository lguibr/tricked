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
    pub raw_unrolled_boards: Tensor,
    pub raw_unrolled_histories: Tensor,
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
            raw_unrolled_boards: pin(
                &[batch_size_limit as i64, unroll_limit as i64, 2],
                tch::Kind::Int64,
            ),
            raw_unrolled_histories: pin(
                &[batch_size_limit as i64, unroll_limit as i64, 14],
                tch::Kind::Int64,
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
    pub raw_unrolled_boards_batch: Tensor,
    pub raw_unrolled_histories_batch: Tensor,

    pub loss_masks_batch: Tensor,
    pub importance_weights_batch: Tensor,
    pub global_indices_sampled: Vec<usize>,
    pub arena: Option<SampleArena>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GameStep {
    pub board_state: [u64; 2],
    pub available_pieces: [i32; 3],
    pub action_taken: i64,
    pub piece_identifier: i64,
    pub value_prefix_received: f32,
    pub policy_target: Vec<f32>,
    pub value_target: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
        batch_size_limit: usize,
        artifacts_dir: Option<String>,
        discount_factor: f32,
        td_lambda: f32,
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

            episodes: crossbeam_queue::SegQueue::new(),
            recent_scores: crossbeam_queue::SegQueue::new(),
            completed_games: AtomicUsize::new(0),
        };

        let state_arc = Arc::new(shared_state);
        let (tx, rx) = crossbeam_channel::bounded::<OwnedGameData>(1024);
        let background_state = state_arc.clone();

        let vault = artifacts_dir.map(crate::train::buffer::vault::VaultManager::new);

        std::thread::Builder::new()
            .name("replay_buffer_writer".into())
            .spawn(move || {
                use crate::train::buffer::vault::Score;
                use std::cmp::Reverse;
                use std::collections::BinaryHeap;

                let mut top_hundred: BinaryHeap<Reverse<Score>> = BinaryHeap::new();

                while let Ok(data) = rx.recv() {
                    if let Some(ref v) = vault {
                        let m = Score(data.episode_score);
                        if top_hundred.len() < 100 || m > top_hundred.peek().unwrap().0 {
                            if top_hundred.len() == 100 {
                                top_hundred.pop();
                            }
                            top_hundred.push(Reverse(m));
                            let _ = v.sender.send(data.clone());
                        }
                    }
                    Self::process_add_game(&background_state, data, discount_factor, td_lambda);
                }
            })
            .unwrap();

        let (arena_tx, arena_rx) = crossbeam_channel::bounded(32);
        for _ in 0..32 {
            let _ = arena_tx.send(SampleArena::new(batch_size_limit, unroll_steps));
        }

        Self {
            state: state_arc,
            background_sender: tx,
            arena_pool: (arena_tx, arena_rx),
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

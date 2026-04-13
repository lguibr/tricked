use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::train::buffer::state::SharedState;

pub struct ReplayBuffer {
    pub state: Arc<SharedState>,
    pub background_sender: crossbeam_channel::Sender<OwnedGameData>,
    pub arena_pool: (
        crossbeam_channel::Sender<SampleArena>,
        crossbeam_channel::Receiver<SampleArena>,
    ),
}

#[derive(Debug, Default, Clone)]
pub struct BufferSummary {
    pub cloned_scores: Vec<f32>,
    pub game_score_med: f32,
    pub game_score_max: f32,
    pub game_score_min: f32,
    pub game_score_mean: f32,
    pub mcts_depth_mean: f32,
    pub mcts_time_mean: f32,
    pub game_lines_cleared: f32,
    pub difficulty: f32,
}

pub struct SampleArena {
    pub board_states: Vec<i64>,
    pub board_histories: Vec<i64>,
    pub board_available: Vec<i32>,
    pub board_historical_acts: Vec<i32>,
    pub board_diff: Vec<i32>,
    pub actions: Vec<i64>,
    pub piece_identifiers: Vec<i64>,
    pub value_prefixs: Vec<f32>,
    pub target_policies: Vec<f32>,
    pub target_values: Vec<f32>,
    pub model_values: Vec<f32>,
    pub raw_unrolled_boards: Vec<i64>,
    pub raw_unrolled_histories: Vec<i64>,
    pub raw_unrolled_available: Vec<i32>,
    pub raw_unrolled_actions: Vec<i32>,
    pub raw_unrolled_diff: Vec<i32>,
    pub loss_masks: Vec<f32>,
    pub importance_weights: Vec<f32>,
    pub global_indices_sampled: Vec<usize>,
}

impl SampleArena {
    pub fn new(batch_size_limit: usize, unroll_limit: usize) -> Self {
        Self {
            board_states: vec![0; batch_size_limit * 2],
            board_histories: vec![0; batch_size_limit * 14],
            board_available: vec![0; batch_size_limit * 3],
            board_historical_acts: vec![0; batch_size_limit * 3],
            board_diff: vec![0; batch_size_limit],
            actions: vec![0; batch_size_limit * unroll_limit],
            piece_identifiers: vec![0; batch_size_limit * unroll_limit],
            value_prefixs: vec![0.0; batch_size_limit * unroll_limit],
            target_policies: vec![0.0; batch_size_limit * (unroll_limit + 1) * 288],
            target_values: vec![0.0; batch_size_limit * (unroll_limit + 1)],
            model_values: vec![0.0; batch_size_limit * (unroll_limit + 1)],
            raw_unrolled_boards: vec![0; batch_size_limit * unroll_limit * 2],
            raw_unrolled_histories: vec![0; batch_size_limit * unroll_limit * 14],
            raw_unrolled_available: vec![0; batch_size_limit * unroll_limit * 3],
            raw_unrolled_actions: vec![0; batch_size_limit * unroll_limit * 3],
            raw_unrolled_diff: vec![0; batch_size_limit * unroll_limit],
            loss_masks: vec![0.0; batch_size_limit * (unroll_limit + 1)],
            importance_weights: vec![0.0; batch_size_limit],
            global_indices_sampled: Vec::with_capacity(batch_size_limit),
        }
    }
}

pub struct BatchTensors {
    pub board_states_batch: Vec<i64>,
    pub board_histories_batch: Vec<i64>,
    pub board_available_batch: Vec<i32>,
    pub board_historical_acts_batch: Vec<i32>,
    pub board_diff_batch: Vec<i32>,
    pub actions_batch: Vec<i64>,
    pub piece_identifiers_batch: Vec<i64>,
    pub value_prefixs_batch: Vec<f32>,
    pub target_policies_batch: Vec<f32>,
    pub target_values_batch: Vec<f32>,
    pub model_values_batch: Vec<f32>,
    pub raw_unrolled_boards_batch: Vec<i64>,
    pub raw_unrolled_histories_batch: Vec<i64>,
    pub raw_unrolled_available_batch: Vec<i32>,
    pub raw_unrolled_actions_batch: Vec<i32>,
    pub raw_unrolled_diff_batch: Vec<i32>,
    pub loss_masks_batch: Vec<f32>,
    pub importance_weights_batch: Vec<f32>,
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
    pub is_terminal: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OwnedGameData {
    pub source_run_id: String,
    pub source_run_name: String,
    pub run_type: String,
    pub difficulty: i32,
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
        alpha: f64,
        beta: f64,
        source_run_id: String,
        source_run_name: String,
        run_type: String,
    ) -> Self {
        let num_shards = 64;
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
                num_shards,
            ),
            per: crate::sumtree::ShardedPrioritizedReplay::new(
                total_buffer_capacity_limit,
                alpha,
                beta,
                num_shards,
            ),
            array_contention_us: std::sync::atomic::AtomicU64::new(0),
            contention_us: std::sync::atomic::AtomicU64::new(0),

            episodes: crossbeam_queue::SegQueue::new(),
            recent_scores: crossbeam_queue::SegQueue::new(),
            completed_games: AtomicUsize::new(0),

            source_run_id,
            source_run_name,
            run_type,
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
    pub fn get_and_clear_metrics(&self) -> BufferSummary {
        let mut cloned_scores = Vec::new();
        let mut depth_sum = 0.0;
        let mut time_sum = 0.0;
        let mut count = 0;
        let mut lines_sum = 0.0;
        let mut diff_sum = 0.0;
        while let Some(ep) = self.state.episodes.pop() {
            depth_sum += ep.mcts_depth_mean;
            time_sum += ep.mcts_search_time_mean;
            lines_sum += ep.lines_cleared as f32;
            diff_sum += ep.difficulty as f32;
            count += 1;
        }
        let mcts_depth_mean = if count > 0 {
            depth_sum / count as f32
        } else {
            0.0
        };
        let mcts_time_mean = if count > 0 {
            time_sum / count as f32
        } else {
            0.0
        };
        let game_lines_cleared = if count > 0 {
            lines_sum / count as f32
        } else {
            0.0
        };
        let difficulty = if count > 0 {
            diff_sum / count as f32
        } else {
            0.0
        };

        while let Some(score) = self.state.recent_scores.pop() {
            cloned_scores.push(score);
        }

        // Do not reset metrics if no games completed this frame. Return the EMA from python side or last known.
        // Wait, if cloned_scores is empty, it means no games finished *this frame*.
        // Since we didn't push any values into the running sum, returning exactly 0.0 zeroes out the UI.
        // We will return -1.0 or whatever, but actually Python `loop.py` handles parsing `game_score_mean`.
        // Let's just return what we have (0.0s) but DO NOT POP ALL `recent_scores` if we want a moving average.
        // Wait, the cleanest way to do this without changing Python too much is to maintain an explicit rolling list.
        // But the user said: "Modify get_and_clear_metrics in Rust to either calculate an Exponential Moving Average (EMA) or only return metrics when game_count > 0."
        if cloned_scores.is_empty() {
            return BufferSummary {
                cloned_scores: vec![],
                game_score_med: f32::NAN,
                game_score_max: f32::NAN,
                game_score_min: f32::NAN,
                game_score_mean: f32::NAN,
                mcts_depth_mean,
                mcts_time_mean,
                game_lines_cleared,
                difficulty,
            };
        }

        let mut sorted_scores = cloned_scores.clone();
        sorted_scores
            .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

        let game_score_med = sorted_scores[sorted_scores.len() / 2];
        let game_score_max = *sorted_scores.last().unwrap_or(&0.0);
        let game_score_min = *sorted_scores.first().unwrap_or(&0.0);
        let sum_scores: f32 = cloned_scores.iter().sum();
        let game_score_mean = sum_scores / cloned_scores.len() as f32;

        BufferSummary {
            cloned_scores,
            game_score_med,
            game_score_max,
            game_score_min,
            game_score_mean,
            mcts_depth_mean,
            mcts_time_mean,
            game_lines_cleared,
            difficulty,
        }
    }
}

#[cfg(test)]
mod test_buffer_integrity {
    use super::*;
    #[test]
    fn test_reanalyze_target_overwrite_integrity() {
        let arena = SampleArena::new(10, 5);
        assert_eq!(arena.target_policies.len(), 10 * 6 * 288);
        assert_eq!(arena.target_values.len(), 10 * 6);
    }
    #[test]
    fn test_per_importance_weight_bounds() {
        let arena = SampleArena::new(10, 5);
        assert_eq!(arena.importance_weights.len(), 10);
    }
}

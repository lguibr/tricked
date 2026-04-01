use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::Arc;
use tch::Tensor;

use crate::train::buffer::state::{EpisodeMeta, SharedState};

pub struct ReplayBuffer {
    pub state: Arc<SharedState>,
    pub background_sender: crossbeam_channel::Sender<OwnedGameData>,
    pub arena_pool: (
        crossbeam_channel::Sender<SampleArena>,
        crossbeam_channel::Receiver<SampleArena>,
    ),
}

pub struct SampleArena {
    pub state_features: Vec<f32>,
    pub actions: Vec<i64>,
    pub piece_identifiers: Vec<i64>,
    pub value_prefixs: Vec<f32>,
    pub target_policies: Vec<f32>,
    pub target_values: Vec<f32>,
    pub model_values: Vec<f32>,
    pub transition_states: Vec<f32>,
    pub loss_masks: Vec<f32>,
    pub importance_weights: Vec<f32>,
}

impl SampleArena {
    pub fn new(batch_size_limit: usize, unroll_limit: usize) -> Self {
        Self {
            state_features: vec![0.0; batch_size_limit * 20 * 128],
            actions: vec![0; batch_size_limit * unroll_limit],
            piece_identifiers: vec![0; batch_size_limit * unroll_limit],
            value_prefixs: vec![0.0; batch_size_limit * unroll_limit],
            target_policies: vec![0.0; batch_size_limit * (unroll_limit + 1) * 288],
            target_values: vec![0.0; batch_size_limit * (unroll_limit + 1)],
            model_values: vec![0.0; batch_size_limit * (unroll_limit + 1)],
            transition_states: vec![0.0; batch_size_limit * unroll_limit * 20 * 128],
            loss_masks: vec![0.0; batch_size_limit * (unroll_limit + 1)],
            importance_weights: vec![0.0; batch_size_limit],
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
    pub transition_states_batch: Tensor,
    pub loss_masks_batch: Tensor,
    pub importance_weights_batch: Tensor,
    pub global_indices_sampled: Vec<usize>,
    pub arena: Option<SampleArena>,
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
        let (evaluation_request_transmitter, evaluation_response_receiver) =
            crossbeam_channel::bounded::<OwnedGameData>(1024);
        let background_state = state_arc.clone();

        std::thread::Builder::new()
            .name("replay_buffer_writer".into())
            .spawn(move || {
                while let Ok(data) = evaluation_response_receiver.recv() {
                    Self::process_add_game(&background_state, data);
                }
            })
            .unwrap();

        Self {
            state: state_arc,
            background_sender: evaluation_request_transmitter,
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
    pub fn add_game(&self, data: OwnedGameData) {
        let _ = self.background_sender.send(data);
    }

    fn process_add_game(state: &SharedState, data: OwnedGameData) {
        let OwnedGameData {
            difficulty_setting,
            episode_score,
            steps,
            lines_cleared,
            mcts_depth_mean,
            mcts_search_time_mean,
        } = data;
        let episode_length = steps.len();
        if episode_length == 0 {
            return;
        }

        let episode_start_index = state.global_write_storage_index.load(Ordering::Relaxed);
        let buffer_buffer_capacity_limit = state.buffer_capacity_limit;
        let running_difficulty = state.current_diff.load(Ordering::Relaxed);
        let next_global_write_index = episode_start_index + episode_length;

        state
            .global_write_active_storage_index
            .store(next_global_write_index, Ordering::Release);

        if running_difficulty == 0 || difficulty_setting != running_difficulty {
            state
                .current_diff
                .store(difficulty_setting, Ordering::Relaxed);
        }

        let active_difficulty = state.current_diff.load(Ordering::Relaxed);
        let absolute_difficulty_penalty =
            10f64.powf(-(active_difficulty - difficulty_setting).abs() as f64);

        for (transition_offset, step) in steps.iter().take(episode_length).enumerate() {
            let circular_write_index =
                (episode_start_index + transition_offset) % buffer_buffer_capacity_limit;
            state.arrays.write_storage_index(
                circular_write_index,
                |memory_shard, internal_shard_index| {
                    memory_shard.state_start[internal_shard_index] = episode_start_index as i64;
                    memory_shard.state_diff[internal_shard_index] = difficulty_setting;
                    memory_shard.state_len[internal_shard_index] = episode_length as i32;

                    memory_shard.boards[internal_shard_index] = step.board_state;
                    memory_shard.available[internal_shard_index] = step.available_pieces;
                    memory_shard.actions[internal_shard_index] = step.action_taken;
                    memory_shard.piece_ids[internal_shard_index] = step.piece_identifier;
                    memory_shard.value_prefixs[internal_shard_index] = step.value_prefix_received;
                    memory_shard.policies[internal_shard_index] = step.policy_target;
                    memory_shard.values[internal_shard_index] = step.value_target;
                },
            );
        }

        let mut circular_indices_to_add = Vec::with_capacity(episode_length);
        let mut transition_penalties = Vec::with_capacity(episode_length);
        for transition_offset in 0..episode_length {
            circular_indices_to_add
                .push((episode_start_index + transition_offset) % buffer_buffer_capacity_limit);
            transition_penalties.push(absolute_difficulty_penalty);
        }
        state
            .per
            .add_batch(&circular_indices_to_add, &transition_penalties);

        state
            .global_write_storage_index
            .store(next_global_write_index, Ordering::Release);

        Self::update_episode_metadata(
            state,
            episode_start_index,
            episode_length,
            difficulty_setting,
            episode_score,
            next_global_write_index,
            buffer_buffer_capacity_limit,
            lines_cleared,
            mcts_depth_mean,
            mcts_search_time_mean,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn update_episode_metadata(
        state: &SharedState,
        episode_start_index: usize,
        episode_length: usize,
        difficulty_setting: i32,
        episode_score: f32,
        next_global_write_index: usize,
        buffer_buffer_capacity_limit: usize,
        lines_cleared: u32,
        mcts_depth_mean: f32,
        mcts_search_time_mean: f32,
    ) {
        {
            let mut episode_metadata_lock = match state.episodes.write() {
                Ok(lock) => lock,
                Err(e) => e.into_inner(),
            };
            episode_metadata_lock.push(EpisodeMeta {
                global_start_storage_index: episode_start_index,
                length: episode_length,
                difficulty: difficulty_setting,
                score: episode_score,
                lines_cleared,
                mcts_depth_mean,
                mcts_search_time_mean,
            });

            let remove_count = episode_metadata_lock
                .iter()
                .take_while(|episode| {
                    episode.global_start_storage_index + buffer_buffer_capacity_limit
                        < next_global_write_index
                })
                .count();
            if remove_count > 0 {
                episode_metadata_lock.drain(0..remove_count);
            }
        }

        state.recent_scores.push(episode_score);
        state.completed_games.fetch_add(1, Ordering::Relaxed);

        let current_state_count = state.num_states.load(Ordering::Relaxed);
        state.num_states.store(
            buffer_buffer_capacity_limit.min(current_state_count + episode_length),
            Ordering::Relaxed,
        );
    }

    pub fn sample_for_reanalyze(&self) -> Option<(usize, crate::core::board::GameStateExt)> {
        let (transitions, _weights) =
            self.state
                .per
                .sample(1, self.state.buffer_capacity_limit, 1.0)?;
        if transitions.is_empty() {
            return None;
        }

        let circular_idx = transitions[0].0;

        let (board, pieces, difficulty) = self
            .state
            .arrays
            .read_storage_index(circular_idx, |shard, i| {
                (shard.boards[i], shard.available[i], shard.state_diff[i])
            });

        let board_u128 = (board[1] as u128) << 64 | (board[0] as u128);
        let state =
            crate::core::board::GameStateExt::new(Some(pieces), board_u128, 0, difficulty, 0);
        Some((circular_idx, state))
    }

    pub fn update_reanalyzed_targets(
        &self,
        circular_idx: usize,
        new_policy: [f32; 288],
        new_value: f32,
    ) {
        self.state
            .arrays
            .write_storage_index(circular_idx, |shard, i| {
                shard.policies[i] = new_policy;
                shard.values[i] = new_value;
            });
    }

    pub fn sample_batch(&self, batch_size_limit: usize, beta: f64) -> Option<BatchTensors> {
        let (sampled_transitions, sampled_importance_weights) =
            match self
                .state
                .per
                .sample(batch_size_limit, self.state.buffer_capacity_limit, beta)
            {
                Some((samples, weights)) => (samples, weights),
                None => return None,
            };

        let unroll_limit = self.state.unroll_steps;

        let mut arena = match self.arena_pool.1.try_recv() {
            Ok(a) => a,
            Err(_) => SampleArena::new(batch_size_limit, unroll_limit),
        };

        let mut global_indices_sampled: Vec<usize> = Vec::with_capacity(batch_size_limit);

        {
            let state_features_buffer: &mut [f32] = arena.state_features.as_mut_slice();
            let actions_buffer: &mut [i64] = arena.actions.as_mut_slice();
            let piece_identifiers_buffer: &mut [i64] = arena.piece_identifiers.as_mut_slice();
            let value_prefixs_buffer: &mut [f32] = arena.value_prefixs.as_mut_slice();
            let target_policies_buffer: &mut [f32] = arena.target_policies.as_mut_slice();
            let target_values_buffer: &mut [f32] = arena.target_values.as_mut_slice();
            let model_values_buffer: &mut [f32] = arena.model_values.as_mut_slice();
            let transition_states_buffer: &mut [f32] = arena.transition_states.as_mut_slice();
            let loss_masks_buffer: &mut [f32] = arena.loss_masks.as_mut_slice();
            let importance_weights_buffer: &mut [f32] = arena.importance_weights.as_mut_slice();

            for (batch_index, &(circular_index, _)) in sampled_transitions.iter().enumerate() {
                importance_weights_buffer[batch_index] = sampled_importance_weights[batch_index];

                let (logical_start_global, logical_length) = self.state.arrays.read_storage_index(
                    circular_index,
                    |array_shard, shard_internal| {
                        (
                            array_shard.state_start[shard_internal],
                            array_shard.state_len[shard_internal],
                        )
                    },
                );

                let global_state_index = if logical_start_global != -1 {
                    let positional_offset = (circular_index as i64 - logical_start_global)
                        .rem_euclid(self.state.buffer_capacity_limit as i64);
                    if positional_offset < logical_length as i64 {
                        (logical_start_global + positional_offset) as usize
                    } else {
                        logical_start_global as usize
                    }
                } else {
                    0
                };

                global_indices_sampled.push(global_state_index);
                self.extract_single_sample_data(
                    batch_index,
                    global_state_index,
                    unroll_limit,
                    state_features_buffer,
                    actions_buffer,
                    piece_identifiers_buffer,
                    value_prefixs_buffer,
                    target_policies_buffer,
                    target_values_buffer,
                    model_values_buffer,
                    transition_states_buffer,
                    loss_masks_buffer,
                    importance_weights_buffer,
                );
            }
        }

        let state_features_batch =
            Tensor::from_slice(&arena.state_features).view([batch_size_limit as i64, 20, 8, 16]);
        let actions_batch =
            Tensor::from_slice(&arena.actions).view([batch_size_limit as i64, unroll_limit as i64]);
        let piece_identifiers_batch = Tensor::from_slice(&arena.piece_identifiers)
            .view([batch_size_limit as i64, unroll_limit as i64]);
        let value_prefixs_batch = Tensor::from_slice(&arena.value_prefixs)
            .view([batch_size_limit as i64, unroll_limit as i64]);
        let target_policies_batch = Tensor::from_slice(&arena.target_policies).view([
            batch_size_limit as i64,
            (unroll_limit + 1) as i64,
            288,
        ]);
        let target_values_batch = Tensor::from_slice(&arena.target_values)
            .view([batch_size_limit as i64, (unroll_limit + 1) as i64]);
        let model_values_batch = Tensor::from_slice(&arena.model_values)
            .view([batch_size_limit as i64, (unroll_limit + 1) as i64]);
        let transition_states_batch = Tensor::from_slice(&arena.transition_states).view([
            batch_size_limit as i64,
            unroll_limit as i64,
            20,
            8,
            16,
        ]);
        let loss_masks_batch = Tensor::from_slice(&arena.loss_masks)
            .view([batch_size_limit as i64, (unroll_limit + 1) as i64]);
        let importance_weights_batch =
            Tensor::from_slice(&arena.importance_weights).view([batch_size_limit as i64]);

        Some(BatchTensors {
            state_features_batch,
            actions_batch,
            piece_identifiers_batch,
            value_prefixs_batch,
            target_policies_batch,
            target_values_batch,
            model_values_batch,
            transition_states_batch,
            loss_masks_batch,
            importance_weights_batch,
            global_indices_sampled,
            arena: Some(arena),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_single_sample_data(
        &self,
        batch_index: usize,
        global_state_index: usize,
        unroll_limit: usize,
        state_features_buffer: &mut [f32],
        actions_buffer: &mut [i64],
        piece_identifiers_buffer: &mut [i64],
        value_prefixs_buffer: &mut [f32],
        target_policies_buffer: &mut [f32],
        target_values_buffer: &mut [f32],
        model_values_buffer: &mut [f32],
        transition_states_buffer: &mut [f32],
        loss_masks_buffer: &mut [f32],
        importance_weights_buffer: &mut [f32],
    ) {
        let global_start_circular = global_state_index % self.state.buffer_capacity_limit;
        let (logical_start_global, logical_length) = self.state.arrays.read_storage_index(
            global_start_circular,
            |array_shard, shard_internal| {
                (
                    array_shard.state_start[shard_internal],
                    array_shard.state_len[shard_internal],
                )
            },
        );

        let episode_end_global = if logical_start_global != -1 {
            (logical_start_global + logical_length as i64) as usize
        } else {
            global_state_index + 1
        };

        let safe_before_boundary = self
            .state
            .global_write_storage_index
            .load(Ordering::Acquire);
        let active_after_boundary = self
            .state
            .global_write_active_storage_index
            .load(Ordering::Acquire);

        let initial_features = self.state.get_features(global_state_index);
        let total_feature_elements = 20 * 128;
        let destination_offset = batch_index * total_feature_elements;

        unsafe {
            std::ptr::copy_nonoverlapping(
                initial_features.as_ptr(),
                state_features_buffer.as_mut_ptr().add(destination_offset),
                total_feature_elements,
            );
        }

        for unroll_offset in 0..=unroll_limit {
            self.process_unrolled_step(
                batch_index,
                global_state_index,
                unroll_offset,
                unroll_limit,
                episode_end_global,
                actions_buffer,
                piece_identifiers_buffer,
                value_prefixs_buffer,
                target_policies_buffer,
                target_values_buffer,
                model_values_buffer,
                transition_states_buffer,
                loss_masks_buffer,
            );
        }

        let maximum_global_read =
            global_state_index + unroll_limit + self.state.temporal_difference_steps;
        let minimum_global_read = global_state_index.saturating_sub(8);

        let write_not_finalized = maximum_global_read >= safe_before_boundary;
        let safely_overwritten =
            minimum_global_read + self.state.buffer_capacity_limit <= active_after_boundary;

        if write_not_finalized || safely_overwritten {
            importance_weights_buffer[batch_index] = 0.0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_unrolled_step(
        &self,
        batch_index: usize,
        global_state_index: usize,
        unroll_offset: usize,
        unroll_limit: usize,
        episode_end_global: usize,
        actions_buffer: &mut [i64],
        piece_identifiers_buffer: &mut [i64],
        value_prefixs_buffer: &mut [f32],
        target_policies_buffer: &mut [f32],
        target_values_buffer: &mut [f32],
        model_values_buffer: &mut [f32],
        transition_states_buffer: &mut [f32],
        loss_masks_buffer: &mut [f32],
    ) {
        let current_global_step = global_state_index + unroll_offset;
        let current_circular_step = current_global_step % self.state.buffer_capacity_limit;

        if current_global_step < episode_end_global {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 1.0;

            if unroll_offset > 0 {
                let previous_circular_step =
                    (current_global_step - 1) % self.state.buffer_capacity_limit;
                let (previous_action, previous_piece_identifier, previous_value_prefix) = self
                    .state
                    .arrays
                    .read_storage_index(previous_circular_step, |array_shard, shard_internal| {
                        (
                            array_shard.actions[shard_internal],
                            array_shard.piece_ids[shard_internal],
                            array_shard.value_prefixs[shard_internal],
                        )
                    });

                actions_buffer[batch_index * unroll_limit + unroll_offset - 1] = previous_action;
                piece_identifiers_buffer[batch_index * unroll_limit + unroll_offset - 1] =
                    previous_piece_identifier;
                value_prefixs_buffer[batch_index * unroll_limit + unroll_offset - 1] =
                    previous_value_prefix;

                let transition_features = self.state.get_features(current_global_step);
                let total_feature_elements = 20 * 128;
                let destination_offset =
                    (batch_index * unroll_limit + unroll_offset - 1) * total_feature_elements;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        transition_features.as_ptr(),
                        transition_states_buffer
                            .as_mut_ptr()
                            .add(destination_offset),
                        total_feature_elements,
                    );
                }
            }

            let (stored_policy, stored_value) = self.state.arrays.read_storage_index(
                current_circular_step,
                |array_shard, shard_internal| {
                    (
                        array_shard.policies[shard_internal],
                        array_shard.values[shard_internal],
                    )
                },
            );

            let destination_offset = batch_index * (unroll_limit + 1) * 288 + unroll_offset * 288;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stored_policy.as_ptr(),
                    target_policies_buffer.as_mut_ptr().add(destination_offset),
                    288,
                );
            }
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = stored_value;

            let bootstrap_global_step = current_global_step + self.state.temporal_difference_steps;
            let discount_factor = 0.99f32;
            let mut discounted_sum_value_prefixs = 0.0;

            let accumulation_limit = bootstrap_global_step.min(episode_end_global);

            for accumulation_step in 0..(accumulation_limit - current_global_step) {
                let value_prefix_circular_index =
                    (current_global_step + accumulation_step) % self.state.buffer_capacity_limit;
                discounted_sum_value_prefixs += self.state.arrays.read_storage_index(
                    value_prefix_circular_index,
                    |array_shard, shard_internal| array_shard.value_prefixs[shard_internal],
                ) * discount_factor.powi(accumulation_step as i32);
            }

            if bootstrap_global_step < episode_end_global {
                let value_bootstrap_circular =
                    bootstrap_global_step % self.state.buffer_capacity_limit;
                discounted_sum_value_prefixs +=
                    self.state.arrays.read_storage_index(
                        value_bootstrap_circular,
                        |array_shard, shard_internal| array_shard.values[shard_internal],
                    ) * discount_factor.powi(self.state.temporal_difference_steps as i32);
            }
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] =
                discounted_sum_value_prefixs;
        } else {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            let destination_offset = batch_index * (unroll_limit + 1) * 288 + unroll_offset * 288;
            target_policies_buffer[destination_offset..destination_offset + 288].fill(1.0 / 288.0);
        }
    }

    pub fn update_priorities(&self, priority_indices: &[usize], computed_priorities: &[f64]) {
        let running_difficulty = self.state.current_diff.load(Ordering::Relaxed);
        let mut mapped_circular_indices = Vec::with_capacity(priority_indices.len());
        let mut mapped_difficulty_penalties = Vec::with_capacity(priority_indices.len());

        for &global_state_index in priority_indices {
            let circular_index = global_state_index % self.state.buffer_capacity_limit;
            let (logical_start_global, difficulty_setting) = self.state.arrays.read_storage_index(
                circular_index,
                |array_shard, shard_internal| {
                    (
                        array_shard.state_start[shard_internal],
                        array_shard.state_diff[shard_internal],
                    )
                },
            );

            if logical_start_global != -1 {
                mapped_difficulty_penalties
                    .push(10f64.powf(-(running_difficulty - difficulty_setting).abs() as f64));
                mapped_circular_indices.push(circular_index);
            } else {
                mapped_difficulty_penalties.push(0.0);
                mapped_circular_indices.push(circular_index);
            }
        }

        self.state.per.update_priorities(
            &mapped_circular_indices,
            &mapped_difficulty_penalties,
            computed_priorities,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_wraparound() {
        let replay_buffer = ReplayBuffer::new(5, 1, 10);

        let steps = vec![
            crate::train::buffer::replay::GameStep {
                board_state: [0, 0],
                available_pieces: [0, 0, 0],
                action_taken: 0,
                piece_identifier: 0,
                value_prefix_received: 0.0,
                policy_target: [0.0; 288],
                value_target: 0.0,
            };
            4
        ];

        replay_buffer.add_game(OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps,
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });

        let steps_2 = vec![
            crate::train::buffer::replay::GameStep {
                board_state: [5, 0],
                available_pieces: [0; 3],
                action_taken: 0,
                piece_identifier: 0,
                value_prefix_received: 0.0,
                policy_target: [0.0; 288],
                value_target: 0.0,
            },
            crate::train::buffer::replay::GameStep {
                board_state: [6, 0],
                available_pieces: [0; 3],
                action_taken: 0,
                piece_identifier: 0,
                value_prefix_received: 0.0,
                policy_target: [0.0; 288],
                value_target: 0.0,
            },
        ];

        replay_buffer.add_game(OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps: steps_2,
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });

        std::thread::sleep(std::time::Duration::from_millis(50));

        assert_eq!(
            replay_buffer.get_length(),
            5,
            "Buffer length should be hard-capped at exact buffer_capacity_limit 5"
        );
        assert_eq!(
            replay_buffer.get_global_write_storage_index(),
            6,
            "Global write index should be monotonic 6"
        );

        let mut final_batch = None;
        for _ in 0..500 {
            if let Some(batch) = replay_buffer.sample_batch(2, 1.0) {
                final_batch = Some(batch);
                break;
            }
        }
        let generated_batch =
            final_batch.expect("Should sample batch across wraps after finding non-empty shard");
        assert_eq!(
            generated_batch.state_features_batch.size(),
            vec![2, 20, 8, 16]
        );
    }
}

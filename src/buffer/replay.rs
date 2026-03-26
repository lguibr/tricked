use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use crate::buffer::state::{EpisodeMeta, SharedState};

pub struct ReplayBuffer {
    pub state: Arc<SharedState>,
}

pub struct BatchTensors {
    pub state_features_batch: Tensor,
    pub actions_batch: Tensor,
    pub piece_identifiers_batch: Tensor,
    pub rewards_batch: Tensor,
    pub target_policies_batch: Tensor,
    pub target_values_batch: Tensor,
    #[allow(dead_code)]
    pub model_values_batch: Tensor,
    pub transition_states_batch: Tensor,
    pub loss_masks_batch: Tensor,
    pub importance_weights_batch: Tensor,
    pub global_indices_sampled: Vec<usize>,
}

impl ReplayBuffer {
    pub fn new(
        total_capacity: usize,
        unroll_steps: usize,
        temporal_difference_steps: usize,
    ) -> Self {
        let shared_state = SharedState {
            capacity: total_capacity,
            unroll_steps,
            td_steps: temporal_difference_steps,
            current_diff: AtomicI32::new(1),
            global_write_idx: AtomicUsize::new(0),
            global_write_active_idx: AtomicUsize::new(0),
            num_states: AtomicUsize::new(0),

            arrays: crate::buffer::state::ShardedStorageArrays::new(total_capacity, 64),
            per: crate::sumtree::ShardedPrioritizedReplay::new(total_capacity, 0.6, 0.4, 64),

            episodes: std::sync::Mutex::new(Vec::new()),
            recent_scores: std::sync::Mutex::new(Vec::new()),
            completed_games: AtomicUsize::new(0),
        };

        Self {
            state: Arc::new(shared_state),
        }
    }

    pub fn get_length(&self) -> usize {
        self.state.num_states.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub fn get_global_write_idx(&self) -> usize {
        self.state.global_write_idx.load(Ordering::Acquire)
    }

    #[allow(dead_code)]
    pub fn get_and_clear_metrics(&self) -> (Vec<f32>, f32, f32, f32) {
        let mut recent_scores_lock = match self.state.recent_scores.lock() {
            Ok(lock) => lock,
            Err(poisoned) => poisoned.into_inner(),
        };
        if recent_scores_lock.is_empty() {
            return (vec![], 0.0, 0.0, 0.0);
        }
        let cloned_scores = std::mem::take(&mut *recent_scores_lock);
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

pub struct AddGameParams<'a> {
    pub difficulty_setting: i32,
    pub episode_score: f32,
    pub board_states: &'a [[u64; 2]],
    pub available_pieces: &'a [[i32; 3]],
    pub actions_taken: &'a [i64],
    pub piece_identifiers: &'a [i64],
    pub rewards_received: &'a [f32],
    pub policy_targets: &'a [[f32; 288]],
    pub value_targets: &'a [f32],
}

impl ReplayBuffer {
    pub fn add_game(&self, params: AddGameParams) {
        let AddGameParams {
            difficulty_setting,
            episode_score,
            board_states,
            available_pieces,
            actions_taken,
            piece_identifiers,
            rewards_received,
            policy_targets,
            value_targets,
        } = params;
        let episode_length = board_states.len();
        if episode_length == 0 {
            return;
        }

        let episode_start_index = self.state.global_write_idx.load(Ordering::Relaxed);
        let buffer_capacity = self.state.capacity;
        let running_difficulty = self.state.current_diff.load(Ordering::Relaxed);
        let next_global_write_index = episode_start_index + episode_length;

        self.state
            .global_write_active_idx
            .store(next_global_write_index, Ordering::Release);

        if running_difficulty == 0 || difficulty_setting != running_difficulty {
            self.state
                .current_diff
                .store(difficulty_setting, Ordering::Relaxed);
        }

        let active_difficulty = self.state.current_diff.load(Ordering::Relaxed);
        let absolute_difficulty_penalty =
            10f64.powf(-(active_difficulty - difficulty_setting).abs() as f64);

        for transition_offset in 0..episode_length {
            let circular_write_index = (episode_start_index + transition_offset) % buffer_capacity;
            self.state.arrays.write_idx(
                circular_write_index,
                |memory_shard, internal_shard_index| {
                    memory_shard.state_start[internal_shard_index] = episode_start_index as i64;
                    memory_shard.state_diff[internal_shard_index] = difficulty_setting;
                    memory_shard.state_len[internal_shard_index] = episode_length as i32;

                    memory_shard.boards[internal_shard_index] = board_states[transition_offset];
                    memory_shard.available[internal_shard_index] =
                        available_pieces[transition_offset];
                    memory_shard.actions[internal_shard_index] = actions_taken[transition_offset];
                    memory_shard.piece_ids[internal_shard_index] =
                        piece_identifiers[transition_offset];
                    memory_shard.rewards[internal_shard_index] =
                        rewards_received[transition_offset];
                    memory_shard.policies[internal_shard_index] = policy_targets[transition_offset];
                    memory_shard.values[internal_shard_index] = value_targets[transition_offset];
                },
            );
        }

        let mut circular_indices_to_add = Vec::with_capacity(episode_length);
        let mut transition_penalties = Vec::with_capacity(episode_length);
        for transition_offset in 0..episode_length {
            circular_indices_to_add
                .push((episode_start_index + transition_offset) % buffer_capacity);
            transition_penalties.push(absolute_difficulty_penalty);
        }
        self.state
            .per
            .add_batch(&circular_indices_to_add, &transition_penalties);

        self.state
            .global_write_idx
            .store(next_global_write_index, Ordering::Release);

        self.update_episode_metadata(
            episode_start_index,
            episode_length,
            difficulty_setting,
            episode_score,
            next_global_write_index,
            buffer_capacity,
        );
    }

    fn update_episode_metadata(
        &self,
        episode_start_index: usize,
        episode_length: usize,
        difficulty_setting: i32,
        episode_score: f32,
        next_global_write_index: usize,
        buffer_capacity: usize,
    ) {
        {
            let mut episode_metadata_lock = match self.state.episodes.lock() {
                Ok(lock) => lock,
                Err(e) => e.into_inner(),
            };
            episode_metadata_lock.push(EpisodeMeta {
                global_start_idx: episode_start_index,
                length: episode_length,
                difficulty: difficulty_setting,
                score: episode_score,
            });

            let mut valid_episodes_retained = Vec::new();
            for episode in episode_metadata_lock.iter() {
                if episode.global_start_idx + buffer_capacity >= next_global_write_index {
                    valid_episodes_retained.push(episode.clone());
                }
            }
            *episode_metadata_lock = valid_episodes_retained;
        }

        let mut recent_scores_lock = match self.state.recent_scores.lock() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        recent_scores_lock.push(episode_score);
        self.state.completed_games.fetch_add(1, Ordering::Relaxed);

        let current_state_count = self.state.num_states.load(Ordering::Relaxed);
        self.state.num_states.store(
            buffer_capacity.min(current_state_count + episode_length),
            Ordering::Relaxed,
        );
    }

    pub fn sample_batch(
        &self,
        batch_size_limit: usize,
        computation_device: Device,
    ) -> Option<BatchTensors> {
        let (sampled_transitions, sampled_importance_weights) =
            match self.state.per.sample(batch_size_limit, self.state.capacity) {
                Some((samples, weights)) => (samples, weights),
                None => return None,
            };

        let unroll_limit = self.state.unroll_steps;
        let mut state_features_buffer: Vec<f32> = vec![0.0; batch_size_limit * 20 * 96];
        let mut actions_buffer: Vec<i64> = vec![0; batch_size_limit * unroll_limit];
        let mut piece_identifiers_buffer: Vec<i64> = vec![0; batch_size_limit * unroll_limit];
        let mut rewards_buffer: Vec<f32> = vec![0.0; batch_size_limit * unroll_limit];
        let mut target_policies_buffer: Vec<f32> =
            vec![0.0; batch_size_limit * (unroll_limit + 1) * 288];
        let mut target_values_buffer: Vec<f32> = vec![0.0; batch_size_limit * (unroll_limit + 1)];
        let mut model_values_buffer: Vec<f32> = vec![0.0; batch_size_limit * (unroll_limit + 1)];
        let mut transition_states_buffer: Vec<f32> =
            vec![0.0; batch_size_limit * unroll_limit * 20 * 96];
        let mut loss_masks_buffer: Vec<f32> = vec![0.0; batch_size_limit * (unroll_limit + 1)];
        let mut importance_weights_buffer: Vec<f32> = vec![0.0; batch_size_limit];
        let mut global_indices_sampled: Vec<usize> = Vec::with_capacity(batch_size_limit);

        for (batch_index, &(circular_index, _)) in sampled_transitions.iter().enumerate() {
            importance_weights_buffer[batch_index] = sampled_importance_weights[batch_index];

            let (logical_start_global, logical_length) =
                self.state
                    .arrays
                    .read_idx(circular_index, |array_shard, shard_internal| {
                        (
                            array_shard.state_start[shard_internal],
                            array_shard.state_len[shard_internal],
                        )
                    });

            let global_state_index = if logical_start_global != -1 {
                let positional_offset = (circular_index as i64 - logical_start_global)
                    .rem_euclid(self.state.capacity as i64);
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
                &mut state_features_buffer,
                &mut actions_buffer,
                &mut piece_identifiers_buffer,
                &mut rewards_buffer,
                &mut target_policies_buffer,
                &mut target_values_buffer,
                &mut model_values_buffer,
                &mut transition_states_buffer,
                &mut loss_masks_buffer,
                &mut importance_weights_buffer,
            );
        }

        Some(self.construct_batch_tensors(
            batch_size_limit,
            unroll_limit,
            computation_device,
            state_features_buffer,
            actions_buffer,
            piece_identifiers_buffer,
            rewards_buffer,
            target_policies_buffer,
            target_values_buffer,
            model_values_buffer,
            transition_states_buffer,
            loss_masks_buffer,
            importance_weights_buffer,
            global_indices_sampled,
        ))
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
        rewards_buffer: &mut [f32],
        target_policies_buffer: &mut [f32],
        target_values_buffer: &mut [f32],
        model_values_buffer: &mut [f32],
        transition_states_buffer: &mut [f32],
        loss_masks_buffer: &mut [f32],
        importance_weights_buffer: &mut [f32],
    ) {
        let global_start_circular = global_state_index % self.state.capacity;
        let (logical_start_global, logical_length) =
            self.state
                .arrays
                .read_idx(global_start_circular, |array_shard, shard_internal| {
                    (
                        array_shard.state_start[shard_internal],
                        array_shard.state_len[shard_internal],
                    )
                });

        let episode_end_global = if logical_start_global != -1 {
            (logical_start_global + logical_length as i64) as usize
        } else {
            global_state_index + 1
        };

        let safe_before_boundary = self.state.global_write_idx.load(Ordering::Acquire);
        let active_after_boundary = self.state.global_write_active_idx.load(Ordering::Acquire);

        let initial_features = self.state.get_features(global_state_index);
        for depth_channel in 0..20 {
            for spatial_position in 0..96 {
                state_features_buffer
                    [batch_index * 20 * 96 + depth_channel * 96 + spatial_position] =
                    initial_features[depth_channel * 96 + spatial_position];
            }
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
                rewards_buffer,
                target_policies_buffer,
                target_values_buffer,
                model_values_buffer,
                transition_states_buffer,
                loss_masks_buffer,
            );
        }

        let maximum_global_read = global_state_index + unroll_limit + self.state.td_steps;
        let minimum_global_read = global_state_index.saturating_sub(8);

        let write_not_finalized = maximum_global_read >= safe_before_boundary;
        let safely_overwritten = minimum_global_read + self.state.capacity <= active_after_boundary;

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
        rewards_buffer: &mut [f32],
        target_policies_buffer: &mut [f32],
        target_values_buffer: &mut [f32],
        model_values_buffer: &mut [f32],
        transition_states_buffer: &mut [f32],
        loss_masks_buffer: &mut [f32],
    ) {
        let current_global_step = global_state_index + unroll_offset;
        let current_circular_step = current_global_step % self.state.capacity;

        if current_global_step < episode_end_global {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 1.0;

            if unroll_offset > 0 {
                let previous_circular_step = (current_global_step - 1) % self.state.capacity;
                let (previous_action, previous_piece_identifier, previous_reward) = self
                    .state
                    .arrays
                    .read_idx(previous_circular_step, |array_shard, shard_internal| {
                        (
                            array_shard.actions[shard_internal],
                            array_shard.piece_ids[shard_internal],
                            array_shard.rewards[shard_internal],
                        )
                    });

                actions_buffer[batch_index * unroll_limit + unroll_offset - 1] = previous_action;
                piece_identifiers_buffer[batch_index * unroll_limit + unroll_offset - 1] =
                    previous_piece_identifier;
                rewards_buffer[batch_index * unroll_limit + unroll_offset - 1] = previous_reward;

                let transition_features = self.state.get_features(current_global_step);
                for depth_channel in 0..20 {
                    for spatial_position in 0..96 {
                        transition_states_buffer[(batch_index * unroll_limit + unroll_offset
                            - 1)
                            * 20
                            * 96
                            + depth_channel * 96
                            + spatial_position] =
                            transition_features[depth_channel * 96 + spatial_position];
                    }
                }
            }

            let (stored_policy, stored_value) =
                self.state
                    .arrays
                    .read_idx(current_circular_step, |array_shard, shard_internal| {
                        (
                            array_shard.policies[shard_internal],
                            array_shard.values[shard_internal],
                        )
                    });

            for spatial_position in 0..288 {
                target_policies_buffer[batch_index * (unroll_limit + 1) * 288
                    + unroll_offset * 288
                    + spatial_position] = stored_policy[spatial_position];
            }
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = stored_value;

            let bootstrap_global_step = current_global_step + self.state.td_steps;
            let discount_factor = 0.99f32;
            let mut discounted_sum_rewards = 0.0;

            let accumulation_limit = bootstrap_global_step.min(episode_end_global);

            for accumulation_step in 0..(accumulation_limit - current_global_step) {
                let reward_circular_index =
                    (current_global_step + accumulation_step) % self.state.capacity;
                discounted_sum_rewards += self
                    .state
                    .arrays
                    .read_idx(reward_circular_index, |array_shard, shard_internal| {
                        array_shard.rewards[shard_internal]
                    })
                    * discount_factor.powi(accumulation_step as i32);
            }

            if bootstrap_global_step < episode_end_global {
                let value_bootstrap_circular = bootstrap_global_step % self.state.capacity;
                discounted_sum_rewards += self
                    .state
                    .arrays
                    .read_idx(value_bootstrap_circular, |array_shard, shard_internal| {
                        array_shard.values[shard_internal]
                    })
                    * discount_factor.powi(self.state.td_steps as i32);
            }
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] =
                discounted_sum_rewards;
        } else {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            for spatial_position in 0..288 {
                target_policies_buffer[batch_index * (unroll_limit + 1) * 288
                    + unroll_offset * 288
                    + spatial_position] = 1.0 / 288.0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn construct_batch_tensors(
        &self,
        batch_size_limit: usize,
        unroll_limit: usize,
        computation_device: Device,
        state_features_buffer: Vec<f32>,
        actions_buffer: Vec<i64>,
        piece_identifiers_buffer: Vec<i64>,
        rewards_buffer: Vec<f32>,
        target_policies_buffer: Vec<f32>,
        target_values_buffer: Vec<f32>,
        model_values_buffer: Vec<f32>,
        transition_states_buffer: Vec<f32>,
        loss_masks_buffer: Vec<f32>,
        importance_weights_buffer: Vec<f32>,
        global_indices_sampled: Vec<usize>,
    ) -> BatchTensors {
        BatchTensors {
            state_features_batch: Tensor::from_slice(&state_features_buffer)
                .view((batch_size_limit as i64, 20, 96))
                .to_kind(Kind::BFloat16)
                .to_device(computation_device),
            actions_batch: Tensor::from_slice(&actions_buffer)
                .view((batch_size_limit as i64, unroll_limit as i64))
                .to_device(computation_device),
            piece_identifiers_batch: Tensor::from_slice(&piece_identifiers_buffer)
                .view((batch_size_limit as i64, unroll_limit as i64))
                .to_device(computation_device),
            rewards_batch: Tensor::from_slice(&rewards_buffer)
                .view((batch_size_limit as i64, unroll_limit as i64))
                .to_device(computation_device)
                .nan_to_num(0.0, Some(0.0), Some(0.0)),
            target_policies_batch: Tensor::from_slice(&target_policies_buffer)
                .view((batch_size_limit as i64, (unroll_limit + 1) as i64, 288))
                .to_device(computation_device)
                .nan_to_num(0.0, Some(0.0), Some(0.0)),
            target_values_batch: Tensor::from_slice(&target_values_buffer)
                .view((batch_size_limit as i64, (unroll_limit + 1) as i64))
                .to_device(computation_device)
                .nan_to_num(0.0, Some(0.0), Some(0.0)),
            model_values_batch: Tensor::from_slice(&model_values_buffer)
                .view((batch_size_limit as i64, (unroll_limit + 1) as i64))
                .to_device(computation_device)
                .nan_to_num(0.0, Some(0.0), Some(0.0)),
            transition_states_batch: Tensor::from_slice(&transition_states_buffer)
                .view((batch_size_limit as i64, unroll_limit as i64, 20, 96))
                .to_kind(Kind::BFloat16)
                .to_device(computation_device),
            loss_masks_batch: Tensor::from_slice(&loss_masks_buffer)
                .view((batch_size_limit as i64, (unroll_limit + 1) as i64))
                .to_device(computation_device),
            importance_weights_batch: Tensor::from_slice(&importance_weights_buffer)
                .view((batch_size_limit as i64,))
                .to_device(computation_device),
            global_indices_sampled,
        }
    }

    pub fn update_priorities(&self, priority_indices: &[usize], computed_priorities: &[f64]) {
        let running_difficulty = self.state.current_diff.load(Ordering::Relaxed);
        let mut mapped_circular_indices = Vec::with_capacity(priority_indices.len());
        let mut mapped_difficulty_penalties = Vec::with_capacity(priority_indices.len());

        for &global_state_index in priority_indices {
            let circular_index = global_state_index % self.state.capacity;
            let (logical_start_global, difficulty_setting) =
                self.state
                    .arrays
                    .read_idx(circular_index, |array_shard, shard_internal| {
                        (
                            array_shard.state_start[shard_internal],
                            array_shard.state_diff[shard_internal],
                        )
                    });

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
    use tch::Device;

    #[test]
    fn test_ring_buffer_wraparound() {
        let replay_buffer = ReplayBuffer::new(5, 1, 10);

        let board_states = vec![[1, 0], [2, 0], [3, 0], [4, 0]];
        let available_pieces = vec![[0; 3]; 4];
        let actions_taken = vec![0; 4];
        let piece_identifiers = vec![0; 4];
        let rewards_received = vec![0.0; 4];
        let policy_targets = vec![[0.0; 288]; 4];
        let value_targets = vec![0.0; 4];

        replay_buffer.add_game(AddGameParams {
            difficulty_setting: 6,
            episode_score: 1.0,
            board_states: &board_states,
            available_pieces: &available_pieces,
            actions_taken: &actions_taken,
            piece_identifiers: &piece_identifiers,
            rewards_received: &rewards_received,
            policy_targets: &policy_targets,
            value_targets: &value_targets,
        });

        let board_states_2 = vec![[5, 0], [6, 0]];
        let available_pieces_2 = vec![[0; 3]; 2];
        let actions_taken_2 = vec![0; 2];
        let piece_identifiers_2 = vec![0; 2];
        let rewards_received_2 = vec![0.0; 2];
        let policy_targets_2 = vec![[0.0; 288]; 2];
        let value_targets_2 = vec![0.0; 2];

        replay_buffer.add_game(AddGameParams {
            difficulty_setting: 6,
            episode_score: 1.0,
            board_states: &board_states_2,
            available_pieces: &available_pieces_2,
            actions_taken: &actions_taken_2,
            piece_identifiers: &piece_identifiers_2,
            rewards_received: &rewards_received_2,
            policy_targets: &policy_targets_2,
            value_targets: &value_targets_2,
        });

        assert_eq!(
            replay_buffer.get_length(),
            5,
            "Buffer length should be hard-capped at exact capacity 5"
        );
        assert_eq!(
            replay_buffer.get_global_write_idx(),
            6,
            "Global write index should be monotonic 6"
        );

        let mut final_batch = None;
        for _ in 0..500 {
            if let Some(batch) = replay_buffer.sample_batch(2, Device::Cpu) {
                final_batch = Some(batch);
                break;
            }
        }
        let generated_batch =
            final_batch.expect("Should sample batch across wraps after finding non-empty shard");
        assert_eq!(generated_batch.state_features_batch.size(), vec![2, 20, 96]);
    }
}

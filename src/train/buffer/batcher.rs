use std::sync::atomic::Ordering;

use crate::core::board::GameStateExt;
use crate::train::buffer::core::{BatchTensors, ReplayBuffer};

impl ReplayBuffer {
    pub fn sample_for_reanalyze(&self) -> Option<(usize, GameStateExt)> {
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
        let state = GameStateExt::new(Some(pieces), board_u128, 0, difficulty, 0);
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

        let arena = match self.arena_pool.1.recv() {
            Ok(a) => a,
            Err(_) => return None,
        };

        let mut global_indices_sampled: Vec<usize> = Vec::with_capacity(batch_size_limit);

        {
            let state_features_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.state_features.data_ptr() as *mut f32,
                    batch_size_limit * 20 * 128,
                )
            };
            let actions_buffer: &mut [i64] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.actions.data_ptr() as *mut i64,
                    batch_size_limit * unroll_limit,
                )
            };
            let piece_identifiers_buffer: &mut [i64] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.piece_identifiers.data_ptr() as *mut i64,
                    batch_size_limit * unroll_limit,
                )
            };
            let value_prefixs_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.value_prefixs.data_ptr() as *mut f32,
                    batch_size_limit * unroll_limit,
                )
            };
            let target_policies_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.target_policies.data_ptr() as *mut f32,
                    batch_size_limit * (unroll_limit + 1) * 288,
                )
            };
            let target_values_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.target_values.data_ptr() as *mut f32,
                    batch_size_limit * (unroll_limit + 1),
                )
            };
            let model_values_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.model_values.data_ptr() as *mut f32,
                    batch_size_limit * (unroll_limit + 1),
                )
            };

            let unrolled_state_features_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.unrolled_state_features.data_ptr() as *mut f32,
                    batch_size_limit * unroll_limit * 20 * 128,
                )
            };

            let loss_masks_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.loss_masks.data_ptr() as *mut f32,
                    batch_size_limit * (unroll_limit + 1),
                )
            };
            let importance_weights_buffer: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    arena.importance_weights.data_ptr() as *mut f32,
                    batch_size_limit,
                )
            };

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
                    unrolled_state_features_buffer,
                    loss_masks_buffer,
                    importance_weights_buffer,
                );
            }
        }

        let state_features_batch = arena.state_features.shallow_clone();
        let actions_batch = arena.actions.shallow_clone();
        let piece_identifiers_batch = arena.piece_identifiers.shallow_clone();
        let value_prefixs_batch = arena.value_prefixs.shallow_clone();
        let target_policies_batch = arena.target_policies.shallow_clone();
        let target_values_batch = arena.target_values.shallow_clone();
        let model_values_batch = arena.model_values.shallow_clone();

        let unrolled_state_features_batch = arena.unrolled_state_features.shallow_clone();

        let loss_masks_batch = arena.loss_masks.shallow_clone();
        let importance_weights_batch = arena.importance_weights.shallow_clone();

        Some(BatchTensors {
            state_features_batch,
            actions_batch,
            piece_identifiers_batch,
            value_prefixs_batch,
            target_policies_batch,
            target_values_batch,
            model_values_batch,
            unrolled_state_features_batch,
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
        unrolled_state_features_buffer: &mut [f32],
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
                unrolled_state_features_buffer,
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
        unrolled_state_features_buffer: &mut [f32],
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

                let _difficulty_setting = self
                    .state
                    .arrays
                    .read_storage_index(current_circular_step, |shard, i| shard.state_diff[i]);

                let (board, _available) = self
                    .state
                    .arrays
                    .read_storage_index(current_circular_step, |shard, i| {
                        (shard.boards[i], shard.available[i])
                    });

                let board_u128 = ((board[1] as u128) << 64) | (board[0] as u128);
                let history_boards = self.state.get_historical_boards(current_circular_step);

                let board_offset = (batch_index * unroll_limit + unroll_offset - 1) * 20 * 128;
                for i in 0..(20 * 128) {
                    unrolled_state_features_buffer[board_offset + i] = 0.0;
                }

                let mut fill_channel = |plane_idx: usize, val0: u64, val1: u64| {
                    let plane_offset = board_offset + plane_idx * 128;
                    for i in 0..64 {
                        if (val0 & (1 << i)) != 0 {
                            unrolled_state_features_buffer[plane_offset + i] = 1.0;
                        }
                        if (val1 & (1 << i)) != 0 {
                            unrolled_state_features_buffer[plane_offset + 64 + i] = 1.0;
                        }
                    }
                };

                fill_channel(0, board[0], board[1]);
                for i in 0..7 {
                    let hist_u128 = if i < history_boards.len() {
                        history_boards[i]
                    } else {
                        board_u128
                    };
                    fill_channel(
                        i + 1,
                        (hist_u128 & 0xFFFFFFFFFFFFFFFF) as u64,
                        (hist_u128 >> 64) as u64,
                    );
                }
            }

            let (stored_policy, stored_value, stored_td) = self.state.arrays.read_storage_index(
                current_circular_step,
                |array_shard, shard_internal| {
                    (
                        array_shard.policies[shard_internal],
                        array_shard.values[shard_internal],
                        array_shard.td_targets[shard_internal],
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

            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = stored_td;
        } else {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            let destination_offset = batch_index * (unroll_limit + 1) * 288 + unroll_offset * 288;
            target_policies_buffer[destination_offset..destination_offset + 288].fill(1.0 / 288.0);
        }
    }
}

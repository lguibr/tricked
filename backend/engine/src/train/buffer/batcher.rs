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
                let mut policy_u8 = [0u8; 288];
                for idx in 0..288 {
                    policy_u8[idx] = (new_policy[idx] * 255.0).round() as u8;
                }
                shard.policies[i] = policy_u8;
                shard.values[i] = half::f16::from_f32(new_value);
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

        let mut arena = match self.arena_pool.1.recv() {
            Ok(a) => a,
            Err(_) => return None,
        };

        let arena_batch_cap = arena.board_states.len() / 2;
        let arena_unroll_cap = arena.actions.len() / arena_batch_cap;

        assert!(
            batch_size_limit <= arena_batch_cap,
            "Requested batch_size_limit {} exceeds SampleArena capacity {}",
            batch_size_limit,
            arena_batch_cap
        );
        assert!(
            unroll_limit <= arena_unroll_cap,
            "Requested unroll_limit {} exceeds SampleArena capacity {}",
            unroll_limit,
            arena_unroll_cap
        );

        arena.global_indices_sampled.clear();

        {
            let board_states_buffer: &mut [i64] = &mut arena.board_states;
            let board_histories_buffer: &mut [i64] = &mut arena.board_histories;
            let board_available_buffer: &mut [i32] = &mut arena.board_available;
            let board_historical_acts_buffer: &mut [i32] = &mut arena.board_historical_acts;
            let board_diff_buffer: &mut [i32] = &mut arena.board_diff;
            let actions_buffer: &mut [i64] = &mut arena.actions;
            let piece_identifiers_buffer: &mut [i64] = &mut arena.piece_identifiers;
            let value_prefixs_buffer: &mut [f32] = &mut arena.value_prefixs;
            let target_policies_buffer: &mut [f32] = &mut arena.target_policies;
            let target_values_buffer: &mut [f32] = &mut arena.target_values;
            let model_values_buffer: &mut [f32] = &mut arena.model_values;

            let raw_unrolled_boards_buffer: &mut [i64] = &mut arena.raw_unrolled_boards;
            let raw_unrolled_histories_buffer: &mut [i64] = &mut arena.raw_unrolled_histories;
            let raw_unrolled_available_buffer: &mut [i32] = &mut arena.raw_unrolled_available;
            let raw_unrolled_actions_buffer: &mut [i32] = &mut arena.raw_unrolled_actions;
            let raw_unrolled_diff_buffer: &mut [i32] = &mut arena.raw_unrolled_diff;

            let loss_masks_buffer: &mut [f32] = &mut arena.loss_masks;
            let importance_weights_buffer: &mut [f32] = &mut arena.importance_weights;

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

                arena.global_indices_sampled.push(global_state_index);
                self.extract_single_sample_data(
                    batch_index,
                    global_state_index,
                    unroll_limit,
                    board_states_buffer,
                    board_histories_buffer,
                    board_available_buffer,
                    board_historical_acts_buffer,
                    board_diff_buffer,
                    actions_buffer,
                    piece_identifiers_buffer,
                    value_prefixs_buffer,
                    target_policies_buffer,
                    target_values_buffer,
                    model_values_buffer,
                    raw_unrolled_boards_buffer,
                    raw_unrolled_histories_buffer,
                    raw_unrolled_available_buffer,
                    raw_unrolled_actions_buffer,
                    raw_unrolled_diff_buffer,
                    loss_masks_buffer,
                    importance_weights_buffer,
                );
            }
        }

        let board_states_batch = arena.board_states.clone();
        let board_histories_batch = arena.board_histories.clone();
        let board_available_batch = arena.board_available.clone();
        let board_historical_acts_batch = arena.board_historical_acts.clone();
        let board_diff_batch = arena.board_diff.clone();
        let actions_batch = arena.actions.clone();
        let piece_identifiers_batch = arena.piece_identifiers.clone();
        let value_prefixs_batch = arena.value_prefixs.clone();
        let target_policies_batch = arena.target_policies.clone();
        let target_values_batch = arena.target_values.clone();
        let model_values_batch = arena.model_values.clone();

        let raw_unrolled_boards_batch = arena.raw_unrolled_boards.clone();
        let raw_unrolled_histories_batch = arena.raw_unrolled_histories.clone();
        let raw_unrolled_available_batch = arena.raw_unrolled_available.clone();
        let raw_unrolled_actions_batch = arena.raw_unrolled_actions.clone();
        let raw_unrolled_diff_batch = arena.raw_unrolled_diff.clone();

        let loss_masks_batch = arena.loss_masks.clone();
        let importance_weights_batch = arena.importance_weights.clone();

        Some(BatchTensors {
            board_states_batch,
            board_histories_batch,
            board_available_batch,
            board_historical_acts_batch,
            board_diff_batch,
            actions_batch,
            piece_identifiers_batch,
            value_prefixs_batch,
            target_policies_batch,
            target_values_batch,
            model_values_batch,
            raw_unrolled_boards_batch,
            raw_unrolled_histories_batch,
            raw_unrolled_available_batch,
            raw_unrolled_actions_batch,
            raw_unrolled_diff_batch,
            loss_masks_batch,
            importance_weights_batch,
            arena: Some(arena),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_single_sample_data(
        &self,
        batch_index: usize,
        global_state_index: usize,
        unroll_limit: usize,
        board_states_buffer: &mut [i64],
        board_histories_buffer: &mut [i64],
        board_available_buffer: &mut [i32],
        board_historical_acts_buffer: &mut [i32],
        board_diff_buffer: &mut [i32],
        actions_buffer: &mut [i64],
        piece_identifiers_buffer: &mut [i64],
        value_prefixs_buffer: &mut [f32],
        target_policies_buffer: &mut [f32],
        target_values_buffer: &mut [f32],
        model_values_buffer: &mut [f32],
        raw_unrolled_boards_buffer: &mut [i64],
        raw_unrolled_histories_buffer: &mut [i64],
        raw_unrolled_available_buffer: &mut [i32],
        raw_unrolled_actions_buffer: &mut [i32],
        raw_unrolled_diff_buffer: &mut [i32],
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

        let _difficulty_setting = self
            .state
            .arrays
            .read_storage_index(global_start_circular, |shard, i| shard.state_diff[i]);

        let (board, available) = self
            .state
            .arrays
            .read_storage_index(global_start_circular, |shard, i| {
                (shard.boards[i], shard.available[i])
            });
        let history_boards = self.state.get_historical_boards(global_start_circular);
        let history_actions = self.state.get_historical_actions(global_start_circular);

        board_diff_buffer[batch_index] = _difficulty_setting;

        let avail_offset = batch_index * 3;
        board_available_buffer[avail_offset] = available[0] as i32;
        board_available_buffer[avail_offset + 1] = available[1] as i32;
        board_available_buffer[avail_offset + 2] = available[2] as i32;

        let acts_offset = batch_index * 3;
        for i in 0..3 {
            board_historical_acts_buffer[acts_offset + i] = if i < history_actions.len() {
                history_actions[i]
            } else {
                -1
            };
        }

        let bs_offset = batch_index * 2;
        board_states_buffer[bs_offset] = board[0] as i64;
        board_states_buffer[bs_offset + 1] = board[1] as i64;

        let bh_offset = batch_index * 14;
        let board_u128 = ((board[1] as u128) << 64) | (board[0] as u128);
        for i in 0..7 {
            let hist_u128 = if i < history_boards.len() {
                history_boards[i]
            } else {
                board_u128
            };
            board_histories_buffer[bh_offset + i * 2] = (hist_u128 & 0xFFFFFFFFFFFFFFFF) as i64;
            board_histories_buffer[bh_offset + i * 2 + 1] = (hist_u128 >> 64) as i64;
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
                raw_unrolled_boards_buffer,
                raw_unrolled_histories_buffer,
                raw_unrolled_available_buffer,
                raw_unrolled_actions_buffer,
                raw_unrolled_diff_buffer,
                loss_masks_buffer,
            );
        }

        let maximum_global_read = std::cmp::min(
            episode_end_global,
            global_state_index + unroll_limit + self.state.temporal_difference_steps,
        );
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
        raw_unrolled_boards_buffer: &mut [i64],
        raw_unrolled_histories_buffer: &mut [i64],
        raw_unrolled_available_buffer: &mut [i32],
        raw_unrolled_actions_buffer: &mut [i32],
        raw_unrolled_diff_buffer: &mut [i32],
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
                    previous_value_prefix.to_f32();

                let _difficulty_setting = self
                    .state
                    .arrays
                    .read_storage_index(current_circular_step, |shard, i| shard.state_diff[i]);

                let (board, available) = self
                    .state
                    .arrays
                    .read_storage_index(current_circular_step, |shard, i| {
                        (shard.boards[i], shard.available[i])
                    });

                let board_u128 = ((board[1] as u128) << 64) | (board[0] as u128);
                let history_boards = self.state.get_historical_boards(current_circular_step);
                let history_actions = self.state.get_historical_actions(current_circular_step);

                let board_offset = (batch_index * unroll_limit + unroll_offset - 1) * 2;
                raw_unrolled_boards_buffer[board_offset] = board[0] as i64;
                raw_unrolled_boards_buffer[board_offset + 1] = board[1] as i64;

                let avail_offset = (batch_index * unroll_limit + unroll_offset - 1) * 3;
                raw_unrolled_available_buffer[avail_offset] = available[0] as i32;
                raw_unrolled_available_buffer[avail_offset + 1] = available[1] as i32;
                raw_unrolled_available_buffer[avail_offset + 2] = available[2] as i32;

                let acts_offset = (batch_index * unroll_limit + unroll_offset - 1) * 3;
                for i in 0..3 {
                    raw_unrolled_actions_buffer[acts_offset + i] = if i < history_actions.len() {
                        history_actions[i]
                    } else {
                        -1
                    };
                }

                raw_unrolled_diff_buffer[batch_index * unroll_limit + unroll_offset - 1] = _difficulty_setting;

                let hist_offset = (batch_index * unroll_limit + unroll_offset - 1) * 14;
                for i in 0..7 {
                    let hist_u128 = if i < history_boards.len() {
                        history_boards[i]
                    } else {
                        board_u128
                    };
                    raw_unrolled_histories_buffer[hist_offset + i * 2] =
                        (hist_u128 & 0xFFFFFFFFFFFFFFFF) as i64;
                    raw_unrolled_histories_buffer[hist_offset + i * 2 + 1] =
                        (hist_u128 >> 64) as i64;
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
            for i in 0..288 {
                target_policies_buffer[destination_offset + i] = (stored_policy[i] as f32) / 255.0;
            }
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] =
                stored_value.to_f32();

            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] =
                stored_td.to_f32();
        } else {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            let destination_offset = batch_index * (unroll_limit + 1) * 288 + unroll_offset * 288;
            target_policies_buffer[destination_offset..destination_offset + 288].fill(1.0 / 288.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::catch_unwind;

    #[test]
    fn test_sample_arena_bounds_panic() {
        let buffer = ReplayBuffer::new(
            100,
            5,
            3,
            10,
            None,
            0.99,
            0.95,
            0.6,
            0.4,
            "test_run_id".to_string(),
            "test_run_name".to_string(),
            "test_run".to_string(),
        );

        // Fill buffer so length > batch_size to bypass the early None return
        let mut steps = vec![];
        for _ in 0..25 {
            steps.push(crate::train::buffer::core::GameStep {
                board_state: [0, 0],
                available_pieces: [0; 3],
                action_taken: 0,
                piece_identifier: 0,
                policy_target: vec![0.0; 288],
                value_target: 0.0,
                value_prefix_received: 0.0,
                is_terminal: false,
            });
        }
        ReplayBuffer::insert_trajectory(&buffer.state, 1, 10.0, steps, 0, 0.0, 0.0, 0.99, 0.95);

        for i in 0..100 {
            buffer.state.per.add(i, 1.0);
        }

        // By default batch_size_limit inside arena is 10.
        // Requesting 20 should trigger our bounds check assertion.
        let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            buffer.sample_batch(20, 1.0).is_some()
        }));
        assert!(
            result.is_err(),
            "Expected panic due to out of bounds arena access. Instead got Ok(is_some={})",
            result.unwrap()
        );
    }
}

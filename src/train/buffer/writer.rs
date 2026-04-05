use std::sync::atomic::Ordering;

use crate::train::buffer::core::{OwnedGameData, ReplayBuffer};
use crate::train::buffer::state::{EpisodeMeta, SharedState};

impl ReplayBuffer {
    pub fn add_game(&self, data: OwnedGameData) {
        let _ = self.background_sender.send(data);
    }

    pub(crate) fn process_add_game(state: &SharedState, data: OwnedGameData) {
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

        let mut precomputed_tds = vec![0.0; episode_length];
        let discount_factor = 0.99f32;
        let lambda = 0.95f32;

        let mut g = if episode_length > 0 {
            steps[episode_length - 1].value_target
        } else {
            0.0
        };

        for t in (0..episode_length).rev() {
            let r = steps[t].value_prefix_received;
            let v_next = if t + 1 < episode_length {
                steps[t + 1].value_target
            } else {
                0.0
            };

            g = r + discount_factor * ((1.0 - lambda) * v_next + lambda * g);
            precomputed_tds[t] = g;
        }

        for (transition_offset, step) in steps.iter().take(episode_length).enumerate() {
            let circular_write_index =
                (episode_start_index + transition_offset) % buffer_buffer_capacity_limit;
            state.arrays.write_storage_index(
                circular_write_index,
                |memory_shard, internal_shard_index| {
                    memory_shard.state_diff[internal_shard_index] = difficulty_setting;
                    memory_shard.state_len[internal_shard_index] = episode_length as i32;

                    memory_shard.boards[internal_shard_index] = step.board_state;
                    memory_shard.available[internal_shard_index] = step.available_pieces;
                    memory_shard.actions[internal_shard_index] = step.action_taken;
                    memory_shard.piece_ids[internal_shard_index] = step.piece_identifier;
                    memory_shard.value_prefixs[internal_shard_index] = step.value_prefix_received;
                    memory_shard.policies[internal_shard_index] = step.policy_target;
                    memory_shard.values[internal_shard_index] = step.value_target;
                    memory_shard.td_targets[internal_shard_index] =
                        precomputed_tds[transition_offset];

                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                    memory_shard.state_start[internal_shard_index] = episode_start_index as i64;
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
    pub(crate) fn update_episode_metadata(
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
}

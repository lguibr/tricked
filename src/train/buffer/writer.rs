use std::sync::atomic::Ordering;

use crate::train::buffer::core::{OwnedGameData, ReplayBuffer};
use crate::train::buffer::state::{EpisodeMeta, SharedState};

pub const ROT_MAP_60: [usize; 96] = [
    7, 8, 18, 19, 31, 32, 46, 47, 62, 5, 6, 16, 17, 29, 30, 44, 45, 60, 61, 75, 3, 4, 14, 15, 27,
    28, 42, 43, 58, 59, 73, 74, 86, 1, 2, 12, 13, 25, 26, 40, 41, 56, 57, 71, 72, 84, 85, 95, 0,
    10, 11, 23, 24, 38, 39, 54, 55, 69, 70, 82, 83, 93, 94, 9, 21, 22, 36, 37, 52, 53, 67, 68, 80,
    81, 91, 92, 20, 34, 35, 50, 51, 65, 66, 78, 79, 89, 90, 33, 48, 49, 63, 64, 76, 77, 87, 88,
];

fn rotate_mask(mask: u128, map: &[usize; 96]) -> u128 {
    let mut res = 0;
    for (i, &mapped_index) in map.iter().enumerate() {
        if (mask & (1 << i)) != 0 {
            res |= 1 << mapped_index;
        }
    }
    res
}

fn map_piece(p: &[u128; 96], map: &[usize; 96]) -> [u128; 96] {
    let mut res = [0; 96];
    for i in 0..96 {
        if p[i] != 0 {
            res[map[i]] = rotate_mask(p[i], map);
        }
    }
    res
}

pub fn rotate_piece_id(id: i32, rot: usize) -> i32 {
    if id == -1 {
        return -1;
    }
    let mut piece = crate::core::constants::STANDARD_PIECES[id as usize];
    for _ in 0..rot {
        piece = map_piece(&piece, &ROT_MAP_60);
    }
    crate::core::constants::STANDARD_PIECES
        .iter()
        .position(|&x| x == piece)
        .unwrap() as i32
}

pub fn rotate_anchor(piece_id: i32, piece_id_rot: i32, anchor: usize, rot: usize) -> usize {
    let original_mask_opt = crate::node::COMPACT_PIECE_MASKS[piece_id as usize]
        .iter()
        .find(|&&(a, _)| a == anchor);

    if original_mask_opt.is_none() {
        return anchor; // Dummy fallback for mock BPTT tests with garbage inputs. Real MCTS ensures bijections.
    }
    let original_mask = original_mask_opt.unwrap().1;

    let mut m_rot = original_mask;
    for _ in 0..rot {
        m_rot = rotate_mask(m_rot, &ROT_MAP_60);
    }

    crate::node::COMPACT_PIECE_MASKS[piece_id_rot as usize]
        .iter()
        .find(|&&(_, m)| m == m_rot)
        .map(|&(a, _)| a)
        .unwrap_or(anchor)
}

pub fn rotate_action(action: i32, piece_id: i32, piece_id_rot: i32, rot: usize) -> i32 {
    let anchor = (action % 96) as usize;
    let anchor_rot = rotate_anchor(piece_id, piece_id_rot, anchor, rot);
    piece_id_rot * 96 + anchor_rot as i32
}

pub fn rotate_policy(policy: &[f32; 288], available_pieces: &[i32; 3], rot: usize) -> [f32; 288] {
    let mut rot_policy = [0.0; 288];

    for (slot, &piece_id) in available_pieces.iter().enumerate() {
        if piece_id == -1 {
            continue;
        }

        let piece_id_rot = rotate_piece_id(piece_id, rot);

        for &(anchor, _) in crate::node::COMPACT_PIECE_MASKS[piece_id as usize].iter() {
            let original_action = (slot * 96) + anchor;
            let prob = policy[original_action];
            if prob > 0.0 {
                let anchor_rot = rotate_anchor(piece_id, piece_id_rot, anchor, rot);
                let rot_action = (slot * 96) + anchor_rot;
                rot_policy[rot_action] = prob;
            }
        }
    }

    rot_policy
}

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

        let mut all_trajectories = vec![steps];
        for r in 1..6 {
            let mut aug = all_trajectories[0].clone();
            for step in aug.iter_mut() {
                let mut b_rot =
                    (step.board_state[0] as u128) | ((step.board_state[1] as u128) << 64);
                for _ in 0..r {
                    b_rot = rotate_mask(b_rot, &ROT_MAP_60);
                }
                step.board_state = [b_rot as u64, (b_rot >> 64) as u64];

                let original_policy: [f32; 288] = step.policy_target.clone().try_into().unwrap();
                step.policy_target =
                    rotate_policy(&original_policy, &step.available_pieces, r).to_vec();

                let p_rot = rotate_piece_id(step.piece_identifier as i32, r);
                step.action_taken = rotate_action(
                    step.action_taken as i32,
                    step.piece_identifier as i32,
                    p_rot,
                    r,
                ) as i64;
                step.piece_identifier = p_rot as i64;

                for p in step.available_pieces.iter_mut() {
                    *p = rotate_piece_id(*p, r);
                }
            }
            all_trajectories.push(aug);
        }

        for trajectory_steps in all_trajectories {
            Self::insert_trajectory(
                state,
                difficulty_setting,
                episode_score,
                trajectory_steps,
                lines_cleared,
                mcts_depth_mean,
                mcts_search_time_mean,
            );
        }
    }

    pub(crate) fn insert_trajectory(
        state: &SharedState,
        difficulty_setting: i32,
        episode_score: f32,
        steps: Vec<crate::train::buffer::core::GameStep>,
        lines_cleared: u32,
        mcts_depth_mean: f32,
        mcts_search_time_mean: f32,
    ) {
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
                    memory_shard.value_prefixs[internal_shard_index] =
                        half::f16::from_f32(step.value_prefix_received);

                    let mut policy_u8 = [0u8; 288];
                    for (i, val) in policy_u8.iter_mut().enumerate() {
                        *val = (step.policy_target[i] * 255.0).round() as u8;
                    }
                    memory_shard.policies[internal_shard_index] = policy_u8;

                    memory_shard.values[internal_shard_index] =
                        half::f16::from_f32(step.value_target);
                    memory_shard.td_targets[internal_shard_index] =
                        half::f16::from_f32(precomputed_tds[transition_offset]);

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
        _next_global_write_index: usize,
        buffer_buffer_capacity_limit: usize,
        lines_cleared: u32,
        mcts_depth_mean: f32,
        mcts_search_time_mean: f32,
    ) {
        state.episodes.push(EpisodeMeta {
            global_start_storage_index: episode_start_index,
            length: episode_length,
            difficulty: difficulty_setting,
            score: episode_score,
            lines_cleared,
            mcts_depth_mean,
            mcts_search_time_mean,
        });

        state.recent_scores.push(episode_score);
        state.completed_games.fetch_add(1, Ordering::Relaxed);

        let current_state_count = state.num_states.load(Ordering::Relaxed);
        state.num_states.store(
            buffer_buffer_capacity_limit.min(current_state_count + episode_length),
            Ordering::Relaxed,
        );
    }
}

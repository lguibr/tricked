use crate::board::GameStateExt;
use crate::constants::STANDARD_PIECES;
use crate::neighbors::NEIGHBOR_MASKS;

const TOTAL_TRIANGLES: usize = 96;

pub fn extract_feature_native(
    state: &GameStateExt,
    history: Option<Vec<u128>>,
    action_history: Option<Vec<i32>>,
    difficulty: i32,
) -> Vec<f32> {
    let mut features_array = vec![0.0_f32; 20 * TOTAL_TRIANGLES];

    fill_history_channels(&mut features_array, state.board, history);
    fill_action_history_channels(&mut features_array, action_history);
    fill_piece_overlay_channels(&mut features_array, &state.available, state.board);
    fill_static_game_channels(&mut features_array, difficulty, state.board);

    features_array
}

fn fill_channel(features_array: &mut [f32], channel_index: usize, board_bits: u128) {
    let memory_offset = channel_index * TOTAL_TRIANGLES;
    for bit_index in 0..96 {
        if (board_bits >> bit_index) & 1 == 1 {
            features_array[memory_offset + bit_index] = 1.0;
        }
    }
}

fn fill_history_channels(
    features_array: &mut [f32],
    current_board_state: u128,
    history_boards: Option<Vec<u128>>,
) {
    let unwrapped_history = history_boards.unwrap_or_default();
    fill_channel(features_array, 0, current_board_state);

    for memory_index in 1..=7 {
        if unwrapped_history.len() >= memory_index {
            let prior_state = unwrapped_history[unwrapped_history.len() - memory_index];
            fill_channel(features_array, memory_index, prior_state);
        } else {
            fill_channel(features_array, memory_index, current_board_state);
        }
    }
}

fn fill_action_history_channels(features_array: &mut [f32], action_history: Option<Vec<i32>>) {
    // channel 8-10: action history
    let unwrapped_actions = action_history.unwrap_or_default();
    for memory_index in 0..3 {
        if unwrapped_actions.len() > memory_index {
            let prior_action = unwrapped_actions[unwrapped_actions.len() - (memory_index + 1)];
            let slot_index = prior_action / (TOTAL_TRIANGLES as i32);
            let map_index = (prior_action % (TOTAL_TRIANGLES as i32)) as usize;

            if map_index < TOTAL_TRIANGLES {
                features_array[(8 + memory_index) * TOTAL_TRIANGLES + map_index] =
                    (slot_index as f32 + 1.0) * 0.33;
            }
        }
    }
}

fn fill_piece_overlay_channels(
    features_array: &mut [f32],
    available_pieces: &[i32; 3],
    current_board_state: u128,
) {
    // channel 11-16: pieces overlay (canonical shape) and valid mask
    for slot_index in 0..3 {
        let piece_identifier = available_pieces[slot_index];
        if piece_identifier == -1 {
            continue;
        }

        let piece_table_index = piece_identifier as usize;
        let mut validity_mask = [0_u8; 96];
        let mut canonical_shape_drawn = false;

        for &piece_mask in &STANDARD_PIECES[piece_table_index] {
            if piece_mask == 0 {
                continue;
            }

            // FIX: Draw ONLY the first valid geometric shape as a "sprite" for the CNN
            if !canonical_shape_drawn {
                for bit_index in 0..96 {
                    if (piece_mask & (1_u128 << bit_index)) != 0 {
                        // Channel 11, 13, 15: The exact shape of the piece
                        features_array[(11 + slot_index * 2) * TOTAL_TRIANGLES + bit_index] = 1.0;
                    }
                }
                canonical_shape_drawn = true;
            }

            // Keep the validity mask so the network knows WHERE it can legally place it
            if (current_board_state & piece_mask) == 0 {
                for (bit_index, valid_mask) in validity_mask.iter_mut().enumerate() {
                    if (piece_mask & (1_u128 << bit_index)) != 0 {
                        *valid_mask = 1;
                    }
                }
            }
        }

        // Channel 12, 14, 16: The legal placement footprint
        for memory_index in 0..TOTAL_TRIANGLES {
            if validity_mask[memory_index] == 1 {
                features_array[(12 + slot_index * 2) * TOTAL_TRIANGLES + memory_index] = 1.0;
            }
        }
    }
}

fn fill_static_game_channels(
    features_array: &mut [f32],
    difficulty_level: i32,
    current_board_state: u128,
) {
    // channel 17: empty
    for memory_index in 0..TOTAL_TRIANGLES {
        features_array[17 * TOTAL_TRIANGLES + memory_index] = 1.0 / 22.0;
    }

    // channel 18: difficulty
    let normalized_difficulty = difficulty_level as f32 / 6.0;
    for memory_index in 0..TOTAL_TRIANGLES {
        features_array[18 * TOTAL_TRIANGLES + memory_index] = normalized_difficulty;
    }

    // channel 19: hole checking
    for memory_index in 0..TOTAL_TRIANGLES {
        let is_position_filled = (current_board_state >> memory_index) & 1 == 1;
        if !is_position_filled {
            let neighbor_mask = NEIGHBOR_MASKS[memory_index];
            if (current_board_state & neighbor_mask) == neighbor_mask {
                features_array[19 * TOTAL_TRIANGLES + memory_index] = 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::GameStateExt;

    #[test]
    fn test_extract_feature_history_padding() {
        let mut state = GameStateExt::new(Some([0, 0, 0]), 0, 0, 6, 0);
        state.board = 0b101;

        let history_boards = vec![0b010];
        let features_array = extract_feature_native(&state, Some(history_boards), None, 6);

        assert_eq!(features_array.len(), 20 * 96);

        assert_eq!(features_array[0], 1.0);
        assert_eq!(features_array[2], 1.0);
        assert_eq!(features_array[1], 0.0);

        let memory_offset_1 = 96;
        assert_eq!(features_array[memory_offset_1 + 1], 1.0);
        assert_eq!(features_array[memory_offset_1], 0.0);

        let memory_offset_2 = 2 * 96;
        assert_eq!(features_array[memory_offset_2], 1.0);
        assert_eq!(features_array[memory_offset_2 + 2], 1.0);
        assert_eq!(features_array[memory_offset_2 + 1], 0.0);
    }

    #[test]
    fn test_extract_feature_hole_detection() {
        let mut state = GameStateExt::new(Some([0, 0, 0]), 0, 0, 6, 0);

        let mask_neighbors_0 = crate::neighbors::NEIGHBOR_MASKS[0];

        state.board = mask_neighbors_0;
        let features_array = extract_feature_native(&state, None, None, 6);
        let memory_offset_19 = 19 * 96;
        assert_eq!(features_array[memory_offset_19], 1.0);

        state.board = mask_neighbors_0 | 1;
        let features_array_2 = extract_feature_native(&state, None, None, 6);
        assert_eq!(features_array_2[memory_offset_19], 0.0);
    }
}

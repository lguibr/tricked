use crate::core::constants::STANDARD_PIECES;
use once_cell::sync::Lazy;
use tch::{Device, Tensor};

pub static CANONICAL_PIECE_MASKS: Lazy<[Vec<usize>; 48]> = Lazy::new(|| {
    let mut masks: [Vec<usize>; 48] = core::array::from_fn(|_| Vec::new());
    for (piece_table_index, mask) in masks.iter_mut().enumerate().take(48) {
        let mut canonical_shape_drawn = false;

        for &(_rotation_index, piece_mask) in &crate::node::COMPACT_PIECE_MASKS[piece_table_index] {
            if !canonical_shape_drawn {
                let mut minimum_row = 8;
                let mut maximum_row = 0;
                let mut minimum_column = 16;
                let mut maximum_column = 0;

                let mut temp_mask = piece_mask & ((1_u128 << 96) - 1);
                while temp_mask != 0 {
                    let bit_index = temp_mask.trailing_zeros() as usize;
                    let (row, column) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[bit_index];
                    minimum_row = minimum_row.min(row);
                    maximum_row = maximum_row.max(row);
                    minimum_column = minimum_column.min(column);
                    maximum_column = maximum_column.max(column);
                    temp_mask &= temp_mask - 1;
                }

                let middle_row = (minimum_row + maximum_row) / 2;
                let middle_column = (minimum_column + maximum_column) / 2;
                let target_row = 3;
                let target_column = 8;

                let mut temp_mask2 = piece_mask & ((1_u128 << 96) - 1);
                while temp_mask2 != 0 {
                    let bit_index = temp_mask2.trailing_zeros() as usize;
                    let (row, column) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[bit_index];
                    let offset_row = (row as isize - middle_row as isize) + target_row as isize;
                    let offset_column =
                        (column as isize - middle_column as isize) + target_column as isize;

                    if (0..8).contains(&offset_row) && (0..16).contains(&offset_column) {
                        mask.push(offset_row as usize * 16 + offset_column as usize);
                    }
                    temp_mask2 &= temp_mask2 - 1;
                }
                canonical_shape_drawn = true;
            }
        }
    }
    masks
});
pub const TOTAL_TRIANGLES: usize = 96;
pub const SPATIAL_ROWS: usize = 8;
pub const SPATIAL_COLS: usize = 16;
pub const SPATIAL_SIZE: usize = SPATIAL_ROWS * SPATIAL_COLS;

pub const HEXAGONAL_TO_CARTESIAN_MAP_ARRAY: [(usize, usize); 96] = [
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (0, 12),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
    (1, 11),
    (1, 12),
    (1, 13),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 10),
    (2, 11),
    (2, 12),
    (2, 13),
    (2, 14),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (3, 12),
    (3, 13),
    (3, 14),
    (3, 15),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 8),
    (4, 9),
    (4, 10),
    (4, 11),
    (4, 12),
    (4, 13),
    (4, 14),
    (4, 15),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (5, 11),
    (5, 12),
    (5, 13),
    (5, 14),
    (6, 3),
    (6, 4),
    (6, 5),
    (6, 6),
    (6, 7),
    (6, 8),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 12),
    (6, 13),
    (7, 4),
    (7, 5),
    (7, 6),
    (7, 7),
    (7, 8),
    (7, 9),
    (7, 10),
    (7, 11),
    (7, 12),
];

#[inline(always)]
pub fn get_spatial_idx(hex_idx: usize) -> usize {
    let (row, column) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[hex_idx];
    row * SPATIAL_COLS + column
}

pub fn get_valid_spatial_mask_8x8(computation_device: Device) -> Tensor {
    let mut mask = vec![0.0_f32; 64];
    for &(row, column) in HEXAGONAL_TO_CARTESIAN_MAP_ARRAY.iter() {
        mask[row * 8 + (column / 2)] = 1.0;
    }
    Tensor::from_slice(&mask)
        .view([1, 1, 8, 8])
        .to_device(computation_device)
}

pub fn extract_feature_native(
    extracted_features_tensor_flat: &mut [f32],
    current_board_state: u128,
    available_pieces: &[i32; 3],
    history_boards: &[u128],
    action_history: &[i32],
    difficulty: i32,
) {
    extracted_features_tensor_flat.fill(0.0);

    fill_history_channels(
        extracted_features_tensor_flat,
        current_board_state,
        history_boards,
    );
    fill_action_history_channels(extracted_features_tensor_flat, action_history);
    fill_piece_overlay_channels(
        extracted_features_tensor_flat,
        available_pieces,
        current_board_state,
    );
    fill_static_game_channels(
        extracted_features_tensor_flat,
        difficulty,
        current_board_state,
    );
}

fn fill_channel(
    extracted_features_tensor_flat: &mut [f32],
    channel_index: usize,
    mut board_bits: u128,
) {
    let memory_offset = channel_index * SPATIAL_SIZE;
    board_bits &= (1_u128 << 96) - 1; // Mask out any bits beyond the 96th structural triangle
    while board_bits != 0 {
        let bit_index = board_bits.trailing_zeros() as usize;
        extracted_features_tensor_flat[memory_offset + get_spatial_idx(bit_index)] = 1.0;
        board_bits &= board_bits - 1;
    }
}

fn fill_history_channels(
    extracted_features_tensor_flat: &mut [f32],
    current_board_state: u128,
    unwrapped_history: &[u128],
) {
    fill_channel(extracted_features_tensor_flat, 0, current_board_state);

    for memory_index in 1..=7 {
        if unwrapped_history.len() >= memory_index {
            let prior_state = unwrapped_history[unwrapped_history.len() - memory_index];
            fill_channel(extracted_features_tensor_flat, memory_index, prior_state);
        } else {
            fill_channel(
                extracted_features_tensor_flat,
                memory_index,
                current_board_state,
            );
        }
    }
}

fn fill_action_history_channels(
    extracted_features_tensor_flat: &mut [f32],
    unwrapped_actions: &[i32],
) {
    // channel 8-10: action history
    for memory_index in 0..3 {
        if unwrapped_actions.len() > memory_index {
            let prior_action = unwrapped_actions[unwrapped_actions.len() - (memory_index + 1)];
            let slot_index = prior_action / (TOTAL_TRIANGLES as i32);
            let map_index = (prior_action % (TOTAL_TRIANGLES as i32)) as usize;

            if map_index < TOTAL_TRIANGLES {
                extracted_features_tensor_flat
                    [(8 + memory_index) * SPATIAL_SIZE + get_spatial_idx(map_index)] =
                    (slot_index as f32 + 1.0) * 0.33;
            }
        }
    }
}

fn fill_piece_overlay_channels(
    extracted_features_tensor_flat: &mut [f32],
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
        let mut validity_mask: u128 = 0;

        for &spatial_idx in &CANONICAL_PIECE_MASKS[piece_table_index] {
            extracted_features_tensor_flat[(11 + slot_index * 2) * SPATIAL_SIZE + spatial_idx] =
                1.0;
        }

        for &(_rotation_index, piece_mask) in &crate::node::COMPACT_PIECE_MASKS[piece_table_index] {
            // Keep the validity mask so the network knows WHERE it can legally place it
            if (current_board_state & piece_mask) == 0 {
                validity_mask |= piece_mask;
            }
        }

        // Channel 12, 14, 16: The legal placement footprint
        validity_mask &= (1_u128 << 96) - 1; // Purge topological off-grid spillovers
        while validity_mask != 0 {
            let bit_index = validity_mask.trailing_zeros() as usize;
            extracted_features_tensor_flat
                [(12 + slot_index * 2) * SPATIAL_SIZE + get_spatial_idx(bit_index)] = 1.0;
            validity_mask &= validity_mask - 1;
        }
    }
}

fn fill_static_game_channels(
    extracted_features_tensor_flat: &mut [f32],
    difficulty_level: i32,
    current_board_state: u128,
) {
    let all_hexes = (1_u128 << TOTAL_TRIANGLES) - 1;
    let normalized_difficulty = difficulty_level as f32 / 6.0;

    // channel 17 & 18: empty and difficulty
    let mut temp = all_hexes;
    while temp != 0 {
        let bit_index = temp.trailing_zeros() as usize;
        let spatial_idx = get_spatial_idx(bit_index);
        extracted_features_tensor_flat[17 * SPATIAL_SIZE + spatial_idx] = 1.0 / 22.0;
        extracted_features_tensor_flat[18 * SPATIAL_SIZE + spatial_idx] = normalized_difficulty;
        temp &= temp - 1;
    }

    // channel 19: explicit dead zone detection
    let mut global_valid_mask = 0_u128;
    for pieces_set in STANDARD_PIECES.iter() {
        for &piece_mask in pieces_set.iter() {
            if piece_mask != 0 && (current_board_state & piece_mask) == 0 {
                global_valid_mask |= piece_mask;
            }
        }
    }

    let dead_zone_mask = !current_board_state & !global_valid_mask & all_hexes;
    let mut temp = dead_zone_mask;
    while temp != 0 {
        let bit_index = temp.trailing_zeros() as usize;
        extracted_features_tensor_flat[19 * SPATIAL_SIZE + get_spatial_idx(bit_index)] = 1.0;
        temp &= temp - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::GameStateExt;

    #[test]
    fn test_extract_feature_history_padding() {
        let mut state = GameStateExt::new(Some([0, 0, 0]), 0, 0, 6, 0);
        state.board_bitmask_u128 = 0b101;

        let history_boards = vec![0b010];
        let mut extracted_features_tensor_flat = vec![0.0; 20 * 128];
        extract_feature_native(
            &mut extracted_features_tensor_flat,
            state.board_bitmask_u128,
            &state.available,
            &history_boards,
            &[],
            6,
        );

        assert_eq!(extracted_features_tensor_flat.len(), 20 * 128);

        assert_eq!(extracted_features_tensor_flat[get_spatial_idx(0)], 1.0);
        assert_eq!(extracted_features_tensor_flat[get_spatial_idx(2)], 1.0);
        assert_eq!(extracted_features_tensor_flat[get_spatial_idx(1)], 0.0);

        let memory_offset_1 = 128;
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_1 + get_spatial_idx(1)],
            1.0
        );
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_1 + get_spatial_idx(0)],
            0.0
        );

        let memory_offset_2 = 2 * 128;
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_2 + get_spatial_idx(0)],
            1.0
        );
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_2 + get_spatial_idx(2)],
            1.0
        );
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_2 + get_spatial_idx(1)],
            0.0
        );
    }
}

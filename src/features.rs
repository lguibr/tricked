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
    let mut feature = vec![0.0_f32; 20 * TOTAL_TRIANGLES];

    let history = history.unwrap_or_default();
    let action_history = action_history.unwrap_or_default();

    let fill_channel = |feat: &mut [f32], channel_idx: usize, board_int: u128| {
        let offset = channel_idx * TOTAL_TRIANGLES;
        for i in 0..96 {
            if (board_int >> i) & 1 == 1 {
                feat[offset + i] = 1.0;
            }
        }
    };

    fill_channel(&mut feature, 0, state.board);

    // channel 1-7: history
    for i in 1..=7 {
        if history.len() >= i {
            fill_channel(&mut feature, i, history[history.len() - i]);
        } else {
            fill_channel(&mut feature, i, state.board);
        }
    }

    // channel 8-10: action history
    for i in 0..3 {
        if action_history.len() > i {
            let act = action_history[action_history.len() - (i + 1)];
            let slot = act / (TOTAL_TRIANGLES as i32);
            let idx = (act % (TOTAL_TRIANGLES as i32)) as usize;
            if idx < TOTAL_TRIANGLES {
                feature[(8 + i) * TOTAL_TRIANGLES + idx] = (slot as f32 + 1.0) * 0.33;
            }
        }
    }

    // channel 11-16: pieces overlay and valid mask
    for slot in 0..3 {
        let p_id = state.available[slot];
        if p_id == -1 {
            continue;
        }
        let p_idx = p_id as usize;
        let mut overlay = [0_u8; 96];
        let mut valid_mask = [0_u8; 96];

        for m in &STANDARD_PIECES[p_idx] {
            if *m == 0 {
                continue;
            }
            // overlay
            for i in 0..96 {
                if (*m & (1 << i)) != 0 {
                    overlay[i] = 1;
                }
            }
            // valid mask
            if (state.board & *m) == 0 {
                for i in 0..96 {
                    if (*m & (1 << i)) != 0 {
                        valid_mask[i] = 1;
                    }
                }
            }
        }

        for i in 0..TOTAL_TRIANGLES {
            if overlay[i] == 1 {
                feature[(11 + slot * 2) * TOTAL_TRIANGLES + i] = 1.0;
            }
            if valid_mask[i] == 1 {
                feature[(12 + slot * 2) * TOTAL_TRIANGLES + i] = 1.0;
            }
        }
    }

    // channel 17: empty
    for i in 0..TOTAL_TRIANGLES {
        feature[17 * TOTAL_TRIANGLES + i] = 1.0 / 22.0;
    }

    // channel 18: difficulty
    let diff_val = difficulty as f32 / 6.0;
    for i in 0..TOTAL_TRIANGLES {
        feature[18 * TOTAL_TRIANGLES + i] = diff_val;
    }

    // channel 19: hole checking
    for i in 0..TOTAL_TRIANGLES {
        let is_filled = (state.board >> i) & 1 == 1;
        if !is_filled {
            let neighbors = NEIGHBOR_MASKS[i];
            if (state.board & neighbors) == neighbors {
                feature[19 * TOTAL_TRIANGLES + i] = 1.0;
            }
        }
    }

    feature
}



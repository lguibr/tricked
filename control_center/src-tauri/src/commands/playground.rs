use tricked_engine::core::board::GameStateExt;

#[derive(serde::Serialize)]
pub struct PlaygroundState {
    pub board_low: String,
    pub board_high: String,
    pub available: [i32; 3],
    pub score: i32,
    pub pieces_left: i32,
    pub terminal: bool,
    pub difficulty: i32,
    pub lines_cleared: i32,
}

impl From<GameStateExt> for PlaygroundState {
    fn from(state: GameStateExt) -> Self {
        let low = (state.board_bitmask_u128 & 0xFFFFFFFFFFFFFFFF) as u64;
        let high = (state.board_bitmask_u128 >> 64) as u64;
        PlaygroundState {
            board_low: low.to_string(),
            board_high: high.to_string(),
            available: state.available,
            score: state.score,
            pieces_left: state.pieces_left,
            terminal: state.terminal,
            difficulty: state.difficulty,
            lines_cleared: state.total_lines_cleared,
        }
    }
}

#[tauri::command]
pub fn playground_start_game(difficulty: i32, clutter: i32) -> Result<PlaygroundState, String> {
    let state = GameStateExt::new(None, 0, 0, difficulty, clutter);
    Ok(state.into())
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub fn playground_apply_move(
    board_low: String,
    board_high: String,
    available: Vec<i32>,
    score: i32,
    slot: usize,
    piece_mask_low: String,
    piece_mask_high: String,
    difficulty: i32,
    lines_cleared: i32,
) -> Result<Option<PlaygroundState>, String> {
    let low = board_low.parse::<u64>().map_err(|e| e.to_string())?;
    let high = board_high.parse::<u64>().map_err(|e| e.to_string())?;
    let board_mask = (low as u128) | ((high as u128) << 64);

    let plow = piece_mask_low.parse::<u64>().map_err(|e| e.to_string())?;
    let phigh = piece_mask_high.parse::<u64>().map_err(|e| e.to_string())?;
    let piece_mask = (plow as u128) | ((phigh as u128) << 64);

    let mut available_arr = [-1; 3];
    for (i, &val) in available.iter().take(3).enumerate() {
        available_arr[i] = val;
    }

    let mut state = GameStateExt {
        board_bitmask_u128: board_mask,
        available: available_arr,
        score,
        pieces_left: available_arr.iter().filter(|&&x| x != -1).count() as i32,
        terminal: false,
        difficulty,
        total_lines_cleared: lines_cleared,
    };

    if let Some(next_state) = state.apply_move_mask(slot, piece_mask) {
        Ok(Some(next_state.into()))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod playground_tests {

    #[test]
    fn test_playground_bigint_string_reconstruction() {
        let low_str = "18446744073709551615";
        let high_str = "1";

        let board_low = low_str.parse::<u64>().unwrap();
        let board_high = high_str.parse::<u64>().unwrap();
        let combined = (board_low as u128) | ((board_high as u128) << 64);

        assert_eq!(
            combined,
            18446744073709551615 | (1 << 64),
            "BigInt string parsing lost precision!"
        );
    }
}

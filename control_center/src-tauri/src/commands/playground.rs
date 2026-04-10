use tricked_engine::core::board::GameStateExt;

#[derive(serde::Serialize, ts_rs::TS)]
#[ts(export, export_to = "../../../control_center/src/bindings/")]
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

#[derive(serde::Deserialize)]
pub struct IncomingGameStep {
    pub board_low: String,
    pub board_high: String,
    pub available: Vec<i32>,
    pub action_taken: i32,
    pub piece_identifier: i32,
}

#[tauri::command]
pub fn playground_commit_to_vault(
    app_handle: tauri::AppHandle,
    steps: Vec<IncomingGameStep>,
    score: f32,
    difficulty: i32,
    lines_cleared: i32,
) -> Result<(), String> {
    use rusqlite::Connection;
    use tauri::Manager;
    use tricked_engine::train::buffer::core::{GameStep, OwnedGameData};

    let db_path = app_handle.path().app_data_dir().unwrap().join("tricked.db");
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;

    let run_id = "PLAYGROUND_HUMAN_RUN";
    let artifacts_dir = "runs/PLAYGROUND_HUMAN_RUN";

    // Ensure run exists
    let existing: Result<String, _> =
        conn.query_row("SELECT id FROM runs WHERE id = ?1", [run_id], |row| {
            row.get(0)
        });
    if existing.is_err() {
        conn.execute(
            "INSERT INTO runs (id, name, type, status, config, artifacts_dir) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            [
               run_id,
               "Playground Session",
               "HUMAN",
               "STOPPED",
               "{}",
               artifacts_dir
            ]
        ).map_err(|e| e.to_string())?;
    }

    let db_path = crate::db::get_db_path();
    let root = db_path.parent().unwrap();
    let abs_artifacts_dir = root.join(artifacts_dir);

    std::fs::create_dir_all(&abs_artifacts_dir).map_err(|e| e.to_string())?;

    let vault_file = abs_artifacts_dir.join("vault.bincode");

    let mut games = vec![];
    if vault_file.exists() {
        if let Ok(file) = std::fs::File::open(&vault_file) {
            let reader = std::io::BufReader::new(file);
            if let Ok(existing_games) = bincode::deserialize_from::<_, Vec<OwnedGameData>>(reader) {
                games = existing_games;
            }
        }
    }

    let mut game_steps = Vec::with_capacity(steps.len());
    for s in steps {
        let low = s.board_low.parse::<u64>().unwrap_or(0);
        let high = s.board_high.parse::<u64>().unwrap_or(0);
        let mut avail = [-1; 3];
        for (i, v) in s.available.iter().take(3).enumerate() {
            avail[i] = *v;
        }
        game_steps.push(GameStep {
            board_state: [low, high],
            available_pieces: avail,
            action_taken: s.action_taken as i64,
            piece_identifier: s.piece_identifier as i64,
            value_prefix_received: 0.0,
            policy_target: vec![],
            value_target: 0.0,
        });
    }

    let g = OwnedGameData {
        difficulty_setting: difficulty,
        episode_score: score,
        steps: game_steps,
        lines_cleared: lines_cleared as u32,
        mcts_depth_mean: 0.0,
        mcts_search_time_mean: 0.0,
    };

    games.push(g);

    let file = std::fs::File::create(&vault_file).map_err(|e| e.to_string())?;
    let writer = std::io::BufWriter::new(file);
    bincode::serialize_into(writer, &games).map_err(|e| e.to_string())?;

    Ok(())
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

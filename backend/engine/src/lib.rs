#![allow(clippy::useless_conversion, clippy::too_many_arguments)]
pub mod cli;
pub mod config;
pub mod core;
pub mod env;
pub mod mcts;
pub mod telemetry;
pub mod train;

pub mod node;
pub mod queue;
pub mod sumtree;

pub mod engine;
pub mod inference_worker;

#[pyo3::prelude::pymodule]
fn tricked_engine(
    m: &pyo3::prelude::Bound<'_, pyo3::types::PyModule>,
) -> pyo3::prelude::PyResult<()> {
    use pyo3::prelude::*;
    m.add_class::<engine::TrickedEngine>()?;
    m.add_function(wrap_pyfunction!(playground_start_game, m)?)?;
    m.add_function(wrap_pyfunction!(playground_apply_move, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_board, m)?)?;
    m.add_function(wrap_pyfunction!(get_global_vault_games, m)?)?;
    m.add_function(wrap_pyfunction!(flush_global_vault, m)?)?;
    m.add_function(wrap_pyfunction!(empty_all_vaults, m)?)?;
    m.add_function(wrap_pyfunction!(remove_vault_game, m)?)?;
    m.add_function(wrap_pyfunction!(commit_human_game, m)?)?;
    Ok(())
}

use crate::core::board::GameStateExt;
use pyo3::prelude::*;

#[pyfunction]
fn playground_start_game(difficulty: i32, clutter: i32) -> PyResult<String> {
    let mut state = GameStateExt::new(None, 0, 0, difficulty, clutter);
    state.refill_tray(); // force pieces pop

    // We split board_u128 to low and high u64 strings so JS can parse easily without BigInt JSON overflow
    let board_low = (state.board_bitmask_u128 as u64).to_string();
    let board_high = ((state.board_bitmask_u128 >> 64) as u64).to_string();

    let json_str = format!(
        r#"{{"board_low": "{}", "board_high": "{}", "available": [{}, {}, {}], "score": {}, "pieces_left": {}, "terminal": {}, "difficulty": {}, "lines_cleared": {}}}"#,
        board_low,
        board_high,
        state.available[0],
        state.available[1],
        state.available[2],
        state.score,
        state.pieces_left,
        state.terminal,
        state.difficulty,
        state.total_lines_cleared
    );
    Ok(json_str)
}

#[pyfunction]
fn playground_apply_move(
    board_low_str: String,
    board_high_str: String,
    available: Vec<i32>,
    score: i32,
    difficulty: i32,
    lines_cleared: i32,
    slot: usize,
    piece_mask_low: String,
    piece_mask_high: String,
) -> PyResult<Option<String>> {
    let low = board_low_str.parse::<u64>().unwrap_or(0) as u128;
    let high = board_high_str.parse::<u64>().unwrap_or(0) as u128;
    let board = low | (high << 64);

    let p_low = piece_mask_low.parse::<u64>().unwrap_or(0) as u128;
    let p_high = piece_mask_high.parse::<u64>().unwrap_or(0) as u128;
    let piece_mask = p_low | (p_high << 64);

    let mut pieces = [-1; 3];
    for (i, p) in available.iter().take(3).enumerate() {
        pieces[i] = *p;
    }

    let mut state = GameStateExt {
        board_bitmask_u128: board,
        available: pieces,
        score,
        pieces_left: pieces.iter().filter(|&&x| x != -1).count() as i32,
        terminal: false,
        difficulty,
        total_lines_cleared: lines_cleared,
    };

    if let Some(next_state) = state.apply_move_mask(slot, piece_mask) {
        let b_low = (next_state.board_bitmask_u128 as u64).to_string();
        let b_high = ((next_state.board_bitmask_u128 >> 64) as u64).to_string();

        let json_str = format!(
            r#"{{"board_low": "{}", "board_high": "{}", "available": [{}, {}, {}], "score": {}, "pieces_left": {}, "terminal": {}, "difficulty": {}, "lines_cleared": {}}}"#,
            b_low,
            b_high,
            next_state.available[0],
            next_state.available[1],
            next_state.available[2],
            next_state.score,
            next_state.pieces_left,
            next_state.terminal,
            next_state.difficulty,
            next_state.total_lines_cleared
        );
        Ok(Some(json_str))
    } else {
        Ok(None)
    }
}

#[pyfunction]
fn evaluate_board(
    board_low_str: String,
    board_high_str: String,
    available: Vec<i32>,
    checkpoint_path: String,
) -> PyResult<String> {
    let low = board_low_str.parse::<u64>().unwrap_or(0) as u128;
    let high = board_high_str.parse::<u64>().unwrap_or(0) as u128;
    let board = low | (high << 64);

    let mut pieces = [-1; 3];
    for (i, p) in available.iter().take(3).enumerate() {
        pieces[i] = *p;
    }

    // Convert to Tensor
    let mut features = vec![
        0.0;
        crate::core::features::NATIVE_FEATURE_CHANNELS
            * crate::core::features::SPATIAL_SIZE
    ];
    crate::core::features::extract_feature_native(&mut features, board, &pieces, &[], &[], 6);

    let f_tensor = tch::Tensor::from_slice(&features)
        .view((
            1,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            8,
            16,
        ))
        .to_device(tch::Device::Cpu);

    let module = tch::CModule::load(&checkpoint_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Evaluate pure NN forward pass
    let output = module
        .forward_is(&[tch::IValue::Tensor(f_tensor)])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut best_action = -1;
    let mut value = 0.0;
    if let tch::IValue::Tuple(tup) = output {
        if let tch::IValue::Tensor(val_tensor) = &tup[1] {
            value = f32::try_from(val_tensor).unwrap_or(0.0);
        }
        if let tch::IValue::Tensor(policy_tensor) = &tup[2] {
            // Apply available pieces mask to prevent choosing invalid piece slots
            let valid_mask = tch::Tensor::zeros([288], (tch::Kind::Float, tch::Device::Cpu));
            for (i, &p) in pieces.iter().enumerate() {
                if p != -1 {
                    let start = (i * 96) as i64;
                    let _ = valid_mask.slice(0, start, start + 96, 1).fill_(1.0);
                }
            }
            let valid_policy = policy_tensor * valid_mask;
            best_action = valid_policy.argmax(Some(-1), false).int64_value(&[0]);
        }
    }

    let json_str = format!(r#"{{"best_action": {}, "value": {}}}"#, best_action, value);
    Ok(json_str)
}

use std::cmp::Reverse;
use std::collections::BinaryHeap;

struct VaultItem {
    score: f32,
    data: crate::train::buffer::core::OwnedGameData,
}

impl PartialEq for VaultItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for VaultItem {}
impl Ord for VaultItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for VaultItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn get_top_100_games_internal() -> Vec<crate::train::buffer::core::OwnedGameData> {
    let mut heap: BinaryHeap<Reverse<VaultItem>> = BinaryHeap::new();

    let scan_file = |path: &std::path::Path, heap: &mut BinaryHeap<Reverse<VaultItem>>| {
        if let Ok(file) = std::fs::File::open(path) {
            let reader = std::io::BufReader::new(file);
            if let Ok(games) = bincode::deserialize_from::<
                _,
                Vec<crate::train::buffer::core::OwnedGameData>,
            >(reader)
            {
                for game in games {
                    let score = game.episode_score;
                    heap.push(Reverse(VaultItem { score, data: game }));
                    if heap.len() > 100 {
                        heap.pop();
                    }
                }
            }
        }
    };

    // 1. Scan run vaults
    if let Ok(entries) = std::fs::read_dir("backend/workspace/runs") {
        for entry in entries.flatten() {
            let mut path = entry.path();
            path.push("artifacts");
            path.push("vault.bincode");
            scan_file(&path, &mut heap);
        }
    }

    // 2. Scan human vault
    scan_file(
        &std::path::PathBuf::from("backend/workspace/human_vault/artifacts/vault.bincode"),
        &mut heap,
    );

    // 3. Scan global vault (where flushed games reside)
    scan_file(
        &std::path::PathBuf::from("backend/workspace/global_vault/artifacts/vault.bincode"),
        &mut heap,
    );

    let mut best_games: Vec<crate::train::buffer::core::OwnedGameData> =
        heap.into_iter().map(|i| i.0.data).collect();

    best_games.sort_by(|a, b| {
        b.episode_score
            .partial_cmp(&a.episode_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    best_games
}

#[pyfunction]
fn get_global_vault_games() -> PyResult<String> {
    let best_games = get_top_100_games_internal();
    Ok(serde_json::to_string(&best_games).unwrap_or_else(|_| "[]".to_string()))
}

#[pyfunction]
fn flush_global_vault() -> PyResult<()> {
    let best_games = get_top_100_games_internal();

    // Delete all vault files
    if let Ok(entries) = std::fs::read_dir("backend/workspace/runs") {
        for entry in entries.flatten() {
            let mut path = entry.path();
            path.push("artifacts");
            path.push("vault.bincode");
            let _ = std::fs::remove_file(&path);
        }
    }
    let _ = std::fs::remove_file("backend/workspace/human_vault/artifacts/vault.bincode");

    // Save to global_vault
    let path = std::path::Path::new("backend/workspace/global_vault/artifacts/vault.bincode");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(file) = std::fs::File::create(path) {
        let writer = std::io::BufWriter::new(file);
        let _ = bincode::serialize_into(writer, &best_games);
    }

    Ok(())
}

#[pyfunction]
fn empty_all_vaults() -> PyResult<()> {
    if let Ok(entries) = std::fs::read_dir("backend/workspace/runs") {
        for entry in entries.flatten() {
            let mut path = entry.path();
            path.push("artifacts");
            path.push("vault.bincode");
            let _ = std::fs::remove_file(&path);
        }
    }
    let _ = std::fs::remove_file("backend/workspace/human_vault/artifacts/vault.bincode");
    let _ = std::fs::remove_file("backend/workspace/global_vault/artifacts/vault.bincode");
    Ok(())
}

#[pyfunction]
fn commit_human_game(json_payload: String) -> PyResult<()> {
    let mut game: crate::train::buffer::core::OwnedGameData =
        match serde_json::from_str(&json_payload) {
            Ok(g) => g,
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid game json: {}",
                    e
                )))
            }
        };

    // Safety injections
    game.source_run_id = "human_playhouse".to_string();
    game.source_run_name = "Human Player".to_string();
    game.run_type = "human".to_string();

    let path = std::path::Path::new("backend/workspace/human_vault/artifacts/vault.bincode");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut existing_games = vec![];
    if let Ok(file) = std::fs::File::open(path) {
        let reader = std::io::BufReader::new(file);
        if let Ok(games) =
            bincode::deserialize_from::<_, Vec<crate::train::buffer::core::OwnedGameData>>(reader)
        {
            existing_games = games;
        }
    }

    existing_games.push(game);
    existing_games.sort_by(|a, b| {
        b.episode_score
            .partial_cmp(&a.episode_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if existing_games.len() > 100 {
        existing_games.truncate(100);
    }

    if let Ok(file) = std::fs::File::create(path) {
        let writer = std::io::BufWriter::new(file);
        let _ = bincode::serialize_into(writer, &existing_games);
    }

    Ok(())
}

#[pyfunction]
fn remove_vault_game(score: f32, steps: usize, run_name: String) -> PyResult<()> {
    let remove_from_file = |path: &std::path::Path| {
        if let Ok(file) = std::fs::File::open(path) {
            let reader = std::io::BufReader::new(file);
            if let Ok(mut games) = bincode::deserialize_from::<
                _,
                Vec<crate::train::buffer::core::OwnedGameData>,
            >(reader)
            {
                let original_len = games.len();
                games.retain(|g| {
                    !((g.episode_score - score).abs() < 0.001
                        && g.steps.len() == steps
                        && g.source_run_name == run_name)
                });
                if games.len() < original_len {
                    if let Ok(file_out) = std::fs::File::create(path) {
                        let writer = std::io::BufWriter::new(file_out);
                        let _ = bincode::serialize_into(writer, &games);
                    }
                }
            }
        }
    };

    if let Ok(entries) = std::fs::read_dir("backend/workspace/runs") {
        for entry in entries.flatten() {
            let mut path = entry.path();
            path.push("artifacts");
            path.push("vault.bincode");
            remove_from_file(&path);
        }
    }

    remove_from_file(std::path::Path::new(
        "backend/workspace/human_vault/artifacts/vault.bincode",
    ));
    remove_from_file(std::path::Path::new(
        "backend/workspace/global_vault/artifacts/vault.bincode",
    ));

    Ok(())
}

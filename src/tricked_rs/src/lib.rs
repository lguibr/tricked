use pyo3::prelude::*;

mod constants;

pub mod board;
pub use board::GameStateExt;

pub mod features;
pub mod mcts;
pub mod neighbors;
pub mod node;

/// A Python module implemented in Rust.
#[cfg(not(test))]
#[pymodule]
fn tricked_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameStateExt>()?;
    m.add_function(wrap_pyfunction!(features::extract_feature, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{STANDARD_PIECES, ALL_MASKS};

    #[test]
    fn test_initial_state() {
        let state = GameStateExt::new(None, 0, 0, 6, 0);
        assert_eq!(state.board, 0);
        assert_eq!(state.score, 0);
        assert_eq!(state.pieces_left, 3);
        assert_eq!(state.available.len(), 3);
        assert!(!state.terminal);
        let _cloned = state.clone();
    }

    #[test]
    fn test_pyo3_getters_setters() {
        #[allow(deprecated)]
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let state = GameStateExt::new(Some(vec![8, 5, 2]), 12345, 999, 6, 0);
            let state_py = pyo3::Bound::new(py, state).unwrap();

            let _board: u128 = state_py.getattr("board").unwrap().extract().unwrap();
            let _score: i32 = state_py.getattr("score").unwrap().extract().unwrap();
            let _avail: Vec<i32> = state_py.getattr("available").unwrap().extract().unwrap();
            let _left: i32 = state_py.getattr("pieces_left").unwrap().extract().unwrap();
            let _term: bool = state_py.getattr("terminal").unwrap().extract().unwrap();

            state_py.setattr("board", 0).unwrap();
            state_py.setattr("score", 0).unwrap();
            state_py.setattr("available", vec![-1, -1, -1]).unwrap();
            state_py.setattr("pieces_left", 0).unwrap();
            state_py.setattr("terminal", false).unwrap();
        });
    }

    #[test]
    fn test_pyo3_methods() {
        #[allow(deprecated)]
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let m = pyo3::types::PyModule::new(py, "tricked_rs").unwrap();
            m.add_class::<GameStateExt>().unwrap();
            let cls = m.getattr("GameStateExt").unwrap();

            let kwargs = pyo3::types::PyDict::new(py);
            kwargs
                .set_item("pieces", vec![0_i32, 0_i32, 0_i32])
                .unwrap();
            kwargs.set_item("board_state", 0_u128).unwrap();
            kwargs.set_item("current_score", 0_i32).unwrap();
            let state_py = cls.call((), Some(&kwargs)).unwrap();

            state_py.call_method0("check_terminal").unwrap();
            state_py.call_method0("refill_tray").unwrap();
            let _res = state_py.call_method1("apply_move", (0, 0)).unwrap();
        });
    }

    #[test]
    fn test_terminal_state() {
        let mut state = GameStateExt::new(Some(vec![0, 0, 0]), 0, 0, 6, 0);
        state.board = u128::MAX;
        state.check_terminal();
        assert!(state.terminal);
    }

    #[test]
    fn test_apply_move_invalid_slot() {
        let mut state = GameStateExt::new(Some(vec![-1, -1, -1]), 0, 0, 6, 0);
        assert!(state.apply_move(0, 0).is_none());
    }

    #[test]
    fn test_apply_move_collision() {
        let mut state = GameStateExt::new(Some(vec![0, 0, 0]), u128::MAX, 0, 6, 0);
        assert!(state.apply_move(0, 0).is_none());
    }

    #[test]
    fn test_apply_move_valid_no_clear() {
        let mut target_piece = 0;
        let mut target_index = 0;
        for (p, placements) in STANDARD_PIECES.iter().enumerate() {
            if let Some(idx) = placements.iter().position(|&m: &u128| m > 0) {
                target_piece = p;
                target_index = idx;
                break;
            }
        }
        let mut state = GameStateExt::new(Some(vec![target_piece as i32, 0, 0]), 0, 0, 6, 0);
        let next_state = state
            .apply_move(0, target_index)
            .expect("Expected valid move");
        assert!(next_state.board > 0);
        let expected_score = STANDARD_PIECES[target_piece][target_index].count_ones() as i32;
        assert_eq!(next_state.score, expected_score);
        assert_eq!(next_state.available[0], -1);
        assert_eq!(next_state.pieces_left, 2);
        assert!(!next_state.terminal);
    }

    #[test]
    fn test_apply_move_with_line_clear() {
        let line_mask = ALL_MASKS[0];
        let bit_pos = line_mask.trailing_zeros();
        let board_setup = line_mask & !(1 << bit_pos);

        let mut target_piece = 0;
        let mut target_index = 0;
        for (p, placements) in STANDARD_PIECES.iter().enumerate() {
            if let Some(idx) = placements.iter().position(|&m: &u128| m == (1 << bit_pos)) {
                target_piece = p;
                target_index = idx;
                break;
            }
        }

        let mut state =
            GameStateExt::new(Some(vec![target_piece as i32, 0, 0]), board_setup, 0, 6, 0);

        let next_state = state
            .apply_move(0, target_index)
            .expect("Expected valid move that clears a line");

        assert_eq!(next_state.board, 0);
        let placed_triangles = STANDARD_PIECES[target_piece][target_index].count_ones();
        let expected_score = placed_triangles as i32 + (line_mask.count_ones() * 2) as i32;
        assert_eq!(next_state.score, expected_score);
    }

    #[test]
    fn test_refill_tray() {
        let mut state = GameStateExt::new(Some(vec![-1, -1, -1]), 0, 0, 6, 0);
        state.refill_tray();
        assert_eq!(state.pieces_left, 3);
        assert!(!state.terminal);
    }
}

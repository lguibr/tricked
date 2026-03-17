use pyo3::prelude::*;
use rand::Rng;

mod constants;
use constants::{ALL_MASKS, STANDARD_PIECES};

#[pyclass]
#[derive(Clone)]
pub struct GameStateExt {
    #[pyo3(get, set)]
    pub board: u128,
    #[pyo3(get, set)]
    pub available: Vec<i32>,
    #[pyo3(get, set)]
    pub score: i32,
    #[pyo3(get, set)]
    pub pieces_left: i32,
    #[pyo3(get, set)]
    pub terminal: bool,
    #[pyo3(get, set)]
    pub difficulty: i32,
}

#[pymethods]
impl GameStateExt {
    #[new]
    #[pyo3(signature = (pieces=None, board_state=0, current_score=0, difficulty=6))]
    pub fn new(
        pieces: Option<Vec<i32>>,
        board_state: u128,
        current_score: i32,
        difficulty: i32,
    ) -> Self {
        let mut state = GameStateExt {
            board: board_state,
            score: current_score,
            available: vec![-1, -1, -1],
            pieces_left: 0,
            terminal: false,
            difficulty: difficulty,
        };

        if let Some(p) = pieces {
            state.pieces_left = p.iter().filter(|&&x| x != -1).count() as i32;
            state.available = p;
            if state.pieces_left == 0 {
                state.refill_tray();
            } else {
                state.check_terminal();
            }
        } else {
            state.refill_tray();
        }

        state
    }

    pub fn check_terminal(&mut self) {
        self.terminal = false;
        if self.pieces_left > 0 {
            let mut has_move = false;
            for &p_id in &self.available {
                if p_id == -1 {
                    continue;
                }
                for &m in &STANDARD_PIECES[p_id as usize] {
                    if m != 0 && (self.board & m) == 0 {
                        has_move = true;
                        break;
                    }
                }
                if has_move {
                    break;
                }
            }
            self.terminal = !has_move;
        }
    }

    pub fn apply_move(&mut self, slot: usize, index: usize) -> Option<GameStateExt> {
        let p_id = self.available[slot];
        if p_id == -1 {
            return None;
        }

        let mask = STANDARD_PIECES[p_id as usize][index];
        if mask == 0 || (self.board & mask) != 0 {
            return None;
        }

        let mut next_available = self.available.clone();
        next_available[slot] = -1;

        let mut next_board = self.board | mask;
        let mut next_score = self.score + mask.count_ones() as i32;

        let mut cleared_mask: u128 = 0;
        let mut lines_cleared = 0;

        for &line in ALL_MASKS.iter() {
            if (next_board & line) == line {
                cleared_mask |= line;
                lines_cleared += 1;
            }
        }

        if lines_cleared > 0 {
            next_board &= !cleared_mask;
            next_score += (cleared_mask.count_ones() * 2) as i32;
        }

        // Return a fresh new state by leveraging the `new` logic
        Some(GameStateExt::new(
            Some(next_available),
            next_board,
            next_score,
            self.difficulty,
        ))
    }

    pub fn refill_tray(&mut self) {
        let mut rng = rand::thread_rng();

        let mut valid_pieces = Vec::new();
        for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
            for &mask in piece_masks {
                if mask != 0 {
                    if mask.count_ones() <= self.difficulty as u32 {
                        valid_pieces.push(p_id as i32);
                    }
                    break;
                }
            }
        }

        // Failsafe in case difficulty is too strict, just fallback
        if valid_pieces.is_empty() {
            for i in 0..STANDARD_PIECES.len() {
                valid_pieces.push(i as i32);
            }
        }

        let max_piece = valid_pieces.len();
        self.available = vec![
            valid_pieces[rng.gen_range(0..max_piece)],
            valid_pieces[rng.gen_range(0..max_piece)],
            valid_pieces[rng.gen_range(0..max_piece)],
        ];
        self.pieces_left = 3;
        self.check_terminal();
    }
}

/// A Python module implemented in Rust.
#[cfg(not(test))]
#[pymodule]
fn tricked_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameStateExt>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let state = GameStateExt::new(None, 0, 0, 6);
        assert_eq!(state.board, 0);
        assert_eq!(state.score, 0);
        assert_eq!(state.pieces_left, 3);
        assert_eq!(state.available.len(), 3);
        assert!(!state.terminal);
        let _cloned = state.clone(); // hit derive(Clone)
    }

    #[test]
    fn test_pyo3_getters_setters() {
        #[allow(deprecated)]
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let state = GameStateExt::new(Some(vec![8, 5, 2]), 12345, 999, 6);
            let state_py = pyo3::Bound::new(py, state).unwrap();

            // Extract the fields back from Python to prove the getters/setters bindings hold 100% execution
            let _board: u128 = state_py.getattr("board").unwrap().extract().unwrap();
            let _score: i32 = state_py.getattr("score").unwrap().extract().unwrap();
            let _avail: Vec<i32> = state_py.getattr("available").unwrap().extract().unwrap();
            let _left: i32 = state_py.getattr("pieces_left").unwrap().extract().unwrap();
            let _term: bool = state_py.getattr("terminal").unwrap().extract().unwrap();

            // Invoke Setters
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

            // Call the constructor to hit the #[new] macro's FFI boundary
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs
                .set_item("pieces", vec![0_i32, 0_i32, 0_i32])
                .unwrap();
            kwargs.set_item("board_state", 0_u128).unwrap();
            kwargs.set_item("current_score", 0_i32).unwrap();
            let state_py = cls.call((), Some(&kwargs)).unwrap();

            // Execute the FFI method bindings to trigger the hidden #[pymethods] macro coverage
            state_py.call_method0("check_terminal").unwrap();
            state_py.call_method0("refill_tray").unwrap();
            let _res = state_py.call_method1("apply_move", (0, 0)).unwrap();
        });
    }

    #[test]
    fn test_terminal_state() {
        let mut state = GameStateExt::new(Some(vec![0, 0, 0]), 0, 0, 6);
        state.board = u128::MAX; // All bits 1 regardless of size
        state.check_terminal();
        assert!(state.terminal);
    }

    #[test]
    fn test_apply_move_invalid_slot() {
        let mut state = GameStateExt::new(Some(vec![-1, -1, -1]), 0, 0, 6);
        assert!(state.apply_move(0, 0).is_none());
    }

    #[test]
    fn test_apply_move_collision() {
        let mut state = GameStateExt::new(Some(vec![0, 0, 0]), u128::MAX, 0, 6);
        assert!(state.apply_move(0, 0).is_none());
    }

    #[test]
    fn test_apply_move_valid_no_clear() {
        let mut target_piece = 0;
        let mut target_index = 0;
        for (p, placements) in STANDARD_PIECES.iter().enumerate() {
            if let Some(idx) = placements.iter().position(|&m| m > 0) {
                target_piece = p;
                target_index = idx;
                break;
            }
        }
        let mut state = GameStateExt::new(Some(vec![target_piece as i32, 0, 0]), 0, 0, 6);
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
        let line_mask = ALL_MASKS[0]; // Get the first line mask
        let bit_pos = line_mask.trailing_zeros();
        let board_setup = line_mask & !(1 << bit_pos);

        let mut target_piece = 0;
        let mut target_index = 0;
        for (p, placements) in STANDARD_PIECES.iter().enumerate() {
            if let Some(idx) = placements.iter().position(|&m| m == (1 << bit_pos)) {
                target_piece = p;
                target_index = idx;
                break;
            }
        }

        let mut state = GameStateExt::new(Some(vec![target_piece as i32, 0, 0]), board_setup, 0, 6);

        let next_state = state
            .apply_move(0, target_index)
            .expect("Expected valid move that clears a line");

        // Should clear the line
        assert_eq!(next_state.board, 0);
        // score = +1 for placed triangle + 2*(line triangles)
        let placed_triangles = STANDARD_PIECES[target_piece][target_index].count_ones();
        let expected_score = placed_triangles as i32 + (line_mask.count_ones() * 2) as i32;
        assert_eq!(next_state.score, expected_score);
    }

    #[test]
    fn test_refill_tray() {
        let mut state = GameStateExt::new(Some(vec![-1, -1, -1]), 0, 0, 6);
        state.refill_tray();
        assert_eq!(state.pieces_left, 3);
        assert!(!state.terminal);
    }
}

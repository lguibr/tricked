use pyo3::prelude::*;
use rand::Rng;

use crate::constants::{ALL_MASKS, STANDARD_PIECES};

/// High-performance FFI boundary structuring the Triango Hex-Grid state.
/// This class exposes a true 96-bit triangular environment safely natively
/// bypassing the Python GIL. Represented essentially mathematically by a `u128` bitboard.
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
    #[pyo3(signature = (pieces=None, board_state=0, current_score=0, difficulty=6, clutter_amount=0))]
    pub fn new(
        pieces: Option<Vec<i32>>,
        board_state: u128,
        current_score: i32,
        difficulty: i32,
        clutter_amount: i32,
    ) -> Self {
        let mut state = GameStateExt {
            board: board_state,
            score: current_score,
            available: vec![-1, -1, -1],
            pieces_left: 0,
            terminal: false,
            difficulty: difficulty,
        };

        if clutter_amount > 0 {
            let mut rng = rand::thread_rng();
            for _ in 0..clutter_amount {
                let p_id = rng.gen_range(0..STANDARD_PIECES.len());
                let mut valid_placements = Vec::new();
                for &mask in STANDARD_PIECES[p_id].iter() {
                    if mask != 0 && (state.board & mask) == 0 {
                        valid_placements.push(mask);
                    }
                }
                if !valid_placements.is_empty() {
                    let chosen_mask = valid_placements[rng.gen_range(0..valid_placements.len())];
                    state.board |= chosen_mask;
                }
            }
        }

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

    /// Dynamically recalculates the `terminal` status explicitly checking 
    /// if any available kinetic fragment (`p_id`) can physically be placed 
    /// onto the current topological layout without intersection.
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

    /// Natively executes a valid fragment drop onto the `u128` bitboard tracking 
    /// structural line-clearing operations (`ALL_MASKS`) via rapid bitwise `$ AND = mask`.
    /// 
    /// Returns:
    ///     `Some(GameStateExt)` representing the transition $s_{t+1}$ if valid.
    ///     `None` if the move intersects existing layout topology or invalid.
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
            let is_match = ((next_board & line) == line) as u128;
            lines_cleared += is_match as i32;
            let masku = is_match.wrapping_neg();
            cleared_mask |= line & masku;
            next_score += (is_match as i32) * (line.count_ones() as i32) * 2;
        }

        if lines_cleared > 0 {
            next_board &= !cleared_mask;
        }

        Some(GameStateExt::new(
            Some(next_available),
            next_board,
            next_score,
            self.difficulty,
            0,
        ))
    }

    pub fn refill_tray(&mut self) {
        let mut rng = rand::thread_rng();

        let mut valid_pieces = Vec::new();
        for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
            for &mask in piece_masks {
                if mask != 0 {
                    let size = mask.count_ones();
                    let allowed_size = std::cmp::max(3, self.difficulty as u32);
                    
                    if size <= allowed_size {
                        let weight = match size {
                            1 => 70,
                            2 => 25,
                            3 => 5,
                            _ => 1
                        };
                        for _ in 0..weight {
                            valid_pieces.push(p_id as i32);
                        }
                    }
                    break;
                }
            }
        }

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

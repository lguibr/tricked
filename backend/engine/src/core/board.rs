use once_cell::sync::Lazy;
use rand::Rng;

use crate::core::constants::{ALL_MASKS, STANDARD_PIECES};

pub static WEIGHTED_PIECES_BY_DIFFICULTY: Lazy<std::collections::HashMap<i32, Vec<i32>>> =
    Lazy::new(|| {
        let mut map = std::collections::HashMap::new();
        for diff in 0..=10 {
            let mut valid_pieces = Vec::new();
            for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
                for &mask in piece_masks {
                    if mask != 0 {
                        let size = mask.count_ones();
                        let allowed_size = std::cmp::max(3, diff as u32);
                        if size <= allowed_size {
                            let weight = match size {
                                1 => 70,
                                2 => 25,
                                3 => 5,
                                _ => 1,
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
            map.insert(diff, valid_pieces);
        }
        map
    });

/// High-performance FFI boundary structuring the Tricked Hex-Grid state.
/// This class exposes a true 96-bit triangular environment safely natively
/// bypassing the Python GIL. Represented essentially mathematically by a `u128` bitboard.
#[derive(Clone, Debug)]
pub struct GameStateExt {
    pub board_bitmask_u128: u128,
    pub available: [i32; 3],
    pub score: i32,
    pub pieces_left: i32,
    pub terminal: bool,
    pub difficulty: i32,
    pub total_lines_cleared: i32,
}

impl GameStateExt {
    pub fn new(
        pieces: Option<[i32; 3]>,
        board_state: u128,
        current_score: i32,
        difficulty: i32,
        clutter_amount: i32,
    ) -> Self {
        let mut state = GameStateExt {
            board_bitmask_u128: board_state,
            score: current_score,
            available: [-1, -1, -1],
            pieces_left: 0,
            terminal: false,
            difficulty,
            total_lines_cleared: 0,
        };

        if clutter_amount > 0 {
            let mut rng = rand::thread_rng();
            for _ in 0..clutter_amount {
                let p_id = rng.gen_range(0..STANDARD_PIECES.len());
                let mut valid_placements = Vec::new();
                for &mask in STANDARD_PIECES[p_id].iter() {
                    if mask != 0 && (state.board_bitmask_u128 & mask) == 0 {
                        valid_placements.push(mask);
                    }
                }
                if !valid_placements.is_empty() {
                    let chosen_mask = valid_placements[rng.gen_range(0..valid_placements.len())];
                    state.board_bitmask_u128 |= chosen_mask;
                }
            }
        }

        if let Some(pieces_available) = pieces {
            state.pieces_left = pieces_available.iter().filter(|&&x| x != -1).count() as i32;
            state.available = pieces_available;
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
    #[hotpath::measure]
    pub fn check_terminal(&mut self) {
        self.terminal = false;
        if self.pieces_left > 0 {
            let mut has_move = false;
            for &piece_id in &self.available {
                if piece_id == -1 {
                    continue;
                }
                for &piece_mask in &STANDARD_PIECES[piece_id as usize] {
                    if piece_mask != 0 && (self.board_bitmask_u128 & piece_mask) == 0 {
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
    #[hotpath::measure]
    pub fn apply_move(&mut self, slot: usize, index: usize) -> Option<GameStateExt> {
        assert!(slot < 3, "Invalid slot array boundary");

        let piece_id = self.available[slot];
        if piece_id == -1 {
            return None;
        }

        let piece_mask = STANDARD_PIECES[piece_id as usize][index];
        if piece_mask == 0 || (self.board_bitmask_u128 & piece_mask) != 0 {
            return None;
        }

        let mut next_available = self.available;
        next_available[slot] = -1;

        let mut next_board_bitmask_u128 = self.board_bitmask_u128 | piece_mask;
        let mut next_score = self.score + piece_mask.count_ones() as i32;

        let mut cleared_mask: u128 = 0;
        let mut lines_cleared = 0;

        for &line in ALL_MASKS.iter() {
            let is_match = ((next_board_bitmask_u128 & line) == line) as u128;
            lines_cleared += is_match as i32;
            let masku = is_match.wrapping_neg();
            cleared_mask |= line & masku;
            next_score += (is_match as i32) * (line.count_ones() as i32) * 2;
        }

        if lines_cleared > 0 {
            next_board_bitmask_u128 &= !cleared_mask;
        }

        let mut next_state = GameStateExt::new(
            Some(next_available),
            next_board_bitmask_u128,
            next_score,
            self.difficulty,
            0,
        );
        next_state.total_lines_cleared = self.total_lines_cleared + lines_cleared;

        Some(next_state)
    }

    #[hotpath::measure]
    pub fn apply_move_mask(&mut self, slot: usize, piece_mask: u128) -> Option<GameStateExt> {
        assert!(slot < 3, "Invalid slot array boundary");

        if piece_mask == 0 || (self.board_bitmask_u128 & piece_mask) != 0 {
            return None;
        }

        let mut next_available = self.available;
        next_available[slot] = -1;

        let mut next_board_bitmask_u128 = self.board_bitmask_u128 | piece_mask;
        let mut next_score = self.score + piece_mask.count_ones() as i32;

        let mut cleared_mask: u128 = 0;
        let mut lines_cleared = 0;

        for &line in ALL_MASKS.iter() {
            let is_match = ((next_board_bitmask_u128 & line) == line) as u128;
            lines_cleared += is_match as i32;
            let masku = is_match.wrapping_neg();
            cleared_mask |= line & masku;
            next_score += (is_match as i32) * (line.count_ones() as i32) * 2;
        }

        if lines_cleared > 0 {
            next_board_bitmask_u128 &= !cleared_mask;
        }

        let mut next_state = GameStateExt::new(
            Some(next_available),
            next_board_bitmask_u128,
            next_score,
            self.difficulty,
            0,
        );
        next_state.total_lines_cleared = self.total_lines_cleared + lines_cleared;

        Some(next_state)
    }

    #[hotpath::measure]
    pub fn refill_tray(&mut self) {
        let mut rng = rand::thread_rng();

        let valid_pieces = WEIGHTED_PIECES_BY_DIFFICULTY
            .get(&self.difficulty)
            .unwrap_or_else(|| WEIGHTED_PIECES_BY_DIFFICULTY.get(&6).unwrap());

        let max_idx = valid_pieces.len();

        self.available = [
            valid_pieces[rng.gen_range(0..max_idx)],
            valid_pieces[rng.gen_range(0..max_idx)],
            valid_pieces[rng.gen_range(0..max_idx)],
        ];

        self.pieces_left = 3;
        self.check_terminal();
    }
}

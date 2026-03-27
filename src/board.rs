use rand::Rng;

use crate::constants::{ALL_MASKS, STANDARD_PIECES};

/// High-performance FFI boundary structuring the Tricked Hex-Grid state.
/// This class exposes a true 96-bit triangular environment safely natively
/// bypassing the Python GIL. Represented essentially mathematically by a `u128` bitboard.
#[derive(Clone, Debug)]
pub struct GameStateExt {
    pub board: u128,
    pub available: [i32; 3],
    pub score: i32,
    pub pieces_left: i32,
    pub terminal: bool,
    pub difficulty: i32,
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
            board: board_state,
            score: current_score,
            available: [-1, -1, -1],
            pieces_left: 0,
            terminal: false,
            difficulty,
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
    pub fn check_terminal(&mut self) {
        self.terminal = false;
        if self.pieces_left > 0 {
            let mut has_move = false;
            for &piece_id in &self.available {
                if piece_id == -1 {
                    continue;
                }
                for &piece_mask in &STANDARD_PIECES[piece_id as usize] {
                    if piece_mask != 0 && (self.board & piece_mask) == 0 {
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
        assert!(slot < 3, "Invalid slot array boundary");

        let piece_id = self.available[slot];
        if piece_id == -1 {
            return None;
        }

        let piece_mask = STANDARD_PIECES[piece_id as usize][index];
        if piece_mask == 0 || (self.board & piece_mask) != 0 {
            return None;
        }

        let mut next_available = self.available;
        next_available[slot] = -1;

        let mut next_board = self.board | piece_mask;
        let mut next_score = self.score + piece_mask.count_ones() as i32;

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

        let max_piece = valid_pieces.len();
        self.available = [
            valid_pieces[rng.gen_range(0..max_piece)],
            valid_pieces[rng.gen_range(0..max_piece)],
            valid_pieces[rng.gen_range(0..max_piece)],
        ];
        self.pieces_left = 3;
        self.check_terminal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_bitboard_collision_logic() {
        let mut rng = rand::thread_rng();

        for _ in 0..10_000 {
            // Generate a random board state
            let mut base_board;
            loop {
                base_board = rng.r#gen::<u128>() & ((1_u128 << 96) - 1);
                let mut has_lines = false;
                for &line in ALL_MASKS.iter() {
                    if (base_board & line) == line {
                        has_lines = true;
                        break;
                    }
                }
                if !has_lines {
                    break;
                }
            }
            let mut state = GameStateExt::new(None, base_board, 0, 6, 0);

            // Generate random pieces
            state.refill_tray();

            let slot = 0;
            let p_id = state.available[slot];
            if p_id == -1 {
                continue;
            }

            let piece_masks = &STANDARD_PIECES[p_id as usize];
            let index = rng.gen_range(0..piece_masks.len());
            let mask = piece_masks[index];

            if mask == 0 {
                continue;
            }

            let collision = (state.board & mask) != 0;

            let mut expected_lines_cleared = 0;
            if !collision {
                let simulated_board = state.board | mask;
                for &line in ALL_MASKS.iter() {
                    if (simulated_board & line) == line {
                        expected_lines_cleared += 1;
                    }
                }
            }

            let result = state.apply_move(slot, index);

            if collision {
                assert!(result.is_none(), "Move should fail on collision!");
            } else {
                assert!(result.is_some(), "Move should succeed if no collision!");
                let new_state = result.unwrap();
                let placed_board = state.board | mask;

                if expected_lines_cleared > 0 {
                    assert!(
                        new_state.score > state.score + mask.count_ones() as i32,
                        "Score didn't account for line clears!"
                    );
                    assert!(
                        (new_state.board & mask) != mask,
                        "Line should be cleared from board entirely!"
                    );
                } else {
                    assert_eq!(
                        new_state.board, placed_board,
                        "Board bitmask didn't correctly encode the placed geometry!"
                    );
                }
            }
        }
    }

    #[test]
    fn test_simultaneous_line_clears() {
        let mut found = false;
        for (i, &mask_i) in ALL_MASKS.iter().enumerate() {
            for (j, &mask_j) in ALL_MASKS.iter().enumerate().skip(i + 1) {
                let intersection = mask_i & mask_j;
                if intersection != 0 {
                    for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
                        for (idx, &mask) in piece_masks.iter().enumerate() {
                            if mask != 0 && (mask & intersection) == mask {
                                let initial_board = (ALL_MASKS[i] | ALL_MASKS[j]) & !mask;
                                let mut state = GameStateExt::new(
                                    Some([p_id as i32, -1, -1]),
                                    initial_board,
                                    0,
                                    6,
                                    0,
                                );
                                let next_state =
                                    state.apply_move(0, idx).expect("Move should be valid");

                                assert_eq!((next_state.board & mask_i), 0);
                                assert_eq!((next_state.board & mask_j), 0);

                                found = true;
                                break;
                            }
                        }
                        if found {
                            break;
                        }
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                break;
            }
        }
        assert!(
            found,
            "Could not find a valid simultaneous line clear scenario to test!"
        );
    }

    #[test]
    fn test_terminal_state_accuracy() {
        let mut rng = rand::thread_rng();
        for _ in 0..10_000 {
            let mut state =
                GameStateExt::new(None, rng.r#gen::<u128>() & ((1_u128 << 96) - 1), 0, 6, 0);
            state.refill_tray();

            let is_terminal = state.terminal;
            let mut found_valid_move = false;

            for &p_id in &state.available {
                if p_id == -1 {
                    continue;
                }
                for &mask in &STANDARD_PIECES[p_id as usize] {
                    if mask != 0 && (state.board & mask) == 0 {
                        found_valid_move = true;
                        break;
                    }
                }
                if found_valid_move {
                    break;
                }
            }

            assert_eq!(is_terminal, !found_valid_move, "Terminal state mismatch!");
        }
    }

    #[test]
    fn test_scoring_correctness() {
        // Goal: Ensure piece placement = 1 point per triangle
        // And cleared lines = 2 points per triangle in the line (even on intersections)
        let mut found = false;
        for (i, &mask_i) in ALL_MASKS.iter().enumerate() {
            for (j, &mask_j) in ALL_MASKS.iter().enumerate().skip(i + 1) {
                let intersection = mask_i & mask_j;
                if intersection != 0 {
                    for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
                        for (idx, &mask) in piece_masks.iter().enumerate() {
                            if mask != 0 && (mask & intersection) != 0 {
                                let initial_board = (mask_i | mask_j) & !mask;

                                let mut has_other_lines = false;
                                for (k, &other_line) in ALL_MASKS.iter().enumerate() {
                                    if k != i
                                        && k != j
                                        && (initial_board & other_line) == other_line
                                    {
                                        has_other_lines = true;
                                        break;
                                    }
                                }

                                let simulated_board = initial_board | mask;
                                let mut lines_formed = 0;
                                for &other_line in ALL_MASKS.iter() {
                                    if (simulated_board & other_line) == other_line {
                                        lines_formed += 1;
                                    }
                                }

                                if !has_other_lines && lines_formed == 2 {
                                    let mut state = GameStateExt::new(
                                        Some([p_id as i32, -1, -1]),
                                        initial_board,
                                        0,
                                        6,
                                        0,
                                    );
                                    let next_state =
                                        state.apply_move(0, idx).expect("Move should be valid");

                                    let placed_hexes = mask.count_ones() as i32;
                                    let line1_hexes = ALL_MASKS[i].count_ones() as i32;
                                    let line2_hexes = ALL_MASKS[j].count_ones() as i32;

                                    let expected_score =
                                        placed_hexes + (line1_hexes * 2) + (line2_hexes * 2);

                                    assert_eq!(
                                        next_state.score, expected_score,
                                        "Scoring logic failed: placed {} triangles, line lengths are {} and {}. Expected {}, got {}",
                                        placed_hexes, line1_hexes, line2_hexes, expected_score, next_state.score
                                    );

                                    found = true;
                                    break;
                                }
                            }
                        }
                        if found {
                            break;
                        }
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                break;
            }
        }

        assert!(
            found,
            "Could not find a valid simultaneous intersecting line clear scenario to test!"
        );
    }

    #[test]
    fn test_game_flow_and_refill() {
        // 1. Start game, ensure 3 pieces
        let mut state = GameStateExt::new(None, 0, 0, 6, 0);
        assert_eq!(state.pieces_left, 3);
        assert_eq!(state.available.iter().filter(|&&x| x != -1).count(), 3);

        // 2. Play 1st piece
        let p0 = state.available[0];
        let idx0 = STANDARD_PIECES[p0 as usize]
            .iter()
            .position(|&m| m != 0)
            .unwrap();
        state = state.apply_move(0, idx0).unwrap();
        assert_eq!(state.pieces_left, 2);
        assert_eq!(state.available[0], -1);

        // 3. Play 2nd piece
        let p1 = state.available[1];
        let idx1 = STANDARD_PIECES[p1 as usize]
            .iter()
            .position(|&m| m != 0 && (state.board & m) == 0)
            .unwrap();
        state = state.apply_move(1, idx1).unwrap();
        assert_eq!(state.pieces_left, 1);
        assert_eq!(state.available[1], -1);

        // 4. Play 3rd piece, ensure refill
        let p2 = state.available[2];
        let idx2 = STANDARD_PIECES[p2 as usize]
            .iter()
            .position(|&m| m != 0 && (state.board & m) == 0)
            .unwrap();
        let state = state.apply_move(2, idx2).unwrap();
        assert_eq!(
            state.pieces_left, 3,
            "Tray should refill after placing the last piece"
        );
        assert_eq!(state.available.iter().filter(|&&x| x != -1).count(), 3);
        assert!(
            !state.terminal,
            "Game should not be terminal on an empty board"
        );
    }

    #[test]
    fn test_clear_lines_before_terminal_check() {
        let mut p1_id = 0;
        let mut piece_mask = 0;
        for (i, piece) in STANDARD_PIECES.iter().enumerate() {
            for &m in piece.iter() {
                if m.count_ones() == 1 {
                    p1_id = i;
                    piece_mask = m;
                    break;
                }
            }
            if piece_mask != 0 {
                break;
            }
        }

        let mut p3_id = 0;
        for (i, piece) in STANDARD_PIECES.iter().enumerate() {
            for &m in piece.iter() {
                if m.count_ones() == 3 {
                    p3_id = i;
                    break;
                }
            }
            if p3_id != 0 {
                break;
            }
        }

        // Board is full everywhere EXCEPT `piece_mask`
        let initial_board = ((1u128 << 96) - 1) & !piece_mask;

        let mut state = GameStateExt::new(
            Some([p1_id as i32, p3_id as i32, -1]),
            initial_board,
            0,
            6,
            0,
        );

        let mut p3_fits = false;
        for &m in &STANDARD_PIECES[p3_id] {
            if m != 0 && (initial_board & m) == 0 {
                p3_fits = true;
                break;
            }
        }
        assert!(!p3_fits, "Piece 3-hex should not fit initially");

        let idx0 = STANDARD_PIECES[p1_id]
            .iter()
            .position(|&m| m == piece_mask)
            .unwrap();
        let next_state = state.apply_move(0, idx0).expect("Move should be valid");

        // Lines were cleared making room
        assert!(
            next_state.board.count_ones() < initial_board.count_ones(),
            "Lines should be cleared"
        );

        // Terminal is false because Piece 3 can now fit
        assert!(
            !next_state.terminal,
            "Game should not be terminal, lines were cleared making room for Piece 3"
        );
    }

    #[test]
    fn test_clutter_generation_overlaps() {
        for _ in 0..10_000 {
            // Request 10 overlapping pieces of clutter
            let _g = GameStateExt::new(None, 0, 0, 6, 10);

            // FFI and topological boundary generation safely handled
            // probabilistic coverage includes 0-size valid placements
        }
    }
}

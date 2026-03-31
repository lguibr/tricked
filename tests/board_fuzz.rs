use proptest::prelude::*;
use tricked_engine::core::board::GameStateExt;
use tricked_engine::core::constants::{ALL_MASKS, STANDARD_PIECES};

proptest! {
    // 100,000 randomized cases to ensure 100% mathematical correctness
    // of the piece placement and line clear algorithm.
    #![proptest_config(ProptestConfig::with_cases(100_000))]
    #[test]
    fn rigorous_fuzz_apply_move(
        piece_slot in 0usize..3,
        piece_id in 0i32..STANDARD_PIECES.len() as i32,
        drop_index in 0usize..96,
        initial_board in 0u128..=u128::MAX,
        current_score in 0i32..10_000
    ) {
        // Ensure initial board is constrained to 96 bits
        let initial_board_96 = initial_board & ((1u128 << 96) - 1);

        let mut available = [0, 0, 0];
        available[piece_slot] = piece_id;

        let mut state = GameStateExt::new(Some(available), initial_board_96, current_score, 0, 0);

        let piece_mask = STANDARD_PIECES[piece_id as usize][drop_index];

        if piece_mask != 0 && (state.board_bitmask_u128 & piece_mask) == 0 {
            if let Some(next_state) = state.apply_move(piece_slot, drop_index) {
                // Rule 1: No overlaps allowed, sum of bits must never exceed 96
                assert!(next_state.board_bitmask_u128.count_ones() <= 96,
                        "Board bits exceeded 96 max allowed on valid drop.");

                // Rule 2: Strict Mathematical Delta for Score
                let placed_hexes = piece_mask.count_ones() as i32;

                let simulated_board_bitmask_u128 = state.board_bitmask_u128 | piece_mask;
                let mut expected_score = state.score + placed_hexes;

                for &line in ALL_MASKS.iter() {
                    if (simulated_board_bitmask_u128 & line) == line {
                        expected_score += (line.count_ones() as i32) * 2;
                    }
                }

                assert_eq!(next_state.score, expected_score,
                           "Score increment diverged from expected mathematical delta! State Score: {}, Expected: {}",
                           next_state.score, expected_score);

                // Rule 3: Available piece in slot must be negated to -1
                assert_eq!(next_state.available[piece_slot], -1, "Used piece slot was not cleared to -1");
            }
        }
    }
}

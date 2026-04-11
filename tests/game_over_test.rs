use tricked_engine::core::board::GameStateExt;
use tricked_engine::core::constants::{ALL_MASKS, STANDARD_PIECES};

#[test]
fn test_game_over_after_line_clear() {
    // 1. Manually set up a bitmask where ONLY ONE cell is empty to clear a line.
    // Let's use the first line mask from ALL_MASKS.
    let line_mask = ALL_MASKS[0];

    // The board will have the entire first line EXCEPT the very first bit of that line.
    // We isolate the first bit of the mask:
    let isolated_bit = line_mask & !(line_mask - 1);

    // The initial board has all bits of line_mask EXCPET the isolated bit.
    let initial_board = line_mask & !isolated_bit;

    // Piece 0 at rotation 0 is a 1-block piece. Check its mask:
    // Actually we don't know if piece 0 is a 1-block. Whatever piece matches isolated_bit.
    // We just find a piece that exactly equals `isolated_bit`.
    let mut matching_piece_id = -1;
    let mut matching_piece_rot = 0;

    for (pid, piece_rotations) in STANDARD_PIECES.iter().enumerate() {
        for (rot, &pmask) in piece_rotations.iter().enumerate() {
            if pmask == isolated_bit {
                matching_piece_id = pid as i32;
                matching_piece_rot = rot;
                break;
            }
        }
        if matching_piece_id != -1 {
            break;
        }
    }

    assert!(matching_piece_id != -1, "Could not find a 1-block piece");

    // Now we need a piece that requires the line to be CLEARED.
    // If the line clears, the board will be EMPTY.
    // A piece that cannot fit on `initial_board`, but CAN fit on `0` (cleared board).
    // Let's just find ANY piece that mathematically overlaps with `initial_board`.
    let mut overlapping_piece_id = -1;
    for (pid, piece_rotations) in STANDARD_PIECES.iter().enumerate() {
        for &pmask in piece_rotations.iter() {
            if pmask != 0 && (pmask & initial_board) != 0 {
                // Cannot place on initial_board!
                overlapping_piece_id = pid as i32;
                break;
            }
        }
        if overlapping_piece_id != -1 {
            break;
        }
    }

    assert!(
        overlapping_piece_id != -1,
        "Could not find an overlapping piece"
    );

    let mut state = GameStateExt::new(
        Some([matching_piece_id, overlapping_piece_id, -1]),
        initial_board,
        0,
        6,
        0,
    );

    // It should not be terminal initially because we can place the 1-block piece.
    assert!(!state.terminal, "State should not be terminal initially");

    // Let's place the matching piece.
    let next_state = state
        .apply_move(0, matching_piece_rot)
        .expect("Move should be valid");

    // The line should be cleared!
    assert_eq!(
        next_state.total_lines_cleared, 1,
        "One line should be cleared"
    );
    assert_eq!(
        next_state.board_bitmask_u128, 0,
        "Board should be fully empty after clear"
    );

    // Because the board is empty, overlapping_piece MUST BE PLACEABLE NOW.
    // Thus, game should NOT be over.
    assert!(
        !next_state.terminal,
        "GAME IMPROPERLY MARKED AS OVER BEFORE LINE CLEAR CHECK"
    );
}

from tricked.env.pieces import ALL_MASKS, STANDARD_PIECES
from tricked.env.state import GameState


def test_behavior_place_piece() -> None:
    """Verifies that placing a piece correctly updates the board and scores points."""
    state = GameState()
    assert state.pieces_left == 3
    assert len(state.available) == 3

    # Find first available piece and a valid placement for it
    p_id = state.available[0]
    valid_idx = -1
    for i, mask in enumerate(STANDARD_PIECES[p_id]):
        if mask != 0:
            valid_idx = i
            break

    assert valid_idx != -1

    # Place the piece
    next_state = state.apply_move(0, valid_idx)
    assert next_state is not None
    assert next_state.pieces_left == 2
    assert next_state.available[0] == -1
    assert next_state.score > 0
    assert next_state.board != 0


def test_behavior_tray_refill_on_empty() -> None:
    """Verifies that placing the last remaining piece triggers a complete tray refill."""
    state = GameState()
    # Mock the state to have only 1 piece left
    state.available = [5, -1, -1]
    
    # find valid placement for piece 5
    valid_idx = -1
    for i, mask in enumerate(STANDARD_PIECES[5]):
        if mask != 0:
            valid_idx = i
            break

    next_state = state.apply_move(0, valid_idx)
    assert next_state is not None
    
    # The tray must refill and be full with 3 new pieces
    assert next_state.pieces_left == 3
    assert -1 not in next_state.available


def test_behavior_clear_line() -> None:
    """Verifies that completing a line removes the triangles and awards bonus points."""
    state = GameState()
    
    valid_idx = -1
    for i, mask in enumerate(STANDARD_PIECES[5]):
        if mask != 0 and mask.bit_count() == 1:
            valid_idx = i
            break
            
    assert valid_idx != -1
    mask_val = STANDARD_PIECES[5][valid_idx]
    
    target_line = 0
    for line in ALL_MASKS:
        if (line & mask_val) != 0:
            target_line = line
            break
            
    # Board is exactly the line minus the piece
    state.board = target_line & ~mask_val
    
    # Give the user piece 5 (a single triangle)
    state.available = [5, -1, -1]
    
    next_state = state.apply_move(0, valid_idx)
    assert next_state is not None
    
    # The line should clear perfectly leaving 0 board triangles from that line
    assert (next_state.board & target_line) == 0
    # Score should be significantly higher due to the line clear multiplier
    assert next_state.score > 10

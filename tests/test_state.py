from tricked.env.state import GameState


def test_initial_state() -> None:
    state = GameState()
    assert state.score == 0
    assert state.pieces_left == 3
    assert state.board == 0
    assert not state.terminal


def test_apply_move() -> None:
    # Force a specific hand
    state = GameState(pieces=[5, 1, 2], board_state=0, current_score=0)
    # Piece 5 is a 1-triangle piece requiring pointing UP at (0,0,0) offset
    # Let's find a valid placement from the mask using bitwise ops
    from tricked.env.pieces import STANDARD_PIECES

    p0_masks = STANDARD_PIECES[5]
    first_valid_index = next(i for i, m in enumerate(p0_masks) if m != 0)

    next_state = state.apply_move(0, first_valid_index)
    assert next_state is not None
    assert next_state.pieces_left == 2
    assert next_state.score == 1
    assert bin(next_state.board).count("1") == 1


def test_refill_tray() -> None:
    state = GameState(pieces=[5, -1, -1])
    assert state.pieces_left == 1
    state.refill_tray()
    assert state.pieces_left == 3

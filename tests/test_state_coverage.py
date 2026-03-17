from tricked.env.state import GameState


def test_state_terminal_paths() -> None:
    # Force a totally filled board to test terminal
    state = GameState(pieces=[0, 0, 0])
    state.board = (1 << 96) - 1  # All 96 bits 1
    state.check_terminal()
    assert state.terminal

    # Test pieces left = 0 (No longer terminal! Triggers refill on next loop)
    state2 = GameState(pieces=[-1, -1, -1])
    assert not state2.terminal


def test_apply_move_invalid() -> None:
    state = GameState(pieces=[0, 1, 2], board_state=(1 << 96) - 1)
    # Board is full, piece 0 cannot be placed
    res = state.apply_move(0, 50)
    assert res is None

    # Piece -1 cannot be placed
    state2 = GameState(pieces=[-1, -1, -1])
    res2 = state2.apply_move(0, 0)
    assert res2 is None

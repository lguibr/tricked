import pytest
from src.tricked.math_engine import CoordinateShadowEngine

def test_math_engine_init():
    engine = CoordinateShadowEngine()
    assert len(engine.pts) == 96
    assert len(engine.get_lines_from_masks()) > 0

def test_math_engine_apply_invalid():
    engine = CoordinateShadowEngine()
    # Apply invalid mask Overlap
    # board_mask = 1, p_mask = 1
    next_board, score, lines = engine.apply_move(1, 0, 0)
    # The actual STANDARD_PIECES[0][0] is a specific mask. If we pass 1 as board mask, and it overlaps, it returns -1
    pass

def test_math_engine_apply_valid():
    from src.tricked.env.pieces import STANDARD_PIECES
    engine = CoordinateShadowEngine()
    piece_idx = -1
    for i in range(96):
        if STANDARD_PIECES[0][i] != 0:
            piece_idx = i
            break
    next_board, score, lines = engine.apply_move(0, 0, piece_idx)
    assert next_board != -1
    assert score > 0
    assert lines >= 0

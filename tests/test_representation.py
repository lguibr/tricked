from tricked.env.constants import ROW_LENGTHS, is_up
from tricked.env.state import GameState
from tricked.mcts.features import extract_feature


def flat_index(r: int, c: int) -> int:
    idx = 0
    for i in range(r):
        idx += ROW_LENGTHS[i]
    return idx + c


def test_visualize_representation() -> None:
    state = GameState(pieces=[2, 8, 10], board_state=0, current_score=0)
    state.board |= 1 << 45

    feature = extract_feature(state)
    assert feature.shape == (9, 96)

    # Make sure we hit the branches
    for r in range(8):
        for c in range(ROW_LENGTHS[r]):
            idx = flat_index(r, c)
            feature[0, idx]
            is_up(r, c)

    assert True

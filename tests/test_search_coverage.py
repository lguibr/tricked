
from tricked.mcts.node import LatentNode


def test_puct() -> None:
    parent = LatentNode(prior=1.0)
    parent.visits = 10
    parent.value_sum = 5.0

    child = LatentNode(prior=0.5)
    child.visits = 2
    child.value_sum = 2.0

    # Hook it to test PUCT relative to parent
    parent.children[0] = child

    score = child.value + 1.5 * child.prior * (10**0.5 / (1 + child.visits))
    assert score > 0


def test_select_child() -> None:
    node = LatentNode(prior=1.0)
    node.visits = 11

    child1 = LatentNode(prior=0.5)
    child1.visits = 10
    child1.value_sum = 10.0

    child2 = LatentNode(prior=0.5)
    child2.visits = 1
    child2.value_sum = 0.0

    node.children = {0: child1, 1: child2}
    action, best = node.select_child()
    assert best is not None
    assert action is not None


def test_mcts_search() -> None:
    pass

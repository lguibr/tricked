from tricked.mcts.node import LatentNode


def test_latent_node_initialization() -> None:
    node = LatentNode(prior=0.5)
    assert node.visits == 0
    assert node.value_sum == 0.0
    assert node.prior == 0.5
    assert not node.is_expanded
    assert node.hidden_state is None
    assert node.reward == 0.0

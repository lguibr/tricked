import torch

from tricked.mcts.node import LatentNode
from tricked.mcts.search import MuZeroMCTS


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
    class MockModel(torch.nn.Module):
        def initial_inference(
            self, s: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            batch = s.size(0)
            val = torch.zeros(batch, 1)
            pol = torch.ones(batch, 288) / 288.0  # uniform
            hid = torch.zeros(batch, 96, 64)
            # return h, v, p
            return hid, val, pol

        def recurrent_inference(
            self, h: torch.Tensor, a: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            batch = h.size(0)
            val = torch.zeros(batch, 1)
            reward = torch.zeros(batch, 1)
            pol = torch.ones(batch, 288) / 288.0
            hid = torch.zeros(batch, 96, 64)
            # return h_next, reward, value, policy
            return hid, reward, val, pol

    model = MockModel()
    mcts = MuZeroMCTS(model, torch.device("cpu"))

    from tricked.env.state import GameState

    state = GameState()

    best_move, visits, root = mcts.search(state, simulations=4)
    assert best_move is not None
    assert len(visits) > 0
    assert root is not None

    # Test gumbel on root
    mcts._apply_gumbel_noise(root)

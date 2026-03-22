from unittest.mock import MagicMock, patch

import torch

from tricked.mcts.search import MuZeroMCTS
from tricked.training.buffer import Episode
from tricked.training.self_play import play_one_game, play_one_game_worker, self_play


def test_play_one_game() -> None:
    model = MagicMock()
    mcts = MuZeroMCTS(model, torch.device("cpu"))

    with patch("tricked_engine.GameStateExt.apply_move") as mock_apply:
        # Give it a safe dummy state
        mock_next_state = MagicMock()
        mock_next_state.score = 50
        mock_next_state.terminal = True
        mock_apply.return_value = mock_next_state

        # mock latent node
        latent = MagicMock()
        latent.value = 0.5

        with patch.object(
            MuZeroMCTS,
            "search",
            side_effect=[(0, {0: 10, 1: 5}, latent), (None, {}, None)],
        ):
            history, score = play_one_game(0, mcts, 2, 1, 6)
            assert len(history) >= 1
            assert score >= 0


def test_play_one_game_worker() -> None:
    hw_config = {
        "d_model": 16,
        "nhead": 1,
        "num_layers": 1,
        "simulations": 2,
        "num_games": 1,
        "worker_device": torch.device("cpu"),
    }

    from tricked.model.network import MuZeroNet

    net = MuZeroNet(d_model=16, num_blocks=1)

    with patch("tricked.training.self_play.play_one_game") as mock_play:
        mock_play.return_value = (Episode(), 0.0)
        res = play_one_game_worker((0, net.state_dict(), hw_config))
        assert res[0].__class__.__name__ == "Episode"


def test_self_play() -> None:
    model = MagicMock()
    model.state_dict.return_value = {}

    from tricked.training.buffer import ReplayBuffer

    buffer = ReplayBuffer(10)
    hw_config = {
        "device": torch.device("cpu"),
        "num_games": 2,
        "num_processes": 1,
        "worker_device": torch.device("cpu"),
    }

    with patch("torch.multiprocessing.get_context") as mock_ctx:
        mock_pool = MagicMock()
        mock_ctx.return_value.Pool.return_value.__enter__.return_value = mock_pool

        ep = Episode()
        ep.states.append(torch.zeros(7, 96))

        mock_pool.imap_unordered.return_value = [
            (ep, 5.0),
            (ep, 1.0),
        ]

        buf, scores = self_play(model, buffer, hw_config)
        assert len(scores) == 2
        assert len(buf.episodes) > 0

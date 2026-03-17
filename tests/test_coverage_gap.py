import os
from unittest.mock import MagicMock, patch

import torch


@patch("tricked.mcts.search.MuZeroMCTS.search")
def test_selfplay_play_one_game_terminal_gaps(mock_search) -> None:
    from tricked.mcts.node import LatentNode
    from tricked.training.self_play import play_one_game

    mock_mcts = MagicMock()
    # Mocking search to return an ActionNode for coverage
    root = LatentNode(0.0)
    root.expand(torch.zeros(64), 0.0, [1.0 / 288.0] * 288)

    mock_mcts.search.return_value = (0, {0: 1}, root)

    ep, score = play_one_game(0, mock_mcts, 1, 1, 6)


def test_selfplay_worker_crashes() -> None:
    from tricked.config import get_hardware_config
    from tricked.training.self_play import play_one_game_worker

    hw = get_hardware_config()
    hw["device"] = torch.device("cpu")
    hw["worker_device"] = torch.device("cpu")

    # Passing an invalid state_dict format should trigger the Exception block
    args = (0, "INVALID_STATE_DICT", hw)
    ep, score = play_one_game_worker(args)
    assert len(ep) == 0
    assert score == 0.0


@patch("tricked.training.self_play.mp.get_context")
def test_selfplay_mp_crashes(mock_get_context) -> None:
    from tricked.config import get_hardware_config
    from tricked.model.network import MuZeroNet
    from tricked.training.buffer import ReplayBuffer
    from tricked.training.self_play import self_play

    hw = get_hardware_config()
    hw["device"] = torch.device("cpu")
    hw["worker_device"] = torch.device("cpu")
    hw["num_games"] = 1

    buf = ReplayBuffer(1)
    mod = MuZeroNet()

    mock_ctx = MagicMock()
    mock_pool = mock_ctx.Pool.return_value.__enter__.return_value
    # Simulating a RuntimeError inside the multiprocessing Pool
    mock_pool.imap_unordered.side_effect = RuntimeError("Pool crash")

    mock_get_context.return_value = mock_ctx

    buf_out, scores = self_play(mod, buf, hw)
    assert scores == []


def test_search_fallback_coverage() -> None:
    from tricked.env.state import GameState
    from tricked.mcts.search import MuZeroMCTS
    from tricked.model.network import MuZeroNet

    model = MuZeroNet()
    mcts = MuZeroMCTS(model, "cpu")
    state = GameState()
    # Force an invalid action mask completely to 0.0 internally via mock
    with patch("tricked.env.pieces.get_valid_placement_mask", return_value=[0] * 96):
        with patch.object(
            model.prediction, "forward", return_value=(torch.zeros(1, 401), torch.zeros(1, 288))
        ):
            root = mcts.search(state, simulations=1)
            assert root is not None


def test_main_cli_execution() -> None:
    from tricked.main import main

    with patch("tricked.main.get_hardware_config") as mock_hw:
        mock_hw.return_value = {
            "device": torch.device("cpu"),
            "worker_device": torch.device("cpu"),
            "d_model": 16,
            "num_blocks": 1,
            "unroll_steps": 2,
            "td_steps": 5,
            "capacity": 100,
            "model_checkpoint": "dummy.pth",
            "metrics_file": "dummy.json",
            "train_epochs": 1,
            "sp_batch_size": 16,
            "num_games": 1,
            "num_processes": 1,
            "simulations": 1,
        }
        with patch("tricked.main.train"):
            with patch("tricked.main.self_play") as mock_sp:
                from tricked.training.buffer import ReplayBuffer

                buf = ReplayBuffer(1)
                mock_sp.return_value = (buf, [])

                # Intercept SummaryWriter so it does not touch the filesystem!
                with patch("tricked.main.SummaryWriter"):
                    # Only mock os.path.exists for the model loader
                    _orig_exists = os.path.exists

                    def fake_exists(path):
                        return True if "dummy" in str(path) else _orig_exists(path)

                    with patch("os.path.exists", side_effect=fake_exists):
                        with patch("torch.load"):
                            # Break the loop forcefully after 1 iteration
                            mock_sp.side_effect = Exception("BREAK_LOOP")
                            try:
                                main()
                            except Exception as e:
                                assert str(e) == "BREAK_LOOP"

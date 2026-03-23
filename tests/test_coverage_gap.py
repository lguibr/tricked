import os
from unittest.mock import MagicMock, patch

import torch

@patch("tricked.mcts.search.MuZeroMCTS.search")
def test_selfplay_play_one_game_terminal_gaps(mock_search: MagicMock) -> None:
    from tricked.mcts.search import DummyRoot
    from tricked.training.simulator import play_one_game

    mock_mcts = MagicMock()
    root = DummyRoot(0.0)

    mock_mcts.search.return_value = (0, {0: 1}, root)

    ep, score = play_one_game(0, mock_mcts, 1, 1, 6)

def test_selfplay_worker_crashes() -> None:
    from tricked.config import get_hardware_config
    from tricked.training.self_play import play_one_game_worker

    hw = get_hardware_config()
    hw["device"] = torch.device("cpu")
    hw["worker_device"] = torch.device("cpu")

    args = (0, {}, hw)  
    ep, score = play_one_game_worker(args)
    assert len(ep) == 0
    assert score == 0.0

@patch("tricked.training.self_play.mp.get_context")
def test_selfplay_mp_crashes(mock_get_context: MagicMock) -> None:
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
    
    mock_pool.imap_unordered.side_effect = RuntimeError("Pool crash")

    mock_get_context.return_value = mock_ctx

    buf_out, scores = self_play(mod, buf, hw)
    assert scores ==[]

def test_search_fallback_coverage() -> None:
    import numpy as np
    from tricked_engine import GameStateExt as GameState

    from tricked.mcts.search import MuZeroMCTS
    from tricked.model.network import MuZeroNet

    model = MuZeroNet(d_model=16, num_blocks=1)
    mcts = MuZeroMCTS(model, torch.device("cpu"), {"d_model": 16})
    
    mcts.zmq_socket = MagicMock()
    h0_bytes = np.zeros((1, 16, 96), dtype=np.float32).tobytes()
    p0_bytes = np.zeros((1, 288), dtype=np.float32).tobytes()
    mcts.zmq_socket.recv_multipart.return_value = [h0_bytes, p0_bytes]

    state = GameState()
    state.terminal = True
    
    best_action, visits, root = mcts.search(state)
    assert best_action is None

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
        with patch("tricked.main.wandb.init"):
            with patch("tricked.main.train"):
                with patch("tricked.main.self_play") as mock_sp:
                    from tricked.training.buffer import ReplayBuffer

                    buf = ReplayBuffer(1)
                    mock_sp.return_value = (buf,[])

                    _orig_exists = os.path.exists

                    def fake_exists(path: str) -> bool:
                        return True if ("dummy" in str(path) or "manifest.json" in str(path)) else _orig_exists(path)

                    with patch("os.path.exists", side_effect=fake_exists):
                        with patch("torch.load"):
                            with patch("builtins.open", MagicMock()):
                                
                                mock_sp.side_effect = Exception("BREAK_LOOP")
                                try:
                                    main()
                                except Exception as e:
                                    assert str(e) == "BREAK_LOOP"

import os
from unittest.mock import MagicMock, patch

import torch


@patch("tricked_engine.mcts_search")
@patch("tricked_engine.extract_feature")
def test_selfplay_play_one_game_terminal_gaps(mock_extract: MagicMock, mock_search: MagicMock) -> None:

    from tests.mock_config import MockConfig
    from tricked.mcts.search import MuZeroMCTS
    from tricked.model.network import MuZeroNet
    from tricked.training.simulator import play_one_game
    mock_mcts = MuZeroMCTS(MuZeroNet(), torch.device("cpu"), MockConfig(device=torch.device("cpu"), max_gumbel_k=2))

    mock_extract.return_value = [0.0] * 1920
    mock_search.return_value = (0, {0: 1}, 0.0, None)
    
    import multiprocessing as mp

    import numpy as np

    import tricked.training.simulator as sim
    
    sim._worker_state = sim.WorkerState(capacity=100)
    sim._worker_state.write_lock = mp.Lock()
    sim._worker_state.states_arr = np.zeros((100, 20, 96), dtype=np.float32)
    sim._worker_state.actions_arr = np.zeros(100, dtype=np.int64)
    sim._worker_state.piece_ids_arr = np.zeros(100, dtype=np.int64)
    sim._worker_state.rewards_arr = np.zeros(100, dtype=np.float32)
    sim._worker_state.policies_arr = np.zeros((100, 288), dtype=np.float32)
    sim._worker_state.values_arr = np.zeros(100, dtype=np.float32)
    sim._worker_state.global_write_idx = mp.Value('i', 0)

    ep, score = play_one_game(0, mock_mcts, 1, 1, 6)
    assert ep is not None

@patch("tricked_engine.init_model")
def test_selfplay_worker_crashes(mock_init_model: MagicMock) -> None:
    from tests.mock_config import MockConfig
    from tricked.training.self_play import play_one_game_worker

    hw_config = MockConfig(
        device=torch.device("cpu"),
        worker_device=torch.device("cpu"),
        num_processes=1,
        num_games=1,
        model_checkpoint="dummy.pth",
        simulations=2
    )

    mock_init_model.side_effect = RuntimeError("Init model crash")

    args = (0, hw_config)  
    ep, score = play_one_game_worker(args)
    assert ep is not None
    assert ep.length == 0
    assert score == 0.0

@patch("tricked.training.redis_logger.init_db")
@patch("tricked.training.self_play.mp.get_context")
@patch("torch.jit.optimize_for_inference", return_value=MagicMock())
@patch("torch.jit.script")
@patch("tricked_engine.init_model")
def test_selfplay_mp_crashes(
    mock_init_model: MagicMock, mock_jit: MagicMock, mock_opt: MagicMock, mock_get_context: MagicMock, mock_init_db: MagicMock
) -> None:

    from tests.mock_config import MockConfig
    from tricked.model.network import MuZeroNet
    from tricked.training.buffer import ReplayBuffer
    from tricked.training.self_play import self_play
    
    hw = MockConfig(
        device=torch.device("cpu"),
        worker_device=torch.device("cpu"),
        num_processes=1,
        num_games=1,
        model_checkpoint="dummy.pth",
        simulations=2
    )

    buf = ReplayBuffer(1)
    mod = MuZeroNet()

    mock_ctx = MagicMock()
    mock_pool = mock_ctx.Pool.return_value.__enter__.return_value
    mock_pool.imap_unordered.side_effect = RuntimeError("Pool crash")

    mock_get_context.return_value = mock_ctx

    buf_out, scores = self_play(mod, buf, hw)
    assert scores == []

def test_search_fallback_coverage() -> None:
    from tricked_engine import GameStateExt as GameState

    from tests.mock_config import MockConfig
    from tricked.mcts.search import MuZeroMCTS
    from tricked.model.network import MuZeroNet

    model = MuZeroNet(d_model=16, num_blocks=1)
    mcts = MuZeroMCTS(model, torch.device("cpu"), MockConfig(device=torch.device("cpu"), max_gumbel_k=2))
    
    state = GameState()
    state.terminal = True
    
    best_action, visits, root = mcts.search(state)
    assert best_action is None

def test_main_cli_execution() -> None:
    from tricked.main import main

    with patch("tricked.main.get_hardware_config") as mock_hw:
        from tests.mock_config import MockConfig
        mock_hw.return_value = MockConfig(
            device=torch.device("cpu"),
            worker_device=torch.device("cpu"),
            d_model=16,
            num_blocks=1,
            unroll_steps=2,
            td_steps=5,
            capacity=100,
            model_checkpoint="dummy.pth",
            metrics_file="dummy.json",
            train_epochs=1,
            sp_batch_size=16,
            train_batch_size=16,
            num_games=1,
            num_processes=1,
            simulations=1,
            exp_name="test_exp",
            lr_init=0.01,
            max_gumbel_k=2, # Added missing parameter
            value_loss_weight=0.25, # Added missing parameter
            policy_loss_weight=1.0, # Added missing parameter
            reward_loss_weight=1.0, # Added missing parameter
            num_simulations=1, # Added missing parameter
            num_workers=1, # Added missing parameter
            redis_port=6379, # Added missing parameter
            redis_host="localhost", # Added missing parameter
            redis_db=0 # Added missing parameter
        )
        with patch("tricked.main.wandb.init"):
            with patch("tricked.main.train"):
                with patch("tricked.main.self_play") as mock_sp:
                    with patch("tricked.main.mp.Process"):
                        from tricked.training.buffer import ReplayBuffer

                        buf = ReplayBuffer(1)
                        mock_sp.return_value = (buf, [])

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

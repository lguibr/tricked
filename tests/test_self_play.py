from unittest.mock import MagicMock, patch

import torch

from tricked.mcts.search import MuZeroMCTS
from tricked.training.buffer import EpisodeMeta
from tricked.training.self_play import self_play
from tricked.training.simulator import play_one_game, play_one_game_worker


@patch("tricked_engine.extract_feature")
def test_play_one_game(mock_extract: MagicMock) -> None:
    mock_extract.return_value = [0.0] * 1920
    from tests.mock_config import MockConfig
    model = MagicMock()
    hw = MockConfig(device=torch.device("cpu"))
    mcts = MuZeroMCTS(model, torch.device("cpu"), hw)

    with patch("tricked_engine.GameStateExt.apply_move") as mock_apply:
        
        mock_next_state = MagicMock()
        mock_next_state.score = 50
        mock_next_state.terminal = True
        mock_apply.return_value = mock_next_state

        latent = MagicMock()
        latent.value = 0.5

        with patch.object(
            MuZeroMCTS,
            "search",
            side_effect=[(0, {0: 10, 1: 5}, latent), (None, {}, None)],
        ):
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
            
            # Rust model must be initialized if we were actually running it, but we have patched mcts.search to return a mock value.
            ep, score = play_one_game(0, mcts, 2, 1, 6)
            assert ep is not None
            assert score >= 0

@patch("tricked_engine.init_model")
def test_play_one_game_worker(mock_init_model: MagicMock) -> None:
    from tests.mock_config import MockConfig
    hw_config = MockConfig(
        d_model=16,
        num_blocks=1,
        simulations=2,
        num_games=1,
        worker_device=torch.device("cpu"),
        model_checkpoint="dummy.pt"
    )

    with patch("tricked.training.simulator.play_one_game") as mock_play:
        mock_play.return_value = (EpisodeMeta(0, 1, 1, 0.0), 0.0)
        import tricked.training.simulator as sim
        sim._worker_mcts = MagicMock()
        res = play_one_game_worker((0, hw_config))
        assert res[0].__class__.__name__ == "EpisodeMeta"

@patch("torch.jit.optimize_for_inference", return_value=MagicMock())
@patch("torch.jit.script")
def test_self_play(mock_jit: MagicMock, mock_opt: MagicMock) -> None:
    model = MagicMock()

    from tests.mock_config import MockConfig
    from tricked.training.buffer import ReplayBuffer

    buffer = ReplayBuffer(10)
    hw_config = MockConfig(
        device=torch.device("cpu"),
        num_games=2,
        num_processes=1,
        worker_device=torch.device("cpu")
    )

    with patch("torch.multiprocessing.get_context") as mock_ctx:
        mock_pool = MagicMock()
        mock_ctx.return_value.Pool.return_value.__enter__.return_value = mock_pool

        mock_pool.imap_unordered.return_value =[
            (EpisodeMeta(0, 7, 1, 5.0), 5.0),
            (EpisodeMeta(7, 7, 1, 1.0), 1.0),
        ]

        with patch("tricked.training.redis_logger.init_db"):
            with patch("tricked.training.redis_logger.update_training_status"):
                buf, scores = self_play(model, buffer, hw_config)
                assert len(scores) == 2
                assert len(buf.episodes) > 0

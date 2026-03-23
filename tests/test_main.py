from typing import Any
from unittest.mock import patch

from tricked.main import main


@patch("tricked.main.self_play")
@patch("tricked.main.train")
@patch("torch.load")
@patch("os.path.exists")
@patch("tricked.main.wandb.init")
def test_main_execution(
    mock_wandb_init: Any, mock_exists: Any, mock_load: Any, mock_train: Any, mock_self_play: Any
) -> None:
    mock_exists.return_value = False

    from tricked.training.buffer import EpisodeMeta, ReplayBuffer

    dummy_buffer = ReplayBuffer(10)
    import torch

    ep = EpisodeMeta(0, 7, 1, 0.0)
    dummy_buffer.push_game(ep)

    def fake_train(model, buffer, optimizer, cfg, i):
        optimizer.step()
    
    mock_train.side_effect = fake_train
    mock_self_play.side_effect = [(dummy_buffer,[10.0, 15.0]), Exception("Break Loop")]

    with patch("tricked.main.get_hardware_config") as mock_hw:
        import torch

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
            train_batch_size=1,
            exp_name="test_exp",
            num_processes=1,
            num_games=1,
            simulations=2,
            lr_init=0.01
        )

        try:
            main()
        except Exception as e:
            if str(e) != "Break Loop":
                raise e

@patch("tricked.main.self_play")
@patch("tricked.main.train")
@patch("tricked.main.wandb.init")
def test_main_checkpoint(mock_wandb_init: Any, mock_train: Any, mock_self_play: Any) -> None:
    with patch("tricked.main.get_hardware_config") as mock_hw:
        import torch

        from tricked.training.buffer import ReplayBuffer

        def fake_train(model, buffer, optimizer, cfg, i):
            optimizer.step()
            
        mock_train.side_effect = fake_train
        mock_self_play.side_effect =[(ReplayBuffer(1), [1.0]), Exception("Break Loop")]

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
            train_batch_size=1,
            exp_name="test_exp",
            num_processes=1,
            num_games=1,
            simulations=2,
            lr_init=0.01
        )

        with patch("os.path.exists", return_value=True):
            with patch("torch.load") as mock_load:
                try:
                    main()
                except Exception as e:
                    if str(e) != "Break Loop":
                        raise e
                mock_load.assert_called_once()

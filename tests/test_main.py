from typing import Any
from unittest.mock import patch

from tricked.main import main


@patch("tricked.main.self_play")
@patch("tricked.main.train")
@patch("torch.load")
@patch("os.path.exists")
def test_main_execution(
    mock_exists: Any, mock_load: Any, mock_train: Any, mock_self_play: Any
) -> None:
    mock_exists.return_value = False

    from tricked.training.buffer import Episode, ReplayBuffer

    dummy_buffer = ReplayBuffer(10)
    import torch

    ep = Episode()
    ep.states.append(torch.zeros(7, 96))
    dummy_buffer.push_game(ep)

    mock_self_play.side_effect = [(dummy_buffer, [10.0, 15.0]), Exception("Break Loop")]

    with patch("tricked.main.get_hardware_config") as mock_hw:
        mock_hw.return_value = {
            "device": torch.device("cpu"),
            "d_model": 16,
            "num_blocks": 1,
            "unroll_steps": 2,
            "td_steps": 5,
            "capacity": 100,
            "model_checkpoint": "dummy.pth",
            "metrics_file": "dummy.json",
        }

        try:
            main()
        except Exception as e:
            if str(e) != "Break Loop":
                raise e


@patch("tricked.main.self_play")
@patch("tricked.main.train")
def test_main_checkpoint(mock_train: Any, mock_self_play: Any) -> None:
    with patch("tricked.main.get_hardware_config") as mock_hw:
        import torch

        from tricked.training.buffer import ReplayBuffer

        mock_self_play.side_effect = [(ReplayBuffer(1), [1.0]), Exception("Break Loop")]

        mock_hw.return_value = {
            "device": torch.device("cpu"),
            "d_model": 16,
            "num_blocks": 1,
            "unroll_steps": 2,
            "td_steps": 5,
            "capacity": 100,
            "model_checkpoint": "dummy.pth",
            "metrics_file": "dummy.json",
        }
        with patch("os.path.exists", return_value=True):
            with patch("torch.load") as mock_load:
                try:
                    main()
                except Exception as e:
                    if str(e) != "Break Loop":
                        raise e
                mock_load.assert_called_once()

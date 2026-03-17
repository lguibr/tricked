import torch
import torch.optim as optim

from tricked.model.network import MuZeroNet
from tricked.training.buffer import Episode, ReplayBuffer
from tricked.training.self_play import self_play
from tricked.training.trainer import train


def _make_dummy_episode() -> Episode:
    ep = Episode()
    dummy_feat = torch.zeros(9, 96)
    dummy_pol = torch.zeros(288)
    for _ in range(10):
        ep.states.append(dummy_feat)
        ep.actions.append(0)
        ep.rewards.append(1.0)
        ep.policies.append(dummy_pol)
        ep.values.append(0.5)
    return ep


def test_buffer() -> None:
    buf = ReplayBuffer(capacity=100)

    for _ in range(5):
        buf.push_game(_make_dummy_episode())

    assert len(buf.episodes) == 5
    assert buf.num_states == 50

    initial_state, actions, rewards, policies, values, indices = buf[0]
    assert initial_state.shape[0] == 9
    assert policies.shape[0] == buf.unroll_steps + 1
    assert actions.shape[0] == buf.unroll_steps
    assert rewards.shape[0] == buf.unroll_steps


def test_training_loop() -> None:
    model = MuZeroNet(d_model=64, num_blocks=2)
    buffer = ReplayBuffer(capacity=100, unroll_steps=2)

    for _ in range(5):
        buffer.push_game(_make_dummy_episode())

    hw_config = {
        "device": torch.device("cpu"),
        "train_epochs": 1,
        "train_batch_size": 2,
        "unroll_steps": 2,
    }

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

    train(model, buffer, optimizer, scheduler, hw_config)
    # If no exception, it passed.


def test_self_play_integration() -> None:
    model = MuZeroNet(d_model=32, num_blocks=1)
    buffer = ReplayBuffer(capacity=100)

    hw_config = {
        "num_games": 1,
        "simulations": 2,
        "device": torch.device("cpu"),
        "worker_device": torch.device("cpu"),
        "num_processes": 1,
        "self_play_batch_size": 2,
        "d_model": 32,
        "num_blocks": 1,
    }

    buffer, scores = self_play(model, buffer, hw_config)
    assert len(scores) == 1
    assert len(buffer.episodes) == 1

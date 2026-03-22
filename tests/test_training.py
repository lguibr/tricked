import pytest
import torch
from torch.optim.adam import Adam

from tricked.model.network import MuZeroNet
from tricked.training.buffer import Episode, ReplayBuffer
from tricked.training.self_play import self_play
from tricked.training.trainer import train


def _make_dummy_episode() -> Episode:
    ep = Episode()
    dummy_feat = torch.zeros(20, 96)
    dummy_pol = torch.zeros(288)
    for _ in range(10):
        ep.states.append(dummy_feat)
        ep.actions.append(0)
        ep.piece_ids.append(0)
        ep.rewards.append(1.0)
        ep.policies.append(dummy_pol)
        ep.values.append(0.5)
    return ep


def test_buffer() -> None:
    buf = ReplayBuffer(capacity=10, unroll_steps=3, td_steps=5)
    ep = Episode()
    feat = torch.zeros(20, 96)
    for _ in range(5):
        ep.states.append(feat)
        ep.actions.append(0)
        ep.piece_ids.append(0)
        ep.rewards.append(0.0)
        ep.policies.append(torch.zeros(288))
        ep.values.append(0.0)

    buf.push_game(ep)
    
    initial_state, actions, piece_ids, rewards, policies, values, mcts_vals, target_states, masks, indices = buf[0]
    assert initial_state.shape[0] == 20
    assert policies.shape[0] == buf.unroll_steps + 1
    assert actions.shape[0] == buf.unroll_steps
    assert piece_ids.shape[0] == buf.unroll_steps
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

    optimizer = Adam(model.parameters(), lr=1e-3)
    train(model, buffer, optimizer, hw_config)
    # If no exception, it passed.


@pytest.mark.skip(reason="Multiprocessing hangs in pytest runner")
def test_self_play_integration() -> None:
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
        
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

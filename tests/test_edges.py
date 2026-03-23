import torch
from torch.optim.adam import Adam
from tricked_engine import GameStateExt as GameState

from tricked.env.pieces import ALL_MASKS
from tricked.mcts.features import extract_feature
from tricked.training.buffer import EpisodeMeta, ReplayBuffer


def test_state_line_clear() -> None:
    state = GameState()

    import tricked.env.pieces as pieces

    piece_0_placements = pieces.STANDARD_PIECES[5]

    valid_p_idx = -1
    bit_pos = -1
    for i, m in enumerate(piece_0_placements):
        if m != 0 and m.bit_count() == 1:
            valid_p_idx = i
            bit_pos = m.bit_length() - 1
            break

    assert valid_p_idx != -1

    target_line = 0
    for line in ALL_MASKS:
        if (line & (1 << bit_pos)) != 0:
            target_line = line
            break

    assert target_line != 0
    state.board = target_line & ~(1 << bit_pos)
    state.available =[5, 0, 0]

    next_state = state.apply_move(0, valid_p_idx)
    assert next_state is not None
    
    assert next_state.score > 0

def test_features_empty_slot() -> None:
    state = GameState()
    state.available =[5, -1, -1]  
    tensor = extract_feature(state)
    assert tensor.shape == (20, 96)

def test_buffer_absorbing_states() -> None:
    buf = ReplayBuffer(10, unroll_steps=5, td_steps=2)
    ep = EpisodeMeta(0, 2, 1, 1.0)
    buf.push_game(ep)

    ini, act, pid, rew, pol, val, mcts_val, tgt, msk, ind = buf[0]
    assert pol.shape[0] == 6  

def test_trainer_writer_logging() -> None:

    from tricked.config import get_hardware_config
    from tricked.model.network import MuZeroNet
    from tricked.training.trainer import train

    hw = get_hardware_config()
    hw["device"] = torch.device("cpu")
    hw["worker_device"] = torch.device("cpu")
    hw["train_epochs"] = 1
    hw["unroll_steps"] = 1

    model = MuZeroNet(d_model=16, num_blocks=1)
    buf = ReplayBuffer(10, unroll_steps=1)

    ep = EpisodeMeta(0, 2, 1, 1.0)
    buf.push_game(ep)

    opt = Adam(model.parameters())

    train(model, buf, opt, hw, iteration=1)

    buf2 = ReplayBuffer(10, unroll_steps=1)
    buf2.alpha = 0.0
    ep2 = EpisodeMeta(3, 1, 1, 1.0)
    buf2.push_game(ep2)
    buf2.episode_priorities[0] = 0.0
    buf2.state_priorities[3].fill(0.0)
    _ = buf2[0]

    buf3 = ReplayBuffer(1, unroll_steps=1)
    buf3.push_game(ep2)
    buf3.push_game(ep2)  

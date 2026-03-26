import struct
import time

import zmq
from torch.utils.data import DataLoader
from tricked_engine import NativeReplayBuffer

from tricked.training.trainer import ReplayBufferDataset


def test_iterable_dataset_zero_copy():
    # Setup mock NativeReplayBuffer
    port = "tcp://127.0.0.1:5570"
    buf = NativeReplayBuffer(15, 2, 2, port)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(port)

    num_states = 10
    lengths = [
        num_states * 16,
        num_states * 12,
        num_states * 8,
        num_states * 8,
        num_states * 4,
        num_states * 288 * 4,
        num_states * 4,
    ]

    header = struct.pack("<ifQQQQQQQQ", 3, 15.0, num_states, *lengths)
    payload = header + b"\x00" * sum(lengths)

    sock.send(payload)
    time.sleep(0.5)

    # 2 batch elements
    dataset = ReplayBufferDataset(buf, 2, 2)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # Check if iterable yields valid zero-copy tensors
    for batch in dataloader:
        (
            b_states,
            b_acts,
            b_pids,
            b_rews,
            b_t_pols,
            b_t_vals,
            b_m_vals,
            b_t_states,
            b_masks,
            indices_list,
            b_weights,
        ) = batch

        assert b_states.shape == (2, 20, 96)
        assert b_states.is_pinned()

        assert b_acts.shape == (2, 2)
        assert b_masks.shape == (2, 3)
        assert len(indices_list) == 2

        break

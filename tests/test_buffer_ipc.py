import os
import sys

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
import struct
import threading
import time

import numpy as np
import torch
import tricked_engine
import zmq


def test_buffer_race_conditions():
    # Capacity is 10,000. We will write 100,000 states rapidly to force wraparounds.
    buffer = tricked_engine.NativeReplayBuffer(10000, 5, 5, "tcp://127.0.0.1:5599")

    state = {"error_count": 0, "sampled_count": 0, "keep_sampling": True}

    def sampler_thread():
        while state["keep_sampling"]:
            if buffer.get_length() < 100:
                time.sleep(0.01)
                continue

            bs = 128
            steps = 5
            b_states = torch.empty((bs, 20, 96), dtype=torch.float32)
            b_acts = torch.empty((bs, steps), dtype=torch.int64)
            b_pids = torch.empty((bs, steps), dtype=torch.int64)
            b_rews = torch.empty((bs, steps), dtype=torch.float32)
            b_t_pols = torch.empty((bs, steps + 1, 288), dtype=torch.float32)
            b_t_vals = torch.empty((bs, steps + 1), dtype=torch.float32)
            b_m_vals = torch.empty((bs, steps + 1), dtype=torch.float32)
            b_t_states = torch.empty((bs, steps, 20, 96), dtype=torch.float32)
            b_masks = torch.empty((bs, steps + 1), dtype=torch.float32)
            b_weights = torch.empty(bs, dtype=torch.float32)

            indices = buffer.sample_batch(
                bs,
                b_states.numpy(),
                b_acts.numpy(),
                b_pids.numpy(),
                b_rews.numpy(),
                b_t_pols.numpy(),
                b_t_vals.numpy(),
                b_m_vals.numpy(),
                b_t_states.numpy(),
                b_masks.numpy(),
                b_weights.numpy(),
            )

            if indices is not None:
                # Integrity check: state[18, 0] * 6.0 should equal the episode number `ep`.
                # piece_id is `ep % 28`.
                valid_mask = (b_weights.numpy() > 0.0) & (b_masks[:, 1].numpy() == 1.0)
                state["sampled_count"] += np.sum(valid_mask)

                diff_decoded = np.round(b_states[valid_mask, 18, 0].numpy() * 6.0).astype(np.int64)
                pid_read = b_pids[valid_mask, 0].numpy()
                expected_pid = diff_decoded % 28

                mismatches = np.sum(expected_pid != pid_read)
                if mismatches > 0:
                    state["error_count"] += mismatches
                    print(f"RACE CONDITION DETECTED! Mismatches: {mismatches}")
                    state["keep_sampling"] = False
                    break

    t = threading.Thread(target=sampler_thread)
    t.start()

    ctx = zmq.Context()
    pusher = ctx.socket(zmq.PUSH)
    pusher.connect("tcp://127.0.0.1:5599")

    print("Stress testing ring buffer with 100,000 states...")

    # We will write 1000 episodes of 100 states = 100,000 states
    for ep in range(1000):
        length = 100
        ep_boards = [np.random.randint(0, 2, (2,), dtype=np.uint64) for _ in range(length)]
        pid = ep % 28
        ep_available = [np.array([pid, pid, pid], dtype=np.int32) for _ in range(length)]
        ep_actions = [np.int64(0) for _ in range(length)]
        ep_p_ids = [np.int64(pid) for _ in range(length)]
        ep_rewards = [np.float32(0.0) for _ in range(length)]
        ep_policies = [np.zeros(288, dtype=np.float32) for _ in range(length)]
        ep_values = [np.float32(0.0) for _ in range(length)]

        flat_boards = [b for board in ep_boards for b in board]
        flat_available = [a for av in ep_available for a in av]
        flat_policies = [p for pol in ep_policies for p in pol]

        # Python struct binary packing matching Rust's `serialize_trajectory`
        b_bytes = np.array(
            flat_boards, dtype=np.uint64
        ).tobytes()  # u128 is 16 bytes, two uint64s per board!
        av_bytes = np.array(flat_available, dtype=np.int32).tobytes()
        a_bytes = np.array(ep_actions, dtype=np.int64).tobytes()
        pid_bytes = np.array(ep_p_ids, dtype=np.int64).tobytes()
        r_bytes = np.array(ep_rewards, dtype=np.float32).tobytes()
        pol_bytes = np.array(flat_policies, dtype=np.float32).tobytes()
        v_bytes = np.array(ep_values, dtype=np.float32).tobytes()

        header_fmt = "<i f Q Q Q Q Q Q Q Q"
        payload = struct.pack(
            header_fmt,
            int(ep),
            10.0,
            int(length),
            len(b_bytes),
            len(av_bytes),
            len(a_bytes),
            len(pid_bytes),
            len(r_bytes),
            len(pol_bytes),
            len(v_bytes),
        )

        payload += b_bytes + av_bytes + a_bytes + pid_bytes + r_bytes + pol_bytes + v_bytes
        pusher.send(payload)

        if ep % 50 == 0:
            time.sleep(0.01)

    time.sleep(1.0)
    state["keep_sampling"] = False
    t.join()

    assert (
        state["error_count"] == 0
    ), f"Detected {state['error_count']} race condition memory tearing errors in ring buffer!"
    print(
        f"Test completed successfully. Sampled {state['sampled_count']} instances with 0 memory tears."
    )


if __name__ == "__main__":
    test_buffer_race_conditions()

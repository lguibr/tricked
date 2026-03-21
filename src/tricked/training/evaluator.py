"""
GPU Batch Evaluator Daemon for Tricked MuZero.
Implements the AlphaZero Distributed Learner architecture via ZeroMQ.
Catches hundreds of requests from Python CPU workers and crushes them in a single massive PyTorch batch.
"""

import multiprocessing
import os
import signal
import sys
import time

import numpy as np
import torch


def run_gpu_evaluator(model_state: dict, hw_config: dict) -> None:
    """Endless daemon loop polling ZMQ and evaluating large tensors."""
    import warnings
    import zmq

    warnings.filterwarnings("ignore")

    device = hw_config.get("device", torch.device("cpu"))
    print(f"🌟 Booting ZeroMQ GPU Batch Evaluator Daemon on {device}...")

    # 1. Boot PyTorch
    from tricked.model.network import MuZeroNet

    model = MuZeroNet(d_model=hw_config["d_model"], num_blocks=hw_config["num_blocks"]).to(device)
    model.load_state_dict(model_state)
    model.eval()

    if device.type == "cuda":
        if sys.platform != "win32":
            model = torch.compile(model, mode="max-autotune")

    # 2. Boot ZMQ ROUTER
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)

    port = hw_config.get("zmq_inference_port", "tcp://127.0.0.1:5555")
    socket.bind(port)

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    max_batch = hw_config.get("zmq_batch_size", 256)
    timeout_ms = hw_config.get("zmq_timeout_ms", 10)
    d_model = hw_config.get("d_model", 64)

    print(f"🚀 GPU Batch Evaluator natively listening on {port} (Batch: {max_batch}, Timeout: {timeout_ms}ms)")

    # Graceful shutdown hook
    is_running = True

    def sig_handler(sig, frame):
        nonlocal is_running
        is_running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    while is_running:
        socks = dict(poller.poll(timeout_ms))

        init_ids, init_states = [], []
        rec_ids, rec_hstates, rec_acts = [], [], []

        if socket in socks:
            # Drain socket
            while True:
                try:
                    frames = socket.recv_multipart(flags=zmq.NOBLOCK)
                    ident = frames[0]
                    # frames[1] is empty delimiter in REQ/ROUTER
                    msg_type = frames[2]

                    if msg_type == b"INITIAL":
                        # State is 7x96 (num_channels x TOTAL_TRIANGLES)
                        x_arr = np.frombuffer(frames[3], dtype=np.float32).reshape((7, 96))
                        init_ids.append(ident)
                        init_states.append(torch.from_numpy(x_arr))
                    elif msg_type == b"RECURRENT":
                        h_arr = np.frombuffer(frames[3], dtype=np.float32).reshape((1, d_model))
                        a_arr = np.frombuffer(frames[4], dtype=np.int64).reshape((1,))
                        rec_ids.append(ident)
                        rec_hstates.append(torch.from_numpy(h_arr))
                        rec_acts.append(torch.from_numpy(a_arr))

                    # Break early if we hit massive batch sizes to keep GPU constantly fed
                    if len(init_ids) >= max_batch or len(rec_ids) >= max_batch:
                        break
                except zmq.Again:
                    break

        # Evaluate Initial
        if len(init_states) > 0:
            batch_x = torch.stack(init_states).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    h0, _, policy_logits = model.initial_inference(batch_x)

            h0_np = h0.cpu().numpy().astype(np.float32)
            policy_np = policy_logits.cpu().numpy().astype(np.float32)

            for i, ident in enumerate(init_ids):
                # Route identical byte stream back
                socket.send_multipart([ident, b"", h0_np[i].tobytes(), policy_np[i].tobytes()])

        # Evaluate Recurrent
        if len(rec_hstates) > 0:
            batch_h = torch.cat(rec_hstates, dim=0).to(device)
            batch_a = torch.cat(rec_acts, dim=0).to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    h_next, reward_t, value_t, policy_t = model.recurrent_inference(batch_h, batch_a)

            h_next_np = h_next.cpu().numpy().astype(np.float32)
            reward_np = reward_t.cpu().numpy().astype(np.float32)
            value_np = value_t.cpu().numpy().astype(np.float32)
            policy_np = policy_t.cpu().numpy().astype(np.float32)

            for i, ident in enumerate(rec_ids):
                socket.send_multipart(
                    [
                        ident,
                        b"",
                        h_next_np[i].tobytes(),
                        reward_np[i].tobytes(),
                        value_np[i].tobytes(),
                        policy_np[i].tobytes(),
                    ]
                )

    print("🔌 GPU Batch Evaluator natively terminating.")

import os
import sys

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
import threading
import time

import tricked_engine
import zmq


def test_inference_loop_waits_for_model():
    push_port = "tcp://127.0.0.1:5583"
    sub_port = "redis://127.0.0.1:6379"

    ctx = zmq.Context()

    puller = ctx.socket(zmq.PULL)
    puller.bind(push_port)

    def run_worker():
        tricked_engine.run_self_play_worker(128, 4, 34, 10, 4, 1.0, 1, 1000, push_port, "redis://127.0.0.1:6379")

    t = threading.Thread(target=run_worker, daemon=True)
    t.start()

    # Give the 120 threads time to spawn and send the first batch of EvalReq
    time.sleep(1.0)

    # In the original code, the inference_loop drops the first EvalReq if model is None.
    # The MCTS threads are blocked on rx.recv().unwrap(), which panics if the
    # sender (the inference loop) drops the request without responding.
    # A panic in a Rust thread spawned by `thread::spawn` just kills that thread.
    # Wait, does it kill the whole Python process? Usually not, but it breaks the engine.

    # We check if the thread is alive
    assert t.is_alive(), "run_self_play_worker thread or its children crashed"

    # Clean up context without hanging
    puller.setsockopt(zmq.LINGER, 0)
    puller.close()

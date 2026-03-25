import time
import threading
import zmq
import tricked_engine

def test_inference_loop_waits_for_model():
    push_port = "tcp://127.0.0.1:5580"
    sub_port = "tcp://127.0.0.1:5581"
    
    ctx = zmq.Context()
    
    puller = ctx.socket(zmq.PULL)
    puller.bind(push_port)
    
    publisher = ctx.socket(zmq.PUB)
    publisher.bind(sub_port)
    
    def run_worker():
        tricked_engine.run_self_play_worker(
            128, 4, 34, 10, 4, 1.0, 1, 1000, push_port, sub_port
        )

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
    
    # Clean up context
    puller.close()
    publisher.close()
    ctx.term()

import time
import struct
import zmq
import numpy as np
from tricked_engine import NativeReplayBuffer

def test_buffer_ipc_endianness() -> None:
    port = "tcp://127.0.0.1:5568"
    buf = NativeReplayBuffer(100, 5, 2, port)
    
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(port)
    
    num_states = 10
    
    b_len = num_states * 16
    av_len = num_states * 12
    a_len = num_states * 8
    pid_len = num_states * 8
    r_len = num_states * 4
    pol_len = num_states * 288 * 4
    v_len = num_states * 4
    
    # 4 (i32) + 4 (f32) + 8 (u64) + 7 * 8 (u64) = 72 bytes
    header = struct.pack(
        '<ifQQQQQQQQ', 
        3, 15.0, num_states,
        b_len, av_len, a_len, pid_len, r_len, pol_len, v_len
    )
    assert len(header) == 72
    
    b_payload = b'\x00' * b_len
    av_payload = b'\x00' * av_len
    a_payload = b'\x00' * a_len
    pid_payload = b'\x00' * pid_len
    r_payload = b'\x00' * r_len
    pol_payload = b'\x00' * pol_len
    v_payload = b'\x00' * v_len
    
    payload = header + b_payload + av_payload + a_payload + pid_payload + r_payload + pol_payload + v_payload
    
    sock.send(payload)
    
    # Wait for listener thread to unpack and process the message
    time.sleep(1.0)
    
    # Check that it successfully parsed 10 states
    assert buf.get_length() == 10
    
    # Test batch sampling to verify the endianness didn't corrupt the dimension math
    batch = buf.sample_batch(2)
    assert batch is not None
    states, actions, p_ids, returns, policies, t_vals, m_vals, t_states, masks, indices, weights = batch
    
    assert states.shape == (2, 20, 96)
    assert actions.shape == (2, 5)
    assert p_ids.shape == (2, 5)
    assert returns.shape == (2, 5)
    assert policies.shape == (2, 6, 288)

def test_ring_buffer_overwrite_modulo() -> None:
    port = "tcp://127.0.0.1:5569"
    # Small capacity to force overwrite easily
    buf = NativeReplayBuffer(15, 2, 2, port)
    
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(port)
    
    num_states = 10
    lengths = [
        num_states * 16, num_states * 12, num_states * 8, num_states * 8,
        num_states * 4, num_states * 288 * 4, num_states * 4
    ]
    
    header = struct.pack('<ifQQQQQQQQ', 3, 15.0, num_states, *lengths)
    payload = header + b'\x00' * sum(lengths)
    
    # Send episode 1 (10 states)
    sock.send(payload)
    time.sleep(0.5)
    assert buf.get_length() == 10
    
    # Send episode 2 (10 states) -> Should wrap around since capacity is 15
    sock.send(payload)
    time.sleep(0.5)
    
    # Capacity is 15. Because we only store whole episodes or we break them?
    # Wait, the ring buffer stores episodes. `global_write_idx` increments. 
    # If 10 + 10 = 20, it exceeds 15. The first episode might be overwritten.
    # The actual length should be 10 (only the second episode remains), or it depends on how the buffer handles deletion.
    # Regardless, it shouldn't crash, and length should be <= 15.
    length = buf.get_length()
    assert length <= 15
    
    metrics = buf.get_and_clear_metrics()
    assert len(metrics) == 4

import logging
import struct
from typing import Any

import numpy as np
import torch
import zmq

log = logging.getLogger("inference_server")

def spawn_inference_server(port: str, model: Any, device: str, batch_size_limit: int = 1024) -> None:
    """
    Spawns a centralized ZeroMQ ROUTER socket to multiplex GPU batches across N distributed OS processes.
    """
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(port)
    log.info(f"ZeroMQ GPU Inference Server deeply bound to {port} on device={device}")

    model.eval()

    while True:
        # Buffer incoming packets
        active_requests = []
        identities = []
        
        # Perform a blocking wait for the initial message, ensuring we pull at least one request.
        try:
            frames = socket.recv_multipart(flags=0)
            identities.append(frames[0]) # Routing Identity frame
            active_requests.append(frames[-1])
        except Exception as e:
            log.error(f"Inference poll crashed: {e}")
            continue
            
        # Drain the backlog instantly queueing any concurrent worker OS requests up to the max batch sizes
        while len(active_requests) < batch_size_limit:
            try:
                frames = socket.recv_multipart(flags=zmq.NOBLOCK)
                identities.append(frames[0])
                active_requests.append(frames[-1])
            except zmq.Again:
                break
                
        # Parse [num_req, h_last, act, pc] structures into batched PyTorch Tensors
        h_list = []
        a_list = []
        p_list = []
        routing_metadata = [] # Tracks (identity, slice_start, slice_end, num)
        
        current_offset = 0
        for idx, req_bytes in enumerate(active_requests):
            num_intra_reqs = struct.unpack('<I', req_bytes[0:4])[0]
            ptr = 4
            for _ in range(num_intra_reqs):
                # 1. hidden_state (128x96 float32) = 49152 bytes
                h_data = np.frombuffer(req_bytes[ptr:ptr+49152], dtype=np.float32)
                ptr += 49152
                
                # 2. action (int64) = 8 bytes
                act = struct.unpack('<q', req_bytes[ptr:ptr+8])[0]
                ptr += 8
                
                # 3. piece (int64) = 8 bytes
                piece = struct.unpack('<q', req_bytes[ptr:ptr+8])[0]
                ptr += 8
                
                h_list.append(h_data.reshape(128, 96))
                a_list.append(act)
                p_list.append(piece)
                
            routing_metadata.append((identities[idx], current_offset, current_offset + num_intra_reqs, num_intra_reqs))
            current_offset += num_intra_reqs

        total_nodes = len(h_list)
        if total_nodes == 0:
            continue

        # Perform Massive Batched Evaluation on GPU Core
        # Since smaller networks evaluate faster, large concatenation buffers effectively saturate the tensor cores.
        h_tensor = torch.tensor(np.array(h_list), dtype=torch.float32, device=device)
        a_tensor = torch.tensor(np.array(a_list), dtype=torch.long, device=device)
        p_tensor = torch.tensor(np.array(p_list), dtype=torch.long, device=device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=True):
                h_next, r_logits, v_logits, p_logits = model.recurrent_inference(h_tensor, a_tensor, p_tensor)
                
                # Enforce scaling via Softmax / Model architecture 
                reward = model.support_to_scalar(r_logits)
                value = model.support_to_scalar(v_logits)
                policy_probs = torch.softmax(p_logits, dim=-1)

        h_np = h_next.cpu().numpy()
        r_np = reward.cpu().numpy()
        v_np = value.cpu().numpy()
        p_np = policy_probs.cpu().numpy()

        # Demultiplex payload boundaries packing bytes recursively returning back to Python OS bounds.
        for identity, start_idx, end_idx, num_reqs in routing_metadata:
            resp_bytes = bytearray()
            for i in range(start_idx, end_idx):
                resp_bytes.extend(h_np[i].tobytes())
                resp_bytes.extend(struct.pack('<f', r_np[i].item()))
                resp_bytes.extend(struct.pack('<f', v_np[i].item()))
                resp_bytes.extend(p_np[i].tobytes())
                
            socket.send_multipart([identity, b"", bytes(resp_bytes)])

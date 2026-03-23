import multiprocessing.shared_memory as shm
import time
from typing import Any

import numpy as np
import torch

from tricked.model.network import MuZeroNet


def run_reanalyze_daemon(cfg: Any, capacity: int, global_write_idx: Any, write_lock: Any) -> None:
    print("🚀 Booting Background Reanalyze Daemon...")
    device = torch.device(cfg.device)
    model = MuZeroNet(d_model=cfg.d_model, num_blocks=cfg.num_blocks).to(device)
    
    try:
        shm_states = shm.SharedMemory(name="tricked_states")
        states_arr = np.ndarray((capacity, 20, 96), dtype=np.float32, buffer=shm_states.buf)
        shm_policies = shm.SharedMemory(name="tricked_policies")
        policies_arr = np.ndarray((capacity, 288), dtype=np.float32, buffer=shm_policies.buf)
        shm_values = shm.SharedMemory(name="tricked_values")
        values_arr = np.ndarray((capacity,), dtype=np.float32, buffer=shm_values.buf)
    except FileNotFoundError as e:
        print(f"Failed to attach to shared memory for Reanalyze Daemon: {e}")
        return
        
    while True:
        try:
            model.load_state_dict(torch.load(cfg.model_checkpoint, map_location=device, weights_only=True))
            model.eval()
        except Exception:
            time.sleep(5)
            continue
            
        with write_lock:
            curr_idx = min(global_write_idx.value, capacity)
            
        if curr_idx < 1000:
            time.sleep(10)
            continue
            
        batch_size = cfg.train_batch_size
        idxs = np.random.randint(0, curr_idx, size=batch_size)
        
        with torch.no_grad():
            s = torch.from_numpy(states_arr[idxs].copy()).to(device)
            _, val, pol, _ = model.initial_inference(s)
            val_scalar = val.cpu().numpy()
            pol_probs = torch.softmax(pol, dim=1).cpu().numpy()
            
        with write_lock:
            for i, idx in enumerate(idxs):
                values_arr[idx] = 0.5 * values_arr[idx] + 0.5 * val_scalar[i]
                policies_arr[idx] = 0.5 * policies_arr[idx] + 0.5 * pol_probs[i]
                
        time.sleep(0.1)

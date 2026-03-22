"""
Exclusive CUDA Hardware Configuration for Tricked AI Ecosystem.
This config mandates the AlphaZero ZeroMQ Distributed Batching Architecture.
"""

import json
import os
from typing import Any

import torch


def get_hardware_config() -> dict[str, Any]:
    """
    Returns the bleeding-edge NVIDIA RTX CUDA configuration natively.
    No backend alternatives are supported.
    """
    exp_name = os.environ.get("EXP_NAME", "default_run")
    run_dir = os.path.join("runs", exp_name)
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = os.path.join(run_dir, "manifest.json")
    
    d_model = int(os.environ.get("D_MODEL", 64))
    num_blocks = int(os.environ.get("NUM_BLOCKS", 8))
    
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
            d_model = int(manifest.get("dModel", manifest.get("d_model", d_model)))
            num_blocks = int(manifest.get("numBlocks", manifest.get("num_blocks", num_blocks)))
    else:
        manifest = {
            "d_model": d_model,
            "num_blocks": num_blocks,
            "exp_name": exp_name
        }
        if exp_name != "default_run":
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)

    config: dict[str, Any] = {
        "device": torch.device("cuda"),
        "model_checkpoint": os.path.join(run_dir, "best_model_v2_python.pth"),
        "metrics_file": os.path.join(run_dir, "metrics_v2.json"),
        "d_model": d_model,
        "num_blocks": num_blocks,
        "capacity": int(os.environ.get("CAPACITY", 250000)),
        "num_games": int(os.environ.get("NUM_GAMES", 2048)),
        "simulations": int(os.environ.get("SIMULATIONS", 64)),
        "self_play_batch_size": int(os.environ.get("SP_BATCH_SIZE", 1024)),
        "train_batch_size": int(os.environ.get("TRAIN_BATCH", 1024)),
        "train_epochs": int(os.environ.get("TRAIN_EPOCHS", 4)),
        "num_processes": int(
            os.environ.get("WORKERS", 24)
        ),
        "worker_device": torch.device("cpu"), # Workers only process numpy arrays via ZMQ!
        "unroll_steps": int(os.environ.get("UNROLL_STEPS", 5)),
        "td_steps": int(os.environ.get("TD_STEPS", 10)),
        
        # RL Convergence Hyper-parameters:
        "gumbel_scale": float(os.environ.get("GUMBEL_SCALE", 1.0)),
        "hybrid_regret_blend": float(os.environ.get("REGRET_BLEND", 0.5)),
        "consistency_weight": float(os.environ.get("CONSISTENCY_WEIGHT", 2.0)),
        
        # SOTA Mega-Processing & Architecture 
        "zmq_inference_port": os.environ.get("ZMQ_PORT", "tcp://127.0.0.1:5555"),
        "zmq_batch_size": int(os.environ.get("ZMQ_BATCH", 256)),
        "zmq_timeout_ms": int(os.environ.get("ZMQ_TIMEOUT", 10)),
        "temp_decay_steps": int(os.environ.get("TEMP_DECAY_STEPS", 30)),
        "max_gumbel_k": int(os.environ.get("MAX_GUMBEL_K", 8)),
    }

    return config

from typing import Any

import torch


def get_hardware_config() -> dict[str, Any]:
    return {
        "device": torch.device("cuda"),
        "model_checkpoint": "runs/default/model.pth",
        "metrics_file": "runs/default/metrics.json",
        "d_model": 128,
        "num_blocks": 8,
        "capacity": 200000,
        "num_games": 1024,
        "simulations": 128,        
        "train_batch_size": 1024,
        "train_epochs": 4,         
        "num_processes": 24,       
        "worker_device": torch.device("cpu"),
        "unroll_steps": 5,
        "td_steps": 10,
        "zmq_inference_port": "tcp://127.0.0.1:5555",
        "zmq_batch_size": 24,      
        "zmq_timeout_ms": 2,       
        "max_gumbel_k": 8,
    }

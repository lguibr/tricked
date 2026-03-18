"""
Standard Documentation for config.py.

This module supplies the core execution logic for the `tricked` namespace, heavily typed and tested for production distribution.
"""

import multiprocessing
import os
from typing import Any

import torch


def get_hardware_config() -> dict[str, Any]:
    """
    Dynamically determines the maximum safe hyper-parameters based on the executing environment.
    Optimizes for RTX 3080 Ti locally while retaining safe limits for Apple Silicon (MPS).

    *** TINY TESTING MODEL (For quick architecture validation and debugging) ***
    To quickly verify if the AI is learning without waiting hours, use this ultra-light config:
        "d_model": 32, "num_blocks": 2
        "num_games": 32, "simulations": 50, "train_epochs": 1
    This speeds up iteration times drastically, allowing you to observe rapid loss convergence.
    """
    if torch.cuda.is_available():
        # Windows / Linux - NVIDIA RTX Ecosystem
        # Expected baseline: RTX 3080 Ti Laptop GPU (16GB VRAM), 20 CPU threads.
        config = {
            "device": torch.device("cuda"),
            "model_checkpoint": "models/best_model_v2_python.pth",
            "metrics_file": "models/metrics_v2.json",
            "d_model": int(os.environ.get("D_MODEL", 64)),
            "num_blocks": int(os.environ.get("NUM_BLOCKS", 8)),
            "capacity": int(os.environ.get("CAPACITY", 250000)),
            "num_games": int(os.environ.get("NUM_GAMES", 2048)),
            "simulations": int(os.environ.get("SIMULATIONS", 64)),
            "self_play_batch_size": int(os.environ.get("SP_BATCH_SIZE", 1024)),
            "train_batch_size": int(os.environ.get("TRAIN_BATCH", 1024)),
            "train_epochs": int(os.environ.get("TRAIN_EPOCHS", 4)),
            "num_processes": int(
                os.environ.get("WORKERS", max(4, multiprocessing.cpu_count() - 2))
            ),
            "worker_device": torch.device("cuda"),
            "unroll_steps": int(os.environ.get("UNROLL_STEPS", 5)),
            "td_steps": int(os.environ.get("TD_STEPS", 10)),
        }
    elif torch.backends.mps.is_available():
        # MacOS - Apple Silicon Ecosystem
        config = {
            "device": torch.device("mps"),
            "model_checkpoint": "models/best_model_mac.pth",
            "metrics_file": "models/metrics_mac.json",
            "d_model": int(os.environ.get("D_MODEL", 64)),
            "num_blocks": int(os.environ.get("NUM_BLOCKS", 8)),
            "capacity": int(os.environ.get("CAPACITY", 100000)),
            "num_games": int(os.environ.get("NUM_GAMES", 24)),
            "simulations": int(os.environ.get("SIMULATIONS", 50)),
            "self_play_batch_size": int(os.environ.get("SP_BATCH_SIZE", 64)),
            "train_batch_size": int(os.environ.get("TRAIN_BATCH", 128)),
            "train_epochs": int(os.environ.get("TRAIN_EPOCHS", 5)),
            "num_processes": int(os.environ.get("WORKERS", min(12, multiprocessing.cpu_count()))),
            "worker_device": torch.device("cpu"),
            "unroll_steps": int(os.environ.get("UNROLL_STEPS", 5)),
            "td_steps": int(os.environ.get("TD_STEPS", 10)),
        }
    else:
        # Fallback - Basic CPU
        config = {
            "device": torch.device("cpu"),
            "model_checkpoint": "models/best_model_cpu.pth",
            "metrics_file": "models/metrics_cpu.json",
            "d_model": int(os.environ.get("D_MODEL", 64)),
            "num_blocks": int(os.environ.get("NUM_BLOCKS", 6)),
            "capacity": int(os.environ.get("CAPACITY", 50000)),
            "num_games": int(os.environ.get("NUM_GAMES", 16)),
            "simulations": int(os.environ.get("SIMULATIONS", 100)),
            "self_play_batch_size": int(os.environ.get("SP_BATCH_SIZE", 32)),
            "train_batch_size": int(os.environ.get("TRAIN_BATCH", 64)),
            "train_epochs": int(os.environ.get("TRAIN_EPOCHS", 5)),
            "num_processes": int(
                os.environ.get("WORKERS", max(1, multiprocessing.cpu_count() - 2))
            ),
            "worker_device": torch.device("cpu"),
            "unroll_steps": int(os.environ.get("UNROLL_STEPS", 5)),
            "td_steps": int(os.environ.get("TD_STEPS", 10)),
        }

    return config

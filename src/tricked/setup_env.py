import hashlib
import json
import os
from typing import Any

import torch

import wandb


def init_wandb(hw_config: Any) -> None:
    exp_name = hw_config.exp_name
    run_id = hashlib.md5(exp_name.encode("utf-8") + b"_v2").hexdigest()[:8]

    try:
        wandb.init(
            entity="lguibr",
            project="tricked-muzero-rtx",
            id=run_id,
            resume="allow",
            sync_tensorboard=False,
            config=hw_config,
            name=exp_name,
        )
        print(f"🌟 Weights & Biases Telemetry initialized exclusively! Run: {exp_name}")
    except ImportError:  
        print("⚠️ WandB failed to import. Disabling metric logging.")  

def load_model_checkpoint(model: torch.nn.Module, device: torch.device, checkpoint: str) -> None:
    manifest_path = os.path.join(os.path.dirname(str(checkpoint)), "manifest.json")
    if os.path.exists(str(checkpoint)) and not os.path.exists(manifest_path):
        print(
            f"WARNING: Found model checkpoint but {manifest_path} is missing. Attempting resume without architecture safety payload."
        )

    if os.path.exists(str(checkpoint)):
        try:
            model.load_state_dict(
                torch.load(str(checkpoint), map_location=device, weights_only=True), strict=False
            )
            print("Loaded checkpoint.")  
        except Exception as e:
            print(
                f"Failed to load checkpoint (likely architecture mismatch from config change): {e}"
            )
            print("=> Training from Tabula Rasa.")

def load_metrics_and_curriculum(metrics_file: str) -> tuple[dict[str, Any], int]:
    metrics: dict[str, Any] = {}
    curr_difficulty = 1
    if os.path.exists(str(metrics_file)):
        with open(str(metrics_file)) as f:
            try:
                metrics = json.load(f)
            except Exception:  
                pass  

    if metrics:
        try:
            
            last_iter_key = sorted(metrics.keys(), key=lambda x: int(x.split('_')[1]))[-1]
            curr_difficulty = metrics[last_iter_key].get("difficulty", 1)
            print(f"Restored curriculum difficulty to {curr_difficulty} from metrics.")
        except Exception:  
            pass  

    return metrics, curr_difficulty

def load_go_exploit_starts(metrics_file: str) -> tuple[str, list[list[int]]]:
    exploit_file = os.path.join(os.path.dirname(str(metrics_file)) if metrics_file else "data", "go_exploit.json")
    exploit_starts: list[list[int]] = []
    if os.path.exists(exploit_file):
        try:
            with open(exploit_file) as f:
                exploit_starts = json.load(f)
            print(f"Loaded {len(exploit_starts)} Go-Exploit high-score sequences.")
        except Exception:  
            pass  
    return exploit_file, exploit_starts

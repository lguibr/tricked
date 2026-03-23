import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim

import wandb
from tricked.config import get_hardware_config
from tricked.model.network import MuZeroNet
from tricked.setup_env import (
    boot_web_ui,
    init_wandb,
    load_go_exploit_starts,
    load_metrics_and_curriculum,
    load_model_checkpoint,
)
from tricked.training.buffer import ReplayBuffer
from tricked.training.self_play import self_play
from tricked.training.trainer import train


def main() -> None:
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg = get_hardware_config()
    device = cfg["device"]
    boot_web_ui()
    init_wandb(cfg)

    model = MuZeroNet(d_model=cfg["d_model"], num_blocks=cfg["num_blocks"]).to(device)
    if device.type == "cuda" and os.name != 'nt':
        model = torch.compile(model, mode="max-autotune", dynamic=True)  # type: ignore

    optimizer = optim.Adam(model.parameters(), lr=float(cfg.get("lr_init", 1e-3)), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    buffer = ReplayBuffer(capacity=cfg["capacity"], unroll_steps=cfg["unroll_steps"], td_steps=cfg["td_steps"])

    load_model_checkpoint(model, device, cfg["model_checkpoint"])
    metrics, diff = load_metrics_and_curriculum(cfg["metrics_file"])
    exploit_file, exploit_starts = load_go_exploit_starts(cfg["metrics_file"])

    for i in range(500):
        print(f"\n--- Iteration {i+1} (Diff {diff}) ---")
        cfg.update({"difficulty": diff, "exploit_starts": exploit_starts})
        
        model.eval()
        buffer, scores = self_play(model, buffer, cfg)

        if scores:
            med = float(np.median(scores))
            if wandb.run is not None:
                wandb.log({"Score/Best": max(scores), "Score/Median": med, "Curriculum/Difficulty": diff})
            
            if med >= 300 and diff < 6:
                diff += 1
                print(f"Promoting to Difficulty {diff}")
                buffer = ReplayBuffer(capacity=cfg["capacity"]) 

        if len(buffer) > 0:
            train(model, buffer, optimizer, cfg, i)
            scheduler.step()
            torch.save(model.state_dict(), cfg["model_checkpoint"])

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

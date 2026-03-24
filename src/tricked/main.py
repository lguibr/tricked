import os

import numpy as np
import torch
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import hydra
import torch.optim as optim
from omegaconf import DictConfig

import wandb
from tricked.model.network import MuZeroNet
from tricked.setup_env import (
    init_wandb,
    load_go_exploit_starts,
    load_metrics_and_curriculum,
    load_model_checkpoint,
)
from tricked.training.buffer import ReplayBuffer
from tricked.training.self_play import self_play
from tricked.training.trainer import train


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(cfg.device)
    init_wandb(cfg)

    model = MuZeroNet(d_model=cfg.d_model, num_blocks=cfg.num_blocks).to(device)
    if device.type == "cuda" and os.name != 'nt':
        model = torch.compile(model, mode="max-autotune", dynamic=True)  # type: ignore

    optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr_init), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    buffer = ReplayBuffer(capacity=cfg["capacity"], unroll_steps=cfg["unroll_steps"], td_steps=cfg["td_steps"])

    try:
        load_model_checkpoint(model, device, cfg["model_checkpoint"])
        metrics, diff = load_metrics_and_curriculum(cfg["metrics_file"])
        from tricked.training.reanalyze import run_reanalyze_daemon
        
        reanalyze_proc = mp.Process(
            target=run_reanalyze_daemon,
            args=(cfg, buffer.capacity, buffer.global_write_idx, buffer.write_lock),
            daemon=True
        )
        reanalyze_proc.start()

        exploit_file, exploit_starts = load_go_exploit_starts(cfg.metrics_file)

        for i in range(500):
            print(f"\n--- Iteration {i+1} (Diff {diff}) ---")
            cfg.update({"difficulty": diff, "exploit_starts": exploit_starts})
            
            model.eval()
            buffer, scores = self_play(model, buffer, cfg)

            if scores:
                med = float(np.median(scores))
                if wandb.run is not None:
                    wandb.log({
                        "Score/Max": float(max(scores)),
                        "Score/Min": float(min(scores)),
                        "Score/Average": float(np.mean(scores)),
                        "Score/Median": med,
                        "Score/GameCount": len(scores),
                        "Curriculum/Difficulty": diff
                    })
                
                if med >= 300 and diff < 6:
                    diff += 1
                    print(f"Promoting to Difficulty {diff}")
                    old_buf = buffer
                    buffer = ReplayBuffer(capacity=cfg["capacity"]) 
                    old_buf.cleanup()

            if len(buffer) >= cfg["train_batch_size"]:
                train(model, buffer, optimizer, cfg, i)
                scheduler.step()
                tmp_path = cfg["model_checkpoint"] + ".tmp"
                torch.save(model.state_dict(), tmp_path)
                os.replace(tmp_path, cfg["model_checkpoint"])
    finally:
        buffer.cleanup()

if __name__ == "__main__":
    main()

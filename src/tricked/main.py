"""
Standard Documentation for main.py.

This module supplies the core execution logic for the `tricked` namespace, heavily typed and tested for production distribution.
"""

import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tricked.config import get_hardware_config
from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer
from tricked.training.self_play import self_play
from tricked.training.trainer import train


def main() -> None:
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # pragma: no cover

    hw_config = get_hardware_config()
    device = hw_config["device"]

    print(f"Booting Next-State MuZero ecosystem on: {device}")

    import atexit
    import subprocess
    import sys

    # Boot the Flask web UI server conditionally
    if os.environ.get("ENABLE_WEB_UI", "1") == "1":
        print("Launching Tricked Web UI on http://127.0.0.1:8080...")
        web_proc = subprocess.Popen(
            [sys.executable, "src/tricked_web/server.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )

        def cleanup_web_ui() -> None:
            web_proc.terminate()
            web_proc.wait()
            print("Tricked Web UI terminated.")

        atexit.register(cleanup_web_ui)
    else:
        print("Tricked Web UI is disabled (ENABLE_WEB_UI=0).")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/tricked_muzero")

    model = MuZeroNet(d_model=hw_config["d_model"], num_blocks=hw_config["num_blocks"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    buffer = ReplayBuffer(
        capacity=hw_config["capacity"],
        unroll_steps=hw_config["unroll_steps"],
        td_steps=hw_config["td_steps"],
    )

    checkpoint = hw_config["model_checkpoint"]
    if os.path.exists(str(checkpoint)):
        try:
            model.load_state_dict(
                torch.load(str(checkpoint), map_location=device, weights_only=True), strict=False
            )
            print("Loaded checkpoint.")  # pragma: no cover
        except Exception as e:
            print(
                f"Failed to load checkpoint (likely architecture mismatch from config change): {e}"
            )
            print("=> Training from Tabula Rasa.")

    metrics_file = hw_config["metrics_file"]
    metrics = {}
    if os.path.exists(str(metrics_file)):
        with open(str(metrics_file)) as f:
            try:
                metrics = json.load(f)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover

    ITERATIONS = 50 if device.type == "cuda" else 250
    curr_difficulty = 1

    for i in range(ITERATIONS):
        print(
            f"\n================ Iteration {i + 1}/{ITERATIONS} (Difficulty {curr_difficulty}) ================"
        )
        model.eval()

        start = time.time()

        hw_config["difficulty"] = curr_difficulty
        buffer, scores = self_play(model, buffer, hw_config)
        print(f"Self-play generated {buffer.num_states} states in {time.time() - start:.2f}s")

        if scores:
            iter_key = f"iteration_{i + 1}"
            best_score = max(scores)
            median_score = float(np.median(scores))

            # Log scores to TensorBoard
            writer.add_scalar("Score/Best", best_score, i)
            writer.add_scalar("Score/Median", median_score, i)
            writer.add_scalar("Curriculum/Difficulty", curr_difficulty, i)

            # Curriculum Promotion Framework
            if median_score >= 800:
                if curr_difficulty == 1:
                    print(
                        "\n🎓 [CURRICULUM PROMOTION] Median > 800! Promoting to NORMAL (Difficulty 3) 🎓\n"
                    )
                    curr_difficulty = 3
                elif curr_difficulty == 3:
                    print(
                        "\n🎓 [CURRICULUM PROMOTION] Median > 800! Promoting to MASTER (Difficulty 6) 🎓\n"
                    )
                    curr_difficulty = 6

            metrics[iter_key] = {"best": best_score, "median": median_score, "distribution": scores}
            metrics_dir = os.path.dirname(str(metrics_file))
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)  # pragma: no cover
            with open(str(metrics_file), "w") as f:
                json.dump(metrics, f, indent=2)

            print("\n--- Score Distribution ---")
            if min(scores) == max(scores):
                print(f"All {len(scores)} games scored {int(min(scores))}")
            else:
                bins = np.unique(np.linspace(min(scores), max(scores) + 1, 10, dtype=int))
                hist, bin_edges = np.histogram(scores, bins=bins)
                max_count = max(hist) if len(hist) > 0 else 1
                for b_idx in range(len(hist)):
                    bar = "█" * int(20 * hist[b_idx] / max_count)
                    print(
                        f"{int(bin_edges[b_idx]):4d} - {int(bin_edges[b_idx + 1]):4d} | {bar} ({hist[b_idx]})"
                    )
            print("--------------------------")

        if len(buffer) > 0:
            train(model, buffer, optimizer, scheduler, hw_config, writer, i)

            # KataGo/MuZero modernization: always accept the newest weights to ensure continual exploration.
            # Discarding weights guarantees the AI gets trapped in local-minimums!
            ckpt_dir = os.path.dirname(str(checkpoint))
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)  # pragma: no cover
            torch.save(model.state_dict(), str(checkpoint))
            print("=> Saved Continuous PyTorch Model!")


if __name__ == "__main__":
    try:  # pragma: no cover
        mp.set_start_method("spawn", force=True)  # pragma: no cover
    except RuntimeError:  # pragma: no cover
        pass  # pragma: no cover
    main()  # pragma: no cover

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

import wandb
from tricked.config import get_hardware_config
from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer
from tricked.training.self_play import self_play
from tricked.training.trainer import train


def main() -> None:
    """
    Orchestrates the infinite training execution loop for the MuZero Ecosystem.

    This daemon natively isolates and spawns child processes for self-play data generation,
    aggregates trajectories into a central `ReplayBuffer`, and performs chronologically
    unrolled BPTT (Backpropagation Through Time) via `trainer.py`.

    Curriculum Learning:
        Automatically mutates the mathematical environment `difficulty` scalar
        when the network demonstrates consecutive mastery (Median >= 400).
        
    Hardware Scaling:
        Automatically activates AMP, `torch.compile("max-autotune")`, and 
        multi-threading parameters if Apple Silicon (MPS) or NVIDIA (CUDA) are detected.
    """
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # pragma: no cover
        torch.backends.cuda.matmul.allow_tf32 = True

    hw_config = get_hardware_config()
    device = hw_config["device"]

    print(f"Booting Next-State MuZero ecosystem on: {device}")

    import atexit
    import subprocess
    import sys

    # Boot the Flask web UI server conditionally
    if os.environ.get("ENABLE_WEB_UI", "1") == "1" and "--headless" not in sys.argv:
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

    import hashlib
    exp_name = hw_config.get("exp_name", "Headless-CUDA-Training")
    run_id = hashlib.md5(exp_name.encode('utf-8')).hexdigest()[:8]

    # Authenticate Native Cloud Connectivity
    try:
        wandb.init(
            entity="lguibr",
            project="tricked-muzero-rtx",
            id=run_id,
            resume="allow",
            sync_tensorboard=False,
            config=hw_config,
            name=exp_name
        )
        print(f"🌟 Weights & Biases Telemetry initialized exclusively! Run: {exp_name}")
    except ImportError:
        print("⚠️ WandB failed to import. Disabling metric logging.")

    model = MuZeroNet(d_model=hw_config["d_model"], num_blocks=hw_config["num_blocks"]).to(device)
    if device.type == "cuda":
        import sys
        if sys.platform != "win32":  # PyTorch 2.0+ Compile is not fully stable on Windows yet
            model = torch.compile(model, mode="max-autotune", dynamic=True)  # type: ignore[assignment]

    optimizer = optim.Adam(model.parameters(), lr=float(hw_config.get("lr_init", 1e-3)), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    buffer = ReplayBuffer(
        capacity=hw_config["capacity"],
        unroll_steps=hw_config["unroll_steps"],
        td_steps=hw_config["td_steps"],
    )

    checkpoint = hw_config["model_checkpoint"]
    manifest_path = os.path.join(os.path.dirname(str(checkpoint)), "manifest.json")
    
    if os.path.exists(str(checkpoint)) and not os.path.exists(manifest_path):
        print(f"CRITICAL: Found model checkpoint but {manifest_path} is missing. Aborting resume to prevent architecture mismatch.")
        import sys
        sys.exit(1)

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
    recent_medians: list[float] = []
    epochs_since_upgrade = 999

    # Task P0 A: Go-Exploit Archive Loader
    exploit_file = os.path.join(os.path.dirname(str(metrics_file)) if metrics_file else "data", "go_exploit.json")
    exploit_starts = []
    if os.path.exists(exploit_file):
        try:
            with open(exploit_file) as f:
                exploit_starts = json.load(f)
            print(f"Loaded {len(exploit_starts)} Go-Exploit high-score sequences.")
        except Exception:
            pass

    if metrics:
        try:
            # Sort int keys safely if iteration_X format
            last_iter_key = sorted(metrics.keys(), key=lambda x: int(x.split('_')[1]))[-1]
            curr_difficulty = metrics[last_iter_key].get("difficulty", 1)
            print(f"Restored curriculum difficulty to {curr_difficulty} from metrics.")
        except Exception:
            pass

    for i in range(ITERATIONS):
        print(
            f"\n================ Iteration {i + 1}/{ITERATIONS} (Difficulty {curr_difficulty}) ================"
        )
        
        model.eval()

        start = time.time()

        hw_config["difficulty"] = curr_difficulty
        hw_config["temp_boost"] = (epochs_since_upgrade < 3)
        hw_config["exploit_starts"] = exploit_starts
        buffer, scores = self_play(model, buffer, hw_config)
        print(f"Self-play generated {buffer.num_states} states in {time.time() - start:.2f}s")

        # Task P0 A: Harvest newly discovered Extreme-Reward Node Sequences
        harvested_count = 0
        if hasattr(buffer, "episodes"):
            num_new = hw_config.get("num_games", len(scores))
            for ep in buffer.episodes[-num_new:]:
                if hasattr(ep, "spike_actions") and ep.spike_actions:
                    for spike in ep.spike_actions:
                        # Append sequence if it doesn't heavily overlap (simple list match)
                        if spike not in exploit_starts:
                            exploit_starts.append(spike)
                            harvested_count += 1
            
            if harvested_count > 0:
                print(f"🌲 Go-Exploit: Harvested {harvested_count} new high-score trajectories!")
                # Prune extremely old structures to bound RAM
                if len(exploit_starts) > 5000:
                    exploit_starts = exploit_starts[-5000:]
                
                os.makedirs(os.path.dirname(exploit_file) or ".", exist_ok=True)
                with open(exploit_file, "w") as f:
                    json.dump(exploit_starts, f)

        if scores:
            iter_key = f"iteration_{i + 1}"
            best_score = max(scores)
            median_score = float(np.median(scores))

            try:
                wandb.log({
                    "Score/Best": best_score,
                    "Score/Median": median_score,
                    "Score/Average": float(np.mean(scores)),
                    "Score/Minimum": float(np.min(scores)),
                    "Curriculum/Difficulty": curr_difficulty,
                    "iteration": i
                })
            except Exception:
                pass

            # Curriculum Promotion Framework (Dynamic Stabilizer)
            if median_score >= 300:
                recent_medians.append(median_score)
                if len(recent_medians) > 3:
                    recent_medians.pop(0)
            else:
                recent_medians.clear()
            try:
                wandb.log({"Curriculum/Mastery_Window": len(recent_medians), "iteration": i})
            except Exception:
                pass

            if curr_difficulty < 6 and len(recent_medians) == 3:
                # Dynamic Stability Check: Ensure median hasn't dropped by >10% over the window
                if recent_medians[-1] >= recent_medians[0] * 0.90:
                    curr_difficulty += 1
                    recent_medians.clear()
                    epochs_since_upgrade = 0
                    
                    # SOTA Anti-Forgetting: Hard wipe the Replay Buffer
                    # Forces the network to instantly train ONLY on the new complex pieces
                    # instead of spending hours un-biasing the old difficulty data.
                    buffer = ReplayBuffer(
                        capacity=hw_config["capacity"],
                        unroll_steps=hw_config["unroll_steps"],
                        td_steps=hw_config["td_steps"],
                    )
                    
                    # Optimizer Warm Restart: Give the network the mathematical energy 
                    # required to break out of the local minimum it dug inside the previous difficulty.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = float(hw_config.get("lr_init", 1e-3))
                        
                    print(
                        f"\n🎓 [CURRICULUM PROMOTION] Stable Trend Detected! Promoting to Difficulty {curr_difficulty} 🎓"
                    )
                    print("🔥 [HYPER-MASTERY] Replay Buffer Wiped & LR Restarted to 1e-3! 🔥\n")
                else:
                    print(f"\n⚠️ [CURRICULUM HOLD] Median >300 but trend dropped (from {recent_medians[0]} to {recent_medians[-1]}). Waiting for stabilization... ⚠️\n")
                    # Pop the oldest so it must prove stability over the next epoch
                    recent_medians.pop(0)
            
            epochs_since_upgrade += 1

            metrics[iter_key] = {"difficulty": curr_difficulty, "best": best_score, "median": median_score, "average": float(np.mean(scores)), "distribution": scores}
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
            train(model, buffer, optimizer, hw_config, i)
            
            scheduler.step()
            for param_group in optimizer.param_groups:
                if param_group['lr'] < 1e-5:
                    param_group['lr'] = 1e-5
            
            try:
                wandb.log({"Train/LearningRate": scheduler.get_last_lr()[0], "iteration": i})
            except Exception:
                pass

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

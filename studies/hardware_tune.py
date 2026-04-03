#!/usr/bin/env python3
import optuna
import optunahub
import subprocess
import time
import pandas as pd
import os
import signal
import sys
import json
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True, help="Path to base config JSON"
)
parser.add_argument(
    "--trials", type=int, default=30, help="Number of hyperparameter suggestions"
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=15,
    help="Number of training steps to evaluate per trial",
)
parser.add_argument(
    "--timeout", type=int, default=400, help="Timeout in seconds before pruning a trial"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    BASE_CONFIG = json.load(f)


# Hardware-focused tuning objective
def objective(trial):
    config = BASE_CONFIG.copy()

    # Hardware Tuning Parameters tailored for an i7 + RTX 3080 Ti (12GB) setup
    config["num_processes"] = trial.suggest_int("num_processes", 8, 32)
    config["inference_batch_size_limit"] = trial.suggest_int(
        "inference_batch_size_limit", 16, 256
    )
    config["inference_timeout_ms"] = trial.suggest_int("inference_timeout_ms", 1, 10)

    # Continuous step distribution makes it friendly for CMA-ES
    config["train_batch_size"] = trial.suggest_int(
        "train_batch_size", 256, 4096, step=256
    )

    # Grouped naming convention so the outputs sort nicely in the /runs/ folder!
    experiment_name = f"tune_3080Ti_trial_{trial.number:03d}"
    metrics_file = f"runs/{experiment_name}/{experiment_name}_metrics.csv"
    config_file = f"runs/{experiment_name}/config.json"

    os.makedirs(f"runs/{experiment_name}", exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config, f)

    run_info = {
        "id": experiment_name,
        "name": experiment_name,
        "type": "TUNING_TRIAL",
        "status": "COMPLETED",
        "config": json.dumps(config),
        "tag": "Hardware Study",
    }
    with open(f"runs/{experiment_name}/run_info.json", "w") as f:
        json.dump(run_info, f)

    cmd = [
        "cargo",
        "run",
        "--release",
        "--features=hotpath,hotpath-alloc",
        "--bin",
        "tricked_engine",
        "--",
        "train",
        "--experiment-name",
        experiment_name,
        "--config",
        config_file,
        "--max-steps",
        str(args.max_steps),
    ]

    print(f"\n[Hardware Tune Trial {trial.number}] Starting with CMD: {' '.join(cmd)}")

    import select
    from subprocess import Popen

    process: "Popen[str]" = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    hotpath_mcts_avg = float("inf")

    def parse_hotpath_line(line):
        nonlocal hotpath_mcts_avg
        if "search::mcts_search" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 4:
                avg_str = parts[3]
                try:
                    if "ms" in avg_str:
                        hotpath_mcts_avg = (
                            float(avg_str.replace("ms", "").strip()) / 1000.0
                        )
                    elif "µs" in avg_str:
                        hotpath_mcts_avg = (
                            float(avg_str.replace("µs", "").strip()) / 1000000.0
                        )
                    elif "ns" in avg_str:
                        hotpath_mcts_avg = (
                            float(avg_str.replace("ns", "").strip()) / 1e9
                        )
                    elif "s" in avg_str:
                        hotpath_mcts_avg = float(avg_str.replace("s", "").strip())
                except ValueError:
                    pass

    stdout = process.stdout
    assert stdout is not None

    try:
        start_time = time.time()
        while process.poll() is None:
            reads, _, _ = select.select([stdout], [], [], 2.0)
            if stdout in reads:
                line = stdout.readline()
                if not line:
                    continue
                # Stream standard output to terminal
                sys.stdout.write(line)
                sys.stdout.flush()
                parse_hotpath_line(line)

            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        # Stop early if VRAM exceeds 3080 Ti limits (~11.5 GB safety buffer)
                        if (
                            "vram_usage_mb" in df.columns
                            and df["vram_usage_mb"].max() > 11500
                        ):
                            print(
                                f"[Trial {trial.number}] PRUNED: VRAM exceeded safety limit (3080 Ti)."
                            )
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            raise optuna.TrialPruned()
                except Exception:
                    pass

            if time.time() - start_time > args.timeout:  # Timeout limit per trial
                print(f"[Trial {trial.number}] Timeout reached, killing...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                raise optuna.TrialPruned()

        # Process any remaining stdout after completion
        for line in stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            parse_hotpath_line(line)

    finally:
        end_time = time.time()
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()

    if process.returncode != 0:
        print(
            f"[Trial {trial.number}] PRUNED: Process crashed with exit code {process.returncode}."
        )
        raise optuna.TrialPruned()

    wall_clock_time = end_time - start_time

    # Our goal is to maximize pure hardware throughput over fixed steps.
    if hotpath_mcts_avg != float("inf"):
        print(
            f"[Trial {trial.number}] Hotpath MCTS Search Time Avg: {hotpath_mcts_avg:.4f}s"
        )
        return wall_clock_time + (hotpath_mcts_avg * 100)
    else:
        return wall_clock_time


if __name__ == "__main__":
    storage_name = "sqlite:///studies/hardware_optuna_study.db"

    print("📦 Loading advanced samplers...")
    try:
        hebo_module = optunahub.load_module("samplers/hebo")
        sampler = hebo_module.HEBOSampler()
        print(
            "✅ Successfully loaded HEBOSampler via OptunaHub (Option 3). Best for noisy RL metrics!"
        )
    except Exception as e:
        print(
            f"⚠️ Could not load HEBOSampler. Error: {e}. Falling back to OptunaHub CMA-ES (Option 1)..."
        )
        try:
            cmaes_module = optunahub.load_module("samplers/cma_es_refinement")
            sampler = cmaes_module.CmaEsRefinementSampler()
            print("✅ Successfully loaded OptunaHub CMA-ES Refinement Sampler.")
        except Exception as e:
            print(
                f"⚠️ Could not load CMA-ES Refinement Sampler. Falling back to default TPE. Error: {e}"
            )
            sampler = None

    try:
        wilcoxon_module = optunahub.load_module("pruners/wilcoxon")
        pruner = wilcoxon_module.WilcoxonPruner(p_threshold=0.1)
    except Exception as e:
        print(f"⚠️ Could not load Wilcoxon Pruner, falling back to default. Error: {e}")
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3)

    study = optuna.create_study(
        study_name="tricked_ai_hardware_tune_rtx3080ti",
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def export_callback(study, trial):
        trials_data = []
        for t in study.trials:
            trials_data.append(
                {
                    "number": t.number,
                    "state": t.state.name,
                    "value": t.value,
                    "params": t.params,
                    "intermediate_values": t.intermediate_values or {},
                }
            )
        try:
            importance = optuna.importance.get_param_importances(study)
        except Exception:
            importance = {}

        tmp_file = "studies/optuna_study.json.tmp"
        os.makedirs("studies", exist_ok=True)
        with open(tmp_file, "w") as f:
            json.dump({"trials": trials_data, "importance": importance}, f)
        os.replace(tmp_file, "studies/optuna_study.json")

    print("⚙️  Starting Hardware Tuning Phase...")
    try:
        study.optimize(objective, n_trials=args.trials, callbacks=[export_callback])
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user.")

    print("\n✅ Hardware Tuning Complete!")
    print("Best Trial (Fastest Parallel Throughput):")
    print(f"  Composite Score: {study.best_trial.value}")
    print("  Optimal Hardware Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

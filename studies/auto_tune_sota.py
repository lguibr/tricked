#!/usr/bin/env python3
import optuna
import subprocess
import time
import pandas as pd
import os
import signal
import sys
import json
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

os.environ["WANDB_MODE"] = "online"

wandbc_auto = WeightsAndBiasesCallback(
    metric_name="final_loss",
    wandb_kwargs={"project": "tricked-ai-auto-tune"},
)


def objective(trial):
    # Base Config structure
    config = {
        "device": "cuda",
        "hidden_dimension_size": 64,
        "num_blocks": 4,
        "buffer_capacity_limit": 204800,
        "train_batch_size": 1024,
        "train_epochs": 4,
        "num_processes": 22,
        "worker_device": "cpu",
        "zmq_batch_size": 11,
        "zmq_timeout_ms": 20,
        "max_gumbel_k": 5,
        "difficulty": 6,
        "temp_boost": False,
    }

    # Suggest hyperparameters (from previous auto_tune bounds)
    config["simulations"] = trial.suggest_categorical("simulations", [16, 32])
    config["temporal_difference_steps"] = trial.suggest_categorical(
        "temporal_difference_steps", [3, 5]
    )
    config["gumbel_scale"] = trial.suggest_float("gumbel_scale", 0.5, 2.0)
    config["lr_init"] = trial.suggest_float("lr_init", 5e-4, 5e-3, log=True)
    config["reanalyze_ratio"] = trial.suggest_float("reanalyze_ratio", 0.0, 0.4)
    config["unroll_steps"] = trial.suggest_int("unroll_steps", 3, 6)
    config["support_size"] = trial.suggest_categorical("support_size", [100, 300])
    config["temp_decay_steps"] = trial.suggest_int("temp_decay_steps", 10, 50)

    experiment_name = f"auto_tune_trial_{trial.number}"
    metrics_file = f"runs/{experiment_name}/{experiment_name}_metrics.csv"
    config_file = f"runs/{experiment_name}/config.json"

    # Write config to disk so engine can pick it up
    os.makedirs(f"runs/{experiment_name}", exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config, f)

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "tricked_engine",
        "--",
        "train",
        "--experiment-name",
        experiment_name,
        "--config",
        config_file,
        "--max-steps",
        "20",
    ]

    print(f"\n[Trial {trial.number}] Starting with CMD: {' '.join(cmd)}")
    sys.stdout.flush()

    import select

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    final_loss = float("inf")
    last_reported_step = -1

    try:
        while process.poll() is None:
            reads, _, _ = select.select([process.stdout], [], [], 2.0)
            if process.stdout in reads:
                line = process.stdout.readline()
                if not line:
                    continue
                # Also echo stdout so Tauri can capture it!
                sys.stdout.write(line)
                sys.stdout.flush()

                if "FINAL_EVAL_SCORE:" in line:
                    try:
                        final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                    except:
                        pass

            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        last_step = df["step"].iloc[-1]
                        last_loss = df["total_loss"].iloc[-1]
                        if last_step > last_reported_step:
                            trial.report(last_loss, last_step)
                            last_reported_step = last_step
                        if trial.should_prune():
                            print(f"[Trial {trial.number}] Pruned at step {last_step}.")
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            raise optuna.TrialPruned()
                except:
                    pass

        # Process any remaining lines
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if "FINAL_EVAL_SCORE:" in line:
                try:
                    final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                except:
                    pass

    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()

    if final_loss == float("inf"):
        print(f"[Trial {trial.number}] Failed to read final loss.")
        raise optuna.TrialPruned()

    return final_loss


if __name__ == "__main__":
    storage_name = "sqlite:///studies/optuna_study.db"
    study = optuna.create_study(
        study_name="tricked_ai_sota_auto_tune",
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=5),
    )
    print("🚀 Starting SOTA Optuna auto-tuning... Press Ctrl+C to stop.")
    try:
        study.optimize(objective, n_trials=40, callbacks=[wandbc_auto])
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user.")

    print("\n✅ Optimization Session Complete!")

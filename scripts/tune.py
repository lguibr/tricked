#!/usr/bin/env python3
import optuna
import subprocess
import time
import pandas as pd
import os
import signal
import sys
from pathlib import Path

# Paths


def objective(trial):
    # Suggest hyperparameters (ranges tailored for rapid low-fidelity evaluation)
    lr_init = trial.suggest_float("lr_init", 1e-4, 5e-3, log=True)
    simulations = trial.suggest_int("simulations", 10, 50)
    unroll_steps = trial.suggest_int("unroll_steps", 5, 10)
    temporal_difference_steps = trial.suggest_int("temporal_difference_steps", 5, 10)
    reanalyze_ratio = trial.suggest_float("reanalyze_ratio", 0.0, 0.5)
    support_size = trial.suggest_int("support_size", 100, 500)
    temp_decay_steps = trial.suggest_int("temp_decay_steps", 1000, 50000)

    # We want to run for a fixed number of steps per trial to evaluate
    # Low-fidelity iteration for the optimization pipeline
    max_steps = 20
    experiment_name = f"optuna_trial_{trial.number}"
    metrics_file = f"runs/{experiment_name}/{experiment_name}_metrics.csv"

    # Ensure runs directory exists
    os.makedirs(f"runs/{experiment_name}", exist_ok=True)

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
        "--lr-init",
        str(lr_init),
        "--simulations",
        str(simulations),
        "--unroll-steps",
        str(unroll_steps),
        "--temporal-difference-steps",
        str(temporal_difference_steps),
        "--reanalyze-ratio",
        str(reanalyze_ratio),
        "--support-size",
        str(support_size),
        "--temp-decay-steps",
        str(temp_decay_steps),
        "--max-steps",
        str(max_steps),
    ]

    print(f"\n[Trial {trial.number}] Starting with CMD: {' '.join(cmd)}")

    import select

    # Start the process in a new process group so we can cleanly kill it
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,  # Line buffered
        preexec_fn=os.setsid,
    )

    final_loss = float("inf")
    last_reported_step = -1

    try:
        # Wait for the process to finish or check periodically for pruning
        while process.poll() is None:
            reads, _, _ = select.select([process.stdout], [], [], 2.0)
            if process.stdout in reads:
                line = process.stdout.readline()
                if "FINAL_EVAL_SCORE:" in line:
                    try:
                        final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                    except ValueError:
                        pass

            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        last_step = df["step"].iloc[-1]
                        last_loss = df["total_loss"].iloc[-1]

                        if last_step > last_reported_step:
                            # Report to Optuna for pruning
                            trial.report(last_loss, last_step)
                            last_reported_step = last_step

                        if trial.should_prune():
                            print(f"[Trial {trial.number}] Pruned at step {last_step}.")
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            raise optuna.TrialPruned()

                except Exception as e:
                    # File might be locked or half-written, ignore and retry next loop
                    pass

        # Final check if finished normally and we missed the line
        for line in process.stdout:
            if "FINAL_EVAL_SCORE:" in line:
                try:
                    final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                except ValueError:
                    pass

    finally:
        # Ensure cleanup of the child process tree
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()

    if final_loss == float("inf"):
        print(
            f"[Trial {trial.number}] Failed to read final loss. Process may have crashed."
        )
        raise optuna.TrialPruned()

    return final_loss


if __name__ == "__main__":
    storage_name = "sqlite:///optuna_study.db"

    # Create study using Median Pruner
    study = optuna.create_study(
        study_name="tricked_ai_optimization",
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=100, n_startup_trials=5),
    )

    print("🚀 Starting Optuna optimization... Press Ctrl+C to stop.")
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user.")

    print("\n✅ Optimization Session Complete!")
    print("Best Trial:")
    print(f"  Value (Final Loss): {study.best_trial.value}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

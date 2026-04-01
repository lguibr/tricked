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
import argparse
from optuna.integration.wandb import WeightsAndBiasesCallback

os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True, help="Path to base config JSON"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    BASE_CONFIG = json.load(f)

wandbc_auto = WeightsAndBiasesCallback(
    metric_name="final_loss",
    wandb_kwargs={"project": "tricked-ai-auto-tune"},
)


def dump_study_summary(study, trial):
    trials_data = []
    for t in study.trials:
        if (
            t.state == optuna.trial.TrialState.COMPLETE
            or t.state == optuna.trial.TrialState.RUNNING
        ):
            trials_data.append(
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": t.state.name,
                }
            )
    os.makedirs("studies", exist_ok=True)
    with open("studies/optuna_study.json", "w") as f:
        json.dump(trials_data, f)


def objective(trial):
    # Base Config structure cloned from UI
    config = BASE_CONFIG.copy()

    # Helper to either suggest range or keep static
    def apply_param(name, is_float=False, log=False, categorical=None):
        range_key = f"{name}_range"
        if (
            range_key in config
            and isinstance(config[range_key], list)
            and len(config[range_key]) == 2
        ):
            low, high = config[range_key]
            if is_float:
                config[name] = trial.suggest_float(
                    name, float(low), float(high), log=log
                )
            else:
                config[name] = trial.suggest_int(name, int(low), int(high))
        elif categorical is not None:
            config[name] = trial.suggest_categorical(name, categorical)

    apply_param("simulations")
    apply_param("max_gumbel_k")
    apply_param("lr_init", is_float=True, log=True)
    apply_param("num_processes")
    apply_param("num_blocks")
    apply_param("train_batch_size", categorical=[128, 256, 512, 1024, 2048, 4096])
    apply_param("temporal_difference_steps")
    apply_param("gumbel_scale", is_float=True)
    apply_param("reanalyze_ratio", is_float=True)
    apply_param("unroll_steps")
    apply_param("support_size")
    apply_param("temp_decay_steps")

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
    dump_study_summary(study, None)
    try:
        study.optimize(
            objective, n_trials=40, callbacks=[wandbc_auto, dump_study_summary]
        )
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user.")

    print("\n✅ Optimization Session Complete!")

#!/usr/bin/env python3
import time
from datetime import datetime
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import requests  # type: ignore
import optuna
from optuna.pruners import PatientPruner, HyperbandPruner
import optunahub

ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL = "http://127.0.0.1:8000/api"
TARGET_TRAINING_STEPS = (
    50  # Reduced to fit comfortably within the 10m hardware timeout!
)

# Hardware parameters locked for maximum throughput
base_config = {
    "device": "cuda",
    "hidden_dimension_size": 128,
    "num_blocks": 4,
    "buffer_capacity_limit": 204800,
    "train_batch_size": 1024,
    "train_epochs": 4,
    "num_processes": 20,
    "worker_device": "cpu",
    "zmq_batch_size": 12,
    "zmq_timeout_ms": 10,
    "max_gumbel_k": 8,
    "difficulty": 6,
    "temp_boost": False,
    "experiment_name_identifier": "tune_sota",
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def stop_engine_and_cooldown():
    try:
        requests.post(f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/stop")
        # Engine tears down quickly now, no need for massive 5s sleeps!
        time.sleep(0.5)
    except Exception:
        pass


def objective(trial: optuna.Trial) -> float:
    configuration = base_config.copy()

    # Base SOTA Sweeps (Reduced cardinality for 2-hour budget)
    simulations = trial.suggest_categorical("simulations", [16, 32])
    temporal_difference_steps = trial.suggest_categorical(
        "temporal_difference_steps", [3, 5]
    )
    gumbel_scale = trial.suggest_float("gumbel_scale", 0.5, 2.0)
    lr_init = trial.suggest_float("lr_init", 5e-4, 5e-3, log=True)

    # Advanced Sweeps
    reanalyze_ratio = trial.suggest_float("reanalyze_ratio", 0.0, 0.4)
    unroll_steps = trial.suggest_int("unroll_steps", 3, 6)
    support_size = trial.suggest_categorical("support_size", [100, 300])
    temp_decay_steps = trial.suggest_int("temp_decay_steps", 10, 50)

    configuration.update(
        {
            "simulations": simulations,
            "temporal_difference_steps": temporal_difference_steps,
            "gumbel_scale": gumbel_scale,
            "lr_init": lr_init,
            "reanalyze_ratio": reanalyze_ratio,
            "unroll_steps": unroll_steps,
            "support_size": support_size,
            "temp_decay_steps": temp_decay_steps,
        }
    )

    timestamp_prefix = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    experiment_name = f"{timestamp_prefix}_t{trial.number}_s{simulations}_td{temporal_difference_steps}_k{unroll_steps}"
    configuration["experiment_name_identifier"] = experiment_name

    print(f"\n[Trial {trial.number}] 🧠 Testing AI Conf: {experiment_name}")

    try:
        response = requests.post(
            f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/start",
            json=configuration,
        )
        if response.status_code != 200:
            print(
                f"❌ Failed to start engine: HTTP {response.status_code}, Body: {response.text}"
            )
            raise optuna.exceptions.TrialPruned()
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Engine API. Is the server running?")
        trial.study.stop()
        return 0.0

    best_mean_score: float = 0.0
    current_steps: int = 0
    start_time: float = time.time()
    max_trial_duration_secs: int = 240  # 4 minutes hard timeout

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        eval_task = progress.add_task(
            "[yellow]Evaluating (Max Score: 0.0)...",
            total=TARGET_TRAINING_STEPS,
        )

        while current_steps < TARGET_TRAINING_STEPS:
            time.sleep(2.0)  # Polling interval

            try:
                status = requests.get(
                    f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/status",
                    timeout=5,
                ).json()

                new_steps = status.get("training_steps", 0)
                if new_steps > current_steps:
                    advance = min(
                        new_steps - current_steps, TARGET_TRAINING_STEPS - current_steps
                    )
                    progress.advance(eval_task, advance=advance)
                    current_steps = new_steps

                top_games = status.get("top_games", [])
                if top_games:
                    # Calculate Mean Score of Top Games
                    scores = [g.get("score", 0.0) for g in top_games]
                    mean_score = sum(scores) / len(scores)
                    if mean_score > best_mean_score:
                        best_mean_score = mean_score
                        progress.update(
                            eval_task,
                            description=f"[green]Evaluating (Max Score: {best_mean_score:.1f})...",
                        )
                else:
                    mean_score = 0.0

                trial.report(mean_score, current_steps)

                if trial.should_prune():
                    print(
                        f"\n✂️  Trial {trial.number} pruned by Wilcoxon test/Hyperband. (Mean Score: {mean_score:.1f})."
                    )
                    stop_engine_and_cooldown()
                    raise optuna.exceptions.TrialPruned()

            except Exception as e:
                if "Failed to establish a new connection" in str(e):
                    print("\n❌ Server crashed. Pruning trial.")
                    stop_engine_and_cooldown()
                    raise optuna.exceptions.TrialPruned()

            if time.time() - start_time > max_trial_duration_secs:
                print("\n⏱️ Trial timed out (Stalled engine?). Pruning.")
                stop_engine_and_cooldown()
                raise optuna.exceptions.TrialPruned()

        progress.console.print(
            f"\n📊 Result for Trial {trial.number}: Max Mean Score = {best_mean_score:.1f}"
        )

    stop_engine_and_cooldown()
    return best_mean_score


if __name__ == "__main__":
    print("\nStarting Tricked AI SOTA Auto-Tuner 🚀")

    # 1. HEBOSampler
    print("Loading HEBO Sampler from OptunaHub...")
    try:
        hebo_module = optunahub.load_module("samplers/hebo")
        sampler = hebo_module.HEBOSampler()
        print("✅ HEBOSampler initialized successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load HEBOSampler: {e}\nFalling back to TPESampler.")
        sampler = optuna.samplers.TPESampler()

    # 2. WilcoxonPruner
    print("Initializing built-in Wilcoxon Pruner...")
    try:
        pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
        print("✅ WilcoxonPruner initialized successfully.")
    except Exception as e:
        print(
            f"⚠️ Failed to initialize WilcoxonPruner: {e}\nFalling back to PatientPruner(Hyperband)."
        )
        hyperband_pruner = HyperbandPruner(
            min_resource=100, max_resource=TARGET_TRAINING_STEPS // 2
        )
        pruner = PatientPruner(hyperband_pruner, patience=3)

    study = optuna.create_study(
        study_name="tricked-ai-tuning-sota-2h",
        storage="sqlite:///autotune.db",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    try:
        study.optimize(objective, n_trials=40)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Gracefully stopping study.")

    best_trial = study.best_trial
    print("\n🏆 OPTIMAL MATHEMATICAL CONFIGURATION ACHEIVED:")
    print(f"Max Mean Score: {best_trial.value:.1f}")
    for k, v in best_trial.params.items():
        print(f"{k}: {v}")

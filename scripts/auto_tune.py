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

ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL = "http://127.0.0.1:8000/api"
TIME_PER_EVALUATION = 90

# Base configuration strictly matching src/config.rs structure
base_config = {
    "device": "cuda",
    "hidden_dimension_size": 128,
    "num_blocks": 4,
    "support_size": 200,
    "buffer_capacity_limit": 200000,
    "simulations": 128,
    "train_batch_size": 512,
    "train_epochs": 4,
    "num_processes": 32,
    "worker_device": "cpu",
    "unroll_steps": 7,
    "temporal_difference_steps": 10,
    "zmq_batch_size": 16,
    "zmq_timeout_ms": 10,
    "max_gumbel_k": 8,
    "gumbel_scale": 1.0,
    "temp_decay_steps": 30,
    "difficulty": 6,
    "temp_boost": False,
    "lr_init": 0.001,
    "experiment_name_identifier": "tune_test",
}

# Ensure Optuna suppresses overly verbose info logs for readability
optuna.logging.set_verbosity(optuna.logging.WARNING)


def stop_engine_and_cooldown():
    try:
        requests.post(f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/stop")
        # VRAM Cooldown wait
        time.sleep(5)
    except Exception:
        pass


def objective(trial: optuna.Trial) -> float:
    configuration = base_config.copy()

    # Define the search space using Optuna suggestions
    num_processes = trial.suggest_categorical("num_processes", [32, 64, 128])
    zmq_batch_size = trial.suggest_categorical("zmq_batch_size", [16, 42, 85])
    hidden_dimension_size = trial.suggest_categorical(
        "hidden_dimension_size", [128, 192, 256]
    )
    simulations = trial.suggest_categorical("simulations", [16, 32, 64])
    train_batch_size = trial.suggest_categorical("train_batch_size", [512, 1024, 2048])

    zmq_batch_size_float = float(zmq_batch_size)
    minimum_required_processes = 1.5 * zmq_batch_size_float
    maximum_allowed_processes = 2.0 * zmq_batch_size_float

    # Golden Ratio filtering logic (hard exclusions handled by throwing TrialPruned)
    if num_processes <= 4:
        if zmq_batch_size != num_processes:
            raise optuna.exceptions.TrialPruned(
                f"Discarding unbalanced config: p={num_processes} / z={int(zmq_batch_size)} (Tiny grids demand 1:1 worker-to-zmq ratio)"
            )
    elif not (minimum_required_processes <= num_processes <= maximum_allowed_processes):
        raise optuna.exceptions.TrialPruned(
            f"Discarding unbalanced config: p={num_processes} / z={int(zmq_batch_size)} (Workers must be 1.5x - 2.0x of max ZMQ batch size)"
        )

    configuration.update(
        {
            "num_processes": num_processes,
            "zmq_batch_size": zmq_batch_size,
            "hidden_dimension_size": hidden_dimension_size,
            "simulations": simulations,
            "train_batch_size": train_batch_size,
            "buffer_capacity_limit": 100 * train_batch_size,
        }
    )

    timestamp_prefix = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    experiment_name = f"{timestamp_prefix}_tune_p{num_processes}_z{zmq_batch_size}_s{simulations}_d{hidden_dimension_size}_b{train_batch_size}"
    configuration["experiment_name_identifier"] = experiment_name

    print(f"\n[Trial {trial.number}] 🚀 Testing Configuration: {experiment_name}")

    try:
        response = requests.post(
            f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/start",
            json=configuration,
        )
        if response.status_code != 200:
            print(
                f"❌ Failed to start engine: HTTP {response.status_code} - {response.text}"
            )
            raise optuna.exceptions.TrialPruned("Failed to start engine")
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Engine API. Is the server running?")
        # Connection error is fatal, completely stop the study
        trial.study.stop()
        return 0.0

    games_per_second = 0.0
    evaluation_interval = 10
    total_steps = TIME_PER_EVALUATION // evaluation_interval

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
            f"[yellow]Evaluating {num_processes}P {zmq_batch_size}Z...",
            total=TIME_PER_EVALUATION,
        )

        for step in range(total_steps):
            for _ in range(evaluation_interval):
                time.sleep(1)
                progress.advance(eval_task)

            try:
                status_response = requests.get(
                    f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/status"
                ).json()
                games_played_count = status_response.get("games_played", 0)
                elapsed_time = (step + 1) * evaluation_interval
                games_per_second = games_played_count / elapsed_time

                # Report intermediate values to Optuna for potential Pruning
                trial.report(games_per_second, step)

                if trial.should_prune():
                    print(
                        f"✂️  Trial {trial.number} pruned after {elapsed_time}s due to poor performance ({games_per_second:.2f} G/s)."
                    )
                    stop_engine_and_cooldown()
                    raise optuna.exceptions.TrialPruned()

            except Exception as e:
                # If we fail to get status but the server is running, we just pass
                if "Failed to establish a new connection" in str(e):
                    print("❌ Server crashed. Pruning trial.")
                    stop_engine_and_cooldown()
                    raise optuna.exceptions.TrialPruned()

        progress.console.print(
            f"📊 Result for Trial {trial.number}: {games_per_second:.2f} Games/Second"
        )

    stop_engine_and_cooldown()
    return games_per_second


if __name__ == "__main__":
    print("\nStarting Tricked AI Auto-Tuner (Optuna Edition) 🚀")
    print("Dashboard available via: `optuna-dashboard sqlite:///autotune.db`")
    print(
        "Tip: You can also right-click autotune.db in VS Code to open the Optuna Dashboard extension!\n"
    )

    # Construct the stateful Optuna study
    study = optuna.create_study(
        study_name="tricked-ai-tuning",
        storage="sqlite:///autotune.db",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, interval_steps=1),
    )

    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Gracefully stopping study.")

    best_trial = study.best_trial

    print("\n🏆 OPTIMAL HARDWARE CONFIGURATION ACHEIVED:")
    print(f"Max Throughput: {best_trial.value:.2f} Games/Second")
    print(f"num_processes: {best_trial.params.get('num_processes')}")
    print(f"zmq_batch_size: {best_trial.params.get('zmq_batch_size')}")
    print(f"train_batch_size: {best_trial.params.get('train_batch_size')}")
    print(f"hidden_dimension_size: {best_trial.params.get('hidden_dimension_size')}")
    print(f"simulations: {best_trial.params.get('simulations')}")

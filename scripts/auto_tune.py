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
import itertools

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

# Define the grid of hardware configurations for Tricked AI Engine
hyperparameter_grid = {
    "num_processes": [1, 64],
    "zmq_batch_size": [1, 42],
    "hidden_dimension_size": [192],
    "simulations": [4, 256],
    "train_batch_size": [512, 2048],
}

keys, values = zip(*hyperparameter_grid.items())
all_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

maximum_games_per_second_achieved = 0.0
optimal_hardware_configuration = None

# Filter and score permutations
valid_permutations = []
for permutation in all_permutations:
    configuration = base_config.copy()
    configuration.update(permutation)

    zmq_batch_size = float(configuration.get("zmq_batch_size", 16))
    num_processes = int(configuration.get("num_processes", 32))

    minimum_required_processes = 1.5 * zmq_batch_size
    maximum_allowed_processes = 2.0 * zmq_batch_size

    if num_processes <= 4:
        if zmq_batch_size != num_processes:
            print(
                f"⚠️  Discarding unbalanced config: p={num_processes} / z={int(zmq_batch_size)} (Tiny grids demand 1:1 worker-to-zmq ratio)"
            )
            continue
    elif not (minimum_required_processes <= num_processes <= maximum_allowed_processes):
        print(
            f"⚠️  Discarding unbalanced config: p={num_processes} / z={int(zmq_batch_size)} (Workers must be 1.5x - 2.0x of max ZMQ batch size)"
        )
        continue

    valid_permutations.append(configuration)

# Sort by compute complexity
valid_permutations.sort(
    key=lambda c: (
        c["num_processes"]
        * c["zmq_batch_size"]
        * c["hidden_dimension_size"]
        * c["simulations"]
        * c["train_batch_size"]
    )
)

for index, config in enumerate(valid_permutations):
    config["intensity_rank"] = index + 1

# Interleave (smallest, largest, second smallest, second largest...)
alternating_permutations = []
left = 0
right = len(valid_permutations) - 1
while left <= right:
    alternating_permutations.append(valid_permutations[left])
    if left != right:
        alternating_permutations.append(valid_permutations[right])
    left += 1
    right -= 1

print(f"Total valid permutations to map: {len(alternating_permutations)}")

print(f"Total valid permutations to map: {len(alternating_permutations)}")

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
) as progress:
    total_task = progress.add_task(
        "[bold green]Total Tuning Progress...", total=len(alternating_permutations)
    )

    for configuration in alternating_permutations:
        num_processes = configuration["num_processes"]
        zmq_batch_size = configuration["zmq_batch_size"]
        hidden_dimension_size = configuration["hidden_dimension_size"]
        simulations = configuration["simulations"]
        train_batch_size = configuration["train_batch_size"]

        # Golden Ratio 2: Replay Buffer Ratio
        configuration["buffer_capacity_limit"] = 100 * train_batch_size
        timestamp_prefix = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        intensity_rank = configuration.get("intensity_rank", 0)
        total_models = len(valid_permutations)
        experiment_name = f"{timestamp_prefix}_W{intensity_rank:03d}of{total_models:03d}_tune_p{num_processes}_z{int(zmq_batch_size)}_s{simulations}_d{hidden_dimension_size}_b{train_batch_size}"
        configuration["experiment_name_identifier"] = experiment_name

        progress.console.print(
            f"\n[bold cyan]🚀 Testing Configuration:[/bold cyan] {experiment_name}"
        )

        # Start Training Session
        try:
            response = requests.post(
                f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/start",
                json=configuration,
            )
            if response.status_code != 200:
                progress.console.print(
                    f"[bold red]❌ Failed to start engine: HTTP {response.status_code}[/bold red] - {response.text}"
                )
                continue
        except requests.exceptions.ConnectionError:
            progress.console.print(
                "[bold red]❌ Failed to connect to Engine API. Is the server running?[/bold red]"
            )
            break

        # Wait for GPU warmup and MCTS stabilization
        model_task = progress.add_task(
            f"[yellow]Evaluating {num_processes}P {int(zmq_batch_size)}Z...",
            total=TIME_PER_EVALUATION,
        )
        for _ in range(TIME_PER_EVALUATION):
            time.sleep(1)
            progress.advance(model_task)
        progress.remove_task(model_task)

        # Gather Telemetry Metrics
        try:
            status_response = requests.get(
                f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/status"
            ).json()
            games_played_count = status_response.get("games_played", 0)
            games_per_second = games_played_count / TIME_PER_EVALUATION

            progress.console.print(
                f"[bold magenta]📊 Result:[/bold magenta] {games_per_second:.2f} Games/Second | Training Steps: {status_response.get('training_steps', 0)}"
            )

            if games_per_second > maximum_games_per_second_achieved:
                maximum_games_per_second_achieved = games_per_second
                optimal_hardware_configuration = configuration
        except Exception as e:
            progress.console.print(f"[bold red]⚠️ Failed to get status: {e}[/bold red]")

        # Stop and VRAM cooldown
        try:
            requests.post(
                f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/stop"
            )
            progress.console.print(
                "[dim]🛑 Stopped engine. Cooling VRAM for 5 seconds...[/dim]"
            )

            cooldown_task = progress.add_task("[dim]Cooling VRAM...[/dim]", total=5)
            for _ in range(5):
                time.sleep(1)
                progress.advance(cooldown_task)
            progress.remove_task(cooldown_task)
        except:
            pass

        progress.advance(total_task)

if optimal_hardware_configuration is not None:
    print(f"\n🏆 OPTIMAL HARDWARE CONFIGURATION:")
    print(f"Max Throughput: {maximum_games_per_second_achieved:.2f} Games/Second")
    print(f"num_processes: {optimal_hardware_configuration.get('num_processes')}")
    print(f"zmq_batch_size: {optimal_hardware_configuration.get('zmq_batch_size')}")
    print(f"train_batch_size: {optimal_hardware_configuration.get('train_batch_size')}")
    print(
        f"hidden_dimension_size: {optimal_hardware_configuration.get('hidden_dimension_size')}"
    )
else:
    print("\n⚠️ No configurations successfully completed the sweep.")

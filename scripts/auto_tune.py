#!/usr/bin/env python3
import time
import requests  # type: ignore
import itertools

ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL = "http://127.0.0.1:8000/api"

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
    "experiment_name_identifier": "tune_test"
}

# Define the grid of hardware configurations for Tricked AI Engine
hyperparameter_grid = {
    "num_processes": [16, 24, 32, 48, 64],
    "zmq_batch_size": [16, 32, 64],
    "hidden_dimension_size": [128, 256],
    "simulations": [50, 128],
    "train_batch_size": [512, 1024]
}

keys, values = zip(*hyperparameter_grid.items())
all_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

maximum_games_per_second_achieved = 0.0
optimal_hardware_configuration = None

print(f"Total permutations to map: {len(all_permutations)}")

for permutation in all_permutations:
    # Clone base config
    configuration = base_config.copy()
    configuration.update(permutation)

    # Golden Ratio 1: Inference Ratio
    # num_processes should be 1.5x to 2.0x of zmq_batch_size
    zmq_batch_size = float(configuration.get("zmq_batch_size", 16)) # type: ignore
    num_processes = int(configuration.get("num_processes", 32)) # type: ignore
    train_batch_size = int(configuration.get("train_batch_size", 512)) # type: ignore
    hidden_dimension_size = int(configuration.get("hidden_dimension_size", 128)) # type: ignore

    minimum_required_processes = 1.5 * zmq_batch_size
    maximum_allowed_processes = 2.0 * zmq_batch_size
    
    if not (minimum_required_processes <= num_processes <= maximum_allowed_processes):
        continue

    # Golden Ratio 2: Replay Buffer Ratio
    configuration["buffer_capacity_limit"] = 100 * train_batch_size
    
    experiment_name = f"tune_p{num_processes}_z{int(zmq_batch_size)}_d{hidden_dimension_size}_b{train_batch_size}"
    
    configuration["experiment_name_identifier"] = experiment_name

    print(f"🚀 Testing Configuration: {experiment_name} ...")

    # Start Training Session
    try:
        response = requests.post(f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/start", json=configuration)
        if response.status_code != 200:
            print(f"❌ Failed to start engine: HTTP {response.status_code} - {response.text}")
            continue
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Engine API. Is the server running?")
        break

    # Wait for GPU warmup and MCTS stabilization
    print("⏳ Waiting for 30 seconds for MCTS to stabilize (shortened for tuning)...")
    time.sleep(30)
    
    # Gather Telemetry Metrics
    try:
        status_response = requests.get(f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/status").json()
        games_played_count = status_response.get("games_played", 0)
        games_per_second = games_played_count / 60.0
        
        print(f"📊 Result: {games_per_second:.2f} Games/Second | Training Steps: {status_response.get('training_steps', 0)}")
        
        if games_per_second > maximum_games_per_second_achieved:
            maximum_games_per_second_achieved = games_per_second
            optimal_hardware_configuration = configuration
    except Exception as e:
        print(f"⚠️ Failed to get status: {e}")

    # Stop and VRAM cooldown
    try:
        requests.post(f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/stop")
        print("🛑 Stopped engine. Cooling VRAM for 5 seconds...\n")
        time.sleep(5)
    except:
        pass

if optimal_hardware_configuration is not None:
    print(f"\n🏆 OPTIMAL HARDWARE CONFIGURATION:")
    print(f"Max Throughput: {maximum_games_per_second_achieved:.2f} Games/Second")
    print(f"num_processes: {optimal_hardware_configuration.get('num_processes')}")
    print(f"zmq_batch_size: {optimal_hardware_configuration.get('zmq_batch_size')}")
    print(f"train_batch_size: {optimal_hardware_configuration.get('train_batch_size')}")
    print(f"hidden_dimension_size: {optimal_hardware_configuration.get('hidden_dimension_size')}")
else:
    print("\n⚠️ No configurations successfully completed the sweep.")

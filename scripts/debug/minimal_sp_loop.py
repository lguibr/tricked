import time
import torch
from tricked.config import get_hardware_config
from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer
from tricked.training.self_play import self_play

def run_minimal_profile():
    print("--- Running Tiny Profile ---")
    hw_config = get_hardware_config()
    
    # Force GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw_config["device"] = device
    hw_config["worker_device"] = device
    hw_config["num_games"] = 2
    hw_config["simulations"] = 25
    hw_config["num_processes"] = 1
    hw_config["d_model"] = 64
    hw_config["num_blocks"] = 2
    hw_config["self_play_batch_size"] = 16

    print(f"Using Device: {device}")
    
    model = MuZeroNet(d_model=hw_config["d_model"], num_blocks=hw_config["num_blocks"])
    model.to(device)
    model.share_memory()
    
    buffer = ReplayBuffer(capacity=100)
    
    start_time = time.time()
    
    # Run a single self-play loop
    buffer, scores = self_play(model, buffer, hw_config)
    
    elapsed = time.time() - start_time
    print(f"Completed {len(scores)} games in {elapsed:.2f} seconds.")
    print(f"Scores: {scores}")

if __name__ == "__main__":
    run_minimal_profile()

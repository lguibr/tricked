import time

import torch

from tricked.model.network import MuZeroNet

def benchmark_network():
    print("--- Simulating MuZero V2 Architecture Dimensions ---")
    
    d_model = 128
    num_blocks = 8
    support_size = 200
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MuZeroNet(d_model=d_model, num_blocks=num_blocks, support_size=support_size).to(device)
    model.eval()

    feature_tensor = torch.randn(batch_size, 20, 96, device=device)
    
    print("Testing Initial Inference...")
    start_time = time.time()
    h, value_scalar, policy, hole_logits = model.initial_inference(feature_tensor)
    init_time = time.time() - start_time
    
    assert h.shape == (batch_size, d_model, 96), f"h shape mismatch: {h.shape}"
    assert value_scalar.shape == (batch_size, 1), f"Value scalar mismatch: {value_scalar.shape}"
    assert policy.shape == (batch_size, 288), f"Policy mismatch: {policy.shape}"
    assert hole_logits.shape == (batch_size, 96), f"Hole logits mismatch: {hole_logits.shape}"
    print(f"Initial Inference Passed! ({init_time:.4f}s)")
    
    actions = torch.randint(0, 288, (batch_size,), device=device)
    piece_ids = torch.randint(0, 48, (batch_size,), device=device)
    
    print("Testing Recurrent Inference (Dynamics + Reward Prefix)...")
    start_time = time.time()
    h_next, reward_scalar, value_scalar_next, policy_next, hole_logits_next = model.recurrent_inference(h, actions, piece_ids)
    rec_time = time.time() - start_time
    
    assert h_next.shape == (batch_size, d_model, 96), f"h_next mismatch: {h_next.shape}"
    assert reward_scalar.shape == (batch_size, 1), f"Reward scalar mismatch: {reward_scalar.shape}"
    print(f"Recurrent Inference Passed! ({rec_time:.4f}s)")
    
    print("--> ARCHITECTURE DIMENSIONS VERIFIED <--")

if __name__ == "__main__":
    benchmark_network()

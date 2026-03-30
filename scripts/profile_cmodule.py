import torch
import sys
from torch.profiler import profile, record_function, ProfilerActivity

def run_profile(model_path, batch_size=1024):
    """
    Profiles the exported TorchScript tracing/compilation model 
    to measure cudaMemcpy vs actual Kernel execution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading TorchScript CModule '{model_path}' onto {device} with Batch Size {batch_size}")
    
    try:
        model = torch.jit.load(model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy tensors matching Tricked Engine state input size (BATCH, 20, 8, 16)
    # This reflects the exact dimension of our `pinned_initial_states`
    dummy_input = torch.randn(batch_size, 20, 8, 16, device=device)
    
    print("Warming up CModule execution...")
    with torch.no_grad():
        for _ in range(10):
            try:
                # The exported model should have the "initial_inference" method 
                # as defined in the python export script
                model.initial_inference(dummy_input)
            except Exception as e:
                print(f"Error during execution call: {e}")
                print("Make sure the loaded .pt file exports 'initial_inference(tensor)'.")
                return

    print("Profiling initial_inference across 50 iterations...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for _ in range(50):
                    model.initial_inference(dummy_input)

    print("\n--- Profiling Results (Sorted by CUDA Time) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    trace_file = "trace_initial.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nExported chrome trace to {trace_file}")
    print("You can view the exact PCIe transfer timeline by opening chrome://tracing and loading this JSON file.")
    print("If cudaMemcpyAsync dominates, ensure Rust `pinned_initial_states.pin_memory(device)` is active.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_cmodule.py <path_to_model.pt> [batch_size]")
        sys.exit(1)
    
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    run_profile(sys.argv[1], bs)

import os
import io
import zmq
import numpy as np
import torch
import ray

import hydra
import torch.optim as optim
from omegaconf import DictConfig

import wandb
from tricked.model.network import MuZeroNet
from tricked.setup_env import (
    init_wandb,
    load_go_exploit_starts,
    load_metrics_and_curriculum,
    load_model_checkpoint,
)

from tricked.training.trainer import train

@ray.remote(num_cpus=0)
def run_export(model_state, d_model, num_blocks, chkpt, device_str):
    import torch
    import os
    from tricked.model.network import MuZeroNet
    
    class InitialWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, s):
            return self.m.initial_inference(s)
            
    class RecurrentWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, h, a, p):
            return self.m.recurrent_inference(h, a, p)
            
    dev = torch.device(device_str)
    m = MuZeroNet(d_model=d_model, num_blocks=num_blocks).to(dev)
    m.load_state_dict(model_state)
    m.eval()
    
    dummy_s = torch.zeros(1, 20, 96).to(dev)
    torch.onnx.export(InitialWrapper(m), dummy_s, chkpt + "_initial.onnx", input_names=['state'], output_names=['h', 'value', 'policy', 'hole'], dynamic_axes={'state': {0: 'batch'}, 'h': {0: 'batch'}, 'value': {0: 'batch'}, 'policy': {0: 'batch'}, 'hole': {0: 'batch'}})
    
    dummy_h = torch.zeros(1, d_model, 96).to(dev)
    dummy_a = torch.zeros(1, dtype=torch.int64).to(dev)
    dummy_p = torch.zeros(1, dtype=torch.int64).to(dev)
    torch.onnx.export(RecurrentWrapper(m), (dummy_h, dummy_a, dummy_p), chkpt + "_recurrent.onnx", input_names=['h_in', 'action', 'piece'], output_names=['h_out', 'reward', 'value', 'policy', 'hole'], dynamic_axes={'h_in': {0: 'batch'}, 'action': {0: 'batch'}, 'piece': {0: 'batch'}, 'h_out': {0: 'batch'}, 'reward': {0: 'batch'}, 'value': {0: 'batch'}, 'policy': {0: 'batch'}, 'hole': {0: 'batch'}})

@ray.remote(num_cpus=2)
class RustSelfPlayActor:
    def __init__(self, cfg):
        self.cfg = cfg

    def start(self):
        import tricked_engine
        tricked_engine.run_self_play_worker(
            self.cfg.get("d_model", 32),
            self.cfg.get("num_blocks", 4),
            self.cfg.get("support_size", 300),
            self.cfg.get("simulations", 50),
            self.cfg.get("max_gumbel_k", 4),
            self.cfg.get("gumbel_scale", 1.0),
            self.cfg.get("difficulty", 1),
            self.cfg.get("temp_decay_steps", 15),
            self.cfg.get("push_port", "tcp://127.0.0.1:5555"),
            self.cfg.get("sub_port", "tcp://127.0.0.1:5557")
        )

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ray.init(ignore_reinit_error=True)
    
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(cfg.device)
    init_wandb(cfg)

    model = MuZeroNet(d_model=cfg.d_model, num_blocks=cfg.num_blocks).to(device)
    ema_model = MuZeroNet(d_model=cfg.d_model, num_blocks=cfg.num_blocks).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    if device.type == "cuda" and os.name != 'nt':
        model = torch.compile(model, mode="max-autotune", dynamic=True)  # type: ignore
        ema_model = torch.compile(ema_model, mode="max-autotune", dynamic=True)  # type: ignore

    optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr_init), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    import tricked_engine
    buffer = tricked_engine.NativeReplayBuffer(cfg["capacity"], cfg["unroll_steps"], cfg["td_steps"])
    buffer.start_reanalyzer(cfg.get("sub_port", "tcp://127.0.0.1:5557"))

    ctx = zmq.Context()
    weight_pub = ctx.socket(zmq.PUB)
    weight_pub.bind(cfg.get("sub_port", "tcp://127.0.0.1:5557"))

    try:
        load_model_checkpoint(model, device, cfg["model_checkpoint"])
        metrics, diff = load_metrics_and_curriculum(cfg["metrics_file"])
        
        from tricked.training.evaluator import EvaluatorActor
        
        evaluator = EvaluatorActor.remote(cfg)
        evaluator.evaluate.remote(model.state_dict())

        exploit_file, exploit_starts = load_go_exploit_starts(cfg.metrics_file)
        
        print("🚀 Spawning Hybrid Rust Engine for continuous Self-Play via Ray!")
        rust_worker = RustSelfPlayActor.remote(cfg)
        rust_worker.start.remote()

        consecutive_drops = 0
        
        for i in range(cfg["train_epochs"]):
            print(f"\n--- Iteration {i+1} (Diff {diff}) ---")
            cfg.update({"difficulty": diff, "exploit_starts": exploit_starts})
            
            model.eval()
            
            # Wait for data to arrive from Rust worker
            while True:
                length = buffer.get_length()
                if length >= cfg["train_batch_size"]:
                    break
                import time
                time.sleep(1)
                
            scores, med, max_s, avg = buffer.get_and_clear_metrics()
            curr_write_idx = buffer.get_global_write_idx()

            if scores:
                med = float(np.median(scores))
                avg = float(np.mean(scores))
                if wandb.run is not None:
                    wandb.log({
                        "Score/Max": float(max(scores)),
                        "Score/Min": float(min(scores)),
                        "Score/Average": avg,
                        "Score/Median": med,
                        "Score/GameCount": len(scores),
                        "Curriculum/Difficulty": diff
                    })
                
                from tricked.training.curriculum import evaluate_curriculum
                diff, consecutive_drops, action = evaluate_curriculum(
                    diff, avg, med, consecutive_drops
                )
                
                if action == "demote":
                    print(f"⚠️ Rolling back Curriculum! Avg score dropped < 50 for 3 epochs!")
                    backup_path = cfg["model_checkpoint"] + ".backup"
                    if os.path.exists(backup_path):
                        load_model_checkpoint(model, device, backup_path)
                        print("Restored model from reliable backup.")
                        del buffer
                        buffer = tricked_engine.NativeReplayBuffer(cfg["capacity"], cfg["unroll_steps"], cfg["td_steps"])
                        buffer.start_reanalyzer(cfg.get("sub_port", "tcp://127.0.0.1:5557"))
                elif action == "promote":
                    print(f"Promoting to Difficulty {diff}")
                    import shutil
                    if os.path.exists(cfg["model_checkpoint"]):
                        shutil.copy(cfg["model_checkpoint"], cfg["model_checkpoint"] + ".backup")
                    del buffer
                    buffer = tricked_engine.NativeReplayBuffer(cfg["capacity"], cfg["unroll_steps"], cfg["td_steps"])
                    buffer.start_reanalyzer(cfg.get("sub_port", "tcp://127.0.0.1:5557"))

            if buffer.get_length() >= cfg["train_batch_size"]:
                train(model, ema_model, buffer, optimizer, cfg, i)
                scheduler.step()
                tmp_path = cfg["model_checkpoint"] + ".tmp"
                torch.save(model.state_dict(), tmp_path)
                os.replace(tmp_path, cfg["model_checkpoint"])
                
                # Update Ray Actors in-memory
                ema_state_dict_ref = ray.put(ema_model.state_dict())
                state_dict_ref = ray.put(model.state_dict())
                # Rust Reanalyzer updates natively via ZMQ sync (see below)
                evaluator.evaluate.remote(state_dict_ref)
                
                # Update Rust Worker via ZMQ Sub
                jit_model = torch.jit.script(model)
                out_buffer = io.BytesIO()
                torch.jit.save(jit_model, out_buffer)
                weights_data = out_buffer.getvalue()
                weight_pub.send(weights_data)
                
                try:
                    run_export.remote(model.state_dict(), cfg["d_model"], cfg["num_blocks"], cfg["model_checkpoint"], "cpu")
                except Exception as e:
                    print(f"ONNX export trigger failed: {e}")
    finally:
        del buffer
        ray.shutdown()

if __name__ == "__main__":
    main()

import os
import sys

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
import hydra
import numpy as np
import ray
import torch
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
def run_export(model_state, d_model, num_blocks, support_size, chkpt, device_str):

    import torch

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
    m = MuZeroNet(d_model=d_model, num_blocks=num_blocks, support_size=support_size).to(dev)
    model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    m.load_state_dict(model_state)
    m.eval()

    dummy_s = torch.zeros(1, 20, 96).to(dev)
    torch.onnx.export(
        InitialWrapper(m),
        dummy_s,
        chkpt + "_initial.onnx",
        input_names=["state"],
        output_names=["h", "value", "policy", "hole"],
        dynamic_axes={
            "state": {0: "batch"},
            "h": {0: "batch"},
            "value": {0: "batch"},
            "policy": {0: "batch"},
            "hole": {0: "batch"},
        },
    )

    dummy_h = torch.zeros(1, d_model, 96).to(dev)
    dummy_a = torch.zeros(1, dtype=torch.int64).to(dev)
    dummy_p = torch.zeros(1, dtype=torch.int64).to(dev)
    torch.onnx.export(
        RecurrentWrapper(m),
        (dummy_h, dummy_a, dummy_p),
        chkpt + "_recurrent.onnx",
        input_names=["h_in", "action", "piece"],
        output_names=["h_out", "reward", "value", "policy", "hole"],
        dynamic_axes={
            "h_in": {0: "batch"},
            "action": {0: "batch"},
            "piece": {0: "batch"},
            "h_out": {0: "batch"},
            "reward": {0: "batch"},
            "value": {0: "batch"},
            "policy": {0: "batch"},
            "hole": {0: "batch"},
        },
    )


@ray.remote(num_cpus=2)
class RustSelfPlayActor:
    def __init__(self, cfg):
        self.cfg = cfg

    def start(self):
        import os
        import sys
        sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
        import torch
        
        # Ray uses fork(), which strips CUDA context. By default, Ray also overrides 
        # CUDA_VISIBLE_DEVICES="" if num_gpus=0. We must restore GPU visibility and 
        # re-initialize the C++ CUDA backend kernels before the Rust JIT loads!
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            del os.environ["CUDA_VISIBLE_DEVICES"]
        
        if torch.cuda.is_available():
            torch.cuda.init()

        import tricked_engine

        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_url = os.environ.get("REDIS_URL", f"redis://{redis_host}:6379")

        tricked_engine.run_self_play_worker(
            self.cfg.get("d_model", 32),
            self.cfg.get("num_blocks", 4),
            self.cfg.get("support_size", 200),
            self.cfg.get("simulations", 50),
            self.cfg.get("max_gumbel_k", 4),
            self.cfg.get("gumbel_scale", 1.0),
            self.cfg.get("difficulty", 1),
            self.cfg.get("temp_decay_steps", 15),
            self.cfg.get("push_port", "ipc:///tmp/tricked_relay.ipc"),
            redis_url,
        )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ray.init(ignore_reinit_error=True, dashboard_host="0.0.0.0")

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Falling back to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    init_wandb(cfg)

    model = MuZeroNet(d_model=cfg.d_model, num_blocks=cfg.num_blocks, support_size=cfg.get("support_size", 200)).to(device)
    ema_model = MuZeroNet(d_model=cfg.d_model, num_blocks=cfg.num_blocks, support_size=cfg.get("support_size", 200)).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    if device.type == "cuda" and os.name != "nt":
        model = torch.compile(model, mode="max-autotune", dynamic=True)  # type: ignore
        ema_model = torch.compile(ema_model, mode="max-autotune", dynamic=True)  # type: ignore

    optimizer = optim.Adam(
        model.parameters(), lr=float(cfg.lr_init), weight_decay=1e-4, capturable=True
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    import tricked_engine
    import uuid

    relay_ipc = f"ipc:///tmp/tricked_relay_{uuid.uuid4().hex}.ipc"
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.push_port = relay_ipc

    buffer = tricked_engine.NativeReplayBuffer(
        cfg["capacity"], 
        cfg["unroll_steps"], 
        cfg["td_steps"], 
        relay_ipc
    )
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_url = os.environ.get("REDIS_URL", f"redis://{redis_host}:6379")
    buffer.start_reanalyzer(redis_url)

    import redis

    redis_client = redis.Redis.from_url(redis_url)

    try:
        load_model_checkpoint(model, device, cfg["model_checkpoint"])
        metrics, diff = load_metrics_and_curriculum(cfg["metrics_file"])

        from tricked.training.evaluator import EvaluatorActor

        evaluator = EvaluatorActor.remote(cfg)
        clean_initial_sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
        evaluator.evaluate.remote(clean_initial_sd)

        exploit_file, exploit_starts = load_go_exploit_starts(cfg.metrics_file)

        import io

        import lz4.frame

        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
        export_model = __import__("tricked.model.network", fromlist=["MuZeroNet"]).MuZeroNet(
            cfg.get("d_model", 128), cfg.get("num_blocks", 8), cfg.get("support_size", 200)
        )
        export_model.load_state_dict(clean_sd)
        export_model.eval()
        for p in export_model.parameters():
            p.requires_grad = False
            
        jit_model = torch.jit.script(export_model.half())
        out_buffer = io.BytesIO()
        torch.jit.save(jit_model, out_buffer)
        compressed_weights = lz4.frame.compress(out_buffer.getvalue())
        redis_client.set("model_weights", compressed_weights)

        print("🚀 Spawning Hybrid Rust Engine for continuous Self-Play via Ray!", flush=True)
        rust_worker = RustSelfPlayActor.remote(cfg)
        rust_worker.start.remote()

        consecutive_drops = 0

        for i in range(cfg["train_epochs"]):
            print(f"\n--- Iteration {i+1} (Diff {diff}) ---", flush=True)
            cfg.update({"difficulty": diff, "exploit_starts": exploit_starts})

            model.eval()

            # Wait for data to arrive from Rust worker
            while True:
                length = buffer.get_length()
                if length >= cfg["train_batch_size"]:
                    print(
                        f"\nBuffer filled: {length}/{cfg['train_batch_size']} items. Proceeding.",
                        flush=True,
                    )
                    break
                print(
                    f"Waiting for games from Rust workers... ({length}/{cfg['train_batch_size']})",
                    end="\r",
                    flush=True,
                )
                import time

                time.sleep(1)

            scores, med, max_s, avg = buffer.get_and_clear_metrics()
            curr_write_idx = buffer.get_global_write_idx()

            if scores:
                med = float(np.median(scores))
                avg = float(np.mean(scores))
                if wandb.run is not None:
                    wandb.log(
                        {
                            "Score/Max": float(max(scores)),
                            "Score/Min": float(min(scores)),
                            "Score/Average": avg,
                            "Score/Median": med,
                            "Score/GameCount": len(scores),
                            "Curriculum/Difficulty": diff,
                        }
                    )

                from tricked.training.curriculum import evaluate_curriculum

                diff, consecutive_drops, action = evaluate_curriculum(
                    diff, avg, med, consecutive_drops
                )

                if action == "demote":
                    print("⚠️ Rolling back Curriculum! Avg score dropped < 50 for 3 epochs!")
                    backup_path = cfg["model_checkpoint"] + ".backup"
                    if os.path.exists(backup_path):
                        load_model_checkpoint(model, device, backup_path)
                        print("Restored model from reliable backup.")
                        del buffer
                        buffer = tricked_engine.NativeReplayBuffer(
                            cfg["capacity"], cfg["unroll_steps"], cfg["td_steps"]
                        )
                        buffer.start_reanalyzer(redis_url)
                elif action == "promote":
                    print(f"Promoting to Difficulty {diff}")
                    import shutil

                    if os.path.exists(cfg["model_checkpoint"]):
                        shutil.copy(cfg["model_checkpoint"], cfg["model_checkpoint"] + ".backup")
                    del buffer
                    buffer = tricked_engine.NativeReplayBuffer(
                        cfg["capacity"], cfg["unroll_steps"], cfg["td_steps"]
                    )
                    buffer.start_reanalyzer(redis_url)

            if buffer.get_length() >= cfg["train_batch_size"]:
                train(model, ema_model, buffer, optimizer, cfg, i)
                scheduler.step()
                os.makedirs(os.path.dirname(cfg["model_checkpoint"]), exist_ok=True)
                tmp_path = cfg["model_checkpoint"] + ".tmp"
                torch.save(model.state_dict(), tmp_path)
                os.replace(tmp_path, cfg["model_checkpoint"])

                # Update Ray Actors in-memory
                clean_ema_sd = {
                    k.replace("_orig_mod.", ""): v for k, v in ema_model.state_dict().items()
                }
                clean_sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}

                ema_state_dict_ref = ray.put(clean_ema_sd)
                state_dict_ref = ray.put(clean_sd)
                # Rust Reanalyzer updates natively via ZMQ sync (see below)
                evaluator.evaluate.remote(state_dict_ref)

                # Update Rust Worker via Redis Sub
                import io

                import lz4.frame

                clean_sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
                export_model = __import__("tricked.model.network", fromlist=["MuZeroNet"]).MuZeroNet(
                    cfg.get("d_model", 128), cfg.get("num_blocks", 8), cfg.get("support_size", 200)
                )
                export_model.load_state_dict(clean_sd)
                export_model.eval()
                for p in export_model.parameters():
                    p.requires_grad = False
                    
                jit_model = torch.jit.script(export_model.half())

                out_buffer = io.BytesIO()
                torch.jit.save(jit_model, out_buffer)
                compressed_weights = lz4.frame.compress(out_buffer.getvalue())

                redis_client.set("model_weights", compressed_weights)
                redis_client.publish("model_updates", b"update")

                try:
                    run_export.remote(
                        model.state_dict(),
                        cfg["d_model"],
                        cfg["num_blocks"],
                        cfg.get("support_size", 200),
                        cfg["model_checkpoint"],
                        "cpu",
                    )
                except Exception as e:
                    print(f"ONNX export trigger failed: {e}")
    finally:
        del buffer
        ray.shutdown()


if __name__ == "__main__":
    main()

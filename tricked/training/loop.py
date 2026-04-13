import torch
import torch.nn.functional as F
import time
import sys
import os
import argparse
import threading
import json
import traceback
import sqlite3
import psutil
import queue
import warnings
import math
import collections

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine", "target", "release")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extensions")))

telemetry_q = queue.Queue(maxsize=1000)

def sqlite_daemon():
    while True:
        metric = telemetry_q.get()
        if metric is None:
            break
        try:
            conn = sqlite3.connect(metric['db_path'], timeout=10)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (
                    run_id, step, total_loss, elapsed_time, value_loss, policy_loss, 
                    reward_loss, policy_entropy, action_space_entropy, representation_drift, gradient_norm,
                    mcts_depth_mean, mcts_search_time_mean, game_score_mean, game_score_med, game_score_max, game_score_min, game_lines_cleared, difficulty, game_count, win_rate, spatial_heatmap,
                    cpu_usage_pct, ram_usage_mb, gpu_usage_pct, vram_usage_mb, disk_usage_pct,
                    network_tx_mbps, network_rx_mbps, disk_read_mbps, disk_write_mbps,
                    queue_saturation_ratio, sps_vs_tps, queue_latency_us, sumtree_contention_us, lr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, metric['args'])
            conn.commit()
            conn.close()
        except Exception:
            pass

threading.Thread(target=sqlite_daemon, daemon=True).start()

def update_training_metrics(db_path, run_id, step, total_loss, elapsed, val_loss, pol_loss, reward_loss, pol_entropy, act_entropy, sim_loss, grad_norm, mcts_depth_mean, mcts_search_time_mean, game_score_mean, game_score_med, game_score_max, game_score_min, game_lines_cleared, difficulty, game_count, win_rate, spatial_heatmap, cpu_pct, ram_mb, gpu_pct, vram_mb, disk_usage_pct, net_tx, net_rx, disk_read, disk_write, q_sat, sps_tps, q_lat, tree_cont, lr):
    try:
        telemetry_q.put_nowait({
            'db_path': db_path,
            'args': (run_id, step, total_loss, elapsed, val_loss, pol_loss, reward_loss, pol_entropy, act_entropy, sim_loss, grad_norm, mcts_depth_mean, mcts_search_time_mean, game_score_mean, game_score_med, game_score_max, game_score_min, game_lines_cleared, difficulty, game_count, win_rate, json.dumps(spatial_heatmap), cpu_pct, ram_mb, gpu_pct, vram_mb, disk_usage_pct, net_tx, net_rx, disk_read, disk_write, q_sat, sps_tps, q_lat, tree_cont, lr)
        })
    except queue.Full:
        pass

def save_checkpoint(net, config_path, step):
    run_dir = os.path.dirname(config_path)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, f"checkpoint_step_{step}.safetensors")
    torch.save(net.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

try:
    import tricked_engine
    import numpy as np
except ImportError as e:
    print(f"Failed to import native extensions. Error: {e}")
    sys.exit(1)

from tricked.models.muzero import MuZeroNet, InitialInferenceModel, RecurrentInferenceModel
from tricked.models.bptt import BPTTKernel
from tricked.extensions.native_features import launch_extract_features

def update_ema_targets(active_net, target_net, tau=0.99):
    with torch.no_grad():
        for param, target_param in zip(active_net.parameters(), target_net.parameters()):
            target_param.data.mul_(tau).add_(param.data, alpha=1.0 - tau)

def run_training(config_json_path, run_id, db_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from tricked.config_schema import TrickedConfig
    with open(config_json_path, "r") as f:
        config_str = f.read()
        
    cfg = TrickedConfig.parse_raw(config_str)
    config_bytes = cfg.get_full_bytes()

    capacity = cfg.optimizer.buffer_capacity_limit
    inf_batch = cfg.hardware.inference_batch_size_limit
    inf_timeout = cfg.hardware.inference_timeout_ms
    batch_size = cfg.optimizer.train_batch_size
    unroll_steps = cfg.optimizer.unroll_steps

    engine = tricked_engine.TrickedEngine(capacity, config_bytes, str(run_id), f"Training {run_id[:8]}", "training")
    
    hidden_dim = cfg.architecture.hidden_dimension_size
    blocks = cfg.architecture.num_blocks
    support = cfg.architecture.value_support_size
    
    print(f"Initializing network with dim={hidden_dim}, blocks={blocks}")
    net = MuZeroNet(hidden_dim, blocks, support, 20).to(device)
    
    bptt = BPTTKernel(hidden_dim, blocks, support, 20).to(device)
    bptt.active_net.load_state_dict(net.state_dict(), strict=False)
    bptt.target_net.load_state_dict(bptt.active_net.state_dict())
    
    print("Exporting networks to TorchScript...")
    model_dir = os.path.dirname(config_json_path)
    os.makedirs(model_dir, exist_ok=True)
    
    initial_eval_model = InitialInferenceModel(bptt.active_net).eval()
    recurrent_eval_model = RecurrentInferenceModel(bptt.active_net).eval()

    initial_scripted = torch.jit.script(initial_eval_model)
    initial_scripted.save(os.path.join(model_dir, "initial_model.pt"))
    recurrent_scripted = torch.jit.script(recurrent_eval_model)
    recurrent_scripted.save(os.path.join(model_dir, "recurrent_model.pt"))

    bptt.train()

    print("Compiling BPTT forward unroll with torch.compile...")
    bptt.forward = torch.compile(bptt.forward)
    
    lr = cfg.optimizer.lr_init
    max_steps = cfg.optimizer.max_steps if cfg.optimizer.max_steps > 0 else 10000000
    optimizer = torch.optim.AdamW(bptt.active_net.parameters(), lr=lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    print("Starting MCTS Engine workers...")
    
    micro_batch_size = 128 # Safe chunk size for 8GB-16GB VRAM cards
    
    engine.start_workers(cfg.hardware.num_processes, os.path.join(model_dir, "initial_model.pt"), os.path.join(model_dir, "recurrent_model.pt"), device.type == "cuda")
    
    print("Engine logic engaged.")
    start_time = time.time()
    checkpoint_interval = 10 
    last_net = psutil.net_io_counters()
    last_disk = psutil.disk_io_counters()
    last_time = time.time()
    
    prefetch_q = queue.Queue(maxsize=3)
    def prefetch_daemon():
        while True:
            b = engine.sample_batch(batch_size)
            if b is not None:
                pinned_batch = {}
                for k, v in b.items():
                    if k == "arena_ptr":
                        pinned_batch[k] = v
                    else:
                        pinned_batch[k] = torch.from_numpy(v).pin_memory()
                prefetch_q.put(pinned_batch)
            else:
                time.sleep(0.1)
                
    threading.Thread(target=prefetch_daemon, daemon=True).start()

    last_telemetry = {
        "mcts_depth_mean": 0.0, "mcts_time_mean": 0.0, "game_score_mean": 0.0,
        "game_score_med": 0.0, "game_score_max": 0.0, "game_score_min": 0.0,
        "game_lines_cleared": 0.0, "difficulty": 0.0, "win_rate": 0.0
    }

    try:
        for step in range(max_steps):

            try:
                batch = prefetch_q.get(timeout=5.0)
            except queue.Empty:
                print("Waiting for buffer to fill...")
                continue
                
            # 1. FIX: Reshape the 1D flat arrays into 2D so they slice by "states" instead of "integers"
            board_states = batch["board_states"].to(device, non_blocking=True).view(batch_size, 2)
            board_histories = batch["board_histories"].to(device, non_blocking=True).view(batch_size, 14)
            board_available = batch["board_available"].to(device, non_blocking=True).view(batch_size, 3)
            board_historical_acts = batch["board_historical_acts"].to(device, non_blocking=True).view(batch_size, 3)
            board_diff = batch["board_diff"].to(device, non_blocking=True).view(batch_size)
            
            actions = batch["actions"].view(batch_size, unroll_steps).to(device, non_blocking=True)
            piece_ids = batch["piece_identifiers"].view(batch_size, unroll_steps).to(device, non_blocking=True)
            vps = batch["value_prefixs"].view(batch_size, unroll_steps).to(device, non_blocking=True)
            pol = batch["target_policies"].view(batch_size, unroll_steps + 1, 288).to(device, non_blocking=True)
            val = batch["target_values"].view(batch_size, unroll_steps + 1).to(device, non_blocking=True)
            loss_mask = batch["loss_masks"].view(batch_size, unroll_steps + 1).to(device, non_blocking=True)
            imp_weights = batch["importance_weights"].view(batch_size).to(device, non_blocking=True)
            
            raw_unrolled_boards = batch["raw_unrolled_boards"].to(device, non_blocking=True).view(batch_size * unroll_steps, 2)
            raw_unrolled_histories = batch["raw_unrolled_histories"].to(device, non_blocking=True).view(batch_size * unroll_steps, 14)
            raw_unrolled_available = batch["raw_unrolled_available"].to(device, non_blocking=True).view(batch_size * unroll_steps, 3)
            raw_unrolled_actions = batch["raw_unrolled_actions"].to(device, non_blocking=True).view(batch_size * unroll_steps, 3)
            raw_unrolled_diff = batch["raw_unrolled_diff"].to(device, non_blocking=True).view(batch_size * unroll_steps)

            optimizer.zero_grad(set_to_none=True)

            num_micro_batches = math.ceil(batch_size / micro_batch_size)
            accumulated_loss = 0.0
            mb_metrics = collections.defaultdict(float)

            for i in range(num_micro_batches):
                start_idx = i * micro_batch_size
                end_idx = min((i + 1) * micro_batch_size, batch_size)
                mb_size = end_idx - start_idx
                if mb_size <= 0: continue
                mb_ratio = mb_size / batch_size

                mb_board_states = board_states[start_idx:end_idx].contiguous()
                mb_board_histories = board_histories[start_idx:end_idx].contiguous()
                mb_board_available = board_available[start_idx:end_idx].contiguous()
                mb_board_historical_acts = board_historical_acts[start_idx:end_idx].contiguous()
                mb_board_diff = board_diff[start_idx:end_idx].contiguous()
                
                mb_state_features = launch_extract_features(mb_board_states, mb_board_available, mb_board_histories, mb_board_historical_acts, mb_board_diff, 1)
                
                mb_raw_unrolled_boards = raw_unrolled_boards[start_idx * unroll_steps : end_idx * unroll_steps].contiguous()
                mb_raw_unrolled_histories = raw_unrolled_histories[start_idx * unroll_steps : end_idx * unroll_steps].contiguous()
                mb_raw_unrolled_avail = raw_unrolled_available[start_idx * unroll_steps : end_idx * unroll_steps].contiguous()
                mb_raw_unrolled_acts = raw_unrolled_actions[start_idx * unroll_steps : end_idx * unroll_steps].contiguous()
                mb_raw_unrolled_diff = raw_unrolled_diff[start_idx * unroll_steps : end_idx * unroll_steps].contiguous()
                unrolled_features_flat = launch_extract_features(mb_raw_unrolled_boards, mb_raw_unrolled_avail, mb_raw_unrolled_histories, mb_raw_unrolled_acts, mb_raw_unrolled_diff, 1)
                mb_synthetic_unrolled_states = unrolled_features_flat.view(mb_size, unroll_steps, 20, 8, 16)

                # Execute with AMP (BFloat16) to drastically reduce memory usage, preventing NaN gradients from exponent overflow
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    mb_final_loss, _, val_l, pol_l, vp_l, pol_entr, act_entr, sim_l = bptt(
                        mb_state_features, 
                        actions[start_idx:end_idx], 
                        piece_ids[start_idx:end_idx], 
                        vps[start_idx:end_idx], 
                        pol[start_idx:end_idx], 
                        val[start_idx:end_idx], 
                        mb_synthetic_unrolled_states, 
                        loss_mask[start_idx:end_idx], 
                        imp_weights[start_idx:end_idx]
                    )
                    scaled_loss = mb_final_loss * mb_ratio

                # Standard accumulation for bfloat16
                scaled_loss.backward()

                accumulated_loss += mb_final_loss.item() * mb_ratio
                mb_metrics['val_l'] += val_l.item() * mb_ratio
                mb_metrics['pol_l'] += pol_l.item() * mb_ratio
                mb_metrics['vp_l'] += vp_l.item() * mb_ratio
                mb_metrics['pol_entr'] += pol_entr.item() * mb_ratio
                mb_metrics['act_entr'] += act_entr.item() * mb_ratio
                mb_metrics['sim_l'] += sim_l.item() * mb_ratio

            grad_norm = torch.nn.utils.clip_grad_norm_(bptt.active_net.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            update_ema_targets(bptt.active_net, bptt.target_net, tau=0.99)

            # Map tracking variables to accumulated sums
            current_loss = accumulated_loss
            val_l_item = mb_metrics['val_l']
            pol_l_item = mb_metrics['pol_l']
            vp_l_item = mb_metrics['vp_l']
            pol_entr_item = mb_metrics['pol_entr']
            act_entr_item = mb_metrics['act_entr']
            sim_l_item = mb_metrics['sim_l']
            elapsed = time.time() - start_time
            print(f"Step {step} | Loss: {current_loss:.4f} | Grad_N: {grad_norm:.4f}", flush=True)

            try:
                tel = engine.get_telemetry()
                
                game_count = tel.get("game_count", 0)
                
                mcts_depth_mean = tel.get("mcts_depth_mean", 0.0)
                if not math.isnan(mcts_depth_mean): last_telemetry["mcts_depth_mean"] = mcts_depth_mean
                mcts_depth_mean = last_telemetry["mcts_depth_mean"]

                mcts_time_mean = tel.get("mcts_search_time_mean", 0.0)
                if not math.isnan(mcts_time_mean): last_telemetry["mcts_time_mean"] = mcts_time_mean
                mcts_time_mean = last_telemetry["mcts_time_mean"]

                game_score_mean = tel.get("game_score_mean", 0.0)
                if not math.isnan(game_score_mean): last_telemetry["game_score_mean"] = game_score_mean
                game_score_mean = last_telemetry["game_score_mean"]

                game_score_med = tel.get("game_score_med", 0.0)
                if not math.isnan(game_score_med): last_telemetry["game_score_med"] = game_score_med
                game_score_med = last_telemetry["game_score_med"]

                game_score_max = tel.get("game_score_max", 0.0)
                if not math.isnan(game_score_max): last_telemetry["game_score_max"] = game_score_max
                game_score_max = last_telemetry["game_score_max"]

                game_score_min = tel.get("game_score_min", 0.0)
                if not math.isnan(game_score_min): last_telemetry["game_score_min"] = game_score_min
                game_score_min = last_telemetry["game_score_min"]

                game_lines_cleared = tel.get("game_lines_cleared", 0.0)
                if not math.isnan(game_lines_cleared): last_telemetry["game_lines_cleared"] = game_lines_cleared
                game_lines_cleared = last_telemetry["game_lines_cleared"]

                difficulty = tel.get("difficulty", 0.0)
                if not math.isnan(difficulty): last_telemetry["difficulty"] = difficulty
                difficulty = last_telemetry["difficulty"]

                win_rate = tel.get("win_rate", 0.0)
                if not math.isnan(win_rate): last_telemetry["win_rate"] = win_rate
                win_rate = last_telemetry["win_rate"]

                spatial_heatmap = tel.get("spatial_heatmap", [])
                
                q_sat = tel.get("queue_saturation_ratio", 0.0)
                sps_tps = tel.get("sps_vs_tps", 0.0)
                q_lat = tel.get("queue_latency_us", 0.0)
                tree_cont = tel.get("sumtree_contention_us", 0.0)
            except Exception as e:
                print(f"Telemetry warning: {e}", flush=True)
                mcts_depth_mean, mcts_time_mean, game_score_mean, win_rate = 0.0, 0.0, 0.0, 0.0
                game_score_med, game_score_max, game_score_min, game_lines_cleared, difficulty, game_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
                spatial_heatmap = []
                q_sat, sps_tps, q_lat, tree_cont = 0.0, 0.0, 0.0, 0.0
            
            cpu_pct = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            ram_mb = mem.used / 1024 / 1024
            disk_pct = psutil.disk_usage('/').percent
            
            now = time.time()
            delta = max(now - last_time, 0.001)
            net_io = psutil.net_io_counters()
            disk = psutil.disk_io_counters()
            
            if step == 0:
                net_rx, net_tx, disk_read, disk_write = 0.0, 0.0, 0.0, 0.0
            else:
                net_rx = ((net_io.bytes_recv - last_net.bytes_recv) * 8) / (1024 * 1024) / delta
                net_tx = ((net_io.bytes_sent - last_net.bytes_sent) * 8) / (1024 * 1024) / delta
                disk_read = (disk.read_bytes - last_disk.read_bytes) / (1024 * 1024) / delta
                disk_write = (disk.write_bytes - last_disk.write_bytes) / (1024 * 1024) / delta
                
            last_net, last_disk, last_time = net_io, disk, now
            
            gpu_pct, vram_mb = 0.0, 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_pct = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                vram_mb = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 / 1024
            except:
                pass

            update_training_metrics(
                db_path, run_id, step, current_loss, elapsed,
                val_l_item, pol_l_item, vp_l_item, 
                pol_entr_item, act_entr_item, sim_l_item, grad_norm.item(),
                mcts_depth_mean, mcts_time_mean, game_score_mean, game_score_med, game_score_max, game_score_min, game_lines_cleared, difficulty, game_count, win_rate, spatial_heatmap,
                cpu_pct, ram_mb, gpu_pct, vram_mb, disk_pct,
                net_tx, net_rx, disk_read, disk_write,
                q_sat, sps_tps, q_lat, tree_cont, current_lr
            )
            
            if step > 0 and step % checkpoint_interval == 0:
                save_checkpoint(net, config_json_path, step)
                
            if "arena_ptr" in batch:
                engine.release_batch_arena(batch["arena_ptr"])
                
        print("Training finished dynamically!")
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        engine.stop_workers()
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            cur = conn.cursor()
            cur.execute("UPDATE runs SET status = 'COMPLETED' WHERE id = ?", (run_id,))
            conn.commit()
            conn.close()
        except:
            pass
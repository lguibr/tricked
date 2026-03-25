"""
Standard Documentation for self_play.py.

This module supplies the core execution logic for the `training` namespace, heavily typed and tested for production distribution.
"""

import struct
import subprocess
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import zmq

if TYPE_CHECKING:
    from tricked.model.network import MuZeroNet
    from tricked.training.buffer import ReplayBuffer

def self_play(
    model: "MuZeroNet", buffer: "ReplayBuffer", hw_config: Any
) -> tuple["ReplayBuffer", list[float]]:
    from tricked.training.redis_logger import init_db
    init_db()

    num_games = hw_config["num_games"]
    print(f"🚀 Spawning Hybrid Rust Engine for {num_games} Self-Play Episodes via ZMQ Pipeline!")

    import torch
    try:
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        m_script = torch.jit.script(base_model.cpu())
        import os
        checkpoint_dir = os.path.dirname(hw_config.model_checkpoint)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        m_script.save(hw_config.model_checkpoint + "_jit.pt")
    except Exception as e:
        print(f"Failed to JIT script model: {e}")
        
    try:
        model.to(torch.device(hw_config.device))
    except Exception:
        pass

    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    puller.bind("tcp://127.0.0.1:5556")

    rust_bin_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "tricked_rs", "target", "release", "self_play_worker")
    if os.path.exists(rust_bin_path):
        rust_proc = subprocess.Popen([rust_bin_path], cwd=os.path.dirname(rust_bin_path))
    else:
        rust_proc = subprocess.Popen(["cargo", "run", "--release", "--bin", "self_play_worker"], cwd=os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "tricked_rs"))

    results = []
    completed_games = 0
    running_scores = []
    
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )
        console = Console()
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn(" | [yellow]Med[/]: {task.fields[med]:.1f} | [red]Max[/]: {task.fields[max]:.0f} | [magenta]Avg[/]: {task.fields[avg]:.1f}"),
            console=console,
            transient=True 
        ) as progress:
            task1 = progress.add_task("Hybrid MCTS Generation", total=num_games, med=0, max=0, avg=0)
            
            while completed_games < num_games:
                try:
                    payload = puller.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.01)
                    if rust_proc.poll() is not None:
                        print("⚠️ Rust Engine Subprocess Crashed!")
                        break
                    continue
                
                # unpack logic
                diff = struct.unpack("<i", payload[0:4])[0]
                score = struct.unpack("<f", payload[4:8])[0]
                length = struct.unpack("<Q", payload[8:16])[0]
                
                if length == 0:
                    continue

                b_len = struct.unpack("<Q", payload[16:24])[0]
                av_len = struct.unpack("<Q", payload[24:32])[0]
                a_len = struct.unpack("<Q", payload[32:40])[0]
                pid_len = struct.unpack("<Q", payload[40:48])[0]
                r_len = struct.unpack("<Q", payload[48:56])[0]
                pol_len = struct.unpack("<Q", payload[56:64])[0]
                v_len = struct.unpack("<Q", payload[64:72])[0]

                offset = 72
                b_np = np.frombuffer(payload, dtype=np.uint64, count=b_len//8, offset=offset).reshape(length, 2)
                offset += b_len
                av_np = np.frombuffer(payload, dtype=np.int32, count=av_len//4, offset=offset).reshape(length, 3)
                offset += av_len
                a_np = np.frombuffer(payload, dtype=np.int64, count=a_len//8, offset=offset)
                offset += a_len
                pid_np = np.frombuffer(payload, dtype=np.int64, count=pid_len//8, offset=offset)
                offset += pid_len
                r_np = np.frombuffer(payload, dtype=np.float32, count=r_len//4, offset=offset)
                offset += r_len
                pol_np = np.frombuffer(payload, dtype=np.float32, count=pol_len//4, offset=offset).reshape(length, 288)
                offset += pol_len
                v_np = np.frombuffer(payload, dtype=np.float32, count=v_len//4, offset=offset)
                
                from tricked.training.buffer import EpisodeMeta
                
                # lock-free writes 
                g_start = buffer.global_write_idx.value
                buffer.global_write_idx.value += length
                cap = buffer.capacity
                s_mod = g_start % cap
                e_mod = (g_start + length) % cap
                
                if s_mod < e_mod:
                    buffer.boards[s_mod:e_mod] = b_np
                    buffer.available[s_mod:e_mod] = av_np
                    buffer.actions[s_mod:e_mod] = a_np
                    buffer.piece_ids[s_mod:e_mod] = pid_np
                    buffer.rewards[s_mod:e_mod] = r_np
                    buffer.policies[s_mod:e_mod] = pol_np
                    buffer.values[s_mod:e_mod] = v_np
                else:
                    p1 = cap - s_mod
                    buffer.boards[s_mod:] = b_np[:p1]
                    buffer.boards[:e_mod] = b_np[p1:]
                    buffer.available[s_mod:] = av_np[:p1]
                    buffer.available[:e_mod] = av_np[p1:]
                    buffer.actions[s_mod:] = a_np[:p1]
                    buffer.actions[:e_mod] = a_np[p1:]
                    buffer.piece_ids[s_mod:] = pid_np[:p1]
                    buffer.piece_ids[:e_mod] = pid_np[p1:]
                    buffer.rewards[s_mod:] = r_np[:p1]
                    buffer.rewards[:e_mod] = r_np[p1:]
                    buffer.policies[s_mod:] = pol_np[:p1]
                    buffer.policies[:e_mod] = pol_np[p1:]
                    buffer.values[s_mod:] = v_np[:p1]
                    buffer.values[:e_mod] = v_np[p1:]
                
                ep_meta = EpisodeMeta(g_start, length, diff, score)
                results.append((ep_meta, score))
                running_scores.append(score)
                completed_games += 1

                curr_med = float(np.median(running_scores))
                curr_max = float(max(running_scores))
                curr_mean = float(np.mean(running_scores))

                try:
                    from tricked.training.redis_logger import update_training_status
                    update_training_status({
                        "stage": f"Simulating Agent Self-Play ({completed_games}/{num_games})",
                        "completed_games": completed_games,
                        "num_games": num_games,
                        "median_score": curr_med,
                        "max_score": curr_max
                    })
                except Exception:
                    pass

                progress.update(task1, advance=1, med=curr_med, max=curr_max, avg=curr_mean)

    except KeyboardInterrupt:
        print("\nInterrupting Self-Play loop...")
    finally:
        puller.close()
        context.term()
        if rust_proc.poll() is None:
            rust_proc.terminate()
            rust_proc.wait()

    scores = [res[1] for res in results]
    for episode_meta, _ in results:
        buffer.push_game(episode_meta)

    return buffer, scores

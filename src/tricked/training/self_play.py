"""
Standard Documentation for self_play.py.

This module supplies the core execution logic for the `training` namespace, heavily typed and tested for production distribution.
"""

from typing import Any

import numpy as np
import torch.multiprocessing as mp

from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer
from tricked.training.simulator import init_worker, play_one_game_worker


def self_play(
    model: MuZeroNet, buffer: ReplayBuffer, hw_config: Any
) -> tuple[ReplayBuffer, list[float]]:

    from tricked.training.redis_logger import init_db

    init_db()

    context = mp.get_context("spawn")
    num_games = hw_config["num_games"]

    import torch

    m_script = torch.jit.script(model.cpu())
    m_opt = torch.jit.optimize_for_inference(m_script)
    m_opt.save("model_temp.pt")
    
    # We must move the model back to the original device 
    try:
        model.to(hw_config.device)
    except Exception:
        pass

    args = [(i, hw_config) for i in range(num_games)]

    results = []

    num_processes = hw_config["num_processes"]
    print(
        f"Spawning {num_processes} concurrent workers targeting '{hw_config['worker_device'].type}' for {num_games} games..."
    )

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
            task1 = progress.add_task("Self-Play Generation", total=num_games, med=0, max=0, avg=0)
            
            with context.Pool(processes=num_processes, initializer=init_worker, initargs=(hw_config, buffer.capacity, buffer.global_write_idx, buffer.write_lock)) as pool:
                
                for episode_meta, final_score in pool.imap_unordered(play_one_game_worker, args):
                    if episode_meta.length > 0:
                        results.append((episode_meta, final_score))
                        running_scores.append(final_score)
                    completed_games += 1
    
                    if len(running_scores) > 0:
                        curr_med = float(np.median(running_scores))
                        curr_max = float(max(running_scores))
                        curr_mean = float(np.mean(running_scores))
                    else:
                        curr_med = curr_max = curr_mean = 0.0  
    
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
    except RuntimeError as e:
        print(f"\nMultiprocessing error: {e}")
        return buffer, []
    finally:
        pass

    scores = [res[1] for res in results]

    for episode_meta, _ in results:
        buffer.push_game(episode_meta)

    return buffer, scores

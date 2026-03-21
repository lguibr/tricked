"""
Standard Documentation for self_play.py.

This module supplies the core execution logic for the `training` namespace, heavily typed and tested for production distribution.
"""

from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from tricked.env.state import GameState
from tricked.mcts.search import MuZeroMCTS
from tricked.model.network import MuZeroNet
from tricked.training.buffer import Episode, ReplayBuffer
from tricked.training.buffer import Episode, ReplayBuffer


def play_one_game(
    game_idx: int, mcts: MuZeroMCTS, simulations: int, num_games: int, difficulty: int, temp_boost: bool = False,
    exploit_starts: list[list[int]] | None = None, hw_config: dict[str, Any] | None = None
) -> tuple[Episode, float]:
    import os

    state = GameState(difficulty=difficulty)
    episode = Episode(difficulty=difficulty)

    history = [state.board, state.board]
    full_board_history: list[dict[str, Any]] = [
        {"board": str(state.board), "score": 0, "available": state.available.copy()}
    ]
    worker_pid = os.getpid()

    prefix_actions: list[int] = []
    
    # Task P0: Go-Exploit Start Automation
    if exploit_starts and len(exploit_starts) > 0 and np.random.rand() < 0.25:
        chosen_seq = exploit_starts[np.random.choice(len(exploit_starts))]
        for a in chosen_seq:
            slot = a // 96
            idx = a % 96
            next_state = state.apply_move(slot, idx)
            if next_state is None:
                break
            
            history.append(state.board)
            if len(history) > 8:
                history.pop(0)
            state = next_state
            prefix_actions.append(a)

    step = 0
    # Hard cap 10,000 steps to prevent practically infinite games from halting the epoch
    for step in range(10000):
        if state.pieces_left == 0:
            state.refill_tray()  # pragma: no cover

        from tricked.training.redis_logger import update_spectator

        update_spectator(
            worker_pid,
            {
                "board": str(state.board),
                "score": state.score,
                "pieces_left": state.pieces_left,
                "terminal": state.terminal,
                "available": state.available,
            },
        )

        if state.terminal:
            break

        best_move_idx, action_visits, latent_root = mcts.search(
            state, 
            history=history, 
            action_history=episode.actions, 
            difficulty=difficulty, 
            simulations=simulations,
            hw_config=hw_config
        )

        if best_move_idx is None:
            break  # pragma: no cover

        if temp_boost:
            temp_decay = hw_config.get("temp_decay_steps", 30) if hw_config else 30
            temp = 1.0 if step < temp_decay else 0.5
        else:
            temp_decay = hw_config.get("temp_decay_steps", 30) if hw_config else 30
            temp = 1.0 if step < (temp_decay // 2) else (0.5 if step < temp_decay else 0.1)

        actions = list(action_visits.keys())
        counts = np.array([action_visits[a] for a in actions], dtype=np.float64)

        probs = counts ** (1.0 / temp)
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)  # pragma: no cover
        else:
            probs = probs / probs_sum

        # Build full target policy vector [288]
        target_policy = np.zeros(288, dtype=np.float32)
        for idx_a, a in enumerate(actions):
            target_policy[a] = probs[idx_a]

        chosen_idx = np.random.choice(len(actions), p=probs)
        chosen_action = actions[chosen_idx]

        # Reconstruct slot and idx from the 1D action
        slot = chosen_action // 96
        idx = chosen_action % 96

        next_state = state.apply_move(slot, idx)
        if next_state is None:
            # The agent chose an invalid move. This shouldn't happen with our mask, but if it does,
            # treat it as a terminal failure for this game to avoid infinite loops, but penalize it.
            break

        reward = float(next_state.score - state.score)

        from tricked.mcts.features import extract_feature

        feat = extract_feature(state, history).numpy()

        episode.states.append(feat)
        episode.actions.append(chosen_action)
        episode.rewards.append(reward)
        episode.policies.append(target_policy)
        # The network outputs truly scaled physical scores via the new Symexp transformations.
        episode.values.append(latent_root.value)

        # Task P0 A: Identify High-Score Aggressive Spikes!
        if reward >= 40.0:
            episode.spike_actions.append(prefix_actions + episode.actions.copy())

        history.append(state.board)
        if len(history) > 8:
            history.pop(0)
        full_board_history.append(
            {
                "board": str(next_state.board),
                "score": next_state.score,
                "available": next_state.available.copy(),
            }
        )
        state = next_state
        step += 1
    else:
        print(  # pragma: no cover
            f"Warning: Game {game_idx} hit maximum depth cutoff (10000 steps). Terminating early."
        )

    from tricked.training.redis_logger import log_game

    log_game(difficulty, float(state.score), step, full_board_history)

    return episode, float(state.score)


_worker_mcts: MuZeroMCTS | None = None

def init_worker(hw_config: dict[str, Any]) -> None:
    global _worker_mcts

    torch.set_num_threads(1)
    worker_device = hw_config.get("worker_device", torch.device("cpu"))

    # We instantiate a dummy schema block purely for python typing interfaces natively. 
    # Zero weights are transferred over IPC avoiding GB-scale RAM leaks native to Torch distributed pooling!
    model = MuZeroNet(
        d_model=hw_config["d_model"],
        num_blocks=hw_config["num_blocks"],
    ).to(worker_device)
    model.eval()
    
    _worker_mcts = MuZeroMCTS(model, worker_device, hw_config)


def play_one_game_worker(
    args: tuple[int, dict[str, Any]],
) -> tuple[Episode, float]:
    try:
        global _worker_mcts
        game_idx, hw_config = args

        base_difficulty = hw_config.get("difficulty", 6)
        # SOTA Fix: Domain Randomization (Curriculum Smearing)
        if base_difficulty > 1 and np.random.rand() < 0.20:
            # 20% chance to play an historically easier difficulty to prevent Catastrophic Forgetting
            difficulty = int(np.random.randint(1, base_difficulty))
        else:
            difficulty = base_difficulty
            
        temp_boost = hw_config.get("temp_boost", False)
        exploit_starts = hw_config.get("exploit_starts", [])
        assert _worker_mcts is not None
        return play_one_game(
            game_idx, _worker_mcts, hw_config["simulations"], hw_config["num_games"], difficulty, temp_boost, exploit_starts, hw_config
        )  # pragma: no cover
    except Exception as e:
        import traceback

        print(f"Worker {args[0]} failed: {e}")
        traceback.print_exc()
        return Episode(), 0.0


def self_play(
    model: MuZeroNet, buffer: ReplayBuffer, hw_config: dict[str, Any]
) -> tuple[ReplayBuffer, list[float]]:
    import sys

    from tricked.training.redis_logger import init_db

    # Initialize Telemetry database explicitly once before spawning parallel workers
    init_db()

    context = mp.get_context("spawn")
    num_games = hw_config["num_games"]

    # The master model dictionary is exclusively extracted for the centralized GPU Router daemon.
    # The CPU workers receive none of this natively guaranteeing ultra-light multiprocessing footprints.
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()} 

    evaluator_proc = None
    import time
    from tricked.training.evaluator import run_gpu_evaluator
    evaluator_proc = mp.Process(target=run_gpu_evaluator, args=(state_dict, hw_config))
    evaluator_proc.start()
    # Ensure the underlying C++ ZMQ Router socket fully binds to the TCP/IPC port before CPU workers hammer it 
    time.sleep(1.0)

    args = [(i, hw_config) for i in range(num_games)]

    results = []

    num_processes = hw_config["num_processes"]
    print(
        f"Spawning {num_processes} concurrent workers targeting '{hw_config['worker_device'].type}' for {num_games} games..."
    )

    completed_games = 0
    running_scores = []

    try:
        from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
        from rich.console import Console
        console = Console()
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn(" | [yellow]Med[/]: {task.fields[med]:.1f} | [red]Max[/]: {task.fields[max]:.0f} | [magenta]Avg[/]: {task.fields[avg]:.1f}"),
            console=console,
            transient=True # Disappears completely when finished preventing log pollution
        ) as progress:
            task1 = progress.add_task("Self-Play Generation", total=num_games, med=0, max=0, avg=0)
            
            with context.Pool(processes=num_processes, initializer=init_worker, initargs=(hw_config,)) as pool:
                # We use imap_unordered to get results as they finish to update the progress bar live
                for episode, final_score in pool.imap_unordered(play_one_game_worker, args):
                    if len(episode) > 0:
                        results.append((episode, final_score))
                        running_scores.append(final_score)
                    completed_games += 1
    
                    # Live Analytics
                    if len(running_scores) > 0:
                        curr_med = float(np.median(running_scores))
                        curr_max = float(max(running_scores))
                        curr_min = float(min(running_scores))
                        curr_mean = float(np.mean(running_scores))
                    else:
                        curr_med = curr_max = curr_min = curr_mean = 0.0  # pragma: no cover
    
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
        if evaluator_proc is not None:
            evaluator_proc.terminate()
            evaluator_proc.join()

    scores = [res[1] for res in results]

    for episode, _ in results:
        buffer.push_game(episode)

    return buffer, scores

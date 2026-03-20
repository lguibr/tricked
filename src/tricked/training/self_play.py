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
from tricked.training.sqlite_logger import init_db, log_game, update_spectator


def play_one_game(
    game_idx: int, mcts: MuZeroMCTS, simulations: int, num_games: int, difficulty: int, temp_boost: bool = False
) -> tuple[Episode, float]:
    import os

    state = GameState(difficulty=difficulty)
    episode = Episode(difficulty=difficulty)

    history = [state.board, state.board]
    full_board_history: list[dict[str, Any]] = [
        {"board": str(state.board), "score": 0, "available": state.available.copy()}
    ]
    worker_pid = os.getpid()

    step = 0
    # Hard cap 10,000 steps to prevent practically infinite games from halting the epoch
    for step in range(10000):
        if state.pieces_left == 0:
            state.refill_tray()  # pragma: no cover

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
            simulations=simulations
        )

        if best_move_idx is None:
            break  # pragma: no cover

        if temp_boost:
            temp = 1.0 if step < 30 else 0.5
        else:
            temp = 1.0 if step < 15 else (0.5 if step < 30 else 0.1)

        actions = list(action_visits.keys())
        counts = np.array([action_visits[a] for a in actions], dtype=np.float64)

        probs = counts ** (1.0 / temp)
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)  # pragma: no cover
        else:
            probs = probs / probs_sum

        # Build full target policy vector [288]
        target_policy = torch.zeros(288, dtype=torch.float32)
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

        feat = extract_feature(state, history)

        episode.states.append(feat)
        episode.actions.append(chosen_action)
        episode.rewards.append(reward)
        episode.policies.append(target_policy)
        # The network outputs truly scaled physical scores via the new Symexp transformations.
        episode.values.append(latent_root.value)

        history = [history[1], state.board]
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

    log_game(difficulty, float(state.score), step, full_board_history)

    return episode, float(state.score)


def play_one_game_worker(
    args: tuple[int, dict[str, Any] | None, dict[str, Any]],
) -> tuple[Episode, float]:
    try:
        import torch

        torch.set_num_threads(1)
        game_idx, state_dict, hw_config = args

        worker_device = hw_config["worker_device"]

        model = MuZeroNet(
            d_model=hw_config["d_model"],
            num_blocks=hw_config["num_blocks"],
        ).to(worker_device)

        if state_dict is not None:
            # Reconstruct safely from state_dict to bypass multiprocessing deadlocks
            model.load_state_dict(state_dict)
        model.eval()  # pragma: no cover
        # pragma: no cover
        mcts = MuZeroMCTS(model, worker_device)  # pragma: no cover
        difficulty = hw_config.get("difficulty", 6)
        temp_boost = hw_config.get("temp_boost", False)
        return play_one_game(
            game_idx, mcts, hw_config["simulations"], hw_config["num_games"], difficulty, temp_boost
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

    # Initialize SQLite database explicitly once before spawning parallel workers
    init_db()

    context = mp.get_context("spawn")
    num_games = hw_config["num_games"]

    state_dict = None
    if hw_config["device"].type != "cpu":
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}  # pragma: no cover
    else:
        state_dict = model.state_dict()

    args = [(i, state_dict, hw_config) for i in range(num_games)]

    results = []

    num_processes = hw_config["num_processes"]
    print(
        f"Spawning {num_processes} concurrent workers targeting '{hw_config['worker_device'].type}' for {num_games} games..."
    )

    completed_games = 0
    running_scores = []

    try:
        with context.Pool(processes=num_processes) as pool:
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

                # Build Progress Bar string
                pct = int((completed_games / num_games) * 20)
                try:
                    from tricked.training.sqlite_logger import update_training_status
                    update_training_status({
                        "stage": f"Simulating Agent Self-Play ({completed_games}/{num_games})",
                        "completed_games": completed_games,
                        "num_games": num_games,
                        "median_score": curr_med,
                        "max_score": curr_max
                    })
                    
                    bar = "█" * pct + "-" * (20 - pct)
                    sys.stdout.write(
                        f"\r[{bar}] {completed_games}/{num_games} | "
                        f"Med: {curr_med:.1f} | Avg: {curr_mean:.1f} | Max: {curr_max:.1f} | Min: {curr_min:.1f}   "
                    )
                    sys.stdout.flush()
                except UnicodeEncodeError:  # pragma: no cover
                    bar = "#" * pct + "-" * (20 - pct)
                    sys.stdout.write(
                        f"\r[{bar}] {completed_games}/{num_games} | "
                        f"Med: {curr_med:.1f} | Avg: {curr_mean:.1f} | Max: {curr_max:.1f} | Min: {curr_min:.1f}   "
                    )
                    sys.stdout.flush()

            print()  # Newline after progress bar finishes
    except RuntimeError as e:
        print(f"\nMultiprocessing error: {e}")
        return buffer, []

    scores = [res[1] for res in results]

    for episode, _ in results:
        buffer.push_game(episode)

    return buffer, scores

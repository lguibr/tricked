import os
from typing import Any

import numpy as np
import torch
from tricked_engine import GameStateExt as GameState

from tricked.mcts.search import MuZeroMCTS
from tricked.model.network import MuZeroNet


def play_one_game(
    game_idx: int, mcts: MuZeroMCTS, simulations: int, num_games: int, difficulty: int, temp_boost: bool = False,
    exploit_starts: list[list[int]] | None = None, hw_config: dict[str, Any] | None = None
) -> tuple[Any, float]:
    state = GameState(difficulty=difficulty)

    history = [state.board, state.board]
    full_board_history: list[dict[str, Any]] = [
        {"board": str(state.board), "score": 0, "available": state.available.copy()}
    ]
    worker_pid = os.getpid()

    prefix_actions: list[int] = []
    prefix_piece_ids: list[int] = []
    
    if exploit_starts and len(exploit_starts) > 0 and np.random.rand() < 0.25:
        chosen_seq = exploit_starts[np.random.choice(len(exploit_starts))]
        for a in chosen_seq:
            slot = a // 96
            idx = a % 96
            prefix_piece_ids.append(state.available[slot])
            next_state = state.apply_move(slot, idx)
            if next_state is None:
                break
            
            history.append(state.board)
            if len(history) > 8:
                history.pop(0)
            state = next_state
            prefix_actions.append(a)

    step = 0
    for step in range(10000):
        if state.pieces_left == 0:
            state.refill_tray()  

        from tricked.mcts.features import extract_feature
        from tricked.training.redis_logger import update_spectator

        feat = extract_feature(state, history)

        best_move_idx, action_visits, latent_root = mcts.search(
            state, 
            history=history, 
            action_history=prefix_actions, 
            difficulty=difficulty, 
            simulations=simulations,
            hw_config=hw_config
        )

        if best_move_idx is None:
            break  

        top_moves = [{"action": int(a), "visits": int(v)} for a, v in action_visits.items()]
        top_moves = sorted(top_moves, key=lambda x: x["visits"], reverse=True)[:5]

        update_spectator(
            worker_pid,
            {
                "board": str(state.board),
                "score": state.score,
                "pieces_left": state.pieces_left,
                "terminal": state.terminal,
                "available": state.available,
                "hole_logits": feat[19].tolist(),
                "mcts_mind": top_moves
            },
        )

        if state.terminal:
            break

        temp_decay = hw_config.get("temp_decay_steps", 30) if hw_config else 30
        if temp_boost:
            temp = 1.0 if step < temp_decay else 0.5
        else:
            temp = 1.0 if step < (temp_decay // 2) else (0.5 if step < temp_decay else 0.1)

        actions = list(action_visits.keys())
        counts = np.array([action_visits[a] for a in actions], dtype=np.float64)

        probs = counts ** (1.0 / temp)
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)  
        else:
            probs = probs / probs_sum

        target_policy = np.zeros(288, dtype=np.float32)
        for idx_a, a in enumerate(actions):
            target_policy[a] = probs[idx_a]

        chosen_idx = np.random.choice(len(actions), p=probs)
        chosen_action = actions[chosen_idx]

        slot = chosen_action // 96
        idx = chosen_action % 96

        next_state = state.apply_move(slot, idx)
        if next_state is None:
            break

        reward = float(next_state.score - state.score)

        from tricked.mcts.features import extract_feature
        feat_np = extract_feature(state, history).numpy()

        piece_id = state.available[slot]
        if piece_id == -1:
            piece_id = 0
        piece_action = piece_id * 96 + idx

        prefix_actions.append(piece_action)
        prefix_piece_ids.append(int(piece_id))
        
        if 'states' not in locals():
            states, actions, p_ids, rewards, policies, values = [], [], [], [], [], []
            
        states.append(feat_np)
        actions.append(piece_action)
        p_ids.append(int(piece_id))
        rewards.append(reward)
        policies.append(target_policy)
        values.append(latent_root.value)

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
        print(f"Warning: Game {game_idx} hit maximum depth cutoff (10000 steps). Terminating early.")  

    from tricked.training.redis_logger import log_game
    log_game(difficulty, float(state.score), step, full_board_history)

    length = len(states) if 'states' in locals() else 0
    from tricked.training.buffer import EpisodeMeta
    if length == 0:
        return EpisodeMeta(0, 0, difficulty, 0.0), 0.0

    s_np = np.stack(states)
    a_np = np.array(actions, dtype=np.int64)
    p_id_np = np.array(p_ids, dtype=np.int64)
    r_np = np.array(rewards, dtype=np.float32)
    p_np = np.stack(policies)
    v_np = np.array(values, dtype=np.float32)

    global _worker_shm_map
    lock = _worker_shm_map["write_lock"]
    global_idx = _worker_shm_map["global_write_idx"]
    capacity = _worker_shm_map["capacity"]

    with lock:
        g_start = global_idx.value
        global_idx.value += length

    start_mod = g_start % capacity
    end_mod = (g_start + length) % capacity

    if start_mod < end_mod:
        _worker_shm_map["states_arr"][start_mod:end_mod] = s_np
        _worker_shm_map["actions_arr"][start_mod:end_mod] = a_np
        _worker_shm_map["piece_ids_arr"][start_mod:end_mod] = p_id_np
        _worker_shm_map["rewards_arr"][start_mod:end_mod] = r_np
        _worker_shm_map["policies_arr"][start_mod:end_mod] = p_np
        _worker_shm_map["values_arr"][start_mod:end_mod] = v_np
    else:
        part1 = capacity - start_mod
        _worker_shm_map["states_arr"][start_mod:] = s_np[:part1]
        _worker_shm_map["states_arr"][:end_mod] = s_np[part1:]
        _worker_shm_map["actions_arr"][start_mod:] = a_np[:part1]
        _worker_shm_map["actions_arr"][:end_mod] = a_np[part1:]
        _worker_shm_map["piece_ids_arr"][start_mod:] = p_id_np[:part1]
        _worker_shm_map["piece_ids_arr"][:end_mod] = p_id_np[part1:]
        _worker_shm_map["rewards_arr"][start_mod:] = r_np[:part1]
        _worker_shm_map["rewards_arr"][:end_mod] = r_np[part1:]
        _worker_shm_map["policies_arr"][start_mod:] = p_np[:part1]
        _worker_shm_map["policies_arr"][:end_mod] = p_np[part1:]
        _worker_shm_map["values_arr"][start_mod:] = v_np[:part1]
        _worker_shm_map["values_arr"][:end_mod] = v_np[part1:]

    return EpisodeMeta(g_start, length, difficulty, float(state.score)), float(state.score)

_worker_mcts: MuZeroMCTS | None = None
_worker_shm_map: dict[str, Any] = {}

def init_worker(hw_config: dict[str, Any], capacity: int = 200000, global_write_idx: Any = None, write_lock: Any = None) -> None:
    global _worker_mcts, _worker_shm_map

    import multiprocessing.shared_memory as shm
    _worker_shm_map = {
        "capacity": capacity,
        "global_write_idx": global_write_idx,
        "write_lock": write_lock,
    }
    
    if global_write_idx is not None:
        _worker_shm_map["shm_states"] = shm.SharedMemory(name="tricked_states")
        _worker_shm_map["states_arr"] = np.ndarray((capacity, 20, 96), dtype=np.float32, buffer=_worker_shm_map["shm_states"].buf)
        _worker_shm_map["shm_actions"] = shm.SharedMemory(name="tricked_actions")
        _worker_shm_map["actions_arr"] = np.ndarray((capacity,), dtype=np.int64, buffer=_worker_shm_map["shm_actions"].buf)
        _worker_shm_map["shm_piece_ids"] = shm.SharedMemory(name="tricked_piece_ids")
        _worker_shm_map["piece_ids_arr"] = np.ndarray((capacity,), dtype=np.int64, buffer=_worker_shm_map["shm_piece_ids"].buf)
        _worker_shm_map["shm_rewards"] = shm.SharedMemory(name="tricked_rewards")
        _worker_shm_map["rewards_arr"] = np.ndarray((capacity,), dtype=np.float32, buffer=_worker_shm_map["shm_rewards"].buf)
        _worker_shm_map["shm_policies"] = shm.SharedMemory(name="tricked_policies")
        _worker_shm_map["policies_arr"] = np.ndarray((capacity, 288), dtype=np.float32, buffer=_worker_shm_map["shm_policies"].buf)
        _worker_shm_map["shm_values"] = shm.SharedMemory(name="tricked_values")
        _worker_shm_map["values_arr"] = np.ndarray((capacity,), dtype=np.float32, buffer=_worker_shm_map["shm_values"].buf)

    torch.set_num_threads(1)
    worker_device = hw_config.get("worker_device", torch.device("cpu"))

    model = MuZeroNet(
        d_model=hw_config["d_model"],
        num_blocks=hw_config["num_blocks"],
    ).to(worker_device)
    model.eval()
    
    _worker_mcts = MuZeroMCTS(model, worker_device, hw_config)

def play_one_game_worker(
    args: tuple[int, dict[str, Any]],
) -> tuple[Any, float]:
    try:
        global _worker_mcts
        game_idx, hw_config = args

        base_difficulty = hw_config.get("difficulty", 6)
        if base_difficulty > 1 and np.random.rand() < 0.20:
            difficulty = int(np.random.randint(1, base_difficulty))
        else:
            difficulty = base_difficulty
            
        temp_boost = hw_config.get("temp_boost", False)
        exploit_starts = hw_config.get("exploit_starts", [])
        assert _worker_mcts is not None
        return play_one_game(
            game_idx, _worker_mcts, hw_config["simulations"], 
            hw_config["num_games"], difficulty, temp_boost, exploit_starts, hw_config
        )  
    except Exception as e:
        import traceback
        print(f"Worker {args[0]} failed: {e}")
        traceback.print_exc()
        return None, 0.0

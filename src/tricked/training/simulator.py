import os
import time
from typing import Any

import numpy as np
import torch
from tricked_engine import GameStateExt as GameState

from tricked.mcts.search import MuZeroMCTS
from tricked.model.network import MuZeroNet


class WorkerState:
    def __init__(self, capacity: int, global_write_idx: Any = None, write_lock: Any = None):
        self.capacity = capacity
        self.global_write_idx = global_write_idx
        self.write_lock = write_lock
        
        if global_write_idx is not None:
            import multiprocessing.shared_memory as shm
            self.shm_states = shm.SharedMemory(name="tricked_states")
            self.states_arr = np.ndarray((capacity, 20, 96), dtype=np.float32, buffer=self.shm_states.buf)
            self.shm_actions = shm.SharedMemory(name="tricked_actions")
            self.actions_arr = np.ndarray((capacity,), dtype=np.int64, buffer=self.shm_actions.buf)
            self.shm_piece_ids = shm.SharedMemory(name="tricked_piece_ids")
            self.piece_ids_arr = np.ndarray((capacity,), dtype=np.int64, buffer=self.shm_piece_ids.buf)
            self.shm_rewards = shm.SharedMemory(name="tricked_rewards")
            self.rewards_arr = np.ndarray((capacity,), dtype=np.float32, buffer=self.shm_rewards.buf)
            self.shm_policies = shm.SharedMemory(name="tricked_policies")
            self.policies_arr = np.ndarray((capacity, 288), dtype=np.float32, buffer=self.shm_policies.buf)
            self.shm_values = shm.SharedMemory(name="tricked_values")
            self.values_arr = np.ndarray((capacity,), dtype=np.float32, buffer=self.shm_values.buf)

def play_one_game(
    game_idx: int, mcts: MuZeroMCTS, simulations: int, num_games: int, difficulty: int, temp_boost: bool = False,
    exploit_starts: list[list[int]] | None = None, hw_config: Any = None
) -> tuple[Any, float]:
    mcts.tree = None
    last_action_taken = None
    state = GameState(difficulty=difficulty)

    history = [state.board, state.board]
    full_board_history: list[dict[str, Any]] = [
        {"board": str(state.board), "score": 0, "available": state.available.copy()}
    ]
    worker_pid = os.getpid()

    prefix_actions: list[int] = []
    prefix_piece_ids: list[int] = []
    last_spectator_update = 0.0
    
    ep_states: list[np.ndarray] = []
    ep_actions: list[int] = []
    ep_p_ids: list[int] = []
    ep_rewards: list[float] = []
    ep_policies: list[np.ndarray] = []
    ep_values: list[float] = []
    
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

        from tricked_engine import extract_feature

        from tricked.training.redis_logger import update_spectator

        feat_list = extract_feature(state, history, prefix_actions, difficulty)
        feat = torch.tensor(feat_list, dtype=torch.float32).reshape(20, 96)

        best_move_idx, action_visits, latent_root = mcts.search(
            state, 
            history=history, 
            action_history=prefix_actions, 
            difficulty=difficulty, 
            simulations=simulations,
            hw_config=hw_config,
            last_action=last_action_taken
        )

        if best_move_idx is None:
            break  

        top_moves = [{"action": int(a), "visits": int(v)} for a, v in action_visits.items()]
        top_moves = sorted(top_moves, key=lambda x: x["visits"], reverse=True)[:5]

        if time.time() - last_spectator_update > 2.0:
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
            last_spectator_update = time.time()

        if state.terminal:
            break

        temp_decay = hw_config.temp_decay_steps if hw_config else 30
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
        last_action_taken = chosen_action

        slot = chosen_action // 96
        idx = chosen_action % 96

        next_state = state.apply_move(slot, idx)
        if next_state is None:
            break
        
        if next_state.pieces_left == 3:
            mcts.tree = None

        reward = float(next_state.score - state.score)

        from tricked_engine import extract_feature
        feat_list2 = extract_feature(state, history, prefix_actions, difficulty)
        feat_np = np.array(feat_list2, dtype=np.float32).reshape(20, 96)

        piece_id = state.available[slot]
        if piece_id == -1:
            piece_id = 0
        piece_action = piece_id * 96 + idx

        prefix_actions.append(piece_action)
        prefix_piece_ids.append(int(piece_id))
        
        ep_states.append(feat_np)
        ep_actions.append(piece_action)
        ep_p_ids.append(int(piece_id))
        ep_rewards.append(reward)
        ep_policies.append(target_policy)
        ep_values.append(latent_root.value)

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

    length = len(ep_states)
    from tricked.training.buffer import EpisodeMeta
    if length == 0:
        return EpisodeMeta(0, 0, difficulty, 0.0), 0.0

    s_np = np.stack(ep_states)
    a_np = np.array(ep_actions, dtype=np.int64)
    p_id_np = np.array(ep_p_ids, dtype=np.int64)
    r_np = np.array(ep_rewards, dtype=np.float32)
    p_np = np.stack(ep_policies)
    v_np = np.array(ep_values, dtype=np.float32)

    global _worker_state
    if _worker_state is None:
        return EpisodeMeta(0, 0, difficulty, 0.0), 0.0

    lock = _worker_state.write_lock
    global_idx = _worker_state.global_write_idx
    capacity = _worker_state.capacity

    with lock:
        g_start = global_idx.value
        global_idx.value += length

        start_mod = g_start % capacity
        end_mod = (g_start + length) % capacity

        if start_mod < end_mod:
            _worker_state.states_arr[start_mod:end_mod] = s_np
            _worker_state.actions_arr[start_mod:end_mod] = a_np
            _worker_state.piece_ids_arr[start_mod:end_mod] = p_id_np
            _worker_state.rewards_arr[start_mod:end_mod] = r_np
            _worker_state.policies_arr[start_mod:end_mod] = p_np
            _worker_state.values_arr[start_mod:end_mod] = v_np
        else:
            part1 = capacity - start_mod
            _worker_state.states_arr[start_mod:] = s_np[:part1]
            _worker_state.states_arr[:end_mod] = s_np[part1:]
            _worker_state.actions_arr[start_mod:] = a_np[:part1]
            _worker_state.actions_arr[:end_mod] = a_np[part1:]
            _worker_state.piece_ids_arr[start_mod:] = p_id_np[:part1]
            _worker_state.piece_ids_arr[:end_mod] = p_id_np[part1:]
            _worker_state.rewards_arr[start_mod:] = r_np[:part1]
            _worker_state.rewards_arr[:end_mod] = r_np[part1:]
            _worker_state.policies_arr[start_mod:] = p_np[:part1]
            _worker_state.policies_arr[:end_mod] = p_np[part1:]
            _worker_state.values_arr[start_mod:] = v_np[:part1]
            _worker_state.values_arr[:end_mod] = v_np[part1:]

    return EpisodeMeta(g_start, length, difficulty, float(state.score)), float(state.score)

_worker_mcts: MuZeroMCTS | None = None
_worker_state: WorkerState | None = None

def init_worker(hw_config: Any, capacity: int = 200000, global_write_idx: Any = None, write_lock: Any = None) -> None:
    global _worker_mcts, _worker_state

    _worker_state = WorkerState(capacity, global_write_idx, write_lock)

    torch.set_num_threads(1)
    worker_device = torch.device(hw_config.worker_device)

    model = MuZeroNet(
        d_model=hw_config.d_model,
        num_blocks=hw_config.num_blocks,
    ).to(worker_device)
    model.eval()
    
    import tricked_engine
    try:
        tricked_engine.init_model(hw_config.model_checkpoint + "_jit.pt")
    except Exception as e:
        print(f"Failed to init Rust LibTorch model: {e}")
    
    _worker_mcts = MuZeroMCTS(model, worker_device, hw_config)

def play_one_game_worker(
    args: tuple[int, Any],
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
        from tricked.training.buffer import EpisodeMeta
        return EpisodeMeta(0, 0, 0, 0.0), 0.0

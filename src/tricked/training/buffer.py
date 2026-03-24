import multiprocessing as mp
import multiprocessing.shared_memory as shm
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeMeta:
    def __init__(self, global_start_idx: int, length: int, difficulty: int, score: float):
        self.global_start_idx = global_start_idx
        self.length = length
        self.difficulty = difficulty
        self.score = score

class SegmentTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree_cap = 1
        while self.tree_cap < capacity:
            self.tree_cap *= 2
        self.tree = np.zeros(2 * self.tree_cap, dtype=np.float32)

    def update(self, idx: int, p: float) -> None:
        tree_idx = idx + self.tree_cap
        change = p - self.tree[tree_idx]
        while tree_idx > 0:
            self.tree[tree_idx] += change
            tree_idx //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def get_leaf(self, v: float) -> tuple[int, float]:
        idx = 1
        while idx < self.tree_cap:
            left = 2 * idx
            if v <= self.tree[left]:
                idx = left
            else:
                v -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.tree_cap
        return data_idx, self.tree[idx]

class ReplayBuffer(Dataset[tuple[Any, ...]]):
    def __init__(
        self, capacity: int, unroll_steps: int = 5, td_steps: int = 10, elite_ratio: float = 0.1
    ):
        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.elite_ratio = elite_ratio

        self.episodes: list[EpisodeMeta] = []
        self.num_states = 0
        
        self.global_write_idx = mp.Value('Q', 0) 
        self.write_lock = mp.Lock()

        def _alloc(name: str, size: int) -> shm.SharedMemory:
            try:
                s = shm.SharedMemory(name=name, create=True, size=size)
            except FileExistsError:
                s = shm.SharedMemory(name=name)
                s.unlink()
                s = shm.SharedMemory(name=name, create=True, size=size)
            return s

        self.shm_states = _alloc('tricked_states', self.capacity * 20 * 96 * 4)
        self.shm_actions = _alloc('tricked_actions', self.capacity * 8)
        self.shm_piece_ids = _alloc('tricked_piece_ids', self.capacity * 8)
        self.shm_rewards = _alloc('tricked_rewards', self.capacity * 4)
        self.shm_policies = _alloc('tricked_policies', self.capacity * 288 * 4)
        self.shm_values = _alloc('tricked_values', self.capacity * 4)

        self.states = np.ndarray((self.capacity, 20, 96), dtype=np.float32, buffer=self.shm_states.buf)
        self.actions = np.ndarray((self.capacity,), dtype=np.int64, buffer=self.shm_actions.buf)
        self.piece_ids = np.ndarray((self.capacity,), dtype=np.int64, buffer=self.shm_piece_ids.buf)
        self.rewards = np.ndarray((self.capacity,), dtype=np.float32, buffer=self.shm_rewards.buf)
        self.policies = np.ndarray((self.capacity, 288), dtype=np.float32, buffer=self.shm_policies.buf)
        self.values = np.ndarray((self.capacity,), dtype=np.float32, buffer=self.shm_values.buf)

        self.state_priorities = np.ones(self.capacity, dtype=np.float32)
        self.sum_tree = SegmentTree(self.capacity)
        self.state_to_ep: list[EpisodeMeta | None] = [None] * self.capacity
        self.current_diff = 1
        self.alpha = 0.6
        self.max_priority = 10.0

    def __del__(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        try:
            self.shm_states.close()
            self.shm_actions.close()
            self.shm_piece_ids.close()
            self.shm_rewards.close()
            self.shm_policies.close()
            self.shm_values.close()
            
            self.shm_states.unlink()
            self.shm_actions.unlink()
            self.shm_piece_ids.unlink()
            self.shm_rewards.unlink()
            self.shm_policies.unlink()
            self.shm_values.unlink()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.num_states

    def make_target(self, global_state_index: int, ep: EpisodeMeta) -> tuple[Any, ...]:
        idx = global_state_index % self.capacity
        ep_end_global = ep.global_start_idx + ep.length
        
        initial_state = self.states[idx].copy()
        actions = []
        piece_ids = []
        rewards = []
        policies = []
        values = []
        mcts_values = []
        target_states = []
        masks = []

        for offset in range(self.unroll_steps + 1):
            curr_global = global_state_index + offset
            curr_idx = curr_global % self.capacity
            
            if curr_global < ep_end_global:
                masks.append(1.0)
                if offset > 0:
                    prev_idx = (curr_global - 1) % self.capacity
                    actions.append(int(self.actions[prev_idx]))
                    piece_ids.append(int(self.piece_ids[prev_idx]))
                    rewards.append(float(self.rewards[prev_idx]))
                    target_states.append(self.states[curr_idx].copy())
                    
                policies.append(self.policies[curr_idx].copy())
                mcts_values.append(float(self.values[curr_idx]))
                
                bootstrap_global = curr_global + self.td_steps
                gamma = 0.99
                val = 0.0
                limit = min(bootstrap_global, ep_end_global)
                
                for i in range(limit - curr_global):
                    r_idx = (curr_global + i) % self.capacity
                    val += float(self.rewards[r_idx]) * (gamma ** i)
                
                if bootstrap_global < ep_end_global:
                    val += float(self.values[bootstrap_global % self.capacity]) * (gamma ** self.td_steps)
                values.append(val)
            else:
                masks.append(0.0)
                if offset > 0:
                    actions.append(0)
                    piece_ids.append(0)
                    rewards.append(0.0)
                    target_states.append(np.zeros_like(initial_state))
                
                dummy_policy = np.ones_like(policies[0]) / len(policies[0]) if len(policies) > 0 else np.ones(288)/288.
                policies.append(dummy_policy)
                values.append(0.0)
                mcts_values.append(0.0)

        return initial_state, actions, piece_ids, rewards, policies, values, mcts_values, target_states, masks

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        if self.sum_tree.total() == 0.0:
            # Fallback if buffer is entirely empty but somehow triggered
            circ_idx = 0
            ep = self.episodes[-1] if len(self.episodes) > 0 else EpisodeMeta(0, 0, 1, 0.0)
            global_state_index = 0
        else:
            v = np.random.uniform(0, self.sum_tree.total())
            circ_idx, p = self.sum_tree.get_leaf(v)
            
            ep_cand = self.state_to_ep[circ_idx]
            if ep_cand is None:
                ep = self.episodes[-1] if len(self.episodes) > 0 else EpisodeMeta(0, 0, 1, 0.0)
                global_state_index = ep.global_start_idx
            else:
                ep = ep_cand
                st = ep.global_start_idx
                offset = (circ_idx - st) % self.capacity
                if offset >= ep.length:
                    offset = 0
                global_state_index = st + offset

        initial_state, actions, piece_ids, rewards, policies, values, mcts_values, target_states, masks = self.make_target(global_state_index, ep)

        return (
            torch.tensor(initial_state, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(piece_ids, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(policies), dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mcts_values, dtype=torch.float32),
            torch.tensor(np.array(target_states), dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor([0, global_state_index], dtype=torch.long),
        )

    def push_game(self, meta: EpisodeMeta) -> None:
        self.episodes.append(meta)
        
        if len(self.episodes) == 1 or meta.difficulty != self.current_diff:
            self.current_diff = meta.difficulty

        diff_penalty = 0.1 ** abs(self.current_diff - meta.difficulty)
        base_p = (self.max_priority ** self.alpha) * diff_penalty

        for i in range(meta.length):
            idx = (meta.global_start_idx + i) % self.capacity
            self.state_to_ep[idx] = meta
            self.sum_tree.update(idx, base_p)
            self.state_priorities[idx] = self.max_priority

        latest_global = meta.global_start_idx + meta.length
        valid_eps = []
        num_valid_states = 0
        for ep in self.episodes:
            if ep.global_start_idx >= latest_global - self.capacity:
                valid_eps.append(ep)
                num_valid_states += ep.length
                
        self.episodes = valid_eps
        self.num_states = min(self.capacity, self.num_states + meta.length)

    def update_priorities(self, indices: torch.Tensor, priorities: np.ndarray[Any, Any]) -> None:
        for i in range(len(indices)):
            global_state_idx = int(indices[i][1].item())
            p = float(priorities[i])

            if p > self.max_priority:
                self.max_priority = p
            if p < 1e-4:
                p = 1e-4

            circ_idx = global_state_idx % self.capacity
            self.state_priorities[circ_idx] = p
            
            ep = self.state_to_ep[circ_idx]
            if ep is not None:
                diff_penalty = 0.1 ** abs(self.current_diff - ep.difficulty)
                self.sum_tree.update(circ_idx, (p ** self.alpha) * diff_penalty)

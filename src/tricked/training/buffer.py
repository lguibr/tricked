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
        self.episode_priorities: list[float] = []
        self.alpha = 0.6
        self.max_priority = 10.0

    def __del__(self) -> None:
        try:
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
        current_diff = self.episodes[-1].difficulty if len(self.episodes) > 0 else 1
        diff_discounts = np.array([0.1 ** abs(current_diff - ep.difficulty) for ep in self.episodes], dtype=np.float32)
        ep_probs = np.array(self.episode_priorities, dtype=np.float32) ** self.alpha
        ep_probs *= diff_discounts

        ep_sum = ep_probs.sum()
        if ep_sum > 0:
            ep_probs /= ep_sum
        else:
            ep_probs = np.ones_like(ep_probs) / len(ep_probs)  

        ep_idx = int(np.random.choice(len(self.episodes), p=ep_probs))
        ep = self.episodes[ep_idx]

        ep_start_mod = ep.global_start_idx % self.capacity
        ep_end_mod = (ep.global_start_idx + ep.length) % self.capacity
        if ep_start_mod <= ep_end_mod or ep.length == 0:
            st_probs = self.state_priorities[ep_start_mod:ep_start_mod + ep.length] ** self.alpha
        else:
            st_probs = np.concatenate((self.state_priorities[ep_start_mod:], self.state_priorities[:ep_end_mod])) ** self.alpha
            
        st_sum = st_probs.sum()
        if st_sum > 0:
            st_probs /= st_sum
        else:
            st_probs = np.ones_like(st_probs) / len(st_probs)  

        state_offset = int(np.random.choice(ep.length, p=st_probs))
        global_state_index = ep.global_start_idx + state_offset

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
            torch.tensor([ep_idx, global_state_index], dtype=torch.long),
        )

    def push_game(self, meta: EpisodeMeta) -> None:
        self.episodes.append(meta)
        self.episode_priorities.append(self.max_priority)
        
        start_mod = meta.global_start_idx % self.capacity
        end_mod = (meta.global_start_idx + meta.length) % self.capacity
        if meta.length > 0:
            if start_mod < end_mod:
                self.state_priorities[start_mod:end_mod] = self.max_priority
            else:
                self.state_priorities[start_mod:] = self.max_priority
                self.state_priorities[:end_mod] = self.max_priority

        latest_global = meta.global_start_idx + meta.length
        valid_eps = []
        valid_ep_prios = []
        num_valid_states = 0
        for ep, ep_prio in zip(self.episodes, self.episode_priorities):
            if ep.global_start_idx >= latest_global - self.capacity:
                valid_eps.append(ep)
                valid_ep_prios.append(ep_prio)
                num_valid_states += ep.length
                
        self.episodes = valid_eps
        self.episode_priorities = valid_ep_prios
        self.num_states = num_valid_states

    def update_priorities(self, indices: torch.Tensor, priorities: np.ndarray[Any, Any]) -> None:
        for i in range(len(indices)):
            ep_idx = int(indices[i][0].item())
            global_state_idx = int(indices[i][1].item())
            p = float(priorities[i])

            if p > self.max_priority:
                self.max_priority = p
            if p < 1e-4:
                p = 1e-4

            self.state_priorities[global_state_idx % self.capacity] = p
            
            if ep_idx < len(self.episodes):
                ep = self.episodes[ep_idx]
                start_mod = ep.global_start_idx % self.capacity
                end_mod = (ep.global_start_idx + ep.length) % self.capacity
                if ep.length > 0:
                    if start_mod < end_mod:
                        mean_prio = np.mean(self.state_priorities[start_mod:end_mod])
                    else:
                        mean_prio = np.mean(np.concatenate((self.state_priorities[start_mod:], self.state_priorities[:end_mod])))
                    self.episode_priorities[ep_idx] = float(mean_prio)

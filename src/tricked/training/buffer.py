"""
Standard Documentation for buffer.py.

This module supplies the core execution logic for the `training` namespace, heavily typed and tested for production distribution.
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class Episode:
    """Stores the chronological sequence of a single game."""

    def __init__(self, difficulty: int = 1) -> None:
        self.difficulty = difficulty
        self.states: list[torch.Tensor] = []  # Root states seen at each step
        self.actions: list[int] = []  # The action chosen at each step
        self.rewards: list[float] = []  # The reward received after each action
        self.policies: list[torch.Tensor] = []  # The MCTS policy at each step
        self.values: list[float] = []  # The MCTS value (or true outcome) at each step

    def __len__(self) -> int:
        return len(self.states)

    def make_target(
        self, state_index: int, unroll_steps: int, td_steps: int
    ) -> tuple[torch.Tensor, list[int], list[float], list[torch.Tensor], list[float], list[torch.Tensor], list[float]]:
        """
        Extracts a sequence of length `unroll_steps` starting from `state_index`.
        Returns:
            initial_state: The real board state at `state_index`.
            actions: The sequence of actions taken.
            rewards: The true rewards observed during the unroll.
            policies: The target policies for each step of the unroll.
            values: The target values (bootstrapped TD-returns or monte-carlo) for each step.
            target_states: The true physical representations for Steps 1..K (used for P1 Alignment).
        """
        initial_state = self.states[state_index]

        actions = []
        rewards = []
        policies = []
        values = []
        target_states = []
        masks = []

        for current_index in range(state_index, state_index + unroll_steps + 1):
            if current_index < len(self):
                masks.append(1.0)
                if current_index > state_index:
                    actions.append(self.actions[current_index - 1])
                    rewards.append(self.rewards[current_index - 1])
                    target_states.append(self.states[current_index])

                policies.append(self.policies[current_index])

                # Task 9: TD(λ) Bootstrapping approximation
                # We use N-step return, but with a precise geometric horizon lock
                bootstrap_index = current_index + td_steps
                gamma = 0.99
                if bootstrap_index < len(self):
                    val = sum(
                        self.rewards[current_index:bootstrap_index]
                    ) + self.values[bootstrap_index] * (gamma**td_steps)
                else:
                    val = sum(self.rewards[current_index:])
                values.append(val)

            else:
                masks.append(0.0)
                if current_index > state_index:
                    actions.append(0)
                    rewards.append(0.0)
                    # Use absolute zero state for out of bounds
                    target_states.append(torch.zeros_like(self.states[0]))

                dummy_policy = torch.ones_like(self.policies[0]) / self.policies[0].numel()
                policies.append(dummy_policy)
                values.append(0.0)

        return initial_state, actions, rewards, policies, values, target_states, masks


class ReplayBuffer(
    Dataset[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]
):
    """
    Stores full Game Episodes instead of independent states, allowing the
    trainer to sample chronological unroll sequences for MuZero BPTT.
    """

    def __init__(
        self, capacity: int, unroll_steps: int = 5, td_steps: int = 10, elite_ratio: float = 0.1
    ):
        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps

        self.episodes: list[Episode] = []
        self.num_states = 0

        self.state_priorities: list[np.ndarray[Any, Any]] = []
        self.episode_priorities: list[float] = []
        self.alpha = 0.6
        self.max_priority = 10.0

    def __len__(self) -> int:
        return self.num_states

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        import numpy as np

        current_diff = self.episodes[-1].difficulty if len(self.episodes) > 0 else 1
        diff_discounts = np.array([0.1 ** abs(current_diff - ep.difficulty) for ep in self.episodes], dtype=np.float32)

        ep_probs = np.array(self.episode_priorities, dtype=np.float32) ** self.alpha
        ep_probs *= diff_discounts

        ep_sum = ep_probs.sum()
        if ep_sum > 0:
            ep_probs /= ep_sum
        else:
            ep_probs = np.ones_like(ep_probs) / len(ep_probs)  # pragma: no cover

        ep_idx = int(np.random.choice(len(self.episodes), p=ep_probs))
        ep = self.episodes[ep_idx]

        st_probs = self.state_priorities[ep_idx] ** self.alpha
        st_sum = st_probs.sum()
        if st_sum > 0:
            st_probs /= st_sum
        else:
            st_probs = np.ones_like(st_probs) / len(st_probs)  # pragma: no cover

        state_idx = int(np.random.choice(len(ep), p=st_probs))

        initial_state, actions, rewards, policies, values, target_states, masks = ep.make_target(
            state_idx, self.unroll_steps, self.td_steps
        )

        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        policies_tensor = torch.stack(policies)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        target_states_tensor = torch.stack(target_states)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)
        indices_tensor = torch.tensor([ep_idx, state_idx], dtype=torch.long)

        return (
            initial_state,
            actions_tensor,
            rewards_tensor,
            policies_tensor,
            values_tensor,
            target_states_tensor,
            masks_tensor,
            indices_tensor,
        )

    def push_game(self, episode: Episode) -> None:
        import numpy as np

        self.episodes.append(episode)

        length = len(episode)
        self.state_priorities.append(np.full(length, self.max_priority, dtype=np.float32))
        self.episode_priorities.append(self.max_priority)

        self.num_states += length

        while self.num_states > self.capacity and len(self.episodes) > 1:
            removed_ep = self.episodes.pop(0)
            self.state_priorities.pop(0)
            self.episode_priorities.pop(0)
            self.num_states -= len(removed_ep)

    def update_priorities(self, indices: torch.Tensor, priorities: np.ndarray[Any, Any]) -> None:
        import numpy as np

        for i in range(len(indices)):
            ep_idx = int(indices[i][0].item())
            st_idx = int(indices[i][1].item())
            p = float(priorities[i])

            if p > self.max_priority:
                self.max_priority = p
            if p < 1e-4:
                p = 1e-4  # pragma: no cover

            if ep_idx < len(self.state_priorities):
                if st_idx < len(self.state_priorities[ep_idx]):
                    self.state_priorities[ep_idx][st_idx] = p
                    self.episode_priorities[ep_idx] = float(np.mean(self.state_priorities[ep_idx]))

"""
Standard Documentation for search.py.

This module supplies the core execution logic for the `mcts` namespace, heavily typed and tested for production distribution.
"""

import numpy as np
import torch

from tricked.env.state import GameState
from tricked.mcts.features import extract_feature
from tricked.mcts.node import LatentNode
from tricked.model.network import MuZeroNet


class MuZeroMCTS:
    def __init__(self, model: MuZeroNet, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def _apply_gumbel_noise(self, node: LatentNode) -> None:
        if not node.is_expanded or not node.children:
            return  # pragma: no cover

        actions = list(node.children.keys())
        # Generate Gumbel continuous stochasticity (-log(-log(Uniform)))
        u = np.random.uniform(1e-6, 1.0 - 1e-6, size=len(actions))
        gumbel = -np.log(-np.log(u))

        for i, act in enumerate(actions):
            node.children[act].gumbel_noise = float(gumbel[i])

    def search(
        self, root_state: GameState, history: list[int] | None = None, simulations: int = 50
    ) -> tuple[int | None, dict[int, int], LatentNode]:
        """
        Executes pure Python MuZero Latent MCTS.
        - Transforms real root_state `s` into root `h_0`.
        - Iteratively expands the tree purely in latent space.
        - Returns best action, visit distribution, and the root node.
        """
        with torch.no_grad():
            # 1. Initial representation (s -> h0, v0, p0)
            target_device = self.device
            x = extract_feature(root_state, history).unsqueeze(0).to(target_device)
            # x shape: [1, 7, 96]

            h0, _, policy_logits = self.model.initial_inference(x)
            policy_probs = policy_logits[0].cpu().numpy().tolist()

            # Note: The network outputs 288 values. We must manually mask out physically invalid actions
            # from the real root state to prevent the latent tree from exploring illegal moves *at the root*.
            # Subsequent latent steps won't be perfectly masked (MuZero paper relies on standard learning for that)
            # but root masking vastly accelerates learning.
            valid_action_mask = self._get_valid_action_mask(root_state)

            # Apply mask to policy probabilities
            masked_probs = [p if valid_action_mask[i] else 0.0 for i, p in enumerate(policy_probs)]
            sum_probs = sum(masked_probs)
            if sum_probs > 0:
                masked_probs = [p / sum_probs for p in masked_probs]
            else:
                # Fallback if network totally collapsed
                num_valid = sum(valid_action_mask)
                masked_probs = [1.0 / num_valid if valid else 0.0 for valid in valid_action_mask]

            # 2. Initialize Root
            root = LatentNode(prior=1.0)
            root.expand(hidden_state=h0, reward=0.0, policy_probs=masked_probs)

            self._apply_gumbel_noise(root)

            # 3. MCTS Latent Loop
            for _ in range(simulations):
                node = root
                search_path = [node]
                actions = []

                # Selection
                while node.is_expanded:
                    action, next_node = node.select_child(is_root=(node == root))
                    if next_node is None:
                        break  # Node has no children, acts as terminal in latent space  # pragma: no cover
                    search_path.append(next_node)
                    actions.append(action)
                    node = next_node

                if len(actions) == 0:
                    # Dead-end at root. Break early.
                    break  # pragma: no cover

                # Expansion & Evaluation
                parent = search_path[-2]
                last_action = actions[-1]

                # Dynamics (h_parent, a) -> h_child, reward, value, policy
                act_tensor = torch.tensor([last_action], dtype=torch.long, device=target_device)
                
                assert parent.hidden_state is not None
                h_next, reward_t, value_t, policy_t = self.model.recurrent_inference(
                    parent.hidden_state, act_tensor
                )

                val = value_t.item()
                reward = reward_t.item()
                p_probs = policy_t[0].cpu().numpy().tolist()

                # Expand the newly reached leaf
                node.expand(hidden_state=h_next, reward=reward, policy_probs=p_probs)

                # Backpropagation
                # MuZero incorporates predicted intermediate rewards in the backup.
                self._backpropagate(search_path, val)

            # 4. Action Selection
            visits = {act: child.visits for act, child in root.children.items() if child.visits > 0}

            if not visits:
                return None, {}, root  # pragma: no cover

            best_action = max(visits.keys(), key=lambda k: visits[k])

            return best_action, visits, root

    def _backpropagate(self, search_path: list[LatentNode], value: float) -> None:
        """
        Backpropagates value up the trajectory.
        Value at state s_t = reward_{t+1} + gamma * value_{t+1}
        Since we operate inside out:
        """
        gamma = 0.99
        v = value

        # Traverse backwards from leaf to root
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += v
            # The value for the next node up is the reward achieved by arriving here + discounted current value
            v = node.reward + gamma * v

    def _get_valid_action_mask(self, state: GameState) -> list[bool]:
        """
        Converts pure tricked bitboard validation into a 288-length boolean array.
        Action mapping: slot * 96 + pos_idx
        """
        from tricked.env.constants import TOTAL_TRIANGLES
        from tricked.env.pieces import STANDARD_PIECES

        mask = [False] * 288

        if state.terminal:
            return mask  # pragma: no cover

        for slot in range(3):
            p_id = state.available[slot]
            if p_id == -1:
                continue  # pragma: no cover

            for idx in range(TOTAL_TRIANGLES):
                m = STANDARD_PIECES[p_id][idx]
                if m != 0 and (state.board & m) == 0:
                    action_idx = slot * 96 + idx
                    # Safety check
                    if action_idx < 288:
                        mask[action_idx] = True
        return mask

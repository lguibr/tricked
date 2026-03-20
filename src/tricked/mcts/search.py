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
        self, 
        root_state: GameState, 
        history: list[int] | None = None, 
        action_history: list[int] | None = None,
        difficulty: int = 1,
        simulations: int = 50
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
            x = extract_feature(root_state, history, action_history, difficulty).unsqueeze(0).to(target_device)
            # x shape: [1, 7, 96]

            h0, _, policy_logits = self.model.initial_inference(x)
            policy_probs = policy_logits[0].cpu().numpy().tolist()

            # Note: The network outputs 288 values. We must manually mask out physically invalid actions
            # from the real root state to prevent the latent tree from exploring illegal moves *at the root*.
            # Subsequent latent steps won't be perfectly masked (MuZero paper relies on standard learning for that)
            # but root masking vastly accelerates learning.
            valid_action_mask = self._get_valid_action_mask(root_state)

            # Apply mask to policy probabilities (max 1e-8 to prevent KeyError from 0.0 priors missing in root.children)
            masked_probs = [max(p, 1e-8) if valid_action_mask[i] else 0.0 for i, p in enumerate(policy_probs)]
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

            # --- Topic 2: Gumbel Sequential Halving (Root Action Selection) ---
            valid_actions = [a for a, m in enumerate(valid_action_mask) if m]
            if not valid_actions:
                return None, {}, root  # pragma: no cover

            num_valid = len(valid_actions)
            
            # Task 10: Gumbel Top-K Dynamic Expansion
            # Density bounds: 0.0 early game (empty board), 1.0 late game (full board)
            # Triango piece slots * 96 triangles = 288 initial actions.
            density = 1.0 - (num_valid / 288.0)
            k_dynamic = 4 + int(8 * density)  # Scales smoothly from k=4 to k=12
            k = min(k_dynamic, num_valid)

            if k == 1:
                return valid_actions[0], {valid_actions[0]: 1}, root

            # 2.1 Sample Top-K using Policy Logits + Gumbel Noise
            gumbels = -np.log(-np.log(np.random.uniform(1e-6, 1.0 - 1e-6, size=len(masked_probs))))
            # Safely compute log_pi
            log_pi = np.array([np.log(p + 1e-8) if m else -np.inf for p, m in zip(masked_probs, valid_action_mask)])
            gumbel_pi = log_pi + gumbels
            
            # Sort valid actions by gumbel prior to select initial K candidates
            candidates = sorted(valid_actions, key=lambda a: float(gumbel_pi[a]), reverse=True)[:k]

            import math
            phases = math.ceil(math.log2(k))
            sims_per_phase = simulations // phases if phases > 0 else simulations
            
            # 2.2 Sequential Halving Allocation
            for phase in range(phases):
                num_candidates = len(candidates)
                if num_candidates == 1:
                    break
                visits_per_candidate = sims_per_phase // num_candidates
                if visits_per_candidate == 0:
                    visits_per_candidate = 1
                
                # Distribute Phase Budget
                for cand_action in candidates:
                    for _ in range(visits_per_candidate):
                        node = root
                        search_path = [node]
                        actions = []
                        
                        # Selection from root strictly forces `cand_action` to consume phase budget
                        child = node.children.get(cand_action)
                        if child is None:
                            break  # pragma: no cover
                        
                        actions.append(cand_action)
                        search_path.append(child)
                        node = child
                        
                        # Proceed with deep Latent MCTS selection
                        while node.is_expanded:
                            act, next_node = node.select_child(is_root=False)
                            if next_node is None:
                                break  # pragma: no cover
                            actions.append(act)
                            search_path.append(next_node)
                            node = next_node
                            
                        # Expansion & Evaluation
                        parent = search_path[-2]
                        last_action = actions[-1]
                        act_tensor = torch.tensor([last_action], dtype=torch.long, device=target_device)
                        
                        assert parent.hidden_state is not None
                        h_next, reward_t, value_t, policy_t = self.model.recurrent_inference(parent.hidden_state, act_tensor)
                        node.expand(hidden_state=h_next, reward=reward_t.item(), policy_probs=policy_t[0].cpu().numpy().tolist())
                        
                        self._backpropagate(search_path, value_t.item())

                # Sort candidates by Q-value to halve them
                candidates = sorted(candidates, key=lambda a: root.children[a].value, reverse=True)
                drop_count = num_candidates // 2
                candidates = candidates[:-drop_count]

            # 2.3 Completed Policy Regularization
            # Softmax exclusively over the surviving evaluated Q-values of candidates
            evaluated_k = [a for a in root.children.keys() if root.children[a].visits > 0]
            if not evaluated_k:
                return candidates[0], {candidates[0]: 1}, root  # pragma: no cover

            q_values = np.array([root.children[a].value for a in evaluated_k])
            max_q = np.max(q_values)
            exp_q = np.exp(q_values - max_q)
            q_probs = exp_q / np.sum(exp_q)

            # Map to visit distribution format compatible with self_play.py scaling
            visits = {a: max(1, int(p * simulations)) for a, p in zip(evaluated_k, q_probs)}
            best_action = candidates[0]
            
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

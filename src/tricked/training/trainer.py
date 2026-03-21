"""
Standard Documentation for trainer.py.

This module supplies the core execution logic for the `training` namespace, heavily typed and tested for production distribution.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import wandb
from torch.utils.data import DataLoader

from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer


def train(
    model: MuZeroNet,
    buffer: ReplayBuffer,
    optimizer: Optimizer,
    hw_config: dict[str, Any],
    iteration: int = 0,
) -> None:
    model.train()

    epochs = hw_config["train_epochs"]
    device = hw_config["device"]
    unroll_steps = hw_config["unroll_steps"]
    consistency_loss_weight = hw_config.get("consistency_weight", 2.0)

    dataloader = DataLoader(buffer, batch_size=hw_config["train_batch_size"], shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        reward_loss_sum = 0.0
        consistency_loss_sum = 0.0

        for initial_states, actions, rewards, target_policies, target_values, mcts_values, target_states, masks, indices in dataloader:
            initial_states = initial_states.to(device)
            actions = actions.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)
            mcts_values = mcts_values.to(device)
            rewards = rewards.to(device)
            target_states = target_states.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            loss = torch.tensor(0.0, device=device)

            use_amp = device.type in ["cuda", "mps"]
            amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                h = model.representation(initial_states)
                pred_value_logits, pred_policy = model.prediction(h)

                target_value_supp_0 = model.scalar_to_support(target_values[:, 0])

                v_loss_0 = -(target_value_supp_0 * F.log_softmax(pred_value_logits, dim=-1)).sum(-1)
                p_loss_0 = -torch.sum(target_policies[:, 0] * torch.log(pred_policy + 1e-8), dim=-1)

                # Task Path P1 B: Hybrid RGSC Formula (Regret-Guided Search Control)
                with torch.no_grad():
                    base_val_pred = model.support_to_scalar(pred_value_logits).squeeze(-1)
                    td_error_0 = torch.abs(target_values[:, 0] - base_val_pred)
                    mcts_regret_0 = torch.abs(mcts_values[:, 0] - base_val_pred)
                    
                    blend = hw_config.get("hybrid_regret_blend", 0.5)
                    hybrid_regret = blend * td_error_0 + (1.0 - blend) * mcts_regret_0
                    step_0_td_error = hybrid_regret.detach().cpu().numpy()

                loss = v_loss_0 + p_loss_0

                value_loss_sum += v_loss_0.mean().item()
                policy_loss_sum += p_loss_0.mean().item()

                for k in range(unroll_steps):
                    act_k = actions[:, k]

                    h, pred_reward_logits = model.dynamics(h, act_k)
                    # Task 5: 1/2 Gradient Scaling
                    h.register_hook(lambda grad: grad * 0.5)

                    pred_value_logits, pred_policy = model.prediction(h)

                    # Task 6: P1 Latent Consistency Alignment Loss (EfficientZero V2 Contrastive Projection)
                    proj_pred = model.project(h)
                    with torch.no_grad():
                        real_h = model.representation(target_states[:, k])
                        proj_target = model.project(real_h)
                    
                    proj_pred_norm = F.normalize(proj_pred, dim=-1)
                    proj_target_norm = F.normalize(proj_target, dim=-1)
                    c_loss_k = -(proj_pred_norm * proj_target_norm).sum(dim=-1)
                    
                    target_reward_supp_k = model.scalar_to_support(rewards[:, k])
                    target_value_supp_k = model.scalar_to_support(target_values[:, k + 1])

                    # M0RV: Advantage-Weighted Targets
                    with torch.no_grad():
                        val_pred_k_scalar = model.support_to_scalar(pred_value_logits).squeeze(-1)
                        advantage_k = torch.abs(target_values[:, k + 1] - val_pred_k_scalar)
                        adv_weight_k = 1.0 + (advantage_k / (torch.max(advantage_k) + 1e-4))

                    r_loss_k = -(target_reward_supp_k * F.log_softmax(pred_reward_logits, dim=-1)).sum(-1)
                    v_loss_k = -(target_value_supp_k * F.log_softmax(pred_value_logits, dim=-1)).sum(-1)
                    v_loss_k = v_loss_k * adv_weight_k  # Scale by Advantage
                    
                    p_loss_k = -torch.sum(target_policies[:, k + 1] * torch.log(pred_policy + 1e-8), dim=-1)

                    mask_k = masks[:, k + 1]
                    step_loss = r_loss_k + v_loss_k + p_loss_k + (c_loss_k * consistency_loss_weight)
                    loss += step_loss * mask_k

                    reward_loss_sum += (r_loss_k * mask_k).mean().item()
                    value_loss_sum += (v_loss_k * mask_k).mean().item()
                    policy_loss_sum += (p_loss_k * mask_k).mean().item()
                    consistency_loss_sum += (c_loss_k * consistency_loss_weight * mask_k).mean().item()

                loss = loss / (unroll_steps + 1)

            priorities = step_0_td_error + 1e-4
            buffer.update_priorities(indices, priorities)

            scalar_loss = loss.mean()
            scalar_loss.backward()
            optimizer.step()

            total_loss += scalar_loss.item()

        num_batches = len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} | "
            f"Total: {total_loss / num_batches:.4f} (CE v: {value_loss_sum / num_batches:.4f}, "
            f"CE r: {reward_loss_sum / num_batches:.4f}, Pol CE: {policy_loss_sum / num_batches:.4f}, "
            f"Align: {consistency_loss_sum / num_batches:.4f})"
        )

        try:
            global_step = iteration * epochs + epoch
            wandb.log({
                "Loss/Total": total_loss / num_batches,
                "Loss/Value_CE": value_loss_sum / num_batches,
                "Loss/Reward_CE": reward_loss_sum / num_batches,
                "Loss/Policy_CE": policy_loss_sum / num_batches,
                "Loss/Consistency": consistency_loss_sum / num_batches,
                "global_step": global_step
            })
        except Exception:
            pass

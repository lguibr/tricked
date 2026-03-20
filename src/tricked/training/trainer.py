"""
Standard Documentation for trainer.py.

This module supplies the core execution logic for the `training` namespace, heavily typed and tested for production distribution.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer


def train(
    model: MuZeroNet,
    buffer: ReplayBuffer,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    hw_config: dict[str, Any],
    writer: Any | None = None,
    iteration: int = 0,
) -> None:
    model.train()

    epochs = hw_config["train_epochs"]
    device = hw_config["device"]
    unroll_steps = hw_config["unroll_steps"]
    consistency_loss_weight = 2.0  # Task 6: P1 Alignment Penalty Weight

    dataloader = DataLoader(buffer, batch_size=hw_config["train_batch_size"], shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        reward_loss_sum = 0.0
        consistency_loss_sum = 0.0

        for initial_states, actions, rewards, target_policies, target_values, target_states, indices in dataloader:
            initial_states = initial_states.to(device)
            actions = actions.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)
            rewards = rewards.to(device)
            target_states = target_states.to(device)

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

                loss = v_loss_0 + p_loss_0

                value_loss_sum += v_loss_0.mean().item()
                policy_loss_sum += p_loss_0.mean().item()

                for k in range(unroll_steps):
                    act_k = actions[:, k]

                    h, pred_reward_logits = model.dynamics(h, act_k)
                    # Task 5: 1/2 Gradient Scaling
                    h.register_hook(lambda grad: grad * 0.5)

                    pred_value_logits, pred_policy = model.prediction(h)

                    # Task 6: P1 Latent Consistency Alignment Loss
                    with torch.no_grad():
                        real_h = model.representation(target_states[:, k])
                    
                    c_loss_k = F.mse_loss(h, real_h.detach(), reduction='none').mean(dim=[1, 2])
                    
                    target_reward_supp_k = model.scalar_to_support(rewards[:, k])
                    target_value_supp_k = model.scalar_to_support(target_values[:, k + 1])

                    r_loss_k = -(target_reward_supp_k * F.log_softmax(pred_reward_logits, dim=-1)).sum(-1)
                    v_loss_k = -(target_value_supp_k * F.log_softmax(pred_value_logits, dim=-1)).sum(-1)
                    p_loss_k = -torch.sum(target_policies[:, k + 1] * torch.log(pred_policy + 1e-8), dim=-1)

                    loss += r_loss_k + v_loss_k + p_loss_k + (c_loss_k * consistency_loss_weight)

                    reward_loss_sum += r_loss_k.mean().item()
                    value_loss_sum += v_loss_k.mean().item()
                    policy_loss_sum += p_loss_k.mean().item()
                    consistency_loss_sum += (c_loss_k * consistency_loss_weight).mean().item()

                loss = loss / (unroll_steps + 1)

            priorities = loss.detach().cpu().numpy()
            buffer.update_priorities(indices, priorities)

            scalar_loss = loss.mean()
            scalar_loss.backward()
            optimizer.step()

            total_loss += scalar_loss.item()

        scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 1e-5:
                param_group['lr'] = 1e-5

        num_batches = len(dataloader)
        print(
            f"Epoch {epoch + 1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Total: {total_loss / num_batches:.4f} (CE v: {value_loss_sum / num_batches:.4f}, "
            f"CE r: {reward_loss_sum / num_batches:.4f}, Pol CE: {policy_loss_sum / num_batches:.4f}, "
            f"Align: {consistency_loss_sum / num_batches:.4f})"
        )

        if writer is not None:
            global_step = iteration * epochs + epoch
            writer.add_scalar("Loss/Total", total_loss / num_batches, global_step)
            writer.add_scalar("Loss/Value_CE", value_loss_sum / num_batches, global_step)
            writer.add_scalar("Loss/Reward_CE", reward_loss_sum / num_batches, global_step)
            writer.add_scalar("Loss/Policy_CE", policy_loss_sum / num_batches, global_step)
            writer.add_scalar("Loss/Consistency", consistency_loss_sum / num_batches, global_step)
            writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], global_step)

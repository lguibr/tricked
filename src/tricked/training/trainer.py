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

    # We use a custom collate function because we return tuples of tensors of different shapes
    dataloader = DataLoader(buffer, batch_size=hw_config["train_batch_size"], shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        total_loss = 0.0
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        reward_loss_sum = 0.0

        for initial_states, actions, rewards, target_policies, target_values, indices in dataloader:
            initial_states = initial_states.to(device)
            # shapes:
            # actions: [Batch, K]
            # rewards: [Batch, K]
            # target_policies: [Batch, K+1, 150] (includes root step 0)
            # target_values: [Batch, K+1]

            actions = actions.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)
            rewards = rewards.to(device)

            optimizer.zero_grad()

            loss = torch.tensor(0.0, device=device)

            use_amp = device.type in ["cuda", "mps"]
            amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Step 0: Representation + Prediction (Real Root State)
                h = model.representation(initial_states)
            pred_value_logits, pred_policy = model.prediction(h)

            # Convert true scalar float points to categorical Two-Hot bins
            target_value_supp_0 = model.scalar_to_support(target_values[:, 0])

            # Loss at step 0 (Shape: [Batch])
            v_loss_0 = -(target_value_supp_0 * F.log_softmax(pred_value_logits, dim=-1)).sum(-1)
            p_loss_0 = -torch.sum(target_policies[:, 0] * torch.log(pred_policy + 1e-8), dim=-1)

            loss = v_loss_0 + p_loss_0

            value_loss_sum += v_loss_0.mean().item()
            policy_loss_sum += p_loss_0.mean().item()

            # BPTT Unrolled Steps 1..K
            for k in range(unroll_steps):
                act_k = actions[:, k]

                # Dynamics + Prediction (h_k -> h_{k+1})
                h, pred_reward_logits = model.dynamics(h, act_k)
                h.register_hook(lambda grad: grad * 0.5)
                pred_value_logits, pred_policy = model.prediction(h)

                target_reward_supp_k = model.scalar_to_support(rewards[:, k])
                target_value_supp_k = model.scalar_to_support(target_values[:, k + 1])

                r_loss_k = -(target_reward_supp_k * F.log_softmax(pred_reward_logits, dim=-1)).sum(
                    -1
                )
                v_loss_k = -(target_value_supp_k * F.log_softmax(pred_value_logits, dim=-1)).sum(-1)
                p_loss_k = -torch.sum(
                    target_policies[:, k + 1] * torch.log(pred_policy + 1e-8), dim=-1
                )

                # Assuming uniform padding for episodes that ended early (target policy sum = 1, rewards = 0)
                # In a production system we'd use an absorbing state mask to zero-out loss beyond terminal.

                # Convert true scalar float points to categorical Two-Hot bins
                target_value_supp_0 = model.scalar_to_support(target_values[:, 0])

                # Loss at step 0 (Shape: [Batch])
                v_loss_0 = -(target_value_supp_0 * F.log_softmax(pred_value_logits, dim=-1)).sum(-1)
                p_loss_0 = -torch.sum(target_policies[:, 0] * torch.log(pred_policy + 1e-8), dim=-1)

                loss = v_loss_0 + p_loss_0

                value_loss_sum += v_loss_0.mean().item()
                policy_loss_sum += p_loss_0.mean().item()

                # BPTT Unrolled Steps 1..K
                for k in range(unroll_steps):
                    act_k = actions[:, k]

                    # Dynamics + Prediction (h_k -> h_{k+1})
                    h, pred_reward_logits = model.dynamics(h, act_k)
                    h.register_hook(lambda grad: grad * 0.5)
                    pred_value_logits, pred_policy = model.prediction(h)

                    target_reward_supp_k = model.scalar_to_support(rewards[:, k])
                    target_value_supp_k = model.scalar_to_support(target_values[:, k + 1])

                    r_loss_k = -(target_reward_supp_k * F.log_softmax(pred_reward_logits, dim=-1)).sum(
                        -1
                    )
                    v_loss_k = -(target_value_supp_k * F.log_softmax(pred_value_logits, dim=-1)).sum(-1)
                    p_loss_k = -torch.sum(
                        target_policies[:, k + 1] * torch.log(pred_policy + 1e-8), dim=-1
                    )

                    # Assuming uniform padding for episodes that ended early (target policy sum = 1, rewards = 0)
                    # In a production system we'd use an absorbing state mask to zero-out loss beyond terminal.

                    loss += r_loss_k + v_loss_k + p_loss_k

                    reward_loss_sum += r_loss_k.mean().item()
                    value_loss_sum += v_loss_k.mean().item()
                    policy_loss_sum += p_loss_k.mean().item()

                # Normalize loss sequence by total unrolled steps
                loss = loss / (unroll_steps + 1)

            # Prioritized Experience Replay - Update buffer with unscaled scalar surprises
            priorities = loss.detach().cpu().numpy()
            buffer.update_priorities(indices, priorities)

            # Compress for backpropagation
            scalar_loss = loss.mean()
            scalar_loss.backward()
            optimizer.step()

            total_loss += scalar_loss.item()

        scheduler.step()
        # Enforce eta_min to strictly prevent categorical freezing
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 1e-5:
                param_group['lr'] = 1e-5

        num_batches = len(dataloader)
        print(
            f"Epoch {epoch + 1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Total: {total_loss / num_batches:.4f} (CE v: {value_loss_sum / num_batches:.4f}, "
            f"CE r: {reward_loss_sum / num_batches:.4f}, Pol CE: {policy_loss_sum / num_batches:.4f})"
        )

        if writer is not None:
            global_step = iteration * epochs + epoch
            writer.add_scalar("Loss/Total", total_loss / num_batches, global_step)
            writer.add_scalar("Loss/Value_CE", value_loss_sum / num_batches, global_step)
            writer.add_scalar("Loss/Reward_CE", reward_loss_sum / num_batches, global_step)
            writer.add_scalar("Loss/Policy_CE", policy_loss_sum / num_batches, global_step)
            writer.add_scalar("Train/LearningRate", scheduler.get_last_lr()[0], global_step)

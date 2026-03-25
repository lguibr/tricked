from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import threading
import queue

import wandb
from tricked.model.network import MuZeroNet



def negative_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    return -(x1 * x2).sum(dim=-1)

import ray
import numpy as np

def train(model: MuZeroNet, ema_model: MuZeroNet, buffer: Any, optimizer: torch.optim.Optimizer, cfg: Any, iteration: int=0) -> None:
    model.train()
    device, steps = torch.device(cfg["device"]), cfg["unroll_steps"]

    for epoch in range(cfg["train_epochs"]):
        batch = buffer.sample_batch(cfg["train_batch_size"])
        if not batch: continue
        
        b_states, b_acts, b_pids, b_rews, b_t_pols, b_t_vals, b_m_vals, b_t_states, b_masks, indices_list, b_weights = batch
        
        states = torch.from_numpy(b_states).to(device, dtype=torch.float32)
        acts = torch.from_numpy(b_acts).to(device, dtype=torch.long)
        pids = torch.from_numpy(b_pids).to(device, dtype=torch.long)
        rews = torch.from_numpy(b_rews).to(device, dtype=torch.float32)
        t_pols = torch.from_numpy(b_t_pols).to(device, dtype=torch.float32)
        t_vals = torch.from_numpy(b_t_vals).to(device, dtype=torch.float32)
        m_vals = torch.from_numpy(b_m_vals).to(device, dtype=torch.float32)
        t_states = torch.from_numpy(b_t_states).to(device, dtype=torch.float32)
        masks = torch.from_numpy(b_masks).to(device, dtype=torch.float32)
        weights = torch.from_numpy(b_weights).to(device, dtype=torch.float32)
        
        indices = [idx[1] for idx in indices_list]
        
        weights = weights / (weights.max() + 1e-8)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            h = model.representation(states)
            h.register_hook(lambda grad: grad * 0.5)
            v_logits, p_probs, h_logits = model.prediction(h)
            
            v_loss_0 = -(model.scalar_to_support(t_vals[:, 0]) * F.log_softmax(v_logits, dim=-1)).sum(-1)
            p_loss_0 = -torch.sum(t_pols[:, 0] * torch.log(p_probs + 1e-8), dim=-1)
            bce_0 = F.binary_cross_entropy_with_logits(h_logits, states[:, 19, :], reduction='none')
            if bce_0.dim() > 1:
                bce_0 = bce_0.flatten(1).mean(1)
            
            loss = v_loss_0 + p_loss_0 + 0.5 * bce_0
            
            tracker_v = v_loss_0.mean().item()
            tracker_p = p_loss_0.mean().item()
            tracker_r = 0.0

            for k in range(steps):
                h, r_logits = model.dynamics(h, acts[:, k], pids[:, k])
                h.register_hook(lambda grad: grad * 0.5)
                
                with torch.no_grad():
                    target_h = ema_model.representation(t_states[:, k])
                    target_proj = ema_model.projector(target_h)
                proj_h = model.projector(h)
                v_l, p_p, h_l = model.prediction(h)

                rl = (-(model.scalar_to_support(rews[:, k]) * F.log_softmax(r_logits, dim=-1)).sum(-1)) * masks[:, k+1]
                vl = (-(model.scalar_to_support(t_vals[:, k+1]) * F.log_softmax(v_l, dim=-1)).sum(-1)) * masks[:, k+1]
                pl = (-torch.sum(t_pols[:, k+1] * torch.log(p_p + 1e-8), dim=-1)) * masks[:, k+1]

                tracker_r += rl.mean().item()
                tracker_v += vl.mean().item()
                tracker_p += pl.mean().item()

                loss += rl + vl + pl
                loss += negative_cosine_similarity(proj_h, target_proj) * masks[:, k+1]
                
                bce_k = F.binary_cross_entropy_with_logits(h_l, t_states[:, k, 19, :], reduction='none')
                if bce_k.dim() > 1:
                    bce_k = bce_k.flatten(1).mean(1)
                loss += 0.5 * bce_k * masks[:, k+1]

            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            
            # EMA Update
            for param, target_param in zip(model.parameters(), ema_model.parameters()):
                target_param.data.copy_(0.99 * target_param.data + 0.01 * param.data)
            
        td_errors = torch.abs(model.scalar_to_support(t_vals[:, 0]) - F.softmax(v_logits.detach(), dim=-1)).sum(-1)
        buffer.update_priorities(indices, td_errors.cpu().numpy().tolist())

        if wandb.run is not None:
            wandb.log({
                "Loss/Total": loss.item(), 
                "Loss/Value": tracker_v,
                "Loss/Policy": tracker_p,
                "Loss/Reward": tracker_r,
                "LR": optimizer.param_groups[0]['lr']
            })

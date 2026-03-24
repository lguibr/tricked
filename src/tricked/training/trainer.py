from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer


def negative_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    return -(x1 * x2).sum(dim=-1)

def train(model: MuZeroNet, buffer: ReplayBuffer, optimizer: torch.optim.Optimizer, cfg: Any, iteration: int=0) -> None:
    model.train()
    device, steps = torch.device(cfg["device"]), cfg["unroll_steps"]
    # CRITICAL: num_workers MUST be 0 so the SumTree is not copied to stale subprocesses
    loader = DataLoader(buffer, batch_size=cfg["train_batch_size"], shuffle=True, num_workers=0)

    for epoch in range(cfg["train_epochs"]):
        for batch in loader:
            states, acts, pids, rews, t_pols, t_vals, m_vals, t_states, masks, indices = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                h = model.representation(states)
                h.register_hook(lambda grad: grad * 0.5)
                v_logits, p_probs, h_logits = model.prediction(h)
                
                loss = -(model.scalar_to_support(t_vals[:, 0]) * F.log_softmax(v_logits, dim=-1)).sum(-1)
                loss += -torch.sum(t_pols[:, 0] * torch.log(p_probs + 1e-8), dim=-1)
                loss += 0.5 * F.binary_cross_entropy_with_logits(h_logits, states[:, 19, :])

                for k in range(steps):
                    h, r_logits = model.dynamics(h, acts[:, k], pids[:, k])
                    
                    with torch.no_grad():
                        target_h = model.representation(t_states[:, k])
                        target_proj = model.projector(target_h)
                    proj_h = model.projector(h)
                    v_l, p_p, h_l = model.prediction(h)

                    loss += (-(model.scalar_to_support(rews[:, k]) * F.log_softmax(r_logits, dim=-1)).sum(-1)) * masks[:, k+1]
                    loss += (-(model.scalar_to_support(t_vals[:, k+1]) * F.log_softmax(v_l, dim=-1)).sum(-1)) * masks[:, k+1]
                    loss += (-torch.sum(t_pols[:, k+1] * torch.log(p_p + 1e-8), dim=-1)) * masks[:, k+1]
                    loss += negative_cosine_similarity(proj_h, target_proj) * masks[:, k+1]
                    loss += 0.5 * F.binary_cross_entropy_with_logits(h_l, t_states[:, k, 19, :]) * masks[:, k+1]

                loss = loss.mean()
                loss.backward()  # type: ignore
                optimizer.step()
                
            td_errors = torch.abs(model.scalar_to_support(t_vals[:, 0]) - F.softmax(v_logits.detach(), dim=-1)).sum(-1)
            buffer.update_priorities(indices.cpu().numpy(), td_errors.cpu().numpy())

        if wandb.run is not None:
            wandb.log({"Loss/Total": loss.item(), "LR": optimizer.param_groups[0]['lr']})

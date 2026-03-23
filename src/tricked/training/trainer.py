from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer

def train(model: MuZeroNet, buffer: ReplayBuffer, optimizer: torch.optim.Optimizer, cfg: dict[str, Any], iteration: int=0) -> None:
    model.train()
    device, steps = cfg["device"], cfg["unroll_steps"]
    loader = DataLoader(buffer, batch_size=cfg["train_batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(cfg["train_epochs"]):
        for batch in loader:
            states, acts, pids, rews, t_pols, t_vals, m_vals, t_states, masks, _ = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                h = model.representation(states)
                v_logits, p_probs, h_logits = model.prediction(h)
                
                loss = -(model.scalar_to_support(t_vals[:, 0]) * F.log_softmax(v_logits, dim=-1)).sum(-1)
                loss += -torch.sum(t_pols[:, 0] * torch.log(p_probs + 1e-8), dim=-1)
                loss += 0.5 * F.binary_cross_entropy_with_logits(h_logits, states[:, 19, :])

                for k in range(steps):
                    h, r_logits = model.dynamics(h, acts[:, k], pids[:, k])
                    h.register_hook(lambda grad: grad * 0.5)
                    v_l, p_p, h_l = model.prediction(h)

                    loss += (-(model.scalar_to_support(rews[:, k]) * F.log_softmax(r_logits, dim=-1)).sum(-1)) * masks[:, k+1]
                    loss += (-(model.scalar_to_support(t_vals[:, k+1]) * F.log_softmax(v_l, dim=-1)).sum(-1)) * masks[:, k+1]
                    loss += (-torch.sum(t_pols[:, k+1] * torch.log(p_p + 1e-8), dim=-1)) * masks[:, k+1]

                loss = loss.mean()
                loss.backward() 
                optimizer.step()

        if wandb.run is not None:
            wandb.log({"Loss/Total": loss.item(), "LR": optimizer.param_groups[0]['lr']})

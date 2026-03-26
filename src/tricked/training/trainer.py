import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import wandb
from tricked.model.network import MuZeroNet


def negative_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)
    return -(x1 * x2).sum(dim=-1)


class ReplayBufferDataset(IterableDataset):
    """
    Zero-copy dataset. Pre-allocates pinned memory using PyTorch, and passes the NumPy
    views to Rust. The Rust backend writes directly to these views bypassing the GIL.
    """

    def __init__(self, buffer: Any, batch_size: int, unroll_steps: int):
        self.buffer = buffer
        self.batch_size = batch_size
        self.unroll_steps = unroll_steps

    def __iter__(self):
        while True:
            b_states = torch.empty((self.batch_size, 20, 96), dtype=torch.float32, pin_memory=True)
            b_acts = torch.empty(
                (self.batch_size, self.unroll_steps), dtype=torch.int64, pin_memory=True
            )
            b_pids = torch.empty(
                (self.batch_size, self.unroll_steps), dtype=torch.int64, pin_memory=True
            )
            b_rews = torch.empty(
                (self.batch_size, self.unroll_steps), dtype=torch.float32, pin_memory=True
            )
            b_t_pols = torch.empty(
                (self.batch_size, self.unroll_steps + 1, 288), dtype=torch.float32, pin_memory=True
            )
            b_t_vals = torch.empty(
                (self.batch_size, self.unroll_steps + 1), dtype=torch.float32, pin_memory=True
            )
            b_m_vals = torch.empty(
                (self.batch_size, self.unroll_steps + 1), dtype=torch.float32, pin_memory=True
            )
            b_t_states = torch.empty(
                (self.batch_size, self.unroll_steps, 20, 96), dtype=torch.float32, pin_memory=True
            )
            b_masks = torch.empty(
                (self.batch_size, self.unroll_steps + 1), dtype=torch.float32, pin_memory=True
            )
            b_weights = torch.empty(self.batch_size, dtype=torch.float32, pin_memory=True)

            indices = self.buffer.sample_batch(
                self.batch_size,
                b_states.numpy(),
                b_acts.numpy(),
                b_pids.numpy(),
                b_rews.numpy(),
                b_t_pols.numpy(),
                b_t_vals.numpy(),
                b_m_vals.numpy(),
                b_t_states.numpy(),
                b_masks.numpy(),
                b_weights.numpy(),
            )

            if indices is None:
                break

            yield b_states, b_acts, b_pids, b_rews, b_t_pols, b_t_vals, b_m_vals, b_t_states, b_masks, indices, b_weights


def make_train_step(
    model: MuZeroNet,
    ema_model: MuZeroNet,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    steps: int,
    device: torch.device,
):
    bs = cfg["train_batch_size"]
    # Static tensors
    s_states = torch.empty((bs, 20, 96), device=device, dtype=torch.float32)
    s_acts = torch.empty((bs, steps), device=device, dtype=torch.int64)
    s_pids = torch.empty((bs, steps), device=device, dtype=torch.int64)
    s_rews = torch.empty((bs, steps), device=device, dtype=torch.float32)
    s_t_pols = torch.empty((bs, steps + 1, 288), device=device, dtype=torch.float32)
    s_t_vals = torch.empty((bs, steps + 1), device=device, dtype=torch.float32)
    s_m_vals = torch.empty((bs, steps + 1), device=device, dtype=torch.float32)
    s_t_states = torch.empty((bs, steps, 20, 96), device=device, dtype=torch.float32)
    s_masks = torch.empty((bs, steps + 1), device=device, dtype=torch.float32)
    s_weights = torch.empty(bs, device=device, dtype=torch.float32)

    # Tracking outputs
    out_loss = torch.empty((), device=device, dtype=torch.float32)
    out_v = torch.empty((), device=device, dtype=torch.float32)
    out_p = torch.empty((), device=device, dtype=torch.float32)
    out_r = torch.empty((), device=device, dtype=torch.float32)
    out_td = torch.empty((bs,), device=device, dtype=torch.float32)

    def forward_backward():
        optimizer.zero_grad(set_to_none=True)
        scaled_weights = s_weights / (s_weights.max() + 1e-8)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            h = model.representation(s_states)
            h.register_hook(lambda grad: grad * 0.5)
            v_logits, p_probs, h_logits = model.prediction(h)

            v_loss_0 = -(
                model.scalar_to_support(s_t_vals[:, 0]) * F.log_softmax(v_logits, dim=-1)
            ).sum(-1)
            p_loss_0 = -torch.sum(s_t_pols[:, 0] * torch.log(p_probs + 1e-8), dim=-1)
            bce_0 = F.binary_cross_entropy_with_logits(
                h_logits, s_states[:, 19, :], reduction="none"
            )
            if bce_0.dim() > 1:
                bce_0 = bce_0.flatten(1).mean(1)

            loss = v_loss_0 + p_loss_0 + 0.5 * bce_0

            tracker_v = v_loss_0.mean()
            tracker_p = p_loss_0.mean()
            tracker_r = torch.zeros_like(tracker_v)

            for k in range(steps):
                h, r_logits = model.dynamics(h, s_acts[:, k], s_pids[:, k])
                h.register_hook(lambda grad: grad * 0.5)

                with torch.no_grad():
                    target_h = ema_model.representation(s_t_states[:, k])
                    target_proj = ema_model.projector(target_h)
                proj_h = model.projector(h)
                v_l, p_p, h_l = model.prediction(h)

                rl = (
                    -(model.scalar_to_support(s_rews[:, k]) * F.log_softmax(r_logits, dim=-1)).sum(
                        -1
                    )
                ) * s_masks[:, k + 1]
                vl = (
                    -(model.scalar_to_support(s_t_vals[:, k + 1]) * F.log_softmax(v_l, dim=-1)).sum(
                        -1
                    )
                ) * s_masks[:, k + 1]
                pl = (-torch.sum(s_t_pols[:, k + 1] * torch.log(p_p + 1e-8), dim=-1)) * s_masks[
                    :, k + 1
                ]

                tracker_r += rl.mean()
                tracker_v += vl.mean()
                tracker_p += pl.mean()

                loss += rl + vl + pl
                loss += negative_cosine_similarity(proj_h, target_proj) * s_masks[:, k + 1]

                bce_k = F.binary_cross_entropy_with_logits(
                    h_l, s_t_states[:, k, 19, :], reduction="none"
                )
                if bce_k.dim() > 1:
                    bce_k = bce_k.flatten(1).mean(1)
                loss += 0.5 * bce_k * s_masks[:, k + 1]

            loss = (loss * scaled_weights).mean() / steps

        loss.backward()
        optimizer.step()

        out_loss.copy_(loss.detach())
        out_v.copy_(tracker_v.detach())
        out_p.copy_(tracker_p.detach())
        out_r.copy_(tracker_r.detach())

        with torch.no_grad():
            td_errors = torch.abs(
                model.scalar_to_support(s_t_vals[:, 0]) - F.softmax(v_logits, dim=-1)
            ).sum(-1)
            out_td.copy_(td_errors)

        # EMA Update
        for param, target_param in zip(model.parameters(), ema_model.parameters()):
            target_param.data.copy_(0.99 * target_param.data + 0.01 * param.data)

    if device.type == "cuda" and cfg.get("use_cuda_graphs", True):
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                forward_backward()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            forward_backward()

        def run_step(
            b_states,
            b_acts,
            b_pids,
            b_rews,
            b_t_pols,
            b_t_vals,
            b_m_vals,
            b_t_states,
            b_masks,
            b_weights,
        ):
            s_states.copy_(b_states, non_blocking=True)
            s_acts.copy_(b_acts, non_blocking=True)
            s_pids.copy_(b_pids, non_blocking=True)
            s_rews.copy_(b_rews, non_blocking=True)
            s_t_pols.copy_(b_t_pols, non_blocking=True)
            s_t_vals.copy_(b_t_vals, non_blocking=True)
            s_m_vals.copy_(b_m_vals, non_blocking=True)
            s_t_states.copy_(b_t_states, non_blocking=True)
            s_masks.copy_(b_masks, non_blocking=True)
            s_weights.copy_(b_weights, non_blocking=True)
            g.replay()
            return out_loss.item(), out_v.item(), out_p.item(), out_r.item(), out_td.tolist()

        return run_step
    else:

        def run_step(
            b_states,
            b_acts,
            b_pids,
            b_rews,
            b_t_pols,
            b_t_vals,
            b_m_vals,
            b_t_states,
            b_masks,
            b_weights,
        ):
            s_states.copy_(b_states, non_blocking=True)
            s_acts.copy_(b_acts, non_blocking=True)
            s_pids.copy_(b_pids, non_blocking=True)
            s_rews.copy_(b_rews, non_blocking=True)
            s_t_pols.copy_(b_t_pols, non_blocking=True)
            s_t_vals.copy_(b_t_vals, non_blocking=True)
            s_m_vals.copy_(b_m_vals, non_blocking=True)
            s_t_states.copy_(b_t_states, non_blocking=True)
            s_masks.copy_(b_masks, non_blocking=True)
            s_weights.copy_(b_weights, non_blocking=True)
            forward_backward()
            return out_loss.item(), out_v.item(), out_p.item(), out_r.item(), out_td.tolist()

        return run_step


def train(
    model: MuZeroNet,
    ema_model: MuZeroNet,
    buffer: Any,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    iteration: int = 0,
) -> None:
    model.train()
    device, steps = torch.device(cfg["device"]), cfg["unroll_steps"]

    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    dataset = ReplayBufferDataset(buffer, cfg["train_batch_size"], steps)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    data_iter = iter(dataloader)

    if not hasattr(model, "_train_step_fn"):
        model._train_step_fn = make_train_step(model, ema_model, optimizer, cfg, steps, device)

    for epoch in range(cfg["train_epochs"]):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        (
            b_states,
            b_acts,
            b_pids,
            b_rews,
            b_t_pols,
            b_t_vals,
            b_m_vals,
            b_t_states,
            b_masks,
            indices_list,
            b_weights,
        ) = batch
        indices = [idx[1] for idx in indices_list]

        loss, v, p, r, td = model._train_step_fn(
            b_states,
            b_acts,
            b_pids,
            b_rews,
            b_t_pols,
            b_t_vals,
            b_m_vals,
            b_t_states,
            b_masks,
            b_weights,
        )
        buffer.update_priorities(indices, td)

        try:
            import io

            import lz4.frame
            import redis

            r_client = redis.Redis.from_url(
                cfg.get(
                    "redis_url",
                    (
                        "redis://localhost:6379"
                        if not os.environ.get("REDIS_HOST")
                        else "redis://redis:6379"
                    ),
                )
            )
            model.half()
            jit_model = torch.jit.script(model)

            try:

                # Freeze the model
                jit_model = torch.jit.freeze(jit_model)

                # Apply native kernel fusion and optimization
                jit_model = torch.jit.optimize_for_inference(jit_model)
            except Exception as e:
                print(f"JIT Optimization Failed: {e}")

            buf_mem = io.BytesIO()
            torch.jit.save(jit_model, buf_mem)
            compressed = lz4.frame.compress(buf_mem.getvalue())
            r_client.set("model_weights", compressed)
            r_client.publish("model_updates", "new_model")
            model.float()
        except Exception as e:
            print(f"Failed to publish model: {e}")

        if wandb.run is not None:
            wandb.log(
                {
                    "Loss/Total": loss,
                    "Loss/Value": v,
                    "Loss/Policy": p,
                    "Loss/Reward": r,
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

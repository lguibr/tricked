"""
Standard Documentation for network.py.

This module supplies the core execution logic for the `model` namespace, heavily typed and tested for production distribution.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")


class GridMapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.row_lengths = [9, 11, 13, 15, 15, 13, 11, 9]
        self.offsets = [3, 2, 1, 0, 0, 1, 2, 3]

    def to_grid(self, flat_tensor: torch.Tensor) -> torch.Tensor:
        # flat_tensor: [Batch, Channels, 96] -> [Batch, Channels, 8, 15]
        B, C, _ = flat_tensor.shape
        grid = torch.zeros(B, C, 8, 15, device=flat_tensor.device, dtype=flat_tensor.dtype)
        idx = 0
        for r, (length, offset) in enumerate(zip(self.row_lengths, self.offsets)):
            grid[:, :, r, offset : offset + length] = flat_tensor[:, :, idx : idx + length]
            idx += length
        return grid


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.mish(out)


class RepresentationNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8):
        super().__init__()
        self.mapper = GridMapper()
        self.conv_in = nn.Conv2d(9, d_model, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(d_model)
        self.blocks = nn.ModuleList([ResBlock(d_model) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [Batch, 9, 96]
        x_grid = self.mapper.to_grid(x)  # [Batch, 9, 8, 15]
        h = F.mish(self.bn_in(self.conv_in(x_grid)))
        for block in self.blocks:
            h = block(h)
        return self._scale_hidden(h)

    def _scale_hidden(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        h_flat = h.view(B, -1)
        h_min = h_flat.min(dim=-1, keepdim=True)[0]
        h_max = h_flat.max(dim=-1, keepdim=True)[0]
        h_scale = h_max - h_min
        h_scale[h_scale < 1e-5] += 1e-5
        h_normalized = (h_flat - h_min) / h_scale
        return h_normalized.view_as(h)


class DynamicsNet(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_actions: int = 288,
        num_blocks: int = 4,
        support_size: int = 200,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.conv_in = nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(d_model)
        self.blocks = nn.ModuleList([ResBlock(d_model) for _ in range(num_blocks)])

        self.reward_fc1 = nn.Linear(d_model, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = h.size(0)
        a_emb = self.action_emb(a)

        a_expanded = a_emb.view(batch_size, -1, 1, 1).expand_as(h)
        x = torch.cat([h, a_expanded], dim=1)

        h_next = F.mish(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            h_next = block(h_next)

        r_pooled = h_next.mean(dim=[2, 3])
        r = F.mish(self.reward_norm(self.reward_fc1(r_pooled)))
        reward_logits = self.reward_fc2(r)

        return self._scale_hidden(h_next), reward_logits

    def _scale_hidden(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        h_flat = h.view(B, -1)
        h_min = h_flat.min(dim=-1, keepdim=True)[0]
        h_max = h_flat.max(dim=-1, keepdim=True)[0]
        h_scale = h_max - h_min
        h_scale[h_scale < 1e-5] += 1e-5
        h_normalized = (h_flat - h_min) / h_scale
        return h_normalized.view_as(h)


class PredictionNet(nn.Module):
    def __init__(self, d_model: int = 128, support_size: int = 200):
        super().__init__()
        self.val_conv = nn.Conv2d(d_model, 1, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 15, 64)
        self.value_fc2 = nn.Linear(64, 2 * support_size + 1)

        self.pol_conv = nn.Conv2d(d_model, 2, kernel_size=1)
        self.pol_bn = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(2 * 8 * 15, 288)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = h.size(0)

        v = F.mish(self.val_bn(self.val_conv(h)))
        v = v.view(batch_size, -1)
        v = F.mish(self.value_fc1(v))
        value_logits = self.value_fc2(v)

        p = F.mish(self.pol_bn(self.pol_conv(h)))
        p = p.view(batch_size, -1)
        policy_logits = self.policy_fc1(p)
        policy_probs = F.softmax(policy_logits, dim=-1)
        return value_logits, policy_probs


class MuZeroNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8, support_size: int = 200):
        super().__init__()
        self.support_size = support_size
        self.representation = RepresentationNet(d_model, num_blocks=num_blocks)
        self.dynamics = DynamicsNet(
            d_model, num_actions=288, num_blocks=max(1, num_blocks // 2), support_size=support_size
        )
        self.prediction = PredictionNet(d_model, support_size=support_size)

        self.register_buffer(
            "support_vector", torch.arange(-support_size, support_size + 1, dtype=torch.float32)
        )

    def support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converts Two-Hot encoded logits back to a physical scalar tensor using Symexp.
        logits shape: [Batch, 2 * support_size + 1] -> scalar [Batch, 1]
        """
        probs = F.softmax(logits, dim=-1)
        sym_scalar = torch.sum(probs * self.support_vector, dim=-1, keepdim=True)  # type: ignore
        
        # Symexp: Approximate inverse of h(x) = sign(x)(sqrt(|x|+1)-1)
        scalar = torch.sign(sym_scalar) * (((torch.abs(sym_scalar) + 1) ** 2) - 1)
        return scalar

    def scalar_to_support(self, scalar: torch.Tensor) -> torch.Tensor:
        """
        Converts a raw physical integer target to its 401-bin Two-Hot distribution via Symlog.
        scalar shape: [Batch] or [Batch, 1] -> probabilities [Batch, 2 * support_size + 1]
        """
        # Symlog: h(x) = sign(x)(sqrt(|x|+1)-1) + epsilon*x
        sym_scalar = torch.sign(scalar) * (torch.sqrt(torch.abs(scalar) + 1.0) - 1.0) + 0.001 * scalar
        
        sym_scalar = sym_scalar.view(-1).clamp(-self.support_size, self.support_size)
        probabilities = torch.zeros(sym_scalar.size(0), 2 * self.support_size + 1, device=sym_scalar.device)

        lower = sym_scalar.floor()
        upper = sym_scalar.ceil()

        p_upper = sym_scalar - lower
        p_lower = 1.0 - p_upper

        lower_idx = (lower + self.support_size).long()
        upper_idx = (upper + self.support_size).long()

        probabilities.scatter_add_(1, lower_idx.unsqueeze(1), p_lower.unsqueeze(1))
        probabilities.scatter_add_(1, upper_idx.unsqueeze(1), p_upper.unsqueeze(1))

        return probabilities

    def initial_inference(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.representation(s)
        value_logits, policy = self.prediction(h)
        value_scalar = self.support_to_scalar(value_logits)
        return h, value_scalar, policy

    def recurrent_inference(
        self, h: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_next, reward_logits = self.dynamics(h, a)
        value_logits, policy = self.prediction(h_next)
        reward_scalar = self.support_to_scalar(reward_logits)
        value_scalar = self.support_to_scalar(value_logits)
        return h_next, reward_scalar, value_scalar, policy

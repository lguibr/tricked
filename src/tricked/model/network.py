"""
Standard Documentation for network.py.

This module supplies the core execution logic for the `model` namespace, heavily typed and tested for production distribution.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricked.env.pieces import ALL_MASKS

warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

class HybridTriAxialBlock(nn.Module):
    """
    Task P2: Tri-Axial Line Convolution.
    Replaces Convolutional geometry by structurally mapping cross-board relationships
    simultaneously over the 24 physics-based line clearing structures.
    """
    def __init__(self, d_model: int):
        super().__init__()
        
        M = torch.zeros(24, 96)
        for idx, mask in enumerate(ALL_MASKS):
            for m in range(96):
                if (mask >> m) & 1:
                    M[idx, m] = 1.0
                    
        self.register_buffer("M", M)
        self.register_buffer("line_lengths", M.sum(dim=1).view(1, 24, 1))

        self.line_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model)
        )
        self.global_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from typing import cast
        # x is [Batch, 96, d_model]
        line_lengths = cast(torch.Tensor, getattr(self, "line_lengths"))
        line_sums = torch.einsum('lm,bmd->bld', self.M, x)
        line_context = line_sums / line_lengths
        
        processed_lines = self.line_processor(line_context)
        tactical_update = torch.einsum('lm,bld->bmd', self.M, processed_lines)
        
        global_context = x.mean(dim=1, keepdim=True)
        strategic_update = self.global_processor(global_context)
        
        out = x + tactical_update + strategic_update
        return cast(torch.Tensor, self.norm(out))


class RepresentationNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8):
        super().__init__()
        # 20 Input Channels from features.py expansion
        self.proj_in = nn.Linear(20, d_model)
        self.blocks = nn.ModuleList([HybridTriAxialBlock(d_model) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self._scale_hidden(h.transpose(1, 2))

    def _scale_hidden(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        h_flat = h.reshape(B, -1)
        h_min = h_flat.min(dim=-1, keepdim=True)[0]
        h_max = h_flat.max(dim=-1, keepdim=True)[0]
        h_scale = h_max - h_min
        h_scale[h_scale < 1e-5] += 1e-5
        h_normalized = (h_flat - h_min) / h_scale
        return h_normalized.reshape_as(h)


class DynamicsNet(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 8,
        support_size: int = 200,
    ):
        super().__init__()
        self.piece_emb = nn.Embedding(12, d_model)
        self.pos_emb = nn.Embedding(96, d_model)
        
        self.proj_in = nn.Linear(d_model * 2, d_model)
        self.blocks = nn.ModuleList([HybridTriAxialBlock(d_model) for _ in range(num_blocks)])

        self.reward_fc1 = nn.Linear(d_model * 2, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)

    def forward(self, h: torch.Tensor, a: torch.Tensor, piece_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_idx = a % 96
        a_emb = self.piece_emb(piece_id) + self.pos_emb(pos_idx)

        # EfficientZero V2: Reward Prefix
        h_t_pooled = h.mean(dim=2)  # h is [B, d_model, 96]
        r_input = torch.cat([h_t_pooled, a_emb], dim=1)
        r = F.mish(self.reward_norm(self.reward_fc1(r_input)))
        reward_logits = self.reward_fc2(r)

        # Dynamics next state transition
        a_expanded = a_emb.unsqueeze(-1).expand(-1, -1, h.size(-1))
        x = torch.cat([h, a_expanded], dim=1)

        h_next = self.proj_in(x.transpose(1, 2))
        for block in self.blocks:
            h_next = block(h_next)

        h_next = h_next.transpose(1, 2)
        return self._scale_hidden(h_next), reward_logits

    def _scale_hidden(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        h_flat = h.reshape(B, -1)
        h_min = h_flat.min(dim=-1, keepdim=True)[0]
        h_max = h_flat.max(dim=-1, keepdim=True)[0]
        h_scale = h_max - h_min
        h_scale[h_scale < 1e-5] += 1e-5
        h_normalized = (h_flat - h_min) / h_scale
        return h_normalized.reshape_as(h)


class PredictionNet(nn.Module):
    def __init__(self, d_model: int = 128, support_size: int = 200, num_actions: int = 288):
        super().__init__()
        self.val_proj = nn.Linear(d_model, d_model // 2)
        self.val_norm = nn.LayerNorm(d_model // 2)
        self.value_fc1 = nn.Linear(d_model // 2, 64)
        self.value_fc2 = nn.Linear(64, 2 * support_size + 1)

        self.pol_proj = nn.Linear(d_model, d_model // 2)
        self.pol_norm = nn.LayerNorm(d_model // 2)
        self.policy_fc1 = nn.Linear(d_model // 2, num_actions)
        
        self.hole_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Mish(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = h.transpose(1, 2)

        v = F.mish(self.val_norm(self.val_proj(x)))
        v = v.mean(dim=1)
        v = F.mish(self.value_fc1(v))
        value_logits = self.value_fc2(v)

        p = F.mish(self.pol_norm(self.pol_proj(x)))
        p = p.mean(dim=1)
        policy_logits = self.policy_fc1(p)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        hole_logits = self.hole_predictor(x).squeeze(-1)
        return value_logits, policy_probs, hole_logits


class ProjectorNet(nn.Module):
    """
    SimSiam-style Projection Head mapping raw latent representations into a compressed 
    feature space for Contrastive Alignment Loss.
    """
    def __init__(self, d_model: int = 128, proj_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model // 2)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.fc1 = nn.Linear(d_model // 2, proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.fc2 = nn.Linear(proj_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.mish(self.norm1(self.proj(h.transpose(1, 2))))
        x = x.mean(dim=1)
        x = F.mish(self.norm2(self.fc1(x)))
        return self.fc2(x)  # type: ignore[no-any-return]


class MuZeroNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8, support_size: int = 200):
        super().__init__()
        self.support_size = support_size
        self.representation = RepresentationNet(d_model, num_blocks=num_blocks)
        self.dynamics = DynamicsNet(
            d_model, num_blocks=num_blocks, support_size=support_size
        )
        self.prediction = PredictionNet(d_model, support_size=support_size)
        self.projector = ProjectorNet(d_model=d_model)

        self.register_buffer(
            "support_vector", torch.arange(-support_size, support_size + 1, dtype=torch.float32)
        )

    def support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        sv = self.support_vector
        assert isinstance(sv, torch.Tensor)
        sym_scalar = torch.sum(probs * sv, dim=-1, keepdim=True)
        
        epsilon = 0.001
        y = torch.abs(sym_scalar)
        
        # Exact inverse for y = sign(x)(sqrt(|x|+1)-1) + eps*x
        # y = sqrt(x+1) - 1 + eps*x.
        # Let z = sqrt(x+1). Then x = z^2 - 1.
        # eps*z^2 + z - (1 + eps + y) = 0
        z = (-1.0 + torch.sqrt(1.0 + 4.0 * epsilon * (1.0 + epsilon + y))) / (2.0 * epsilon)
        x = z ** 2 - 1.0
        
        scalar = torch.sign(sym_scalar) * x
        return scalar

    def scalar_to_support(self, scalar: torch.Tensor) -> torch.Tensor:
        sym_scalar = torch.sign(scalar) * (torch.sqrt(torch.abs(scalar) + 1.0) - 1.0) + 0.001 * scalar
        sym_scalar = sym_scalar.reshape(-1).clamp(-self.support_size, self.support_size)
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

    def initial_inference(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.representation(s)
        value_logits, policy, hole_logits = self.prediction(h)
        value_scalar = self.support_to_scalar(value_logits)
        return h, value_scalar, policy, hole_logits

    def recurrent_inference(
        self, h: torch.Tensor, a: torch.Tensor, piece_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_next, reward_logits = self.dynamics(h, a, piece_id)
        value_logits, policy, hole_logits = self.prediction(h_next)
        reward_scalar = self.support_to_scalar(reward_logits)
        value_scalar = self.support_to_scalar(value_logits)
        return h_next, reward_scalar, value_scalar, policy, hole_logits

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)  # type: ignore[no-any-return]

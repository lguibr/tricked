"""
Standard Documentation for network.py.

This module supplies the core execution logic for the `model` namespace, heavily typed and tested for production distribution.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

class TransformerBlock(nn.Module):
    """
    Task P2: Pre-LN Transformer Self-Attention.
    Replaces Convolutional geometry by structurally mapping cross-board relationships
    simultaneously without graph decay boundaries.
    """
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = res + attn_out
        
        res = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        return res + mlp_out  # type: ignore[no-any-return]


class RepresentationNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8):
        super().__init__()
        # 20 Input Channels from features.py expansion
        self.proj_in = nn.Linear(20, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_blocks)])

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
        num_actions: int = 288,
        num_blocks: int = 8,  # Task 7: Deepen Capacity (removed division by 2)
        support_size: int = 200,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.proj_in = nn.Linear(d_model * 2, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_blocks)])

        self.reward_fc1 = nn.Linear(d_model, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_emb = self.action_emb(a)

        # Expand action embedding across the dynamic spatial topologies cleanly
        a_expanded = a_emb.unsqueeze(-1).expand(-1, -1, h.size(-1))
        x = torch.cat([h, a_expanded], dim=1)

        h_next = self.proj_in(x.transpose(1, 2))
        for block in self.blocks:
            h_next = block(h_next)

        h_next = h_next.transpose(1, 2)

        # Global average pool over 96 nodes
        r_pooled = h_next.mean(dim=2)
        r = F.mish(self.reward_norm(self.reward_fc1(r_pooled)))
        reward_logits = self.reward_fc2(r)

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

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = h.transpose(1, 2)

        v = F.mish(self.val_norm(self.val_proj(x)))
        v = v.mean(dim=1)  # Topology-agnostic Global Average Pool
        v = F.mish(self.value_fc1(v))
        value_logits = self.value_fc2(v)

        p = F.mish(self.pol_norm(self.pol_proj(x)))
        p = p.mean(dim=1)  # Topology-agnostic Global Average Pool
        policy_logits = self.policy_fc1(p)
        policy_probs = F.softmax(policy_logits, dim=-1)
        return value_logits, policy_probs


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
        x = x.mean(dim=1)  # Topology-agnostic Global Average Pool
        x = F.mish(self.norm2(self.fc1(x)))
        return self.fc2(x)  # type: ignore[no-any-return]


class MuZeroNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8, support_size: int = 200):
        super().__init__()
        self.support_size = support_size
        self.representation = RepresentationNet(d_model, num_blocks=num_blocks)
        # Task 7: Pass full num_blocks to prevent representation decay horizontally!
        self.dynamics = DynamicsNet(
            d_model, num_actions=288, num_blocks=num_blocks, support_size=support_size
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
        
        # Deep Dive 3: Newton-Raphson Inverse for Symlog Transformation
        epsilon = 0.001
        y = torch.abs(sym_scalar)
        
        # Extract initial approximation based on standard quadratic
        x = ((y + 1.0) ** 2) - 1.0
        
        # 3 hardware-accelerated iterations converge symmetrically
        for _ in range(3):
            sqrt_x_1 = torch.sqrt(x + 1.0)
            g = sqrt_x_1 - 1.0 + epsilon * x - y
            g_prime = 0.5 / sqrt_x_1 + epsilon
            x = x - g / g_prime
            
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

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Projects latent state into structural alignment space."""
        return self.projector(h)  # type: ignore[no-any-return]

"""
Standard Documentation for network.py.

This module supplies the core execution logic for the `model` namespace, heavily typed and tested for production distribution.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricked.env.constants import build_adjacency_matrix

warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

class GraphConv1d(nn.Module):
    """
    Task 8 / P0: 1D Adjacency Mask.
    Performs pure Graph Convolution along the strict 96-node Triangular Adjacency Matrix
    preventing padding-bleed hallucination common in 2D Convolutions on irregular grids.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.W = nn.Linear(in_c, out_c, bias=False)
        self.register_buffer("A", build_adjacency_matrix())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [Batch, C, 96] -> [Batch, 96, C]
        x_t = x.transpose(1, 2)
        # Message passing A [96, 96] x x_t [B, 96, C] -> [B, 96, C]
        A = self.A
        assert isinstance(A, torch.Tensor)
        msg = torch.matmul(A, x_t)
        out = self.W(msg)
        return out.transpose(1, 2)  # type: ignore[no-any-return]

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = GraphConv1d(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = GraphConv1d(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.mish(out)


class RepresentationNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8):
        super().__init__()
        # 20 Input Channels from features.py expansion
        self.conv_in = GraphConv1d(20, d_model)
        self.bn_in = nn.BatchNorm1d(d_model)
        self.blocks = nn.ModuleList([ResBlock(d_model) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.mish(self.bn_in(self.conv_in(x)))
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
        num_blocks: int = 8,  # Task 7: Deepen Capacity (removed division by 2)
        support_size: int = 200,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.conv_in = GraphConv1d(d_model * 2, d_model)
        self.bn_in = nn.BatchNorm1d(d_model)
        self.blocks = nn.ModuleList([ResBlock(d_model) for _ in range(num_blocks)])

        self.reward_fc1 = nn.Linear(d_model, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_emb = self.action_emb(a)

        # Expand action embedding across the 96 spatial topologies cleanly
        a_expanded = a_emb.unsqueeze(-1).expand(-1, -1, 96)
        x = torch.cat([h, a_expanded], dim=1)

        h_next = F.mish(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            h_next = block(h_next)

        # Global average pool over 96 nodes
        r_pooled = h_next.mean(dim=2)
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
        self.val_conv = GraphConv1d(d_model, 1)
        self.val_bn = nn.BatchNorm1d(1)
        self.value_fc1 = nn.Linear(96, 64)
        self.value_fc2 = nn.Linear(64, 2 * support_size + 1)

        self.pol_conv = GraphConv1d(d_model, 2)
        self.pol_bn = nn.BatchNorm1d(2)
        self.policy_fc1 = nn.Linear(2 * 96, 288)

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


class ProjectorNet(nn.Module):
    """
    SimSiam-style Projection Head mapping raw latent representations into a compressed 
    feature space for Contrastive Alignment Loss.
    """
    def __init__(self, d_model: int = 128, proj_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.conv1 = GraphConv1d(d_model, d_model // 2)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.fc1 = nn.Linear((d_model // 2) * 96, proj_dim)
        self.bn2 = nn.BatchNorm1d(proj_dim)
        self.fc2 = nn.Linear(proj_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        x = F.mish(self.bn1(self.conv1(h)))
        x = x.view(B, -1)
        x = F.mish(self.bn2(self.fc1(x)))
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

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Projects latent state into structural alignment space."""
        return self.projector(h)  # type: ignore[no-any-return]

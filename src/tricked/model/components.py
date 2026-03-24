import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricked.env.constants import get_neighbors


class GraphConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        A = torch.zeros(96, 96)
        for i in range(96):
            A[i, i] = 1.0
            
        for i in range(96):
            for j in get_neighbors(i):
                A[i, j] = 1.0

        D = A.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat = torch.diag(D_inv_sqrt)
        A_norm = D_mat @ A @ D_mat
        self.register_buffer("A_norm", A_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        msg = torch.matmul(x, self.A_norm)  # type: ignore
        return F.linear(msg.transpose(1, 2), self.weight, self.bias).transpose(1, 2)


class FlattenedResNetBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = GraphConv1d(d_model, d_model)
        self.conv2 = GraphConv1d(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = x.transpose(1, 2)
        x = self.conv1(x).transpose(1, 2)
        x = F.mish(self.norm1(x))
        
        x = x.transpose(1, 2)
        x = self.conv2(x).transpose(1, 2)
        x = self.norm2(x)
        
        return residual + x

class RepresentationNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(20, d_model)
        self.blocks = nn.ModuleList([FlattenedResNetBlock(d_model) for _ in range(num_blocks)])

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
    def __init__(self, d_model: int = 128, num_blocks: int = 8, support_size: int = 200):
        super().__init__()
        self.piece_emb = nn.Embedding(48, d_model)
        self.pos_emb = nn.Embedding(96, d_model)
        
        self.proj_in = nn.Linear(d_model * 2, d_model)
        self.blocks = nn.ModuleList([FlattenedResNetBlock(d_model) for _ in range(num_blocks)])

        # REMOVED GRU. Replaced with standard MLP for Markov Decision Process
        self.reward_fc1 = nn.Linear(d_model * 2, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)

    def forward(self, h: torch.Tensor, a: torch.Tensor, piece_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_idx = a % 96
        a_emb = self.piece_emb(piece_id) + self.pos_emb(pos_idx)

        h_t_pooled = h.mean(dim=2)
        # Standard MLP concatenation
        r_input = torch.cat([h_t_pooled, a_emb], dim=1)
        r = F.mish(self.reward_norm(self.reward_fc1(r_input)))
        reward_logits = self.reward_fc2(r)

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
        v = F.mish(self.val_norm(self.val_proj(x))).mean(dim=1)
        v = F.mish(self.value_fc1(v))
        value_logits = self.value_fc2(v)

        p = F.mish(self.pol_norm(self.pol_proj(x))).mean(dim=1)
        policy_logits = self.policy_fc1(p)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        hole_logits = self.hole_predictor(x).squeeze(-1)
        return value_logits, policy_probs, hole_logits

class ProjectorNet(nn.Module):
    def __init__(self, d_model: int = 128, proj_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model // 2)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.fc1 = nn.Linear(d_model // 2, proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.fc2 = nn.Linear(proj_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.mish(self.norm1(self.proj(h.transpose(1, 2)))).mean(dim=1)
        x = F.mish(self.norm2(self.fc1(x)))
        return self.fc2(x)  # type: ignore

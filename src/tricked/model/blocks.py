import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricked.env.constants import get_neighbors


class GraphConv1d(nn.Module):
    gather_indices: torch.Tensor
    gather_weights: torch.Tensor

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
            for j in get_neighbors(i):
                A[i, j] = 1.0

        D = A.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat = torch.diag(D_inv_sqrt)
        A_norm = D_mat @ A @ D_mat
        
        max_deg = 5
        indices = torch.zeros(96, max_deg, dtype=torch.long)
        weights = torch.zeros(96, max_deg)
        for i in range(96):
            neighbors = [i] + get_neighbors(i)
            for k, j in enumerate(neighbors):
                indices[i, k] = j
                weights[i, k] = A_norm[i, j]
                
        self.register_buffer("gather_indices", indices)
        self.register_buffer("gather_weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        msg = (x[:, :, self.gather_indices] * self.gather_weights).sum(dim=-1)
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

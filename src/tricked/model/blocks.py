import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricked.env.constants import get_neighbors


class GraphConv1d(nn.Module):
    A_sparse: torch.Tensor

    def __init__(self, in_channels: int, out_channels: int, grid_size: int = 96):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        A = torch.zeros(grid_size, grid_size)
        for i in range(grid_size):
            A[i, i] = 1.0
            for j in get_neighbors(i):
                if j < grid_size:
                    A[i, j] = 1.0

        D = A.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat = torch.diag(D_inv_sqrt)
        A_norm = D_mat @ A @ D_mat

        self.register_buffer("A_dense", A_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, in_channels, grid_size]
        B = x.size(0)
        # Reshape to [grid_size, B * in_channels] for dense matmul
        x_reshaped = x.transpose(1, 2).reshape(self.grid_size, -1)

        # Perform rock-solid dense matrix multiplication. 
        # Natively leverages Tensor Cores in FP16 under autocast, bypassing the
        # catastrophic segfaults of PyTorch's FP16 sparse_csr multi-threading.
        msg_fp32 = torch.matmul(self.A_dense, x_reshaped)
        
        # Reshape back to [B, in_channels, grid_size]
        msg = msg_fp32.view(self.grid_size, B, self.in_channels).permute(1, 2, 0)

        # Linear projection
        return F.linear(msg.transpose(1, 2), self.weight, self.bias).transpose(1, 2)


class FlattenedResNetBlock(nn.Module):
    def __init__(self, d_model: int, grid_size: int = 96):
        super().__init__()
        self.conv1 = GraphConv1d(d_model, d_model, grid_size)
        self.conv2 = GraphConv1d(d_model, d_model, grid_size)
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

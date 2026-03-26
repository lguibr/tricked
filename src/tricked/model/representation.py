import torch
import torch.nn as nn

from tricked.model.blocks import FlattenedResNetBlock


class RepresentationNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(20, d_model)
        self.blocks = nn.ModuleList([FlattenedResNetBlock(d_model) for _ in range(num_blocks)])
        self.scale_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.scale_norm(h).transpose(1, 2)

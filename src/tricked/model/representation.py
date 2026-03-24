import torch
import torch.nn as nn

from tricked.model.blocks import FlattenedResNetBlock


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

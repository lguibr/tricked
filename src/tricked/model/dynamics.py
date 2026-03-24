import torch
import torch.nn as nn
import torch.nn.functional as F

from tricked.model.blocks import FlattenedResNetBlock


class DynamicsNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8, support_size: int = 30):
        super().__init__()
        self.piece_emb = nn.Embedding(48, d_model)
        self.pos_emb = nn.Embedding(96, d_model)
        
        self.proj_in = nn.Linear(d_model * 2, d_model)
        self.blocks = nn.ModuleList([FlattenedResNetBlock(d_model) for _ in range(num_blocks)])

        self.reward_cond = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.reward_fc1 = nn.Linear(d_model, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)

    def forward(self, h: torch.Tensor, a: torch.Tensor, piece_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_idx = a % 96
        a_emb = self.piece_emb(piece_id) + self.pos_emb(pos_idx)

        a_expanded = a_emb.unsqueeze(-1).expand(-1, -1, h.size(-1))
        x = torch.cat([h, a_expanded], dim=1)

        r_conv = F.mish(self.reward_cond(x)) 
        h_t_pooled = r_conv.mean(dim=2) 
        
        r = F.mish(self.reward_norm(self.reward_fc1(h_t_pooled)))
        reward_logits = self.reward_fc2(r)

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

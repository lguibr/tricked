import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionNet(nn.Module):
    def __init__(self, d_model: int = 128, support_size: int = 30, num_actions: int = 288):
        super().__init__()
        self.val_proj = nn.Linear(d_model, d_model // 2)
        self.val_norm = nn.LayerNorm(d_model // 2)
        self.value_fc1 = nn.Linear(d_model // 2, 64)
        self.value_fc2 = nn.Linear(64, 2 * support_size + 1)

        self.pol_proj = nn.Linear(d_model, d_model // 2)
        self.pol_norm = nn.LayerNorm(d_model // 2)
        self.policy_fc1 = nn.Linear(d_model // 2, num_actions)

        self.hole_predictor = nn.Sequential(nn.Linear(d_model, 64), nn.Mish(), nn.Linear(64, 1))

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

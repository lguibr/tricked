"""
Standard Documentation for network.py.

This module supplies the core execution logic for the `model` namespace, heavily typed and tested for production distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricked.model.dynamics import DynamicsNet
from tricked.model.prediction import PredictionNet, ProjectorNet
from tricked.model.representation import RepresentationNet


class MuZeroNet(nn.Module):
    def __init__(self, d_model: int = 128, num_blocks: int = 8, support_size: int = 200):
        super().__init__()
        self.support_size = support_size
        self.epsilon = 0.001
        self.representation = RepresentationNet(d_model, num_blocks=num_blocks)
        self.dynamics = DynamicsNet(d_model, num_blocks=num_blocks, support_size=support_size)
        self.prediction = PredictionNet(d_model, support_size=support_size)
        self.projector = ProjectorNet(d_model=d_model)

        self.register_buffer(
            "support_vector", torch.arange(-support_size, support_size + 1, dtype=torch.float32)
        )

    def support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)
        sv = self.support_vector.float()
        assert isinstance(sv, torch.Tensor)
        sym_scalar = torch.sum(probs * sv, dim=-1)

        epsilon = self.epsilon
        y = torch.abs(sym_scalar)
        y = torch.clamp(y, min=0.0, max=float(self.support_size))

        z = (-1.0 + torch.sqrt(1.0 + 4.0 * epsilon * (1.0 + epsilon + y))) / (2.0 * epsilon)
        x = z**2 - 1.0

        scalar = torch.sign(sym_scalar) * x
        return scalar

    def scalar_to_support(self, scalar: torch.Tensor) -> torch.Tensor:
        scalar = torch.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)
        sym_scalar = (
            torch.sign(scalar) * (torch.sqrt(torch.abs(scalar) + 1.0) - 1.0) + self.epsilon * scalar
        )
        sym_scalar = sym_scalar.reshape(-1).clamp(-self.support_size, self.support_size)
        probabilities = torch.zeros(
            sym_scalar.size(0), 2 * self.support_size + 1, device=sym_scalar.device
        )

        lower = sym_scalar.floor()
        upper = sym_scalar.ceil()

        p_upper = sym_scalar - lower
        p_lower = 1.0 - p_upper

        lower_idx = (lower + self.support_size).long()
        upper_idx = (upper + self.support_size).long()

        probabilities.scatter_add_(1, lower_idx.unsqueeze(1), p_lower.unsqueeze(1))
        probabilities.scatter_add_(1, upper_idx.unsqueeze(1), p_upper.unsqueeze(1))

        return probabilities

    def initial_inference(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.representation(s)
        value_logits, policy, hole_logits = self.prediction(h)
        value_scalar = self.support_to_scalar(value_logits)
        return h, value_scalar, policy, hole_logits

    @torch.jit.export
    def initial_inference_jit(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.representation(s)
        value_logits, policy, hole_logits = self.prediction(h)
        value_scalar = self.support_to_scalar(value_logits)
        policy_probs = torch.softmax(policy, dim=-1)
        return h, value_scalar, policy_probs, hole_logits

    def recurrent_inference(
        self, h: torch.Tensor, a: torch.Tensor, piece_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_next, reward_logits = self.dynamics(h, a, piece_id)
        value_logits, policy, hole_logits = self.prediction(h_next)
        reward_scalar = self.support_to_scalar(reward_logits)
        value_scalar = self.support_to_scalar(value_logits)
        return h_next, reward_scalar, value_scalar, policy, hole_logits

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)  # type: ignore

    @torch.jit.export
    def forward(
        self, h: torch.Tensor, a: torch.Tensor, piece_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Native entry point for LibTorch (Rust).
        Maps directly to the recurrent inference step required by the MCTS unroll.
        """
        h_next, reward_scalar, value_scalar, policy, hole_logits = self.recurrent_inference(
            h, a, piece_id
        )
        policy_probs = torch.softmax(policy, dim=-1)
        return h_next, reward_scalar, value_scalar, policy_probs, hole_logits

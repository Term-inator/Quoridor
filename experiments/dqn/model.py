"""Masked Q network for the local Double DQN experiment."""

from __future__ import annotations

import torch
from torch import nn

from experiments.model import board_encoder


class MaskedQNetwork(nn.Module):
    """Estimate action values and select only legal actions."""

    def __init__(self, action_count: int = 209) -> None:
        super().__init__()
        self.features = board_encoder()
        self.q_head = nn.Linear(256, action_count)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.features(observation))

    @torch.no_grad()
    def select_actions(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        epsilon: float,
    ) -> torch.Tensor:
        legal = action_mask.bool()
        if legal.ndim != 2 or not legal.any(dim=1).all():
            raise ValueError("every observation must have at least one legal action")
        q_values = self(observation).masked_fill(~legal, -torch.inf)
        greedy = q_values.argmax(dim=1)
        if epsilon <= 0:
            return greedy
        random_actions = torch.multinomial(legal.float(), num_samples=1).squeeze(1)
        if epsilon >= 1:
            return random_actions
        explore = torch.rand(len(observation), device=observation.device) < epsilon
        return torch.where(explore, random_actions, greedy)

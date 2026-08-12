"""Policy-value network for the AlphaZero experiment."""

from __future__ import annotations

import torch
from torch import nn

from experiments.model import board_encoder


class PolicyValueNetwork(nn.Module):
    """Compact shared network predicting action logits and terminal value."""

    def __init__(self, action_count: int = 209) -> None:
        super().__init__()
        self.features = board_encoder()
        self.policy_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observation)
        logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features)).squeeze(-1)
        return logits, value

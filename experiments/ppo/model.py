"""Shared masked policy and value network for the PPO experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from experiments.model import board_encoder


class MaskedActorCritic(nn.Module):
    """A compact shared CNN with policy and value heads."""

    def __init__(self, action_count: int = 209) -> None:
        super().__init__()
        self.features = board_encoder()
        self.policy_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observation)
        logits = self.policy_head(features)
        logits = logits.masked_fill(~action_mask.bool(), -torch.inf)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        """Estimate the return for one or more canonical observations."""
        return self.value_head(self.features(observation)).squeeze(-1)

    def action_and_value(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self(observation, action_mask)
        distribution = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            values,
        )

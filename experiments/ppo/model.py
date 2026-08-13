"""PPO 实验共享参数、带合法动作掩码的策略—价值网络。"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from experiments.model import board_encoder


class MaskedActorCritic(nn.Module):
    """共享紧凑卷积特征，并分别输出策略 logits 与状态价值。"""

    def __init__(self, action_count: int = 209) -> None:
        """创建共享棋盘编码器、固定动作策略头和标量价值头。"""
        super().__init__()
        self.features = board_encoder()
        self.policy_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算策略与价值，并把非法动作 logits 设为负无穷。"""
        features = self.features(observation)
        logits = self.policy_head(features)
        logits = logits.masked_fill(~action_mask.bool(), -torch.inf)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        """估计一个或一批规范视角观测的期望回报。"""
        return self.value_head(self.features(observation)).squeeze(-1)

    def action_and_value(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样或确定性选择动作，同时返回对数概率、熵和价值。"""
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

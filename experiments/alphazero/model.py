"""AlphaZero 实验使用的策略—价值网络。"""

from __future__ import annotations

import torch
from torch import nn

from experiments.model import board_encoder


class PolicyValueNetwork(nn.Module):
    """共享紧凑棋盘特征，同时预测动作 logits 和有界终局价值。"""

    def __init__(self, action_count: int = 209) -> None:
        """创建共享编码器、固定动作策略头和标量价值头。"""
        super().__init__()
        self.features = board_encoder()
        self.policy_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回未掩码策略 logits，以及经 ``tanh`` 限制到 [-1, 1] 的价值。"""
        features = self.features(observation)
        logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features)).squeeze(-1)
        return logits, value

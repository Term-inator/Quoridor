"""本地 Double DQN 实验使用的带掩码 Q 网络。"""

from __future__ import annotations

import torch
from torch import nn

from experiments.model import board_encoder
from quoridor_rl.codec import ActionCodec


class MaskedQNetwork(nn.Module):
    """估计全部动作价值，并保证选择结果只落在合法动作中。"""

    def __init__(self, action_count: int = ActionCodec.action_count) -> None:
        """以共享棋盘编码器连接 209 维线性 Q 值头。"""
        super().__init__()
        self.features = board_encoder()
        self.q_head = nn.Linear(256, action_count)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """返回批量观测中每个固定动作编号的未掩码 Q 值。"""
        return self.q_head(self.features(observation))

    @torch.no_grad()
    def select_actions(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        epsilon: float,
    ) -> torch.Tensor:
        """按 ε-greedy 策略逐样本选择合法动作。"""
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

"""DQN 自我对弈使用的冻结历史对手池。"""

from __future__ import annotations

import copy
import random
from collections import deque

from experiments.dqn.model import MaskedQNetwork


class OpponentPool:
    """保留最近策略快照，并以一定概率使用随机对手作为稳定锚点。"""

    def __init__(
        self,
        capacity: int,
        *,
        random_probability: float,
        seed: int,
    ) -> None:
        """创建有界先进先出快照队列和独立随机源。"""
        if capacity <= 0:
            raise ValueError("opponent capacity must be positive")
        if not 0 <= random_probability <= 1:
            raise ValueError("random probability must be between zero and one")
        self.capacity = capacity
        self.random_probability = random_probability
        self._snapshots: deque[MaskedQNetwork] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        """返回当前冻结策略快照数。"""
        return len(self._snapshots)

    def add(self, model: MaskedQNetwork) -> None:
        """深复制并冻结当前模型，避免后续在线训练污染历史对手。"""
        snapshot = copy.deepcopy(model).eval()
        snapshot.requires_grad_(False)
        self._snapshots.append(snapshot)

    def sample(self) -> MaskedQNetwork | None:
        """返回 ``None`` 代表随机对手，否则返回一个冻结网络快照。"""
        if not self._snapshots or self._random.random() < self.random_probability:
            return None
        return self._random.choice(tuple(self._snapshots))

"""Frozen historical opponents for DQN self-play."""

from __future__ import annotations

import copy
import random
from collections import deque

from experiments.dqn.model import MaskedQNetwork


class OpponentPool:
    """Keep recent frozen policies and retain random play as an anchor."""

    def __init__(
        self,
        capacity: int,
        *,
        random_probability: float,
        seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("opponent capacity must be positive")
        if not 0 <= random_probability <= 1:
            raise ValueError("random probability must be between zero and one")
        self.capacity = capacity
        self.random_probability = random_probability
        self._snapshots: deque[MaskedQNetwork] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._snapshots)

    def add(self, model: MaskedQNetwork) -> None:
        snapshot = copy.deepcopy(model).eval()
        snapshot.requires_grad_(False)
        self._snapshots.append(snapshot)

    def sample(self) -> MaskedQNetwork | None:
        """Return None for a random opponent or a frozen network snapshot."""
        if not self._snapshots or self._random.random() < self.random_probability:
            return None
        return self._random.choice(tuple(self._snapshots))

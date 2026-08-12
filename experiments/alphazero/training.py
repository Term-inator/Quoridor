"""Replay storage and policy-value updates for AlphaZero."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from experiments.alphazero.model import PolicyValueNetwork


@dataclass(frozen=True, slots=True)
class AlphaZeroConfig:
    seed: int = 0
    max_plies: int = 512
    simulations_per_move: int = 32
    evaluation_simulations: int = 8
    evaluation_workers: int = 4
    maximum_search_actions: int = 16
    pawn_only_curriculum_games: int = 32
    curriculum_progress_prior: float = 0.75
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.03
    root_noise_fraction: float = 0.25
    temperature_plies: int = 30
    replay_capacity: int = 50_000
    replay_warmup: int = 256
    batch_size: int = 256
    updates_per_game: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_gradient_norm: float = 5.0
    torch_threads: int = 4


@dataclass(frozen=True, slots=True)
class TrainingExample:
    observation: np.ndarray
    policy: np.ndarray
    value: float


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    observations: torch.Tensor
    policies: torch.Tensor
    values: torch.Tensor

    def to(self, device: torch.device) -> TrainingBatch:
        return TrainingBatch(
            observations=self.observations.to(device),
            policies=self.policies.to(device),
            values=self.values.to(device),
        )


class ReplayBuffer:
    """Fixed-capacity replay for canonical observations and search targets."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._observations = np.empty((capacity, 6, 9, 9), dtype=np.uint8)
        self._policies = np.empty((capacity, 209), dtype=np.float16)
        self._values = np.empty(capacity, dtype=np.float32)
        self._size = 0
        self._position = 0
        self._random = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, example: TrainingExample) -> None:
        if example.observation.shape != (6, 9, 9):
            raise ValueError("observation must have shape (6, 9, 9)")
        if example.policy.shape != (209,):
            raise ValueError("policy must have shape (209,)")
        if not np.isclose(example.policy.sum(), 1.0):
            raise ValueError("policy target must sum to one")
        if not -1 <= example.value <= 1:
            raise ValueError("value target must be between -1 and one")
        index = self._position
        self._observations[index] = np.rint(example.observation * 10).astype(np.uint8)
        self._policies[index] = example.policy.astype(np.float16)
        self._values[index] = example.value
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TrainingBatch:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if batch_size > self._size:
            raise ValueError("not enough replay examples for this batch")
        indices = self._random.choice(self._size, size=batch_size, replace=False)
        return TrainingBatch(
            observations=torch.from_numpy(
                self._observations[indices].astype(np.float32) / 10
            ),
            policies=torch.from_numpy(self._policies[indices].astype(np.float32)),
            values=torch.from_numpy(self._values[indices].copy()),
        )


class PolicyValueUpdater:
    """Optimize policy cross-entropy and terminal-value regression."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        config: AlphaZeroConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def update(self, batch: TrainingBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.train()
        logits, values = self.model(batch.observations)
        policy_loss = (
            -(batch.policies * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        )
        value_loss = nn.functional.mse_loss(values, batch.values)
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("AlphaZero loss became NaN or infinite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_gradient_norm
        )
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "value_mean": float(values.mean().item()),
            "gradient_norm": float(gradient_norm.item()),
        }

"""Replay and update logic for the masked Double DQN experiment."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from experiments.dqn.model import MaskedQNetwork


@dataclass(frozen=True, slots=True)
class DQNConfig:
    seed: int = 0
    environment_count: int = 4
    max_plies: int = 512
    collection_size: int = 256
    replay_capacity: int = 200_000
    replay_warmup: int = 10_000
    batch_size: int = 512
    update_interval: int = 4
    target_sync_interval: int = 5_000
    opponent_snapshot_interval: int = 50_000
    opponent_pool_capacity: int = 8
    random_opponent_probability: float = 0.2
    learning_rate: float = 1e-4
    gamma: float = 0.99
    max_gradient_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_transitions: int = 200_000


@dataclass(frozen=True, slots=True)
class Transition:
    observation: torch.Tensor
    action_mask: torch.Tensor
    action: int
    reward: float
    next_observation: torch.Tensor
    next_action_mask: torch.Tensor
    done: bool
    episode: int = 0
    agent: str = "player_0"


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    observations: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    next_action_masks: torch.Tensor
    done: torch.Tensor

    def to(self, device: torch.device) -> TransitionBatch:
        return TransitionBatch(
            observations=self.observations.to(device),
            action_masks=self.action_masks.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_observations=self.next_observations.to(device),
            next_action_masks=self.next_action_masks.to(device),
            done=self.done.to(device),
        )


class ReplayBuffer:
    """Fixed-capacity uniform replay with compact lossless board storage."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._observations = torch.empty((capacity, 6, 9, 9), dtype=torch.uint8)
        self._action_masks = torch.empty((capacity, 209), dtype=torch.bool)
        self._actions = torch.empty(capacity, dtype=torch.int64)
        self._rewards = torch.empty(capacity, dtype=torch.float32)
        self._next_observations = torch.empty((capacity, 6, 9, 9), dtype=torch.uint8)
        self._next_action_masks = torch.empty((capacity, 209), dtype=torch.bool)
        self._done = torch.empty(capacity, dtype=torch.bool)
        self._size = 0
        self._position = 0
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        index = self._position
        self._observations[index].copy_(_pack_observation(transition.observation))
        self._action_masks[index].copy_(transition.action_mask.bool())
        self._actions[index] = transition.action
        self._rewards[index] = transition.reward
        self._next_observations[index].copy_(
            _pack_observation(transition.next_observation)
        )
        self._next_action_masks[index].copy_(transition.next_action_mask.bool())
        self._done[index] = transition.done
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if batch_size > self._size:
            raise ValueError("not enough replay transitions for this batch")
        indices = torch.randperm(self._size, generator=self._generator)[:batch_size]
        return TransitionBatch(
            observations=_unpack_observation(self._observations[indices]),
            action_masks=self._action_masks[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=_unpack_observation(self._next_observations[indices]),
            next_action_masks=self._next_action_masks[indices],
            done=self._done[indices],
        )


def _pack_observation(observation: torch.Tensor) -> torch.Tensor:
    if observation.shape != (6, 9, 9):
        raise ValueError("observation must have shape (6, 9, 9)")
    return observation.mul(10).round().to(dtype=torch.uint8, device="cpu")


def _unpack_observation(observation: torch.Tensor) -> torch.Tensor:
    return observation.float().div(10)


class DQNUpdater:
    """Apply masked one-step Double DQN updates."""

    def __init__(
        self,
        online: MaskedQNetwork,
        config: DQNConfig,
        device: torch.device,
    ) -> None:
        self.online = online
        self.config = config
        self.device = device
        self.target = copy.deepcopy(online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            online.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        chosen_are_legal = batch.action_masks.gather(
            1, batch.actions.unsqueeze(1)
        ).squeeze(1)
        if not chosen_are_legal.all():
            raise ValueError("replay contains an illegal chosen action")
        self.online.train()
        chosen_values = (
            self.online(batch.observations)
            .gather(1, batch.actions.unsqueeze(1))
            .squeeze(1)
        )
        targets = double_dqn_targets(
            self.online,
            self.target,
            batch.next_observations,
            batch.next_action_masks,
            rewards=batch.rewards,
            done=batch.done,
            gamma=self.config.gamma,
        )
        loss = nn.functional.smooth_l1_loss(chosen_values, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN loss became NaN or infinite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.max_gradient_norm
        )
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "q_mean": float(chosen_values.mean().item()),
            "target_mean": float(targets.mean().item()),
            "gradient_norm": float(gradient_norm.item()),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())


@torch.no_grad()
def double_dqn_targets(
    online: MaskedQNetwork,
    target: MaskedQNetwork,
    next_observation: torch.Tensor,
    next_action_mask: torch.Tensor,
    *,
    rewards: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Build masked one-step Double DQN targets for a transition batch."""
    result = rewards.clone()
    active = ~done.bool()
    if not active.any():
        return result
    active_masks = next_action_mask[active].bool()
    if not active_masks.any(dim=1).all():
        raise ValueError("non-terminal next states must have a legal action")
    online_values = online(next_observation[active]).masked_fill(
        ~active_masks,
        -torch.inf,
    )
    next_actions = online_values.argmax(dim=1, keepdim=True)
    target_values = target(next_observation[active]).gather(1, next_actions).squeeze(1)
    result[active] += gamma * target_values
    return result

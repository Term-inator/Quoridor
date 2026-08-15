"""只记录学习方决策的 DQN 自我对弈转移采集器。"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from experiments.dqn.model import MaskedQNetwork
from experiments.dqn.opponents import OpponentPool
from experiments.dqn.training import DQNConfig, Transition
from experiments.reward import PotentialRewardWrapper
from quoridor_rl.env import env

AGENTS = ("player_0", "player_1")


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    """一局结束后用于实验汇总的学习方视角统计。"""

    episode: int
    learner_agent: str
    plies: int
    winner: str | None
    terminated: bool
    truncated: bool
    shaped_return: float


@dataclass(frozen=True, slots=True)
class Collection:
    """一次采集返回的学习转移和完整对局统计。"""

    transitions: list[Transition]
    episodes: list[EpisodeStats]


@dataclass(slots=True)
class _PendingTransition:
    """等待对手行动结束后才能补齐下一状态的学习方决策。"""

    observation: torch.Tensor
    action_mask: torch.Tensor
    action: int
    reward: float = 0.0


@dataclass(slots=True)
class _Slot:
    """一个并行轮询的环境槽及其局内采集状态。"""

    environment: PotentialRewardWrapper
    learner_agent: str
    opponent: MaskedQNetwork | None
    episode: int
    shaped_return: float = 0.0
    pending: _PendingTransition | None = None


class SelfPlayCollector:
    """让在线学习方对战冻结对手，只采集学习方产生的转移。

    AEC 环境在双方之间交替，因此学习方动作的 ``next_observation`` 必须等对手走完后
    才能确定；``_PendingTransition`` 用于跨过对手回合累计奖励并闭合该转移。
    """

    def __init__(
        self,
        online: MaskedQNetwork,
        opponents: OpponentPool,
        config: DQNConfig,
        device: torch.device,
    ) -> None:
        """创建指定数量的环境槽，并为每局随机分配学习方角色。"""
        self.online = online
        self.opponents = opponents
        self.config = config
        self.device = device
        self._random = random.Random(config.seed)
        self._next_episode = 0
        self._slots = [self._make_slot() for _ in range(config.environment_count)]

    def collect(self, minimum_transitions: int, *, epsilon: float) -> Collection:
        """轮询所有环境槽，直到至少收集指定数量的学习方转移。"""
        if minimum_transitions <= 0:
            raise ValueError("minimum transitions must be positive")
        transitions: list[Transition] = []
        episodes: list[EpisodeStats] = []
        while len(transitions) < minimum_transitions:
            for slot in self._slots:
                self._step_slot(slot, epsilon, transitions, episodes)
        return Collection(transitions=transitions, episodes=episodes)

    def _make_slot(self) -> _Slot:
        """创建并重置一个带塑形奖励的新环境槽。"""
        environment = PotentialRewardWrapper(env(max_plies=self.config.max_plies))
        environment.reset()
        slot = _Slot(
            environment=environment,
            learner_agent=self._random.choice(AGENTS),
            opponent=self.opponents.sample(),
            episode=self._next_episode,
        )
        self._next_episode += 1
        return slot

    def _reset_slot(self, slot: _Slot) -> None:
        """复用已结束环境槽，并重新抽取角色和对手。"""
        slot.environment.reset()
        slot.learner_agent = self._random.choice(AGENTS)
        slot.opponent = self.opponents.sample()
        slot.episode = self._next_episode
        slot.shaped_return = 0.0
        slot.pending = None
        self._next_episode += 1

    def _step_slot(
        self,
        slot: _Slot,
        epsilon: float,
        transitions: list[Transition],
        episodes: list[EpisodeStats],
    ) -> None:
        """推进环境槽一个 AEC 行动，必要时闭合转移或记录整局统计。"""
        acting_agent = slot.environment.agent_selection
        observation, action_mask = _observe(slot.environment, acting_agent)
        if acting_agent == slot.learner_agent:
            if slot.pending is not None:
                transitions.append(
                    _close_pending(
                        slot,
                        observation,
                        action_mask,
                        done=False,
                    )
                )
            action = int(
                self.online.select_actions(
                    observation.unsqueeze(0).to(self.device),
                    action_mask.unsqueeze(0).to(self.device),
                    epsilon=epsilon,
                )[0].item()
            )
            slot.pending = _PendingTransition(observation, action_mask, action)
        elif slot.opponent is None:
            legal_actions = np.flatnonzero(action_mask.numpy()).tolist()
            action = self._random.choice(legal_actions)
        else:
            action = int(
                slot.opponent.select_actions(
                    observation.unsqueeze(0).to(self.device),
                    action_mask.unsqueeze(0).to(self.device),
                    epsilon=0.0,
                )[0].item()
            )

        slot.environment.step(action)
        reward = float(slot.environment.rewards[slot.learner_agent])
        slot.shaped_return += reward
        if slot.pending is not None:
            slot.pending.reward += reward

        terminated = all(slot.environment.terminations.values())
        truncated = all(slot.environment.truncations.values())
        if not terminated and not truncated:
            return

        if slot.pending is not None:
            terminal_observation, terminal_mask = _observe(
                slot.environment, slot.learner_agent
            )
            transitions.append(
                _close_pending(
                    slot,
                    terminal_observation,
                    terminal_mask,
                    done=True,
                )
            )
        winner = slot.environment.unwrapped.position.winner
        episodes.append(
            EpisodeStats(
                episode=slot.episode,
                learner_agent=slot.learner_agent,
                plies=slot.environment.unwrapped.plies,
                winner=None if winner is None else f"player_{int(winner)}",
                terminated=terminated,
                truncated=truncated,
                shaped_return=slot.shaped_return,
            )
        )
        self._reset_slot(slot)


def _observe(
    environment: PotentialRewardWrapper,
    agent: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把环境的 NumPy 观测和动作掩码转换为 CPU 张量。"""
    observed = environment.observe(agent)
    return (
        torch.from_numpy(observed["observation"]).float(),
        torch.from_numpy(observed["action_mask"]).bool(),
    )


def _close_pending(
    slot: _Slot,
    next_observation: torch.Tensor,
    next_action_mask: torch.Tensor,
    *,
    done: bool,
) -> Transition:
    """用下一决策点信息闭合待定转移，并清空槽内暂存。"""
    pending = slot.pending
    if pending is None:
        raise RuntimeError("cannot close an absent learner transition")
    slot.pending = None
    return Transition(
        observation=pending.observation,
        action_mask=pending.action_mask,
        action=pending.action,
        reward=pending.reward,
        next_observation=next_observation,
        next_action_mask=next_action_mask,
        done=done,
        episode=slot.episode,
        agent=slot.learner_agent,
    )

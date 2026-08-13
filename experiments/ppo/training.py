"""本地 PPO 学习验证的完整轨迹采集与裁剪目标更新。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.envs import PettingZooWrapper
from torchrl.envs.utils import step_mdp
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from experiments.ppo.model import MaskedActorCritic
from experiments.reward import PotentialRewardWrapper
from quoridor_rl import env

AGENTS = ("player_0", "player_1")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO 环境、采样、GAE 和优化相关超参数。"""
    seed: int = 0
    environment_count: int = 4
    rollout_size: int = 4096
    max_plies: int = 512
    minibatch_size: int = 512
    update_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5


@dataclass(frozen=True, slots=True)
class Transition:
    """某玩家两个相邻决策点之间的一条 on-policy 转移。"""
    trajectory: int
    agent: str
    observation: torch.Tensor
    action_mask: torch.Tensor
    action: int
    log_probability: float
    value: float
    reward: float
    next_value: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    """完整自我对弈局的长度、结果和双方塑形回报。"""
    plies: int
    winner: str | None
    terminated: bool
    truncated: bool
    shaped_returns: dict[str, float]


@dataclass(frozen=True, slots=True)
class Rollout:
    """一次采样阶段产生的完整转移与对局集合。"""
    transitions: list[Transition]
    episodes: list[EpisodeStats]


class PPOUpdater:
    """把 TorchRL 的裁剪 PPO 损失应用到 AEC 轨迹。

    适配器把项目模型接入 TorchRL 的 actor/critic 接口；更新前按玩家与对局重建序列并
    计算 GAE，再打散成小批次重复优化。
    """

    def __init__(
        self,
        model: MaskedActorCritic,
        config: PPOConfig,
        device: torch.device,
    ) -> None:
        """组装策略、价值、PPO 损失、GAE 与共享参数优化器。"""
        self.model = model
        self.config = config
        self.device = device
        actor = ProbabilisticActor(
            TensorDictModule(
                _PolicyAdapter(model),
                in_keys=["observation", "action_mask"],
                out_keys=["logits"],
            ),
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )
        critic = ValueOperator(
            _ValueAdapter(model),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        self.loss = ClipPPOLoss(
            actor,
            critic,
            clip_epsilon=config.clip_epsilon,
            entropy_coeff=config.entropy_coefficient,
            critic_coeff=config.value_coefficient,
            normalize_advantage=True,
            functional=False,
        ).to(device)
        self.gae = GAE(
            gamma=config.gamma,
            lmbda=config.gae_lambda,
            value_network=None,
            time_dim=0,
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

    def update(self, transitions: list[Transition]) -> dict[str, float]:
        """计算优势后执行多轮小批量更新并返回平均诊断指标。"""
        batch = _training_batch(transitions, self.gae).to(self.device)
        metric_totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0
        self.model.eval()
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(len(batch), device=self.device)
            for start in range(0, len(batch), self.config.minibatch_size):
                minibatch = batch[
                    permutation[start : start + self.config.minibatch_size]
                ]
                losses = self.loss(minibatch)
                total_loss = (
                    losses["loss_objective"]
                    + losses["loss_critic"]
                    + losses["loss_entropy"]
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("PPO loss became NaN or infinite")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_gradient_norm,
                )
                self.optimizer.step()
                metric_totals["policy_loss"] += float(losses["loss_objective"].item())
                metric_totals["value_loss"] += float(losses["loss_critic"].item())
                metric_totals["entropy"] += float(losses["entropy"].item())
                metric_totals["clip_fraction"] += float(losses["clip_fraction"].item())
                update_count += 1
        return {name: total / update_count for name, total in metric_totals.items()}


class _PolicyAdapter(nn.Module):
    """把联合模型适配为 TorchRL 只返回策略 logits 的模块。"""
    def __init__(self, model: MaskedActorCritic) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(observation, action_mask)[0]


class _ValueAdapter(nn.Module):
    """把联合模型适配为 TorchRL 要求末维为一的价值模块。"""
    def __init__(self, model: MaskedActorCritic) -> None:
        super().__init__()
        self.model = model

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model.value(observation).unsqueeze(-1)


@dataclass(slots=True)
class _PendingTransition:
    """等待同一玩家下一决策点以补齐奖励和下一价值的转移。"""
    trajectory: int
    agent: str
    observation: torch.Tensor
    action_mask: torch.Tensor
    action: int
    log_probability: float
    value: float
    reward: float = 0.0


@dataclass(slots=True)
class _EnvironmentSlot:
    """一个 TorchRL 包装环境及其未闭合转移和局统计。"""
    index: int
    aec: PotentialRewardWrapper
    torch_env: PettingZooWrapper
    state: TensorDict
    pending: dict[str, _PendingTransition] = field(default_factory=dict)
    shaped_returns: dict[str, float] = field(
        default_factory=lambda: {agent: 0.0 for agent in AGENTS}
    )
    episode_number: int = 0
    active: bool = True

    @property
    def trajectory(self) -> int:
        """生成跨环境槽和复用局均唯一的轨迹编号。"""
        return self.episode_number * 10_000 + self.index


def collect_rollout(
    model: MaskedActorCritic,
    config: PPOConfig,
    device: torch.device,
) -> Rollout:
    """使用双方共享的当前策略采集完整自我对弈局，直至满足轨迹规模。"""
    slots = [_make_slot(index, config) for index in range(config.environment_count)]
    transitions: list[Transition] = []
    episodes: list[EpisodeStats] = []
    model.eval()

    while any(slot.active for slot in slots):
        acting_slots = [slot for slot in slots if slot.active]
        observations = torch.stack(
            [_current_observation(slot) for slot in acting_slots]
        ).to(device)
        action_masks = torch.stack(
            [_current_action_mask(slot) for slot in acting_slots]
        ).to(device)
        with torch.no_grad():
            actions, log_probabilities, _, values = model.action_and_value(
                observations,
                action_masks,
            )

        for batch_index, slot in enumerate(acting_slots):
            agent = slot.aec.agent_selection
            if agent in slot.pending:
                transitions.append(
                    _close_pending(
                        slot.pending.pop(agent),
                        next_value=float(values[batch_index].item()),
                        terminated=False,
                        truncated=False,
                    )
                )

            action = int(actions[batch_index].item())
            slot.pending[agent] = _PendingTransition(
                trajectory=slot.trajectory,
                agent=agent,
                observation=observations[batch_index].detach().cpu(),
                action_mask=action_masks[batch_index].detach().cpu(),
                action=action,
                log_probability=float(log_probabilities[batch_index].item()),
                value=float(values[batch_index].item()),
            )
            slot.state.set(
                (agent, "action"),
                torch.tensor([action], dtype=torch.int64),
            )
            stepped = slot.torch_env.step(slot.state)
            slot.state = step_mdp(stepped)
            for rewarded_agent in AGENTS:
                reward = float(slot.state[(rewarded_agent, "reward")].item())
                slot.shaped_returns[rewarded_agent] += reward
                pending = slot.pending.get(rewarded_agent)
                if pending is not None:
                    pending.reward += reward

            if bool(slot.state["done"].item()):
                _finish_episode(slot, model, device, transitions, episodes)
                if len(transitions) >= config.rollout_size:
                    slot.active = False
                else:
                    _reset_slot(slot)

    return Rollout(transitions=transitions, episodes=episodes)


def _make_slot(index: int, config: PPOConfig) -> _EnvironmentSlot:
    """创建带塑形奖励并启用动作掩码的 TorchRL 环境槽。"""
    aec = PotentialRewardWrapper(env(max_plies=config.max_plies))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="PettingZoo in TorchRL is tested using version",
        )
        torch_env = PettingZooWrapper(
            aec,
            use_mask=True,
            categorical_actions=True,
        )
    return _EnvironmentSlot(
        index=index,
        aec=aec,
        torch_env=torch_env,
        state=torch_env.reset(),
    )


def _reset_slot(slot: _EnvironmentSlot) -> None:
    """复用环境槽开始下一局，并清空所有局内累计状态。"""
    slot.episode_number += 1
    slot.pending.clear()
    slot.shaped_returns = {agent: 0.0 for agent in AGENTS}
    slot.state = slot.torch_env.reset()


def _current_observation(slot: _EnvironmentSlot) -> torch.Tensor:
    """从多智能体 TensorDict 取出当前行动方的棋盘观测。"""
    agent = slot.aec.agent_selection
    return slot.state[(agent, "observation", "observation")][0]


def _current_action_mask(slot: _EnvironmentSlot) -> torch.Tensor:
    """从多智能体 TensorDict 取出当前行动方的合法动作掩码。"""
    agent = slot.aec.agent_selection
    return slot.state[(agent, "action_mask")][0]


def _close_pending(
    pending: _PendingTransition,
    *,
    next_value: float,
    terminated: bool,
    truncated: bool,
) -> Transition:
    """把待定决策补齐为不可变训练转移。"""
    return Transition(
        trajectory=pending.trajectory,
        agent=pending.agent,
        observation=pending.observation,
        action_mask=pending.action_mask,
        action=pending.action,
        log_probability=pending.log_probability,
        value=pending.value,
        reward=pending.reward,
        next_value=next_value,
        terminated=terminated,
        truncated=truncated,
    )


def _finish_episode(
    slot: _EnvironmentSlot,
    model: MaskedActorCritic,
    device: torch.device,
    transitions: list[Transition],
    episodes: list[EpisodeStats],
) -> None:
    """闭合一局剩余转移；截断局用价值自举，真实终局价值归零。"""
    terminated = all(slot.aec.terminations.values())
    truncated = all(slot.aec.truncations.values())
    next_values = {agent: 0.0 for agent in AGENTS}
    if truncated:
        terminal_observations = torch.stack(
            [slot.state[(agent, "observation", "observation")][0] for agent in AGENTS]
        ).to(device)
        with torch.no_grad():
            values = model.value(terminal_observations)
        next_values = {
            agent: float(values[index].item()) for index, agent in enumerate(AGENTS)
        }

    for agent, pending in tuple(slot.pending.items()):
        transitions.append(
            _close_pending(
                pending,
                next_value=next_values[agent],
                terminated=terminated,
                truncated=truncated,
            )
        )
    slot.pending.clear()
    winner = slot.aec.unwrapped.position.winner
    episodes.append(
        EpisodeStats(
            plies=slot.aec.unwrapped.plies,
            winner=None if winner is None else f"player_{int(winner)}",
            terminated=terminated,
            truncated=truncated,
            shaped_returns=dict(slot.shaped_returns),
        )
    )


def _training_batch(transitions: list[Transition], gae: GAE) -> TensorDict:
    """按“对局—玩家”恢复时序，逐序列计算 GAE 后拼成训练批次。"""
    trajectories: dict[tuple[int, str], list[Transition]] = {}
    for transition in transitions:
        trajectories.setdefault((transition.trajectory, transition.agent), []).append(
            transition
        )

    sequence_batches: list[TensorDict] = []
    for sequence in trajectories.values():
        length = len(sequence)
        sequence_batch = TensorDict(
            {
                "observation": torch.stack(
                    [transition.observation for transition in sequence]
                ),
                "action_mask": torch.stack(
                    [transition.action_mask for transition in sequence]
                ),
                "action": torch.tensor(
                    [transition.action for transition in sequence],
                    dtype=torch.int64,
                ),
                "action_log_prob": torch.tensor(
                    [transition.log_probability for transition in sequence],
                    dtype=torch.float32,
                ),
                "state_value": torch.tensor(
                    [[transition.value] for transition in sequence],
                    dtype=torch.float32,
                ),
                "next": TensorDict(
                    {
                        "reward": torch.tensor(
                            [[transition.reward] for transition in sequence],
                            dtype=torch.float32,
                        ),
                        "state_value": torch.tensor(
                            [[transition.next_value] for transition in sequence],
                            dtype=torch.float32,
                        ),
                        "done": torch.tensor(
                            [
                                [transition.terminated or transition.truncated]
                                for transition in sequence
                            ],
                            dtype=torch.bool,
                        ),
                        "terminated": torch.tensor(
                            [[transition.terminated] for transition in sequence],
                            dtype=torch.bool,
                        ),
                    },
                    batch_size=[length],
                ),
            },
            batch_size=[length],
        )
        with torch.no_grad():
            gae(sequence_batch)
        sequence_batches.append(sequence_batch)
    return torch.cat(sequence_batches, dim=0)

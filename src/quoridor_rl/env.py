"""双人围墙棋规则的 PettingZoo AEC 环境适配器。

规则状态仍由不可变的 :class:`Position` 管理；本模块只负责 PettingZoo 的轮流行动
协议、张量观测、动作掩码、奖励以及终止/截断信号。
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from quoridor_rl.codec import ActionCodec, ObservationCodec
from quoridor_rl.game import IllegalActionError, Player, Position
from quoridor_rl.language import Language
from quoridor_rl.render import render_ascii

Agent = str
Observation = dict[str, np.ndarray]


def env(
    *,
    max_plies: int = 512,
    render_mode: str | None = None,
    language: Language = Language.CHINESE,
) -> AECEnv:
    """创建带行动顺序检查包装器的公共 PettingZoo AEC 环境。"""
    return OrderEnforcingWrapper(
        QuoridorEnv(max_plies=max_plies, render_mode=render_mode, language=language)
    )


class QuoridorEnv(AECEnv[Agent, Observation, int | None]):
    """由不可变 ``Position`` 驱动、尚未包装的 AEC 环境实现。"""

    metadata: ClassVar[dict[str, Any]] = {
        "name": "quoridor_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        *,
        max_plies: int = 512,
        render_mode: str | None = None,
        language: Language = Language.CHINESE,
    ) -> None:
        """配置回合上限和渲染模式，并声明固定的动作/观测空间。"""
        super().__init__()
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        if not isinstance(language, Language):
            raise TypeError("language must be a Language value")

        self.max_plies = max_plies
        self.render_mode = render_mode
        self.language = language
        self.possible_agents = ["player_0", "player_1"]
        self.action_spaces: dict[Agent, spaces.Discrete] = {
            agent: spaces.Discrete(ActionCodec.action_count)
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(6, 9, 9),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.MultiBinary(ActionCodec.action_count),
                }
            )
            for agent in self.possible_agents
        }
        self._codec = ActionCodec()
        self._observation_codec = ObservationCodec()
        self._position = Position.initial()
        self.plies = 0
        self.agents: list[Agent] = []
        self.agent_selection = "player_0"
        self.rewards: dict[Agent, float] = {}
        self._cumulative_rewards: dict[Agent, float] = {}
        self.terminations: dict[Agent, bool] = {}
        self.truncations: dict[Agent, bool] = {}
        self.infos: dict[Agent, dict[str, Any]] = {}
        self._agent_selector = AgentSelector(self.possible_agents)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """重置规则局面及全部 AEC 逐局状态。

        环境本身没有随机初始状态，因此当前忽略 ``seed`` 和 ``options``。
        """
        del seed, options
        self._position = Position.initial()
        self.plies = 0
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observation_space(self, agent: Agent) -> spaces.Space:
        """返回指定智能体的静态观测空间。"""
        return self.observation_spaces[agent]

    @property
    def position(self) -> Position:
        """返回当前不可变规则局面。"""
        return self._position

    def action_space(self, agent: Agent) -> spaces.Space:
        """返回指定智能体的固定 209 维离散动作空间。"""
        return self.action_spaces[agent]

    def observe(self, agent: Agent) -> Observation:
        """生成规范视角观测和仅在该智能体可行动时有效的动作掩码。"""
        player = _player_for(agent)
        observation = self._observation_codec.encode(self.position, player)

        action_mask = np.zeros(ActionCodec.action_count, dtype=np.int8)
        if (
            agent == self.agent_selection
            and agent in self.agents
            and not self.terminations.get(agent, True)
            and not self.truncations.get(agent, True)
        ):
            for action in self.position.legal_actions():
                action_mask[self._codec.encode(action, player)] = 1

        return {"observation": observation, "action_mask": action_mask}

    def step(self, action: int | None) -> None:
        """执行当前智能体的一步，并更新 AEC 奖励与结束状态。

        无法解码或违反规则的动作会立即判当前玩家负、对手胜；达到 ``max_plies``
        则双方截断而不判胜负。已结束智能体的空动作交给 PettingZoo 基类处理。
        """
        current_agent = self.agent_selection
        if self.terminations[current_agent] or self.truncations[current_agent]:
            self._was_dead_step(action)
            return

        player = _player_for(current_agent)
        next_agent = self._agent_selector.next()
        self._cumulative_rewards[current_agent] = 0.0
        self._clear_rewards()

        # 解码错误和规则错误统一视作非法动作，但不让底层异常泄漏到训练循环。
        semantic_action = None
        if action is not None:
            try:
                semantic_action = self._codec.decode(action, player)
            except (TypeError, ValueError):
                pass

        next_position = None
        if semantic_action is not None:
            try:
                next_position = self.position.play(semantic_action)
            except IllegalActionError:
                pass

        if next_position is None:
            self.rewards[current_agent] = -1.0
            self.rewards[next_agent] = 1.0
            self.terminations = {agent: True for agent in self.agents}
            self.infos[current_agent] = {"illegal_action": True}
            self.agent_selection = next_agent
            self._accumulate_rewards()
            return

        self._position = next_position
        self.plies += 1

        if self.position.winner is not None:
            self.rewards[current_agent] = 1.0
            self.rewards[next_agent] = -1.0
            self.terminations = {agent: True for agent in self.agents}
        elif self.plies >= self.max_plies:
            self.truncations = {agent: True for agent in self.agents}

        self.agent_selection = next_agent
        self._accumulate_rewards()

    def render(self) -> str | None:
        """在 ``ansi`` 模式下返回文本棋盘，否则不产生渲染结果。"""
        if self.render_mode != "ansi":
            return None
        return render_ascii(self.position, language=self.language)


def _player_for(agent: Agent) -> Player:
    """把 PettingZoo 智能体名称映射到规则层玩家枚举。"""
    if agent == "player_0":
        return Player.PLAYER_0
    if agent == "player_1":
        return Player.PLAYER_1
    raise ValueError(f"unknown agent: {agent!r}")

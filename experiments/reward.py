"""各本地学习实验共用、仅训练阶段启用的势函数奖励塑形。"""

from __future__ import annotations

from typing import cast

from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper

from quoridor_rl.env import QuoridorEnv
from quoridor_rl.game import Player, Position

Agent = str


class PotentialRewardWrapper(BaseWrapper[Agent, dict, int | None]):
    """给合法状态转移叠加有界、零和的最短路径势函数奖励。

    使用 ``γΦ(s') - Φ(s)`` 保留原任务的最优策略不变；双方塑形奖励互为相反数，
    不会凭空改变一局的总收益。终局势能归零，以免遗漏最后一次势能回收。
    """

    def __init__(
        self,
        environment: AECEnv,
        *,
        discount: float = 0.99,
        scale: float = 0.01,
        clip: float = 0.05,
    ) -> None:
        """设置折扣、缩放与单步裁剪上限。"""
        super().__init__(environment)
        self.discount = discount
        self.scale = scale
        self.clip = clip
        self.last_shaping_rewards = {"player_0": 0.0, "player_1": 0.0}
        self._potentials = {"player_0": 0.0, "player_1": 0.0}

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        """重置包装环境，并以初始局面势能作为下一步基线。"""
        super().reset(seed=seed, options=options)
        self.last_shaping_rewards = {"player_0": 0.0, "player_1": 0.0}
        self._potentials = _potentials(self._position, terminal=False)

    def step(self, action: int | None) -> None:
        """先推进基础环境，再把势能差加入即时及累计奖励。"""
        acting_agent = self.agent_selection
        was_finished = self.terminations[acting_agent] or self.truncations[acting_agent]
        super().step(action)
        self.last_shaping_rewards = {"player_0": 0.0, "player_1": 0.0}

        if was_finished:
            return
        if self.infos.get(acting_agent, {}).get("illegal_action", False):
            raise RuntimeError(
                f"training policy selected an illegal action: {action!r}"
            )

        terminal = all(self.terminations.values()) or all(self.truncations.values())
        next_potentials = _potentials(self._position, terminal=terminal)
        player_0_shaping = _clamp(
            self.scale
            * (
                self.discount * next_potentials["player_0"]
                - self._potentials["player_0"]
            ),
            self.clip,
        )
        shaping = {
            "player_0": player_0_shaping,
            "player_1": -player_0_shaping,
        }
        for agent, reward in shaping.items():
            self.rewards[agent] += reward
            self._cumulative_rewards[agent] += reward
        self.last_shaping_rewards = shaping
        self._potentials = next_potentials

    @property
    def _position(self) -> Position:
        """穿透包装器取得规则层不可变局面。"""
        return cast(QuoridorEnv, self.env.unwrapped).position


def _potentials(position: Position, *, terminal: bool) -> dict[Agent, float]:
    """用双方最短路之差计算反对称势能；终局势能固定为零。"""
    if terminal:
        return {"player_0": 0.0, "player_1": 0.0}
    player_0_distance = position.shortest_path_length(Player.PLAYER_0)
    player_1_distance = position.shortest_path_length(Player.PLAYER_1)
    player_0 = float(player_1_distance - player_0_distance)
    return {"player_0": player_0, "player_1": -player_0}


def _clamp(value: float, limit: float) -> float:
    """把数值对称裁剪到 ``[-limit, limit]``。"""
    return max(-limit, min(limit, value))

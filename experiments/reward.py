"""Training-only potential reward shared by local learning experiments."""

from __future__ import annotations

from typing import cast

from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper

from quoridor_rl.env import QuoridorEnv
from quoridor_rl.game import Player, Position

Agent = str


class PotentialRewardWrapper(BaseWrapper[Agent, dict, int | None]):
    """Add bounded, zero-sum shortest-path shaping to legal transitions."""

    def __init__(
        self,
        environment: AECEnv,
        *,
        discount: float = 0.99,
        scale: float = 0.01,
        clip: float = 0.05,
    ) -> None:
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
        super().reset(seed=seed, options=options)
        self.last_shaping_rewards = {"player_0": 0.0, "player_1": 0.0}
        self._potentials = _potentials(self._position, terminal=False)

    def step(self, action: int | None) -> None:
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
        return cast(QuoridorEnv, self.env.unwrapped).position


def _potentials(position: Position, *, terminal: bool) -> dict[Agent, float]:
    if terminal:
        return {"player_0": 0.0, "player_1": 0.0}
    player_0_distance = position.shortest_path_length(Player.PLAYER_0)
    player_1_distance = position.shortest_path_length(Player.PLAYER_1)
    player_0 = float(player_1_distance - player_0_distance)
    return {"player_0": player_0, "player_1": -player_0}


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))

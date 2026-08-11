"""PettingZoo AEC adapter for the two-player Quoridor rules."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from quoridor_rl.codec import ActionCodec
from quoridor_rl.game import IllegalActionError, Orientation, Player, Position, Square
from quoridor_rl.render import render_ascii

Agent = str
Observation = dict[str, np.ndarray]


def env(*, max_plies: int = 512, render_mode: str | None = None) -> AECEnv:
    """Create the public, order-enforced PettingZoo AEC environment."""
    return OrderEnforcingWrapper(
        QuoridorEnv(max_plies=max_plies, render_mode=render_mode)
    )


class QuoridorEnv(AECEnv[Agent, Observation, int | None]):
    """Unwrapped AEC implementation backed by an immutable ``Position``."""

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
    ) -> None:
        super().__init__()
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.max_plies = max_plies
        self.render_mode = render_mode
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
        return self.observation_spaces[agent]

    @property
    def position(self) -> Position:
        """Return the current immutable rule position."""
        return self._position

    def action_space(self, agent: Agent) -> spaces.Space:
        return self.action_spaces[agent]

    def observe(self, agent: Agent) -> Observation:
        player = _player_for(agent)
        opponent = Player(1 - player)
        observation = np.zeros((6, 9, 9), dtype=np.float32)

        own_row, own_col = _square_in_view(self.position.pawns[player], player)
        opponent_row, opponent_col = _square_in_view(
            self.position.pawns[opponent], player
        )
        observation[0, own_row, own_col] = 1.0
        observation[1, opponent_row, opponent_col] = 1.0

        for player_walls in self.position.placed_walls_by_player:
            for wall in player_walls:
                row, col = _anchor_in_view(wall.anchor.row, wall.anchor.col, player)
                plane = 2 if wall.orientation is Orientation.HORIZONTAL else 3
                observation[plane, row, col] = 1.0

        observation[4].fill(self.position.walls_remaining[player] / 10.0)
        observation[5].fill(self.position.walls_remaining[opponent] / 10.0)

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
        current_agent = self.agent_selection
        if self.terminations[current_agent] or self.truncations[current_agent]:
            self._was_dead_step(action)
            return

        player = _player_for(current_agent)
        next_agent = self._agent_selector.next()
        self._cumulative_rewards[current_agent] = 0.0
        self._clear_rewards()

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
        if self.render_mode != "ansi":
            return None
        return render_ascii(self.position)


def _player_for(agent: Agent) -> Player:
    if agent == "player_0":
        return Player.PLAYER_0
    if agent == "player_1":
        return Player.PLAYER_1
    raise ValueError(f"unknown agent: {agent!r}")


def _square_in_view(square: Square, player: Player) -> tuple[int, int]:
    if player is Player.PLAYER_0:
        return square.row, square.col
    return 8 - square.row, 8 - square.col


def _anchor_in_view(row: int, col: int, player: Player) -> tuple[int, int]:
    if player is Player.PLAYER_0:
        return row, col
    return 7 - row, 7 - col

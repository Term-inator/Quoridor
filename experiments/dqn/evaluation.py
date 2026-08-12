"""Deterministic evaluation of a frozen Q network against random play."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from experiments.dqn.model import MaskedQNetwork
from quoridor_rl import env


@dataclass(frozen=True, slots=True)
class RoleResult:
    games: int
    wins: int
    losses: int
    truncated: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    as_player_0: RoleResult
    as_player_1: RoleResult
    illegal_actions: int
    requested_games: int

    @property
    def games(self) -> int:
        return self.as_player_0.games + self.as_player_1.games

    @property
    def wins(self) -> int:
        return self.as_player_0.wins + self.as_player_1.wins

    @property
    def losses(self) -> int:
        return self.as_player_0.losses + self.as_player_1.losses

    @property
    def truncated(self) -> int:
        return self.as_player_0.truncated + self.as_player_1.truncated

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def truncation_rate(self) -> float:
        return self.truncated / self.games if self.games else 0.0

    @property
    def complete(self) -> bool:
        return self.games == self.requested_games


def evaluate(
    model: MaskedQNetwork,
    *,
    games: int,
    device: torch.device,
    max_plies: int = 512,
    seed: int = 10_000,
    deadline: float | None = None,
    progress: bool = False,
) -> EvaluationResult:
    """Evaluate deterministic masked argmax play with equal role assignment."""
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even integer")
    counts = {
        "player_0": {"games": 0, "wins": 0, "losses": 0, "truncated": 0},
        "player_1": {"games": 0, "wins": 0, "losses": 0, "truncated": 0},
    }
    illegal_actions = 0
    model.eval()

    for game_index in range(games):
        if deadline is not None and time.monotonic() >= deadline:
            break
        trained_agent = f"player_{game_index % 2}"
        counts[trained_agent]["games"] += 1
        random = np.random.default_rng(seed + game_index)
        environment = env(max_plies=max_plies)
        environment.reset(seed=seed + game_index)

        while True:
            observation, _, terminated, truncated, info = environment.last()
            if info.get("illegal_action", False):
                illegal_actions += 1
            if terminated or truncated:
                break
            if environment.agent_selection == trained_agent:
                observation_tensor = (
                    torch.from_numpy(observation["observation"]).unsqueeze(0).to(device)
                )
                mask_tensor = (
                    torch.from_numpy(observation["action_mask"])
                    .bool()
                    .unsqueeze(0)
                    .to(device)
                )
                selected_action = int(
                    model.select_actions(
                        observation_tensor,
                        mask_tensor,
                        epsilon=0.0,
                    )[0].item()
                )
            else:
                legal_actions = np.flatnonzero(observation["action_mask"])
                selected_action = int(random.choice(legal_actions))
            environment.step(selected_action)

        winner = environment.unwrapped.position.winner
        if winner is None:
            counts[trained_agent]["truncated"] += 1
        elif f"player_{int(winner)}" == trained_agent:
            counts[trained_agent]["wins"] += 1
        else:
            counts[trained_agent]["losses"] += 1
        if progress and (game_index + 1) % 20 == 0:
            completed = counts["player_0"]["games"] + counts["player_1"]["games"]
            wins = counts["player_0"]["wins"] + counts["player_1"]["wins"]
            unresolved = (
                counts["player_0"]["truncated"] + counts["player_1"]["truncated"]
            )
            print(
                f"evaluation {completed}/{games} wins={wins} unresolved={unresolved}",
                flush=True,
            )

    return EvaluationResult(
        as_player_0=RoleResult(**counts["player_0"]),
        as_player_1=RoleResult(**counts["player_1"]),
        illegal_actions=illegal_actions,
        requested_games=games,
    )

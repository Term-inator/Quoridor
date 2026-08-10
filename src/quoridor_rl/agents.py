"""Small reference agents for exercising the rules and environment."""

import random

from quoridor_rl.game import Action, Position


class RandomAgent:
    """A reproducible agent that samples uniformly from legal semantic actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def choose_action(self, position: Position) -> Action:
        actions = position.legal_actions()
        if not actions:
            raise ValueError("cannot choose an action from a terminal position")
        return self._random.choice(actions)

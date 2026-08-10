import random

import numpy as np

from quoridor_rl.env import env


def test_random_legal_policies_finish_or_truncate_without_hanging() -> None:
    randomizer = random.Random(19)

    for _ in range(3):
        environment = env(max_plies=512)
        environment.reset()
        for agent in environment.agent_iter(max_iter=1028):
            observation, _, terminated, truncated, _ = environment.last()
            if terminated or truncated:
                environment.step(None)
                continue
            legal_ids = np.flatnonzero(observation["action_mask"])
            environment.step(int(randomizer.choice(legal_ids)))

        assert environment.agents == []

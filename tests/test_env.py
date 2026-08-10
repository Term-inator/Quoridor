import numpy as np
from pettingzoo.test import api_test

from quoridor_rl.codec import ActionCodec
from quoridor_rl.env import env
from quoridor_rl.game import (
    MovePawn,
    Orientation,
    PlaceWall,
    Player,
    Square,
    WallAnchor,
)


def test_reset_exposes_standard_spaces_masks_and_canonical_observations() -> None:
    environment = env()
    environment.reset(seed=7)

    assert environment.possible_agents == ["player_0", "player_1"]
    assert environment.agents == ["player_0", "player_1"]
    assert environment.agent_selection == "player_0"
    assert environment.action_space("player_0").n == 209

    player_0 = environment.observe("player_0")
    player_1 = environment.observe("player_1")

    assert player_0["observation"].shape == (6, 9, 9)
    assert player_0["observation"].dtype == np.float32
    assert player_0["action_mask"].shape == (209,)
    assert player_0["action_mask"].dtype == np.int8
    assert player_0["action_mask"].sum() == 131
    assert player_1["action_mask"].sum() == 0
    assert player_0["observation"][0, 8, 4] == 1
    assert player_0["observation"][1, 0, 4] == 1
    assert player_1["observation"][0, 8, 4] == 1
    assert player_1["observation"][1, 0, 4] == 1
    assert np.all(player_0["observation"][4:] == 1)
    assert environment.observation_space("player_0").contains(player_0)
    assert environment.observation_space("player_1").contains(player_1)


def test_step_rotates_the_next_players_view_and_clears_inactive_mask() -> None:
    environment = env()
    environment.reset()
    action_id = ActionCodec().encode(MovePawn(Square(7, 4)), Player.PLAYER_0)

    environment.step(np.int64(action_id))  # type: ignore[arg-type]

    assert environment.agent_selection == "player_1"
    assert environment.rewards == {"player_0": 0, "player_1": 0}
    assert environment.observe("player_0")["action_mask"].sum() == 0
    player_1 = environment.observe("player_1")
    assert player_1["observation"][0, 8, 4] == 1
    assert player_1["observation"][1, 1, 4] == 1
    assert player_1["action_mask"].sum() > 0


def test_wall_planes_and_counts_rotate_with_the_observing_player() -> None:
    environment = env()
    environment.reset()
    codec = ActionCodec()
    wall = PlaceWall(WallAnchor(0, 0), Orientation.HORIZONTAL)

    environment.step(codec.encode(wall, Player.PLAYER_0))

    player_0 = environment.observe("player_0")["observation"]
    player_1 = environment.observe("player_1")["observation"]
    assert player_0[2, 0, 0] == 1
    assert player_1[2, 7, 7] == 1
    assert np.all(player_0[4] == 0.9)
    assert np.all(player_0[5] == 1.0)
    assert np.all(player_1[4] == 1.0)
    assert np.all(player_1[5] == 0.9)


def test_goal_is_a_zero_sum_termination_with_dead_agent_steps() -> None:
    environment = env()
    environment.reset()
    codec = ActionCodec()
    targets = (
        Square(7, 4),
        Square(0, 3),
        Square(6, 4),
        Square(0, 4),
        Square(5, 4),
        Square(0, 3),
        Square(4, 4),
        Square(0, 4),
        Square(3, 4),
        Square(0, 3),
        Square(2, 4),
        Square(0, 4),
        Square(1, 4),
        Square(0, 3),
        Square(0, 4),
    )
    for target in targets:
        player = (
            Player.PLAYER_0
            if environment.agent_selection == "player_0"
            else Player.PLAYER_1
        )
        environment.step(codec.encode(MovePawn(target), player))

    _, reward, terminated, truncated, _ = environment.last()
    assert environment.agent_selection == "player_1"
    assert reward == -1
    assert terminated is True
    assert truncated is False

    environment.step(None)
    _, reward, terminated, truncated, _ = environment.last()
    assert environment.agent_selection == "player_0"
    assert reward == 1
    assert terminated is True
    assert truncated is False

    environment.step(None)
    assert environment.agents == []


def test_max_plies_is_a_zero_reward_truncation() -> None:
    environment = env(max_plies=1)
    environment.reset()
    action_id = ActionCodec().encode(MovePawn(Square(7, 4)), Player.PLAYER_0)

    environment.step(action_id)

    assert environment.terminations == {"player_0": False, "player_1": False}
    assert environment.truncations == {"player_0": True, "player_1": True}
    assert environment.rewards == {"player_0": 0, "player_1": 0}
    assert environment.observe("player_1")["action_mask"].sum() == 0


def test_illegal_action_immediately_loses_for_the_acting_agent() -> None:
    environment = env()
    environment.reset()

    environment.step(0)

    assert environment.agent_selection == "player_1"
    assert environment.terminations == {"player_0": True, "player_1": True}
    assert environment.truncations == {"player_0": False, "player_1": False}
    assert environment.rewards == {"player_0": -1, "player_1": 1}
    assert environment.infos["player_0"] == {"illegal_action": True}


def test_environment_conforms_to_pettingzoo_aec_api() -> None:
    api_test(env(), num_cycles=200, verbose_progress=False)

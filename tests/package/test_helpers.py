import pytest

from quoridor_rl import Language
from quoridor_rl.agents import RandomAgent
from quoridor_rl.env import env
from quoridor_rl.game import Orientation, PlaceWall, Position, WallAnchor
from quoridor_rl.render import render_ascii


def test_seeded_random_agent_returns_reproducible_legal_actions() -> None:
    position = Position.initial()

    first = RandomAgent(seed=42).choose_action(position)
    second = RandomAgent(seed=42).choose_action(position)

    assert first == second
    assert first in position.legal_actions()


def test_ascii_renderer_shows_pawns_walls_coordinates_and_inventory() -> None:
    position = Position.initial().play(
        PlaceWall(WallAnchor(7, 3), Orientation.HORIZONTAL)
    )

    rendered = render_ascii(position)

    assert " 1 " in rendered
    assert " 2 " in rendered
    assert "---" in rendered
    assert "a   b   c   d   e   f   g   h   i" in rendered
    assert "玩家 1 = 9，玩家 2 = 10" in rendered


def test_ansi_environment_render_returns_the_shared_board() -> None:
    environment = env(render_mode="ansi")
    environment.reset()

    rendered = environment.render()

    assert isinstance(rendered, str)
    assert "玩家 1 = 10，玩家 2 = 10" in rendered


def test_ascii_renderer_and_environment_support_english() -> None:
    position = Position.initial()

    assert "Walls: player_0 = 10, player_1 = 10" in render_ascii(
        position,
        language=Language.ENGLISH,
    )

    environment = env(render_mode="ansi", language=Language.ENGLISH)
    environment.reset()

    assert environment.render() is not None
    assert "Walls: player_0 = 10, player_1 = 10" in environment.render()


def test_text_interfaces_reject_unknown_language_values() -> None:
    with pytest.raises(TypeError, match="Language"):
        render_ascii(Position.initial(), language="fr")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Language"):
        env(language="fr")  # type: ignore[arg-type]

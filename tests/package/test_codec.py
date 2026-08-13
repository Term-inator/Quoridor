import pytest

from quoridor_rl.codec import ActionCodec
from quoridor_rl.game import (
    MovePawn,
    Orientation,
    PlaceWall,
    Player,
    Square,
    WallAnchor,
)


def test_all_action_ids_round_trip_for_both_player_perspectives() -> None:
    codec = ActionCodec()

    for player in Player:
        for action_id in range(codec.action_count):
            action = codec.decode(action_id, player)
            assert codec.encode(action, player) == action_id


def test_action_ranges_and_player_one_rotation_are_stable() -> None:
    codec = ActionCodec()

    assert codec.decode(0, Player.PLAYER_0) == MovePawn(Square(0, 0))
    assert codec.decode(80, Player.PLAYER_0) == MovePawn(Square(8, 8))
    assert codec.decode(0, Player.PLAYER_1) == MovePawn(Square(8, 8))
    assert codec.decode(80, Player.PLAYER_1) == MovePawn(Square(0, 0))

    assert codec.decode(81, Player.PLAYER_0) == PlaceWall(
        WallAnchor(0, 0), Orientation.HORIZONTAL
    )
    assert codec.decode(81, Player.PLAYER_1) == PlaceWall(
        WallAnchor(7, 7), Orientation.HORIZONTAL
    )
    assert codec.decode(145, Player.PLAYER_0) == PlaceWall(
        WallAnchor(0, 0), Orientation.VERTICAL
    )
    assert codec.decode(208, Player.PLAYER_1) == PlaceWall(
        WallAnchor(0, 0), Orientation.VERTICAL
    )


def test_codec_rejects_ids_outside_discrete_action_space() -> None:
    codec = ActionCodec()

    with pytest.raises(ValueError):
        codec.decode(-1, Player.PLAYER_0)
    with pytest.raises(ValueError):
        codec.decode(codec.action_count, Player.PLAYER_0)

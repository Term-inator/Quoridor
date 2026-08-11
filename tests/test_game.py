import pytest

from quoridor_rl.game import (
    IllegalActionError,
    IllegalActionReason,
    MovePawn,
    Orientation,
    PlaceWall,
    Player,
    Position,
    Square,
    WallAnchor,
)


def test_initial_position_matches_official_two_player_setup() -> None:
    position = Position.initial()

    assert position.pawns == (Square(8, 4), Square(0, 4))
    assert position.walls_remaining == (10, 10)
    assert position.placed_walls_by_player == (frozenset(), frozenset())
    assert position.to_move is Player.PLAYER_0
    assert position.winner is None


def test_initial_position_lists_pawn_moves_and_every_wall_slot() -> None:
    actions = Position.initial().legal_actions()

    pawn_moves = {action for action in actions if isinstance(action, MovePawn)}
    wall_placements = {action for action in actions if isinstance(action, PlaceWall)}

    assert pawn_moves == {
        MovePawn(Square(7, 4)),
        MovePawn(Square(8, 3)),
        MovePawn(Square(8, 5)),
    }
    assert len(wall_placements) == 128
    assert PlaceWall(WallAnchor(0, 0), Orientation.HORIZONTAL) in wall_placements
    assert PlaceWall(WallAnchor(7, 7), Orientation.VERTICAL) in wall_placements


def test_playing_a_pawn_move_returns_a_new_position_and_switches_turn() -> None:
    initial = Position.initial()

    moved = initial.play(MovePawn(Square(7, 4)))

    assert initial.pawns == (Square(8, 4), Square(0, 4))
    assert initial.to_move is Player.PLAYER_0
    assert moved.pawns == (Square(7, 4), Square(0, 4))
    assert moved.to_move is Player.PLAYER_1


def test_adjacent_pawn_can_be_jumped_in_a_straight_line() -> None:
    position = Position.initial()
    for target in (
        Square(7, 4),
        Square(1, 4),
        Square(6, 4),
        Square(2, 4),
        Square(5, 4),
        Square(3, 4),
        Square(4, 4),
    ):
        position = position.play(MovePawn(target))

    assert MovePawn(Square(5, 4)) in position.legal_actions()
    assert MovePawn(Square(4, 3)) not in position.legal_actions()
    assert MovePawn(Square(4, 5)) not in position.legal_actions()


def test_placing_a_wall_spends_inventory_and_blocks_both_covered_edges() -> None:
    wall = PlaceWall(WallAnchor(7, 3), Orientation.HORIZONTAL)

    position = Position.initial().play(wall)

    assert position.placed_walls_by_player == (frozenset({wall}), frozenset())
    assert position.walls_remaining == (9, 10)
    assert position.to_move is Player.PLAYER_1

    position = position.play(MovePawn(Square(1, 4)))
    assert MovePawn(Square(7, 4)) not in position.legal_actions()
    assert MovePawn(Square(8, 3)) in position.legal_actions()
    assert MovePawn(Square(8, 5)) in position.legal_actions()


def test_placed_walls_preserve_each_player_identity() -> None:
    player_0_wall = PlaceWall(WallAnchor(3, 2), Orientation.HORIZONTAL)
    player_1_wall = PlaceWall(WallAnchor(5, 5), Orientation.VERTICAL)

    initial = Position.initial()
    after_player_0 = initial.play(player_0_wall)
    after_player_1 = after_player_0.play(player_1_wall)

    assert initial.placed_walls_by_player == (frozenset(), frozenset())
    assert after_player_0.placed_walls_by_player == (
        frozenset({player_0_wall}),
        frozenset(),
    )
    assert after_player_1.placed_walls_by_player == (
        frozenset({player_0_wall}),
        frozenset({player_1_wall}),
    )
    assert after_player_1.walls_remaining == (9, 9)


def test_wall_candidates_exclude_crossing_and_overlapping_placements() -> None:
    placed = PlaceWall(WallAnchor(4, 3), Orientation.HORIZONTAL)
    legal = set(Position.initial().play(placed).legal_actions())

    assert placed not in legal
    assert PlaceWall(WallAnchor(4, 2), Orientation.HORIZONTAL) not in legal
    assert PlaceWall(WallAnchor(4, 4), Orientation.HORIZONTAL) not in legal
    assert PlaceWall(WallAnchor(4, 3), Orientation.VERTICAL) not in legal
    assert PlaceWall(WallAnchor(4, 1), Orientation.HORIZONTAL) in legal
    assert PlaceWall(WallAnchor(4, 5), Orientation.HORIZONTAL) in legal


def test_wall_may_not_remove_a_players_last_path_to_the_goal() -> None:
    position = Position.initial()
    for col in (2, 4):
        position = position.play(PlaceWall(WallAnchor(7, col), Orientation.VERTICAL))

    closing_wall = PlaceWall(WallAnchor(7, 3), Orientation.HORIZONTAL)

    assert closing_wall not in position.legal_actions()
    with pytest.raises(IllegalActionError) as error:
        position.play(closing_wall)
    assert error.value.reason is IllegalActionReason.WALL_BLOCKS_PATH


def test_board_edge_behind_opponent_enables_diagonal_moves() -> None:
    position = Position.initial()
    for target in (
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
    ):
        position = position.play(MovePawn(target))
    position = position.play(PlaceWall(WallAnchor(4, 0), Orientation.HORIZONTAL))

    legal = set(position.legal_actions())

    assert MovePawn(Square(0, 3)) in legal
    assert MovePawn(Square(0, 5)) in legal
    assert MovePawn(Square(0, 4)) not in legal


def test_wall_behind_opponent_enables_diagonal_instead_of_straight_jump() -> None:
    position = Position.initial()
    actions = (
        PlaceWall(WallAnchor(4, 3), Orientation.HORIZONTAL),
        MovePawn(Square(1, 4)),
        MovePawn(Square(8, 3)),
        MovePawn(Square(2, 4)),
        MovePawn(Square(7, 3)),
        MovePawn(Square(3, 4)),
        MovePawn(Square(6, 3)),
        MovePawn(Square(3, 3)),
        MovePawn(Square(5, 3)),
        MovePawn(Square(3, 4)),
        MovePawn(Square(5, 2)),
        MovePawn(Square(3, 3)),
        MovePawn(Square(4, 2)),
        MovePawn(Square(3, 4)),
        MovePawn(Square(4, 3)),
        PlaceWall(WallAnchor(0, 0), Orientation.VERTICAL),
        MovePawn(Square(4, 4)),
    )
    for action in actions:
        position = position.play(action)

    legal = set(position.legal_actions())

    assert MovePawn(Square(4, 3)) in legal
    assert MovePawn(Square(4, 5)) in legal
    assert MovePawn(Square(5, 4)) not in legal


def test_reaching_goal_row_wins_and_makes_position_terminal() -> None:
    position = Position.initial()
    for target in (
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
    ):
        position = position.play(MovePawn(target))

    won = position.play(MovePawn(Square(0, 4)))

    assert won.winner is Player.PLAYER_0
    assert won.to_move is None
    assert won.legal_actions() == ()
    with pytest.raises(IllegalActionError) as error:
        won.play(MovePawn(Square(0, 5)))
    assert error.value.reason is IllegalActionReason.GAME_OVER


def test_domain_values_reject_coordinates_outside_the_official_board() -> None:
    with pytest.raises(ValueError):
        Square(-1, 4)
    with pytest.raises(ValueError):
        Square(9, 4)
    with pytest.raises(ValueError):
        WallAnchor(0, 8)
    with pytest.raises(TypeError):
        Square(1.5, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MovePawn("e2")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PlaceWall(WallAnchor(0, 0), "horizontal")  # type: ignore[arg-type]


def test_illegal_action_reports_a_stable_reason() -> None:
    position = Position.initial()

    with pytest.raises(IllegalActionError) as error:
        position.play(MovePawn(Square(6, 4)))

    assert error.value.action == MovePawn(Square(6, 4))
    assert error.value.reason is IllegalActionReason.ILLEGAL_PAWN_MOVE


def test_players_cannot_place_more_than_their_ten_walls() -> None:
    position = Position.initial()
    for _ in range(20):
        wall = next(
            action
            for action in position.legal_actions()
            if isinstance(action, PlaceWall)
        )
        position = position.play(wall)

    assert position.walls_remaining == (0, 0)
    assert not any(isinstance(action, PlaceWall) for action in position.legal_actions())
    with pytest.raises(IllegalActionError) as error:
        position.play(PlaceWall(WallAnchor(7, 7), Orientation.VERTICAL))
    assert error.value.reason is IllegalActionReason.NO_WALLS_REMAINING

"""The public, framework-independent Quoridor rules interface."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Self, TypeAlias


class Player(IntEnum):
    """The two players in turn order."""

    PLAYER_0 = 0
    PLAYER_1 = 1


class IllegalActionReason(Enum):
    """Stable categories for well-formed actions rejected by the rules."""

    GAME_OVER = "game_over"
    ILLEGAL_PAWN_MOVE = "illegal_pawn_move"
    NO_WALLS_REMAINING = "no_walls_remaining"
    WALL_CONFLICT = "wall_conflict"
    WALL_BLOCKS_PATH = "wall_blocks_path"


@dataclass(frozen=True, slots=True, order=True)
class Square:
    """A square in absolute board coordinates."""

    row: int
    col: int

    def __post_init__(self) -> None:
        if type(self.row) is not int or type(self.col) is not int:
            raise TypeError("square coordinates must be integers")
        if not (0 <= self.row < 9 and 0 <= self.col < 9):
            raise ValueError("square coordinates must each be between 0 and 8")


@dataclass(frozen=True, slots=True, order=True)
class WallAnchor:
    """The upper-left board cell used to locate a wall."""

    row: int
    col: int

    def __post_init__(self) -> None:
        if type(self.row) is not int or type(self.col) is not int:
            raise TypeError("wall anchor coordinates must be integers")
        if not (0 <= self.row < 8 and 0 <= self.col < 8):
            raise ValueError("wall anchor coordinates must each be between 0 and 7")


class Orientation(Enum):
    """The two possible wall orientations."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass(frozen=True, slots=True, order=True)
class MovePawn:
    """Move the current pawn to a target square."""

    target: Square

    def __post_init__(self) -> None:
        if not isinstance(self.target, Square):
            raise TypeError("target must be a Square")


@dataclass(frozen=True, slots=True)
class PlaceWall:
    """Place a wall at an anchor with an orientation."""

    anchor: WallAnchor
    orientation: Orientation

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, WallAnchor):
            raise TypeError("anchor must be a WallAnchor")
        if not isinstance(self.orientation, Orientation):
            raise TypeError("orientation must be an Orientation")


Action: TypeAlias = MovePawn | PlaceWall


class IllegalActionError(ValueError):
    """Raised when a well-formed action is not legal in a position."""

    def __init__(self, action: Action, reason: IllegalActionReason) -> None:
        self.action = action
        self.reason = reason
        super().__init__(f"illegal action ({reason.value}): {action!r}")


@dataclass(frozen=True, slots=True, init=False)
class Position:
    """An immutable Quoridor position."""

    _pawns: tuple[Square, Square]
    _walls_remaining: tuple[int, int]
    _placed_walls_by_player: tuple[
        frozenset[PlaceWall],
        frozenset[PlaceWall],
    ]
    _to_move: Player | None
    _winner: Player | None

    @classmethod
    def initial(cls) -> Self:
        """Return the official two-player starting position."""
        return cls._from_parts(
            pawns=(Square(8, 4), Square(0, 4)),
            walls_remaining=(10, 10),
            placed_walls_by_player=(frozenset(), frozenset()),
            to_move=Player.PLAYER_0,
            winner=None,
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        pawns: tuple[Square, Square],
        walls_remaining: tuple[int, int],
        placed_walls_by_player: tuple[
            frozenset[PlaceWall],
            frozenset[PlaceWall],
        ],
        to_move: Player | None,
        winner: Player | None,
    ) -> Self:
        position = object.__new__(cls)
        object.__setattr__(position, "_pawns", pawns)
        object.__setattr__(position, "_walls_remaining", walls_remaining)
        object.__setattr__(
            position,
            "_placed_walls_by_player",
            placed_walls_by_player,
        )
        object.__setattr__(position, "_to_move", to_move)
        object.__setattr__(position, "_winner", winner)
        return position

    @property
    def pawns(self) -> tuple[Square, Square]:
        return self._pawns

    @property
    def walls_remaining(self) -> tuple[int, int]:
        return self._walls_remaining

    @property
    def placed_walls_by_player(
        self,
    ) -> tuple[frozenset[PlaceWall], frozenset[PlaceWall]]:
        return self._placed_walls_by_player

    @property
    def to_move(self) -> Player | None:
        return self._to_move

    @property
    def winner(self) -> Player | None:
        return self._winner

    def legal_actions(self) -> tuple[Action, ...]:
        if self._to_move is None:
            return ()

        moves = tuple(MovePawn(target) for target in self._legal_pawn_targets())
        if self._walls_remaining[self._to_move] == 0:
            return moves

        walls: list[Action] = []
        for orientation in (Orientation.HORIZONTAL, Orientation.VERTICAL):
            for row in range(8):
                for col in range(8):
                    candidate = PlaceWall(WallAnchor(row, col), orientation)
                    if not self._wall_conflicts(
                        candidate
                    ) and self._wall_preserves_paths(candidate):
                        walls.append(candidate)
        return moves + tuple(walls)

    def shortest_path_length(self, player: Player) -> int:
        """Return the wall-only shortest distance from ``player`` to its goal row."""
        distance = self._shortest_path_length(player, self._all_placed_walls())
        assert distance is not None
        return distance

    def _wall_conflicts(self, candidate: PlaceWall) -> bool:
        for placed in self._all_placed_walls():
            if candidate.orientation is placed.orientation:
                if candidate.orientation is Orientation.HORIZONTAL:
                    if (
                        candidate.anchor.row == placed.anchor.row
                        and abs(candidate.anchor.col - placed.anchor.col) <= 1
                    ):
                        return True
                elif (
                    candidate.anchor.col == placed.anchor.col
                    and abs(candidate.anchor.row - placed.anchor.row) <= 1
                ):
                    return True
            elif candidate.anchor == placed.anchor:
                return True
        return False

    def _wall_preserves_paths(self, candidate: PlaceWall) -> bool:
        walls = self._all_placed_walls() | {candidate}
        return all(self._has_path(player, walls) for player in Player)

    def _has_path(
        self,
        player: Player,
        walls: frozenset[PlaceWall],
    ) -> bool:
        return self._shortest_path_length(player, walls) is not None

    def _shortest_path_length(
        self,
        player: Player,
        walls: frozenset[PlaceWall],
    ) -> int | None:
        goal_row = 0 if player is Player.PLAYER_0 else 8
        frontier = deque([(self._pawns[player], 0)])
        visited = {self._pawns[player]}

        while frontier:
            square, distance = frontier.popleft()
            if square.row == goal_row:
                return distance
            for row_delta, col_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                neighbor_row = square.row + row_delta
                neighbor_col = square.col + col_delta
                if not _coordinates_on_board(neighbor_row, neighbor_col):
                    continue
                neighbor = Square(neighbor_row, neighbor_col)
                if neighbor not in visited and not _is_blocked_by(
                    walls, square, neighbor
                ):
                    visited.add(neighbor)
                    frontier.append((neighbor, distance + 1))

        return None

    def _legal_pawn_targets(self) -> tuple[Square, ...]:
        assert self._to_move is not None
        pawn = self._pawns[self._to_move]
        opponent = self._pawns[Player(1 - self._to_move)]
        targets: set[Square] = set()

        for row_delta, col_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            adjacent_row = pawn.row + row_delta
            adjacent_col = pawn.col + col_delta
            if not _coordinates_on_board(adjacent_row, adjacent_col):
                continue
            adjacent = Square(adjacent_row, adjacent_col)
            if self._is_blocked(pawn, adjacent):
                continue
            if adjacent != opponent:
                targets.add(adjacent)
                continue

            behind_row = opponent.row + row_delta
            behind_col = opponent.col + col_delta
            if _coordinates_on_board(behind_row, behind_col) and not self._is_blocked(
                opponent,
                Square(behind_row, behind_col),
            ):
                behind = Square(behind_row, behind_col)
                targets.add(behind)
                continue

            perpendicular = ((0, -1), (0, 1)) if row_delta else ((-1, 0), (1, 0))
            for side_row_delta, side_col_delta in perpendicular:
                diagonal_row = opponent.row + side_row_delta
                diagonal_col = opponent.col + side_col_delta
                if not _coordinates_on_board(diagonal_row, diagonal_col):
                    continue
                diagonal = Square(diagonal_row, diagonal_col)
                if not self._is_blocked(opponent, diagonal):
                    targets.add(diagonal)

        return tuple(sorted(targets))

    def _is_blocked(self, first: Square, second: Square) -> bool:
        return _is_blocked_by(self._all_placed_walls(), first, second)

    def _all_placed_walls(self) -> frozenset[PlaceWall]:
        return (
            self._placed_walls_by_player[Player.PLAYER_0]
            | self._placed_walls_by_player[Player.PLAYER_1]
        )

    def play(self, action: Action) -> Self:
        if action not in self.legal_actions():
            raise IllegalActionError(action, self._illegal_action_reason(action))
        assert self._to_move is not None
        current_player = self._to_move

        pawns = list(self._pawns)
        walls_remaining = list(self._walls_remaining)
        placed_walls_by_player = list(self._placed_walls_by_player)
        if isinstance(action, MovePawn):
            pawns[current_player] = action.target
        else:
            walls_remaining[current_player] -= 1
            placed_walls_by_player[current_player] = placed_walls_by_player[
                current_player
            ] | {action}

        goal_row = 0 if current_player is Player.PLAYER_0 else 8
        winner = (
            current_player
            if isinstance(action, MovePawn) and action.target.row == goal_row
            else None
        )

        return self._from_parts(
            pawns=(pawns[0], pawns[1]),
            walls_remaining=(walls_remaining[0], walls_remaining[1]),
            placed_walls_by_player=(
                placed_walls_by_player[Player.PLAYER_0],
                placed_walls_by_player[Player.PLAYER_1],
            ),
            to_move=None if winner is not None else Player(1 - current_player),
            winner=winner,
        )

    def _illegal_action_reason(self, action: Action) -> IllegalActionReason:
        if self._to_move is None:
            return IllegalActionReason.GAME_OVER
        if isinstance(action, MovePawn):
            return IllegalActionReason.ILLEGAL_PAWN_MOVE
        if self._walls_remaining[self._to_move] == 0:
            return IllegalActionReason.NO_WALLS_REMAINING
        if self._wall_conflicts(action):
            return IllegalActionReason.WALL_CONFLICT
        return IllegalActionReason.WALL_BLOCKS_PATH


def _coordinates_on_board(row: int, col: int) -> bool:
    return 0 <= row < 9 and 0 <= col < 9


def _is_blocked_by(
    walls: frozenset[PlaceWall],
    first: Square,
    second: Square,
) -> bool:
    if first.row != second.row:
        anchor_row = min(first.row, second.row)
        col = first.col
        return any(
            wall.orientation is Orientation.HORIZONTAL
            and wall.anchor.row == anchor_row
            and wall.anchor.col in (col - 1, col)
            for wall in walls
        )

    anchor_col = min(first.col, second.col)
    row = first.row
    return any(
        wall.orientation is Orientation.VERTICAL
        and wall.anchor.col == anchor_col
        and wall.anchor.row in (row - 1, row)
        for wall in walls
    )

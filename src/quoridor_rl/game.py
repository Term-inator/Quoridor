"""围墙棋（Quoridor）的公共规则层。

本模块不依赖强化学习框架或图形界面，只负责表示棋盘状态、枚举合法动作并执行
规则校验。棋盘坐标采用从上到下、从左到右的零基 ``(row, col)``；状态对象是不可变
的，因此每次落子都会返回一个新的 :class:`Position`。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Self, TypeAlias

_BOARD_MASK = (1 << 81) - 1
_RIGHT_COLUMN_MASK = sum(1 << (row * 9 + 8) for row in range(9))
_GOAL_ROW_MASKS = ((1 << 9) - 1, ((1 << 9) - 1) << 72)
# 两个整数分别记录“上下相邻格”和“左右相邻格”之间被墙阻断的边。使用位图既能
# 避免在寻路内层循环构造对象，也能一次扩展一整层 BFS 前沿。
BlockedEdges: TypeAlias = tuple[int, int]


class Player(IntEnum):
    """两名玩家；枚举值同时也是元组索引和固定的行动顺序。"""

    PLAYER_0 = 0
    PLAYER_1 = 1


class IllegalActionReason(Enum):
    """语法正确但违反规则的动作类别；值可稳定用于界面或日志。"""

    GAME_OVER = "game_over"
    ILLEGAL_PAWN_MOVE = "illegal_pawn_move"
    NO_WALLS_REMAINING = "no_walls_remaining"
    WALL_CONFLICT = "wall_conflict"
    WALL_BLOCKS_PATH = "wall_blocks_path"


@dataclass(frozen=True, slots=True, order=True)
class Square:
    """棋子所在格的绝对坐标，行列范围均为 0～8。"""

    row: int
    col: int

    def __post_init__(self) -> None:
        """拒绝布尔值、浮点数及越界坐标，保证值对象始终有效。"""
        if type(self.row) is not int or type(self.col) is not int:
            raise TypeError("square coordinates must be integers")
        if not (0 <= self.row < 9 and 0 <= self.col < 9):
            raise ValueError("square coordinates must each be between 0 and 8")


@dataclass(frozen=True, slots=True, order=True)
class WallAnchor:
    """墙的左上锚点；墙跨两个格边，因此行列范围均为 0～7。"""

    row: int
    col: int

    def __post_init__(self) -> None:
        """在构造边界检查锚点，避免非法坐标进入规则计算。"""
        if type(self.row) is not int or type(self.col) is not int:
            raise TypeError("wall anchor coordinates must be integers")
        if not (0 <= self.row < 8 and 0 <= self.col < 8):
            raise ValueError("wall anchor coordinates must each be between 0 and 7")


class Orientation(Enum):
    """墙的两个放置方向。"""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass(frozen=True, slots=True, order=True)
class MovePawn:
    """把当前玩家的棋子移动到目标格。"""

    target: Square

    def __post_init__(self) -> None:
        """确保动作只携带经过校验的格子值对象。"""
        if not isinstance(self.target, Square):
            raise TypeError("target must be a Square")


@dataclass(frozen=True, slots=True)
class PlaceWall:
    """在锚点处按指定方向放置一堵跨越两个格边的墙。"""

    anchor: WallAnchor
    orientation: Orientation

    def __post_init__(self) -> None:
        """确保墙动作由合法的锚点和方向值组成。"""
        if not isinstance(self.anchor, WallAnchor):
            raise TypeError("anchor must be a WallAnchor")
        if not isinstance(self.orientation, Orientation):
            raise TypeError("orientation must be an Orientation")


Action: TypeAlias = MovePawn | PlaceWall


class IllegalActionError(ValueError):
    """动作结构有效、但在给定局面中不合法时抛出。"""

    def __init__(self, action: Action, reason: IllegalActionReason) -> None:
        """保留原动作及机器可读原因，供调用方精确处理。"""
        self.action = action
        self.reason = reason
        super().__init__(f"illegal action ({reason.value}): {action!r}")


@dataclass(frozen=True, slots=True, init=False)
class Position:
    """不可变的围墙棋局面。

    所有按玩家区分的数据都按 ``Player`` 枚举值索引。终局时 ``to_move`` 为
    ``None`` 且 ``winner`` 非空；非终局则恰好相反。
    """

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
        """创建官方双人规则的初始局面：双方各十堵墙、棋子位于底边中央。"""
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
        """由已验证的内部部件直接构造状态。

        这是不可变数据类的唯一内部构造入口；调用者必须维持字段间的不变量。
        """
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
        """返回两名玩家棋子的绝对坐标。"""
        return self._pawns

    @property
    def walls_remaining(self) -> tuple[int, int]:
        """返回两名玩家各自尚可放置的墙数。"""
        return self._walls_remaining

    @property
    def placed_walls_by_player(
        self,
    ) -> tuple[frozenset[PlaceWall], frozenset[PlaceWall]]:
        """返回按放置者分组的墙集合。"""
        return self._placed_walls_by_player

    @property
    def to_move(self) -> Player | None:
        """返回当前行动方；终局时为 ``None``。"""
        return self._to_move

    @property
    def winner(self) -> Player | None:
        """返回获胜玩家；对局进行中时为 ``None``。"""
        return self._winner

    def legal_actions(self) -> tuple[Action, ...]:
        """枚举当前局面的全部合法动作，终局返回空元组。

        动作顺序是确定的，便于测试和可复现实验：先列棋子移动，再按方向、行、列
        列出不会冲突且不会封死任一玩家路径的放墙动作。
        """
        if self._to_move is None:
            return ()

        moves = tuple(MovePawn(target) for target in self._legal_pawn_targets())
        if self._walls_remaining[self._to_move] == 0:
            return moves

        placed_walls = self._all_placed_walls()
        blocked_edges = _blocked_edges(placed_walls)
        walls: list[Action] = []
        for orientation in (Orientation.HORIZONTAL, Orientation.VERTICAL):
            for row in range(8):
                for col in range(8):
                    candidate = PlaceWall(WallAnchor(row, col), orientation)
                    if not self._wall_conflicts(
                        candidate, placed_walls
                    ) and self._wall_preserves_paths(candidate, blocked_edges):
                        walls.append(candidate)
        return moves + tuple(walls)

    def shortest_path_length(self, player: Player) -> int:
        """返回 ``player`` 到目标行的最短距离，只考虑墙、不考虑另一枚棋子。"""
        distance = self._shortest_path_length(player, self._all_placed_walls())
        assert distance is not None
        return distance

    def _wall_conflicts(
        self,
        candidate: PlaceWall,
        placed_walls: frozenset[PlaceWall] | None = None,
    ) -> bool:
        """判断候选墙是否与已有墙重叠、部分重叠或交叉。"""
        if placed_walls is None:
            placed_walls = self._all_placed_walls()
        for placed in placed_walls:
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

    def _wall_preserves_paths(
        self,
        candidate: PlaceWall,
        blocked_edges: BlockedEdges | None = None,
    ) -> bool:
        """判断放置候选墙后双方是否仍至少保有一条到终点的路径。"""
        if blocked_edges is None:
            blocked_edges = _blocked_edges(self._all_placed_walls())
        candidate_edges = _blocked_edges(frozenset((candidate,)))
        combined_edges = (
            blocked_edges[0] | candidate_edges[0],
            blocked_edges[1] | candidate_edges[1],
        )
        return all(
            self._shortest_path_length_through_edges(player, combined_edges) is not None
            for player in Player
        )

    def _has_path(
        self,
        player: Player,
        walls: frozenset[PlaceWall],
    ) -> bool:
        """返回玩家在给定墙布局中是否仍能抵达目标行。"""
        return self._shortest_path_length(player, walls) is not None

    def _shortest_path_length(
        self,
        player: Player,
        walls: frozenset[PlaceWall],
    ) -> int | None:
        """把墙转换为阻断边位图后计算最短路。"""
        return self._shortest_path_length_through_edges(player, _blocked_edges(walls))

    def _shortest_path_length_through_edges(
        self,
        player: Player,
        blocked_edges: BlockedEdges,
    ) -> int | None:
        """用位并行广度优先搜索计算到目标行的距离。

        ``frontier`` 的每一位代表本层可达的一个格子；四次移位同时生成整层的四向
        邻居，再用阻断边和已访问集合过滤。路径不存在时返回 ``None``。
        """
        start = self._pawns[player].row * 9 + self._pawns[player].col
        frontier = 1 << start
        visited = frontier
        goal = _GOAL_ROW_MASKS[player]
        horizontal, vertical = blocked_edges
        distance = 0

        while frontier:
            if frontier & goal:
                return distance
            left = (frontier >> 1) & ~_RIGHT_COLUMN_MASK & ~vertical
            right = ((frontier & ~_RIGHT_COLUMN_MASK & ~vertical) << 1) & _BOARD_MASK
            up = (frontier >> 9) & ~horizontal
            down = ((frontier & ~horizontal) << 9) & _BOARD_MASK
            frontier = (left | right | up | down) & ~visited
            visited |= frontier
            distance += 1

        return None

    def _legal_pawn_targets(self) -> tuple[Square, ...]:
        """按围墙棋的直走、越子和斜走规则计算当前棋子的落点。"""
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

            # 相邻格被对手占据时，优先尝试沿原方向越过对手。
            behind_row = opponent.row + row_delta
            behind_col = opponent.col + col_delta
            if _coordinates_on_board(behind_row, behind_col) and not self._is_blocked(
                opponent,
                Square(behind_row, behind_col),
            ):
                behind = Square(behind_row, behind_col)
                targets.add(behind)
                continue

            # 对手身后越界或被墙挡住时，规则允许从其两侧斜绕。
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
        """判断两个相邻格之间是否有墙；调用方负责保证两格相邻。"""
        return _is_blocked_by(self._all_placed_walls(), first, second)

    def _all_placed_walls(self) -> frozenset[PlaceWall]:
        """合并双方墙集合；墙的几何效果与放置者无关。"""
        return (
            self._placed_walls_by_player[Player.PLAYER_0]
            | self._placed_walls_by_player[Player.PLAYER_1]
        )

    def play(self, action: Action) -> Self:
        """校验并执行一个语义动作，返回新的不可变局面。

        非法动作抛出带原因的 :class:`IllegalActionError`；原局面永远不会被修改。
        """
        if self._to_move is None:
            raise IllegalActionError(action, IllegalActionReason.GAME_OVER)
        if isinstance(action, MovePawn):
            if action.target not in self._legal_pawn_targets():
                raise IllegalActionError(
                    action,
                    IllegalActionReason.ILLEGAL_PAWN_MOVE,
                )
        else:
            if self._walls_remaining[self._to_move] == 0:
                raise IllegalActionError(
                    action,
                    IllegalActionReason.NO_WALLS_REMAINING,
                )
            if self._wall_conflicts(action):
                raise IllegalActionError(action, IllegalActionReason.WALL_CONFLICT)
            if not self._wall_preserves_paths(action):
                raise IllegalActionError(
                    action,
                    IllegalActionReason.WALL_BLOCKS_PATH,
                )
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


def _coordinates_on_board(row: int, col: int) -> bool:
    """判断坐标是否落在 9×9 棋盘内。"""
    return 0 <= row < 9 and 0 <= col < 9


def _is_blocked_by(
    walls: frozenset[PlaceWall],
    first: Square,
    second: Square,
) -> bool:
    """通过阻断边位图判断两个相邻格之间是否有墙。"""
    first_id = first.row * 9 + first.col
    second_id = second.row * 9 + second.col
    horizontal, vertical = _blocked_edges(walls)
    upper_or_left = min(first_id, second_id)
    mask = horizontal if first.row != second.row else vertical
    return bool(mask & (1 << upper_or_left))


def _blocked_edges(walls: frozenset[PlaceWall]) -> BlockedEdges:
    """把墙集合编码为水平、垂直两张阻断边位图。

    位索引采用对应边的上方格或左侧格编号。一堵墙跨两条边，因此每次设置两位。
    """
    horizontal = 0
    vertical = 0
    for wall in walls:
        row = wall.anchor.row
        col = wall.anchor.col
        upper_left = row * 9 + col
        if wall.orientation is Orientation.HORIZONTAL:
            horizontal |= (1 << upper_left) | (1 << (upper_left + 1))
        else:
            vertical |= (1 << upper_left) | (1 << (upper_left + 9))
    return horizontal, vertical

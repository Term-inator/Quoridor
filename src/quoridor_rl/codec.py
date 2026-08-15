"""强化学习适配层使用的动作编号与观测张量编码。

编码统一采用“当前玩家向上进攻”的规范视角，从而让同一策略网络可以无差别地控制
双方。玩家 1 的绝对坐标会绕棋盘中心旋转 180° 后再编码。
"""

from operator import index

import numpy as np

from quoridor_rl.constants import (
    BOARD_SIZE,
    INITIAL_WALLS_PER_PLAYER,
    WALL_ANCHOR_GRID_SIZE,
)
from quoridor_rl.game import (
    Action,
    MovePawn,
    Orientation,
    PlaceWall,
    Player,
    Position,
    Square,
    WallAnchor,
)

_PAWN_ACTION_COUNT = BOARD_SIZE**2
"""棋子可选择的目标格动作数量。"""

_WALL_ACTION_COUNT = WALL_ANCHOR_GRID_SIZE**2
"""每种方向的放墙动作数量。"""

_HORIZONTAL_WALL_ACTION_OFFSET = _PAWN_ACTION_COUNT
"""横墙动作编号的起始偏移。"""

_VERTICAL_WALL_ACTION_OFFSET = (
    _HORIZONTAL_WALL_ACTION_OFFSET + _WALL_ACTION_COUNT
)
"""竖墙动作编号的起始偏移。"""

_OBSERVATION_CHANNEL_COUNT = 6
"""规范视角观测的通道总数。"""

_OWN_PAWN_CHANNEL = 0
"""己方棋子位置通道。"""

_OPPONENT_PAWN_CHANNEL = 1
"""对方棋子位置通道。"""

_HORIZONTAL_WALL_CHANNEL = 2
"""横墙几何布局通道。"""

_VERTICAL_WALL_CHANNEL = 3
"""竖墙几何布局通道。"""

_OWN_WALL_COUNT_CHANNEL = 4
"""己方剩余墙数比例通道。"""

_OPPONENT_WALL_COUNT_CHANNEL = 5
"""对方剩余墙数比例通道。"""


class ActionCodec:
    """在语义动作与固定的 209 维策略空间之间转换。

    编号布局为 81 个棋子目标格、64 个横墙锚点、64 个竖墙锚点。编号空间包含当前
    局面下的非法动作，调用方应配合动作掩码使用。
    """

    action_count = _VERTICAL_WALL_ACTION_OFFSET + _WALL_ACTION_COUNT
    """完整离散动作空间包含棋子、横墙和竖墙三段动作。"""

    def encode(self, action: Action, player: Player) -> int:
        """把绝对坐标动作编码为 ``player`` 规范视角下的动作编号。"""
        if isinstance(action, MovePawn):
            row, col = _square_in_view(action.target, player)
            return row * BOARD_SIZE + col

        row, col = _anchor_in_view(action.anchor, player)
        offset = (
            _HORIZONTAL_WALL_ACTION_OFFSET
            if action.orientation is Orientation.HORIZONTAL
            else _VERTICAL_WALL_ACTION_OFFSET
        )
        return offset + row * WALL_ANCHOR_GRID_SIZE + col

    def decode(self, action_id: int, player: Player) -> Action:
        """把 ``player`` 视角的动作编号解码成绝对坐标语义动作。"""
        if isinstance(action_id, bool):
            raise TypeError(
                f"action ID must be an integer between 0 and {self.action_count - 1}"
            )
        try:
            action_id = index(action_id)
        except TypeError as error:
            raise TypeError(
                f"action ID must be an integer between 0 and {self.action_count - 1}"
            ) from error
        if not 0 <= action_id < self.action_count:
            raise ValueError(f"action ID must be between 0 and {self.action_count - 1}")

        if action_id < _HORIZONTAL_WALL_ACTION_OFFSET:
            row, col = divmod(action_id, BOARD_SIZE)
            if player is Player.PLAYER_1:
                row, col = BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col
            return MovePawn(Square(row, col))

        if action_id < _VERTICAL_WALL_ACTION_OFFSET:
            orientation = Orientation.HORIZONTAL
            anchor_id = action_id - _HORIZONTAL_WALL_ACTION_OFFSET
        else:
            orientation = Orientation.VERTICAL
            anchor_id = action_id - _VERTICAL_WALL_ACTION_OFFSET
        row, col = divmod(anchor_id, WALL_ANCHOR_GRID_SIZE)
        if player is Player.PLAYER_1:
            row, col = (
                WALL_ANCHOR_GRID_SIZE - 1 - row,
                WALL_ANCHOR_GRID_SIZE - 1 - col,
            )
        return PlaceWall(WallAnchor(row, col), orientation)


class ObservationCodec:
    """从指定玩家的规范视角把局面编码为六通道观测。

    通道依次为己方棋子、对方棋子、横墙、竖墙、己方剩余墙比例、对方剩余墙比例。
    墙通道只表达几何布局，不区分墙由谁放置。
    """

    shape = (_OBSERVATION_CHANNEL_COUNT, BOARD_SIZE, BOARD_SIZE)
    """观测由固定数量的通道和完整棋盘平面组成。"""

    def encode(self, position: Position, player: Player) -> np.ndarray:
        """返回 ``player`` 视角、形状为 ``(6, 9, 9)`` 的浮点观测。"""
        opponent = Player(1 - player)
        observation = np.zeros(self.shape, dtype=np.float32)

        own_row, own_col = _square_in_view(position.pawns[player], player)
        opponent_row, opponent_col = _square_in_view(position.pawns[opponent], player)
        observation[_OWN_PAWN_CHANNEL, own_row, own_col] = 1.0
        observation[_OPPONENT_PAWN_CHANNEL, opponent_row, opponent_col] = 1.0

        for player_walls in position.placed_walls_by_player:
            for wall in player_walls:
                row, col = _anchor_in_view(wall.anchor, player)
                plane = (
                    _HORIZONTAL_WALL_CHANNEL
                    if wall.orientation is Orientation.HORIZONTAL
                    else _VERTICAL_WALL_CHANNEL
                )
                observation[plane, row, col] = 1.0

        observation[_OWN_WALL_COUNT_CHANNEL].fill(
            position.walls_remaining[player] / INITIAL_WALLS_PER_PLAYER
        )
        observation[_OPPONENT_WALL_COUNT_CHANNEL].fill(
            position.walls_remaining[opponent] / INITIAL_WALLS_PER_PLAYER
        )
        return observation


def _square_in_view(square: Square, player: Player) -> tuple[int, int]:
    """把棋子绝对坐标转换到玩家的规范视角。"""
    if player is Player.PLAYER_0:
        return square.row, square.col
    return BOARD_SIZE - 1 - square.row, BOARD_SIZE - 1 - square.col


def _anchor_in_view(anchor: WallAnchor, player: Player) -> tuple[int, int]:
    """把 8×8 墙锚点坐标转换到玩家的规范视角。"""
    if player is Player.PLAYER_0:
        return anchor.row, anchor.col
    return (
        WALL_ANCHOR_GRID_SIZE - 1 - anchor.row,
        WALL_ANCHOR_GRID_SIZE - 1 - anchor.col,
    )

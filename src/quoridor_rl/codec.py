"""强化学习适配层使用的动作编号与观测张量编码。

编码统一采用“当前玩家向上进攻”的规范视角，从而让同一策略网络可以无差别地控制
双方。玩家 1 的绝对坐标会绕棋盘中心旋转 180° 后再编码。
"""

from operator import index

import numpy as np

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


class ActionCodec:
    """在语义动作与固定的 209 维策略空间之间转换。

    编号布局为 81 个棋子目标格、64 个横墙锚点、64 个竖墙锚点。编号空间包含当前
    局面下的非法动作，调用方应配合动作掩码使用。
    """

    action_count = 209

    def encode(self, action: Action, player: Player) -> int:
        """把绝对坐标动作编码为 ``player`` 规范视角下的动作编号。"""
        if isinstance(action, MovePawn):
            row, col = _square_in_view(action.target, player)
            return row * 9 + col

        row, col = _anchor_in_view(action.anchor, player)
        offset = 81 if action.orientation is Orientation.HORIZONTAL else 145
        return offset + row * 8 + col

    def decode(self, action_id: int, player: Player) -> Action:
        """把 ``player`` 视角的动作编号解码成绝对坐标语义动作。"""
        if isinstance(action_id, bool):
            raise TypeError("action ID must be an integer between 0 and 208")
        try:
            action_id = index(action_id)
        except TypeError as error:
            raise TypeError("action ID must be an integer between 0 and 208") from error
        if not 0 <= action_id < self.action_count:
            raise ValueError("action ID must be between 0 and 208")

        if action_id < 81:
            row, col = divmod(action_id, 9)
            if player is Player.PLAYER_1:
                row, col = 8 - row, 8 - col
            return MovePawn(Square(row, col))

        if action_id < 145:
            orientation = Orientation.HORIZONTAL
            anchor_id = action_id - 81
        else:
            orientation = Orientation.VERTICAL
            anchor_id = action_id - 145
        row, col = divmod(anchor_id, 8)
        if player is Player.PLAYER_1:
            row, col = 7 - row, 7 - col
        return PlaceWall(WallAnchor(row, col), orientation)


class ObservationCodec:
    """从指定玩家的规范视角把局面编码为六通道观测。

    通道依次为己方棋子、对方棋子、横墙、竖墙、己方剩余墙比例、对方剩余墙比例。
    墙通道只表达几何布局，不区分墙由谁放置。
    """

    shape = (6, 9, 9)

    def encode(self, position: Position, player: Player) -> np.ndarray:
        """返回 ``player`` 视角、形状为 ``(6, 9, 9)`` 的浮点观测。"""
        opponent = Player(1 - player)
        observation = np.zeros(self.shape, dtype=np.float32)

        own_row, own_col = _square_in_view(position.pawns[player], player)
        opponent_row, opponent_col = _square_in_view(position.pawns[opponent], player)
        observation[0, own_row, own_col] = 1.0
        observation[1, opponent_row, opponent_col] = 1.0

        for player_walls in position.placed_walls_by_player:
            for wall in player_walls:
                row, col = _anchor_in_view(wall.anchor, player)
                plane = 2 if wall.orientation is Orientation.HORIZONTAL else 3
                observation[plane, row, col] = 1.0

        observation[4].fill(position.walls_remaining[player] / 10.0)
        observation[5].fill(position.walls_remaining[opponent] / 10.0)
        return observation


def _square_in_view(square: Square, player: Player) -> tuple[int, int]:
    """把棋子绝对坐标转换到玩家的规范视角。"""
    if player is Player.PLAYER_0:
        return square.row, square.col
    return 8 - square.row, 8 - square.col


def _anchor_in_view(anchor: WallAnchor, player: Player) -> tuple[int, int]:
    """把 8×8 墙锚点坐标转换到玩家的规范视角。"""
    if player is Player.PLAYER_0:
        return anchor.row, anchor.col
    return 7 - anchor.row, 7 - anchor.col

"""Numeric action encoding for reinforcement-learning adapters."""

from operator import index

from quoridor_rl.game import (
    Action,
    MovePawn,
    Orientation,
    PlaceWall,
    Player,
    Square,
    WallAnchor,
)


class ActionCodec:
    """Convert between semantic actions and the fixed 209-action policy space."""

    action_count = 209

    def encode(self, action: Action, player: Player) -> int:
        """Encode an absolute-coordinate action in ``player``'s perspective."""
        if isinstance(action, MovePawn):
            row, col = _square_in_view(action.target, player)
            return row * 9 + col

        row, col = _anchor_in_view(action.anchor, player)
        offset = 81 if action.orientation is Orientation.HORIZONTAL else 145
        return offset + row * 8 + col

    def decode(self, action_id: int, player: Player) -> Action:
        """Decode an action ID from ``player``'s perspective to absolute coordinates."""
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


def _square_in_view(square: Square, player: Player) -> tuple[int, int]:
    if player is Player.PLAYER_0:
        return square.row, square.col
    return 8 - square.row, 8 - square.col


def _anchor_in_view(anchor: WallAnchor, player: Player) -> tuple[int, int]:
    if player is Player.PLAYER_0:
        return anchor.row, anchor.col
    return 7 - anchor.row, 7 - anchor.col

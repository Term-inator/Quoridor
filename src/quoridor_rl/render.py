"""环境与命令行游戏共用的纯文本棋盘渲染。"""

from quoridor_rl.game import Orientation, Player, Position


def render_ascii(position: Position) -> str:
    """把绝对坐标局面渲染成人类可读的棋盘字符串。

    棋子用 ``1``/``2`` 表示；竖墙占用相邻两行的分隔符，横墙占用相邻两列的
    分隔符。显示坐标采用棋类习惯的 ``a1``～``i9``，与内部零基坐标方向相反。
    """
    lines: list[str] = []
    walls = (
        position.placed_walls_by_player[Player.PLAYER_0]
        | position.placed_walls_by_player[Player.PLAYER_1]
    )

    for row in range(9):
        cells: list[str] = []
        for col in range(9):
            marker = "."
            if (
                position.pawns[Player.PLAYER_0].row == row
                and position.pawns[Player.PLAYER_0].col == col
            ):
                marker = "1"
            elif (
                position.pawns[Player.PLAYER_1].row == row
                and position.pawns[Player.PLAYER_1].col == col
            ):
                marker = "2"
            cells.append(f" {marker} ")
            if col < 8:
                blocked = any(
                    wall.orientation is Orientation.VERTICAL
                    and wall.anchor.col == col
                    and wall.anchor.row in (row - 1, row)
                    for wall in walls
                )
                cells.append("|" if blocked else " ")
        lines.append(f"{9 - row:>2}  {''.join(cells)}")

        if row < 8:
            horizontal_segments = []
            for col in range(9):
                blocked = any(
                    wall.orientation is Orientation.HORIZONTAL
                    and wall.anchor.row == row
                    and wall.anchor.col in (col - 1, col)
                    for wall in walls
                )
                horizontal_segments.append("---" if blocked else "   ")
            lines.append(f"    {'+'.join(horizontal_segments)}")

    lines.append("     a   b   c   d   e   f   g   h   i")
    lines.append(
        "墙数：玩家 1 = "
        f"{position.walls_remaining[Player.PLAYER_0]}，玩家 2 = "
        f"{position.walls_remaining[Player.PLAYER_1]}"
    )
    return "\n".join(lines)

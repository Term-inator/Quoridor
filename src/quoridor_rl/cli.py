"""供人工验收和游玩的交互式终端入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from quoridor_rl.agents import RandomAgent
from quoridor_rl.game import (
    Action,
    IllegalActionError,
    MovePawn,
    Orientation,
    PlaceWall,
    Player,
    Position,
    Square,
    WallAnchor,
)
from quoridor_rl.render import render_ascii


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并运行人类对局或人机对局。"""
    parser = argparse.ArgumentParser(description="在终端中玩双人围墙棋")
    parser.add_argument(
        "--opponent",
        choices=("human", "random"),
        default="human",
        help="玩家 2 由人类或随机智能体控制（默认：human）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机智能体的种子")
    parser.add_argument(
        "--max-plies",
        type=int,
        default=512,
        help="最多行动数（默认：512）",
    )
    args = parser.parse_args(argv)
    if args.max_plies <= 0:
        parser.error("--max-plies 必须为正整数")

    position = Position.initial()
    random_agent = RandomAgent(args.seed)
    plies = 0

    print("命令：move e2；wall d4 horizontal（也可写 h/v）；quit")
    print(render_ascii(position))

    while position.winner is None and plies < args.max_plies:
        assert position.to_move is not None
        if args.opponent == "random" and position.to_move is Player.PLAYER_1:
            action = random_agent.choose_action(position)
            print(f"随机智能体：{_format_action(action)}")
        else:
            try:
                command = input(f"玩家 {int(position.to_move) + 1}> ")
            except EOFError:
                print("输入结束，对局已退出。")
                return 0
            if command.strip().casefold() in {"quit", "q", "exit"}:
                print("对局已退出。")
                return 0
            try:
                action = _parse_action(command)
            except ValueError as error:
                print(f"输入错误：{error}")
                continue

        try:
            position = position.play(action)
        except IllegalActionError as error:
            print(f"不合法动作：{error.reason.value}")
            continue

        plies += 1
        print(render_ascii(position))

    if position.winner is not None:
        print(f"玩家 {int(position.winner) + 1} 获胜。")
    else:
        print(f"达到 {args.max_plies} 手上限，对局截断。")
    return 0


def _parse_action(command: str) -> Action:
    """把终端命令解析为规则层语义动作。"""
    parts = command.casefold().split()
    if len(parts) == 2 and parts[0] in {"move", "m"}:
        return MovePawn(_parse_square(parts[1]))
    if len(parts) == 3 and parts[0] in {"wall", "w"}:
        anchor = _parse_anchor(parts[1])
        orientations = {
            "h": Orientation.HORIZONTAL,
            "horizontal": Orientation.HORIZONTAL,
            "v": Orientation.VERTICAL,
            "vertical": Orientation.VERTICAL,
        }
        try:
            orientation = orientations[parts[2]]
        except KeyError as error:
            raise ValueError("墙方向必须为 horizontal/h 或 vertical/v") from error
        return PlaceWall(anchor, orientation)
    raise ValueError("请使用 move e2 或 wall d4 horizontal")


def _parse_square(value: str) -> Square:
    """把人类棋盘坐标（如 ``e2``）转换成内部棋子坐标。"""
    if len(value) != 2 or value[0] not in "abcdefghi" or value[1] not in "123456789":
        raise ValueError("格子必须为 a1 到 i9")
    return Square(9 - int(value[1]), ord(value[0]) - ord("a"))


def _parse_anchor(value: str) -> WallAnchor:
    """把人类棋盘坐标转换成 8×8 的内部墙锚点坐标。"""
    if len(value) != 2 or value[0] not in "abcdefgh" or value[1] not in "12345678":
        raise ValueError("墙锚点必须为 a1 到 h8")
    return WallAnchor(8 - int(value[1]), ord(value[0]) - ord("a"))


def _format_action(action: Action) -> str:
    """把语义动作格式化成可再次输入终端的命令。"""
    if isinstance(action, MovePawn):
        square = f"{chr(ord('a') + action.target.col)}{9 - action.target.row}"
        return f"move {square}"
    anchor = f"{chr(ord('a') + action.anchor.col)}{8 - action.anchor.row}"
    return f"wall {anchor} {action.orientation.value}"


if __name__ == "__main__":
    raise SystemExit(main())

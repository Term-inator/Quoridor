"""供人工验收和游玩的交互式终端入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from string import ascii_lowercase

from quoridor_rl.agents import RandomAgent
from quoridor_rl.constants import BOARD_SIZE, WALL_ANCHOR_GRID_SIZE
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
from quoridor_rl.language import Language
from quoridor_rl.render import render_ascii

CLI_TEXT = {
    Language.CHINESE: {
        "description": "在终端中玩双人围墙棋",
        "language_help": "界面语言（默认：zh）",
        "opponent_help": "玩家 2 由人类或随机智能体控制（默认：human）",
        "seed_help": "随机智能体的种子",
        "max_plies_help": "最多行动数（默认：512）",
        "max_plies_error": "--max-plies 必须为正整数",
        "commands": "命令：move e2；wall d4 horizontal（也可写 h/v）；quit",
        "random_agent": "随机智能体：{action}",
        "prompt": "玩家 {player}> ",
        "eof": "输入结束，对局已退出。",
        "quit": "对局已退出。",
        "input_error": "输入错误：{error}",
        "illegal": "不合法动作：{reason}",
        "winner": "玩家 {player} 获胜。",
        "truncated": "达到 {max_plies} 手上限，对局截断。",
        "orientation_error": "墙方向必须为 horizontal/h 或 vertical/v",
        "action_error": "请使用 move e2 或 wall d4 horizontal",
        "square_error": "格子必须为 a1 到 i9",
        "anchor_error": "墙锚点必须为 a1 到 h8",
    },
    Language.ENGLISH: {
        "description": "Play two-player Quoridor in the terminal",
        "language_help": "interface language (default: zh)",
        "opponent_help": "control player_1 with a human or random agent (default: human)",
        "seed_help": "seed for the random agent",
        "max_plies_help": "maximum number of plies (default: 512)",
        "max_plies_error": "--max-plies must be a positive integer",
        "commands": "Commands: move e2; wall d4 horizontal (or h/v); quit",
        "random_agent": "Random agent: {action}",
        "prompt": "player_{player}> ",
        "eof": "Input ended; game exited.",
        "quit": "Game exited.",
        "input_error": "Input error: {error}",
        "illegal": "Illegal action: {reason}",
        "winner": "player_{player} wins.",
        "truncated": "Reached the {max_plies}-ply limit; game truncated.",
        "orientation_error": "wall orientation must be horizontal/h or vertical/v",
        "action_error": "use move e2 or wall d4 horizontal",
        "square_error": "square must be between a1 and i9",
        "anchor_error": "wall anchor must be between a1 and h8",
    },
}

assert CLI_TEXT[Language.CHINESE].keys() == CLI_TEXT[Language.ENGLISH].keys()

_BOARD_FILES = ascii_lowercase[:BOARD_SIZE]
_BOARD_RANKS = "".join(str(rank) for rank in range(1, BOARD_SIZE + 1))
_WALL_FILES = ascii_lowercase[:WALL_ANCHOR_GRID_SIZE]
_WALL_RANKS = "".join(str(rank) for rank in range(1, WALL_ANCHOR_GRID_SIZE + 1))


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并运行人类对局或人机对局。"""
    language_parser = argparse.ArgumentParser(add_help=False)
    language_parser.add_argument(
        "--language",
        choices=tuple(language.value for language in Language),
        default=Language.CHINESE.value,
    )
    language_args, _ = language_parser.parse_known_args(argv)
    language = Language(language_args.language)
    text = CLI_TEXT[language]

    parser = argparse.ArgumentParser(description=text["description"])
    parser.add_argument(
        "--language",
        choices=tuple(item.value for item in Language),
        default=language.value,
        help=text["language_help"],
    )
    parser.add_argument(
        "--opponent",
        choices=("human", "random"),
        default="human",
        help=text["opponent_help"],
    )
    parser.add_argument("--seed", type=int, default=None, help=text["seed_help"])
    parser.add_argument(
        "--max-plies",
        type=int,
        default=512,
        help=text["max_plies_help"],
    )
    args = parser.parse_args(argv)
    if args.max_plies <= 0:
        parser.error(text["max_plies_error"])

    position = Position.initial()
    random_agent = RandomAgent(args.seed)
    plies = 0

    print(text["commands"])
    print(render_ascii(position, language=language))

    while position.winner is None and plies < args.max_plies:
        assert position.to_move is not None
        if args.opponent == "random" and position.to_move is Player.PLAYER_1:
            action = random_agent.choose_action(position)
            print(text["random_agent"].format(action=_format_action(action)))
        else:
            try:
                prompt_player = (
                    int(position.to_move) + 1
                    if language is Language.CHINESE
                    else int(position.to_move)
                )
                command = input(text["prompt"].format(player=prompt_player))
            except EOFError:
                print(text["eof"])
                return 0
            if command.strip().casefold() in {"quit", "q", "exit"}:
                print(text["quit"])
                return 0
            try:
                action = _parse_action(command, language=language)
            except ValueError as error:
                print(text["input_error"].format(error=error))
                continue

        try:
            position = position.play(action)
        except IllegalActionError as error:
            print(text["illegal"].format(reason=error.reason.value))
            continue

        plies += 1
        print(render_ascii(position, language=language))

    if position.winner is not None:
        winner = (
            int(position.winner) + 1
            if language is Language.CHINESE
            else int(position.winner)
        )
        print(text["winner"].format(player=winner))
    else:
        print(text["truncated"].format(max_plies=args.max_plies))
    return 0


def _parse_action(
    command: str,
    *,
    language: Language = Language.CHINESE,
) -> Action:
    """把终端命令解析为规则层语义动作。"""
    parts = command.casefold().split()
    if len(parts) == 2 and parts[0] in {"move", "m"}:
        return MovePawn(_parse_square(parts[1], language=language))
    if len(parts) == 3 and parts[0] in {"wall", "w"}:
        anchor = _parse_anchor(parts[1], language=language)
        orientations = {
            "h": Orientation.HORIZONTAL,
            "horizontal": Orientation.HORIZONTAL,
            "v": Orientation.VERTICAL,
            "vertical": Orientation.VERTICAL,
        }
        try:
            orientation = orientations[parts[2]]
        except KeyError as error:
            raise ValueError(CLI_TEXT[language]["orientation_error"]) from error
        return PlaceWall(anchor, orientation)
    raise ValueError(CLI_TEXT[language]["action_error"])


def _parse_square(
    value: str,
    *,
    language: Language = Language.CHINESE,
) -> Square:
    """把人类棋盘坐标（如 ``e2``）转换成内部棋子坐标。"""
    if len(value) != 2 or value[0] not in _BOARD_FILES or value[1] not in _BOARD_RANKS:
        raise ValueError(CLI_TEXT[language]["square_error"])
    return Square(BOARD_SIZE - int(value[1]), ord(value[0]) - ord("a"))


def _parse_anchor(
    value: str,
    *,
    language: Language = Language.CHINESE,
) -> WallAnchor:
    """把人类棋盘坐标转换成 8×8 的内部墙锚点坐标。"""
    if len(value) != 2 or value[0] not in _WALL_FILES or value[1] not in _WALL_RANKS:
        raise ValueError(CLI_TEXT[language]["anchor_error"])
    return WallAnchor(WALL_ANCHOR_GRID_SIZE - int(value[1]), ord(value[0]) - ord("a"))


def _format_action(action: Action) -> str:
    """把语义动作格式化成可再次输入终端的命令。"""
    if isinstance(action, MovePawn):
        square = f"{chr(ord('a') + action.target.col)}{BOARD_SIZE - action.target.row}"
        return f"move {square}"
    anchor = (
        f"{chr(ord('a') + action.anchor.col)}"
        f"{WALL_ANCHOR_GRID_SIZE - action.anchor.row}"
    )
    return f"wall {anchor} {action.orientation.value}"


if __name__ == "__main__":
    raise SystemExit(main())

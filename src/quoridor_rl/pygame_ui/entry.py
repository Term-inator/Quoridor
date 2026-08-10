"""Lightweight entry point that keeps Pygame optional."""

from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    """Start the Pygame application or explain how to install it."""
    if importlib.util.find_spec("pygame") is None:
        print(
            '未安装图形界面依赖。请运行：pip install "quoridor-rl[pygame]"',
            file=sys.stderr,
        )
        return 2

    from quoridor_rl.pygame_ui.app import run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())

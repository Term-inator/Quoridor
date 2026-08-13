"""保持 Pygame 为可选依赖的轻量桌面入口。"""

from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    """启动 Pygame 应用；缺少可选依赖时输出明确安装说明。"""
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

"""用于验证规则和环境的轻量参考智能体。"""

import random

from quoridor_rl.game import Action, Position


class RandomAgent:
    """从合法语义动作中均匀采样、可通过种子复现的随机智能体。"""

    def __init__(self, seed: int | None = None) -> None:
        """创建独立随机数生成器，避免污染进程级随机状态。"""
        self._random = random.Random(seed)

    def choose_action(self, position: Position) -> Action:
        """从局面的合法动作中等概率选择一个；终局无法选择时抛错。"""
        actions = position.legal_actions()
        if not actions:
            raise ValueError("cannot choose an action from a terminal position")
        return self._random.choice(actions)

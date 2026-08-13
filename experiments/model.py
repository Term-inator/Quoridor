"""本地学习实验共用的小型神经网络构件。"""

from torch import nn


def board_encoder() -> nn.Sequential:
    """创建 PPO、DQN 与 AlphaZero 共用的紧凑棋盘特征编码器。"""
    return nn.Sequential(
        nn.Conv2d(6, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 9 * 9, 256),
        nn.ReLU(),
    )

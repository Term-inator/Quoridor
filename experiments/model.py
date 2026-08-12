"""Small neural-network building blocks shared by local experiments."""

from torch import nn


def board_encoder() -> nn.Sequential:
    """Create the compact board encoder used by the PPO and DQN experiments."""
    return nn.Sequential(
        nn.Conv2d(6, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 9 * 9, 256),
        nn.ReLU(),
    )

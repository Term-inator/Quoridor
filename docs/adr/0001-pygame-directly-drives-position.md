# Pygame 直接驱动规则状态

Pygame 对局直接维护不可变的 `Position`，并接收对局参与者提交的语义动作，而不通过 PettingZoo AEC 环境驱动。这样人类输入和语义智能体无需承担训练环境的生命周期与整数编码；未来的训练模型 adapter 负责复用 observation 编码和 `ActionCodec`，把模型输入输出转换到同一语义动作边界。

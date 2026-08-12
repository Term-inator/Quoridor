# 实现 Masked Double DQN 核心训练

Status: ready-for-agent

实现共享 board encoder、masked Q 网络、epsilon-greedy、uniform replay、Double DQN 更新和历史对手池，并以公共行为测试覆盖。

## Acceptance Criteria

- 所有动作选择和 TD target 都排除非法动作。
- terminal 与 truncation transition 都不 bootstrap。
- replay 与 opponent pool 遵循锁定配置。

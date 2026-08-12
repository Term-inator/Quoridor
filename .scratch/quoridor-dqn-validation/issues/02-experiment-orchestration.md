# 实现 DQN 自博弈与实验编排

Status: ready-for-agent

实现 learner-only transition 采集、随机先后手、冻结对手、均衡随机对手评估、checkpoint、deadline 和 smoke gate。

## Acceptance Criteria

- smoke 在五分钟内覆盖所有训练基础设施。
- 正式实验按 15/30/60/120 分钟验证并正确选择最佳 checkpoint。
- 硬故障留下可诊断结果。

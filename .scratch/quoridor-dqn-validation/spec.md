# Quoridor Masked Double DQN 单 Seed 学习验证规格

Status: implemented

## Problem Statement

现有 masked PPO 单 seed 实验未达到探索目标。需要在不改变围墙棋规则、基础环境奖励和发行包 API 的前提下，增加一个独立的 value-based 强化学习实验，验证另一条训练路线在相同墙钟预算和随机智能体评估协议下的表现。

本实验与 PPO 是两个平行的 seed 0 探索，不用于证明算法优劣或统计显著性。

## Solution

在 `experiments/dqn/` 中实现 Masked Double DQN 自博弈。在线网络只学习自己控制的玩家身份，每局随机分配先后手；对手为随机智能体或最近八个冻结网络快照之一。训练 transition 从在线方一次决策延伸到它的下一次决策，因此包含中间对手动作产生的奖励。终局和达到行动上限的未决对局均不 bootstrap。

网络复用 PPO 的两层 64-channel CNN 和 256-unit board encoder，输出 209 个 Q 值。动作选择、online argmax 和 target lookup 都应用合法动作 mask。

## Locked Configuration

- seed 0；四个逻辑环境；最大 512 plies。
- Adam 学习率 `1e-4`，`gamma=0.99`，Huber loss，gradient clip `10.0`。
- uniform replay 容量 200,000，warm-up 10,000，batch 512，每四个 learner transitions 更新一次。
- target network 每 5,000 learner transitions 硬同步。
- epsilon 在前 200,000 learner transitions 从 `1.0` 线性降至 `0.05`。
- 每 50,000 learner transitions 保存对手快照，FIFO 保留最近八个。快照池为空时只用随机对手；建立后 20% 对局使用随机对手，80% 均匀选择历史快照。
- 在线网络每局随机控制一个玩家身份；replay 只接收在线网络的 transition。
- 训练最多累计 120 分钟，整个训练与评估流程最多 150 分钟。

## Validation Contract

- smoke 在五分钟内验证 replay warm-up、梯度更新、target 同步、历史对手、checkpoint 恢复、短评估、合法动作和固定小批样本过拟合。
- 约 15、30、60、120 分钟保存 checkpoint，并以确定性 masked argmax 对随机智能体评估 200 局，双方身份各半。
- 选择验证胜率最高的完整 checkpoint；胜率相同时选择未决率更低者。
- 最佳 checkpoint 最终评估 1,000 局，双方身份各 500 局。
- 探索目标为胜率至少 70%、未决率不超过 5%、非法动作数为零。
- 仅 NaN/Inf、非法动作、CUDA/存储错误、checkpoint 无法恢复或总 deadline 提前终止实验；性能波动不触发提前停止。

## Artifacts

可提交结果位于 `experiments/dqn/results/seed-0/`。checkpoint、TensorBoard 日志和 smoke 产物位于 Git 忽略的 `experiments/dqn/artifacts/`。比较报告只并列本次 DQN 与既有 PPO 的配置、运行环境和观测指标。

## Out of Scope

- 修改基础环境奖励、observation 或 action schema。
- prioritized replay、dueling network、n-step return、MCTS 或 AlphaZero。
- 通用多算法实验平台。
- 多 seed 结论或算法级优劣声明。
- 将训练依赖或实验代码放入发行 wheel。

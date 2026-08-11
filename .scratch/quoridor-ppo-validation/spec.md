# Quoridor PPO 单 Seed 学习验证规格

Status: implemented

## Problem Statement

规则测试和 PettingZoo conformance 只能证明环境接口正确，不能证明外部深度强化学习代码能从 observation、action mask 和奖励信号中学到有效策略。PyPI 发布前需要一次有严格资源上限、可复现且不污染基础环境语义的本地学习验证。

## Solution

在 `experiments/ppo/` 中实现不会进入 wheel 的共享策略 masked PPO 自博弈实验。双方共用同一个小型 CNN；训练使用最短路径差构造的零和 potential reward wrapper，基础环境继续保持终局 `+1/-1` 与非终局零奖励。训练通过 TorchRL 的 PettingZoo adapter、GAE 和 clipped PPO loss 消费现有 AEC 环境。

规则核心公开 `Position.shortest_path_length(player)`，返回忽略棋子占位的墙体最短路径；`environment.unwrapped.position` 是只读的不可变当前局面。训练依赖放在独立 `train` dependency group 中。

## Validation Contract

- 首次只运行 seed 0；四个逻辑环境，rollout 4096，minibatch 512，四轮更新。
- Adam 学习率 `2.5e-4`，`gamma=0.99`，GAE `lambda=0.95`，PPO clip `0.2`，entropy `0.01`，value coefficient `0.5`，gradient clip `0.5`。
- 最多训练 120 分钟；包括评估在内总流程不超过 150 分钟。
- 约 15、30、60、120 分钟保存 checkpoint，并以确定性 masked argmax 对随机智能体评估 200 局，双方身份各 100 局。
- 最佳 checkpoint 最终评估 1,000 局，双方身份各 500 局。探索性目标为胜率至少 70%、未决率不超过 5%、非法动作数为零。
- NaN/Inf、非法动作、CUDA 错误、连续两次没有正常终局、连续三次胜率下降或总 deadline 均提前停止并留下诊断。

## Reward Contract

对玩家身份 `p`，`Phi(p) = opponent_shortest_path - own_shortest_path`，终局 potential 为零。单步塑形为 `clip(0.01 * (0.99 * Phi(next) - Phi(current)), -0.05, 0.05)`；另一玩家身份得到严格相反的值。非法动作不获得塑形。

## Artifacts

可提交结果位于 `experiments/ppo/results/seed-0/`，包含锁定配置、指标 JSON、Markdown 摘要和学习曲线。checkpoint 与 TensorBoard 原始日志位于 Git 忽略的 `experiments/ppo/artifacts/seed-0/`。结果记录 git commit、Python/依赖版本、CUDA、GPU、驱动和 seed。

## Out of Scope

- 修改基础环境奖励或 observation/action schema。
- 将训练依赖、实验代码或 checkpoint 放入发行 wheel。
- 单 seed 结果直接触发 PyPI 发布。
- 本轮实现历史对手池、AlphaZero、MCTS 或多算法实验平台。

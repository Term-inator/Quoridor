# Quoridor AlphaZero 单 Seed 学习验证规格

Status: implemented

## Problem Statement

共享 masked PPO 与 Masked Double DQN 两轮探索均未达到随机智能体胜率至少 70%、未决率不超过 5% 的目标。需要在相同基础环境合同和两小时训练预算下验证一条搜索增强的学习路线。

## Solution

在 `experiments/alphazero/` 中实现从随机网络开始的 AlphaZero 自博弈实验。双方共享一个策略价值网络；PUCT MCTS 使用合法动作 mask、策略先验和当前行动方价值选择分支。根节点加入 Dirichlet noise，前 30 手按访问次数采样，之后选择访问次数最多的动作。

每个正常终局自博弈局面保存 canonical observation、MCTS 根访问次数分布 `pi` 和最终胜负 `z`。正常获胜时胜者视角为 `+1`、败者视角为 `-1`。达到 512 手的未决对局仍记录指标，但不进入 replay；领域合同明确它不是围墙棋平局，因此不把截断误作零价值监督。训练不使用逐步奖励塑形。

从随机网络直接搜索时，约 131 个合法动作远多于 32 次 simulation，PUCT 只能横向试探一层。训练先运行 32 盘仅允许棋子移动的课程，课程期将最短路径进展以 75% 比例混入搜索先验，从而稳定产生正常终局；之后恢复完整合法动作。完整搜索在每个节点保留所有棋子动作，并按网络先验选取墙动作，将总候选限制为 16。根噪声仍负责在候选内探索墙策略。

训练与评测均使用固定模拟次数的 MCTS。MCTS 直接消费不可变 `Position`；数值 observation 由规则适配层的 `ObservationCodec` 生成，不通过 PettingZoo AEC 生命周期驱动搜索。

## Locked Configuration

- seed 0；最大 512 plies；双方共享 PPO/DQN 同容量的两层 64-channel CNN 与 256-unit encoder。
- 每步训练为 32 次、评测为 8 次 PUCT simulation；评测在 CPU 上并行四局；每节点最多 16 个候选；`c_puct=1.5`。
- 前 32 盘为 pawn-only curriculum，课程搜索先验混合 75% 最短路径进展；课程结束后不再使用该先验。
- 根 Dirichlet `alpha=0.03`，噪声占比 `0.25`；前 30 plies 使用温度 1，之后温度 0。
- replay 容量 50,000，warm-up 256，batch 256；仅正常终局对局入 replay，每盘后更新 8 次。
- AdamW 学习率 `1e-3`，weight decay `1e-4`，gradient clip `5.0`。
- 最多累计训练 120 分钟；包括评估在内总流程不超过 150 分钟。

## Validation Contract

- smoke 在五分钟内验证合法搜索、根访问分布、纯终局标签、replay、梯度更新、checkpoint 恢复和先后手均衡短评测。
- 约 15、30、60、120 分钟保存 checkpoint，并以无根噪声、温度 0 的 MCTS 对随机智能体评估 200 局，双方身份各半。
- 选择验证胜率最高的完整 checkpoint；胜率相同时选择未决率更低者。
- 最佳 checkpoint 最终评估 1,000 局，双方身份各半。
- 探索目标为胜率至少 70%、未决率不超过 5%、非法动作数为零。
- NaN/Inf、非法动作、CUDA/存储错误、checkpoint 无法恢复或总 deadline 提前终止实验；性能波动不触发提前停止。

## Artifacts

可提交结果位于 `experiments/alphazero/results/seed-0/`。checkpoint、TensorBoard 日志和 smoke 产物位于 Git 忽略的 `experiments/alphazero/artifacts/`。比较报告只并列三次独立 single-seed 探索，不作算法级结论。

## Out of Scope

- 修改基础奖励、observation 或 action schema。
- 奖励塑形、PPO 权重预热、启发式教师、resignation 或多 seed 结论。
- MuZero、通用博弈搜索框架或将训练实现放入发行 wheel。

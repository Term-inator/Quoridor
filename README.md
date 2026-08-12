# Quoridor RL

一个遵循官方标准双人规则的围墙棋（Quoridor）Python 实现，包含：

- 与训练框架无关、不可变且可哈希的规则核心；
- 标准 PettingZoo AEC 环境；
- 固定的 209 维离散动作空间和合法动作 mask；
- 双方共用网络所需的 canonical observation；
- 人类对人类、人类对随机智能体的终端对局；
- 可选的中文 Pygame 桌面界面，支持本地游玩和随机智能体观战。

首版固定为 9×9 棋盘、每人 10 面墙，不包含四人制、可变棋盘、训练算法或奖励塑形。项目暂未发布到 PyPI。

## 开发环境

项目使用 uv 管理，开发解释器为 Python 3.14，包支持 Python 3.11–3.14。

```bash
uv sync --python 3.14
uv run pytest
```

基础安装不会安装 Pygame。需要图形界面时使用：

```bash
uv sync --extra pygame
uv run quoridor-pygame
```

## PettingZoo AEC 环境

AEC（Agent Environment Cycle）表示智能体依次行动：每轮只给 `agent_selection` 指定的玩家调用 `step()`。终止或截断后的玩家必须以 `None` 完成 dead step。

```python
import numpy as np

from quoridor_rl import env

environment = env(max_plies=512)
environment.reset(seed=0)

for agent in environment.agent_iter():
    observation, reward, terminated, truncated, info = environment.last()
    if terminated or truncated:
        action = None
    else:
        legal_ids = np.flatnonzero(observation["action_mask"])
        action = int(np.random.choice(legal_ids))
    environment.step(action)
```

### 动作空间

动作空间是 `Discrete(209)`，编号始终使用当前玩家的 canonical 视角：

| ID | 含义 |
| --- | --- |
| 0–80 | 9×9 棋子目标格，row-major |
| 81–144 | 8×8 水平墙锚点，row-major |
| 145–208 | 8×8 垂直墙锚点，row-major |

玩家 1（`player_1`）的绝对棋盘会旋转 180°，因此两个玩家在 observation 中都从棋盘下方向上前进。动作、观察与 mask 使用同一视角，调用者不需要自行旋转。

### Observation

每次观察是一个字典：

- `observation`：`float32`、形状 `(6, 9, 9)`；
- `action_mask`：`int8`、形状 `(209,)`，仅当前玩家含合法动作，其余玩家全为零。

六个 plane 依次为：己方棋子、对方棋子、水平墙锚点、垂直墙锚点、己方剩余墙数、对方剩余墙数。墙数除以 10 后广播到整个 9×9 plane；墙锚点使用左上 8×8，最后一行和一列补零。

### 奖励与结束

- 正常获胜：胜者 `+1`，败者 `-1`，属于 termination；
- 其他合法行动：双方 `0`；
- 达到 `max_plies`：双方 `0`，属于 truncation；
- 非法动作：行动者立即以 `-1` 失败，对手得到 `+1`。

稀疏奖励是环境对真实零和目标的标准定义，并非声称它对所有训练算法都最容易。它让不同算法的结果可比较，也直接支持 MCTS/AlphaZero 的终局价值目标。若实验需要 dense shaping，应在独立 wrapper 中显式添加，避免改变基础环境的任务含义。

当前不可变规则局面可通过 `environment.unwrapped.position` 只读访问，便于训练 wrapper 和搜索算法复用规则核心；给该属性赋值会失败。

## 直接使用规则核心

MCTS、bot 和测试可以绕过 RL 编号，直接处理语义动作：

```python
from quoridor_rl import MovePawn, Player, Position, Square

position = Position.initial()
assert MovePawn(Square(7, 4)) in position.legal_actions()

next_position = position.play(MovePawn(Square(7, 4)))
assert position != next_position  # 原状态没有被修改
assert position.shortest_path_length(Player.PLAYER_0) == 8
```

绝对坐标使用从上到下的 `row=0..8`、从左到右的 `col=0..8`。非法但结构完整的动作会抛出 `IllegalActionError`，并携带稳定的 `reason`。

`shortest_path_length()` 返回只考虑墙体、忽略棋子占位时，到对应目标行的最少步数。它与墙合法性检查复用同一份寻路规则。

## 本地 PPO 学习验证

仓库包含一个不会进入发行 wheel 的探索性 masked PPO 自博弈实验，用于在发布前验证 observation、action mask 和奖励信号是否能够产生学习。基础环境仍保持稀疏的终局零和奖励；dense potential reward 只存在于本地实验共用的训练 wrapper 中。

```bash
uv sync --group train

# 短 CPU/CUDA 链路验证
uv run --group train python -m experiments.ppo.train --smoke --device cuda

# seed 0，最长训练 120 分钟，总流程不超过 150 分钟
uv run --group train python -m experiments.ppo.train --device cuda
```

正式实验在 15、30、60、120 分钟进行 200 局先后手均衡验证，并用最佳 checkpoint 对随机智能体评估 1,000 局。可提交的指标、摘要和曲线写入 `experiments/ppo/results/seed-0/`；checkpoint 与 TensorBoard 日志写入被 Git 忽略的 `experiments/ppo/artifacts/seed-0/`。单 seed 达标只是继续多 seed 验证的依据，不构成 PyPI 发布结论。

## 本地 Masked Double DQN 学习验证

仓库还包含一个与 PPO 平行的 Masked Double DQN 单 seed 实验。在线网络每局随机控制一个玩家身份，只将自己的决策写入 uniform replay，并与随机智能体或最近的冻结策略快照对弈。该实验沿用 PPO 的 observation、合法动作 mask、CNN 容量、potential reward、两小时训练预算和随机智能体评估协议；两个结果只作简单并列观察，不构成算法优劣结论。

```bash
# CUDA smoke gate
uv run --group train python -m experiments.dqn.train --smoke --device cuda

# seed 0，最长训练 120 分钟，总流程不超过 150 分钟
uv run --group train python -m experiments.dqn.train --device cuda
```

可提交结果写入 `experiments/dqn/results/seed-0/`，checkpoint、TensorBoard 日志和 smoke 产物写入 Git 忽略的 `experiments/dqn/artifacts/`。

## 本地 AlphaZero 学习验证

第三条 single-seed 探索使用共享策略价值网络和固定模拟次数的 PUCT MCTS。它从随机网络开始，以 MCTS 根访问次数作为策略目标、正常终局胜负作为价值目标；未决对局不进入 replay，也不使用 PPO/DQN 的逐步奖励塑形。

```bash
# 搜索、自博弈、更新、checkpoint 与短评测 smoke gate
uv run --group train python -m experiments.alphazero.train --smoke --device cpu

# seed 0，最长训练 120 分钟，总流程不超过 150 分钟
uv run --group train python -m experiments.alphazero.train --device cpu
```

训练每步执行 32 次 MCTS simulation，评测每步固定执行 8 次并在 CPU 上并行四局；每节点保留全部棋子动作和网络先验最高的墙动作，总候选上限为 16。前 32 盘通过 pawn-only curriculum 建立终局信号，课程结束后恢复完整动作并移除进展先验。自博弈根节点加入探索噪声，正式评测关闭噪声并确定性选择访问次数最多的动作。可提交结果写入 `experiments/alphazero/results/seed-0/`，大型训练产物写入被 Git 忽略的 `experiments/alphazero/artifacts/`。

## 在终端游玩

```bash
# 两个人类玩家
uv run quoridor --opponent human

# 对随机智能体，可固定随机种子
uv run quoridor --opponent random --seed 42
```

棋盘输入采用左下角为 `a1` 的人类坐标：

```text
move e2
wall d4 horizontal
wall d4 v
quit
```

## 使用 Pygame 图形界面

开始界面提供人类对人类、人类对随机智能体、随机智能体对随机智能体三种模式。人类对随机智能体时可以选择先手或后手；包含随机智能体的模式可以填写整数种子并选择播放速度。

- 移动：直接点击棋盘上显示的合法目标；
- 放墙：选择“横墙”或“竖墙”，鼠标会吸附到最近的墙锚点，单击确认；
- 玩家身份墙：已放置墙使用与对应棋子一致的身份色，合法预览也会显示当前行动方的颜色；
- 墙库存：右侧两张玩家身份状态卡同时显示准确剩余数量和 10 格墙条，当前行动方会高亮；
- 行动记录：右侧实时保存完整行动历史，最新一手置顶，双方操作使用各自的玩家身份色；记录超出可见区域后可用鼠标滚轮浏览；
- 终局回放：对局正常结束后可从结果弹窗进入回放，点击任一行动或初始局面即可查看当时的棋盘，回放不会改变最终结果；
- 非法墙：红色预览、叉线和中文原因会同时显示，回合不会推进；
- `Esc`：返回移动模式；
- 空格：暂停或继续智能体对局；
- 右方向键：暂停时单步执行一个智能体动作。

达到 512 手仍无胜者时，界面将该局显示为“未决”，而不是围墙棋规则中的平局。当前项目只在 Linux 上验证图形界面；pygame-ce 虽为其他主流平台提供 wheel，本项目尚未声称已验证这些平台。

界面随包原样分发 Noto Sans SC Regular 字体，其 SIL Open Font License 1.1 文本位于图形资源包中。字体资源会包含在基础 wheel 中，但 pygame-ce 仍只通过 `pygame` extra 安装。

## 验证

```bash
uv run pytest
uv run --extra pygame pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

测试覆盖普通移动、直跳、墙/边界受阻后的斜跳、墙冲突、双方路径保留、终局、动作编码往返、canonical observation、AEC 生命周期、随机完整对局和 CLI。PettingZoo 官方 `api_test` 也包含在测试套件中。

## License

[MIT](LICENSE)

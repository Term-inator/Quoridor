# 围墙棋 Pygame 交互原型

> **PROTOTYPE — 可丢弃，不是正式 UI 实现。**

这个真实窗口原型只回答四个问题：棋盘与侧栏比例是否舒服、墙锚点吸附是否自然、合法/非法提示是否清楚、窗口缩放后信息是否仍易读。规则全部来自正式的 `Position`，原型没有复制移动、放墙或路径判断。

## 运行

在仓库根目录执行：

```bash
UV_CACHE_DIR=/tmp/quoridor-uv-cache uv run --with 'pygame-ce>=2.5.8,<3' python .scratch/quoridor-pygame-prototype/pygame_ui_prototype.py
```

依赖只进入 uv 的临时运行环境，不会修改项目依赖。也可以用 `--variant B` 或 `--variant C` 指定启动方案。

## 三个方案

- `A — 棋盘优先`：最大棋盘，传统纵向侧栏；优先验证下棋是否舒展。
- `B — 均衡工作台`：棋盘缩小约一成，反馈和操作前置；优先验证动作反馈是否更快被看到。
- `C — 规则调试台`：棋盘缩小约两成，状态和规则原因占主导；优先验证观战/调试密度。

点击底部 `A / B / C`，或按 `← / →`，可以在同一局面中切换方案。底栏是原型工具，不属于候选界面。

## 五分钟体验路径

1. 在方案 A 点击绿色棋子目标 `e2`，确认橙色起点、连线和终点能否清楚表达最近移动。
2. 点击“横墙”，在棋盘中央移动鼠标。观察预览是否稳定吸附到最近的 8×8 锚点，以及绿色是否足够明显。
3. 点击中央合法锚点（例如 `d5`）放墙。下一回合再次选择“横墙”，悬停并点击同一锚点。
4. 确认非法预览同时使用红色、白色叉线和文字原因；点击后手数不增加、行动方不变化。
5. 保持当前局面，用 A/B/C 比较层级。再把窗口拖到最小尺寸 960×640、默认尺寸约 1280×800，以及尽可能大的尺寸。

快捷键：`Esc` 回到移动模式，`R` 重置局面，`← / →` 切换方案。

## 记录结论

把亲手体验后的结论填到 [HANDOFF.md](HANDOFF.md)。如果仍有争议，继续改这个原型；如果四个问题已经回答，再把结论带回正式 Pygame 模块设计和增量规格。不要把本文件直接演变成生产模块。

## 自动渲染（仅供检查）

下面的命令会在指定目录生成三个方案、三个尺寸的 PNG，然后退出：

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy UV_CACHE_DIR=/tmp/quoridor-uv-cache uv run --with 'pygame-ce>=2.5.8,<3' python .scratch/quoridor-pygame-prototype/pygame_ui_prototype.py --screenshots /tmp/quoridor-prototype-screens
```

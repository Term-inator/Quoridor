# Issue tracker：本地 Markdown

本仓库的 issue 和规格以 Markdown 文件形式保存在 `.scratch/` 下。

## 约定

- 每项功能使用一个目录：`.scratch/<feature-slug>/`
- 规格文件为 `.scratch/<feature-slug>/spec.md`
- 实现 issue 分别写入 `.scratch/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 开始编号；不得合并为单一 tickets 文件
- Triage 状态记录在每个 issue 文件顶部附近的 `Status:` 行中；角色字符串见 `triage-labels.md`
- 评论和讨论历史追加在文件末尾的 `## Comments` 标题下

## 当技能要求“发布到 issue tracker”时

在 `.scratch/<feature-slug>/` 下创建新文件；目录不存在时一并创建。

## 当技能要求“获取相关 ticket”时

读取用户提供的路径或 issue 编号对应的文件。

## Wayfinding 操作

供 `/wayfinder` 使用。每项工作包含一个 map 文件，以及每个 ticket 对应的 child 文件。

- **Map**：`.scratch/<effort>/map.md`，正文包含 Notes、Decisions-so-far 和 Fog
- **Child ticket**：`.scratch/<effort>/issues/NN-<slug>.md`，从 `01` 开始编号；`Type:` 记录类型（`research`、`prototype`、`grilling` 或 `task`），`Status:` 记录 `claimed` 或 `resolved`
- **Blocking**：文件顶部附近用 `Blocked by: NN, NN` 记录；列出的文件全部为 `resolved` 后才算解除阻塞
- **Frontier**：扫描 `.scratch/<effort>/issues/`，选择尚未解决、未阻塞且未认领的文件；编号最小者优先
- **Claim**：开始工作前将 `Status:` 改为 `claimed` 并保存
- **Resolve**：在 `## Answer` 下追加答案，将 `Status:` 改为 `resolved`，然后把摘要和链接追加到 `map.md` 的 Decisions-so-far

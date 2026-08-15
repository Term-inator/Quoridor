# 发布到 PyPI

`quoridor-rl` 使用 GitHub Actions 和 PyPI Trusted Publishing 发布，不保存 API token。只有符合 `vX.Y.Z` 格式的 tag 会触发 `.github/workflows/publish.yml`。

## 首次配置

1. 在 GitHub 仓库的 **Settings → Environments** 中创建 `pypi` environment。
2. 为 `pypi` 配置 required reviewer，使上传在构建与验证完成后等待人工确认。
3. 登录 PyPI，在账户的 **Publishing** 页面添加 pending GitHub publisher：

   | 字段 | 值 |
   | --- | --- |
   | PyPI project name | `quoridor-rl` |
   | GitHub owner | `Term-inator` |
   | Repository name | `Quoridor` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

Pending publisher 不会预留项目名。应在发布工作流已进入 `main` 后再配置，并尽快完成首次发布。

## 发布版本

1. 在 `pyproject.toml` 中设置目标版本并更新 `uv.lock`。tag 必须是同一版本加 `v` 前缀。
2. 确认工作树干净，并运行本地门禁：

   ```bash
   uv sync --locked --extra pygame
   uv run pytest tests/package tests/pygame
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run python scripts/check-wheel.py
   ```

3. 将发布准备提交合并到 `main`，等待 CI 全部通过。
4. 在最新 `main` 提交上创建并只推送目标 tag：

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

5. 等待发布工作流的 build job 通过，然后批准 `pypi` environment deployment。
6. 确认 PyPI 页面显示正确版本、项目链接、MIT 许可证、wheel 和 sdist。
7. 从 PyPI 而不是本地项目执行最终冒烟验证：

   ```bash
   uv run --isolated --no-project --with quoridor-rl==0.1.0 \
     python -c "from quoridor_rl import Position; assert len(Position.initial().legal_actions()) == 131"
   uvx --from quoridor-rl==0.1.0 quoridor --help
   ```

## 发布失败

- tag 与 `pyproject.toml` 版本不一致时，工作流会在构建前失败；修正版本并创建新的正确 tag。
- 上传中断时可以重新运行同一个 publish job；`uv publish` 会跳过内容完全相同的已上传文件。
- PyPI 不允许覆盖同版本文件。若已发布的版本存在问题，应 yank 该版本、修复后提升 patch 版本，不能重新上传 `0.1.0`。
- 首次上传前若 `quoridor-rl` 已被其他账户占用，停止发布并重新确定项目名，不创建兼容别名。

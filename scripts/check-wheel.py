"""构建并在隔离环境中验证发行 wheel、sdist 和可选依赖。"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    """构建发行包、检查内容，并隔离验证两种包格式和 Pygame extra。"""
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()

    with tempfile.TemporaryDirectory(prefix="quoridor-wheel-") as temporary:
        root = Path(temporary)
        distributions = root / "dist"
        virtual_environment = root / "venv"

        _run(
            "uv",
            "build",
            "--no-sources",
            "--out-dir",
            str(distributions),
            cwd=repository,
            environment=environment,
        )
        wheels = tuple(distributions.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {len(wheels)}")
        source_distributions = tuple(distributions.glob("*.tar.gz"))
        if len(source_distributions) != 1:
            raise RuntimeError(
                f"expected one source distribution, found {len(source_distributions)}"
            )

        _check_wheel_contents(wheels[0])
        _check_sdist_contents(source_distributions[0])

        for name, distribution in (
            ("wheel", wheels[0]),
            ("sdist", source_distributions[0]),
        ):
            _check_base_install(
                distribution,
                virtual_environment.with_name(f"venv-{name}"),
                root=root,
                environment=environment,
            )

        _check_pygame_extra(
            wheels[0],
            virtual_environment.with_name("venv-pygame"),
            root=root,
            environment=environment,
        )


def _check_wheel_contents(wheel: Path) -> None:
    """确认 wheel 只包含发行代码，并携带字体与两份许可证。"""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    _check_archive_contents(names)
    _require_suffix(names, ".dist-info/licenses/LICENSE")


def _check_sdist_contents(source_distribution: Path) -> None:
    """确认 sdist 可独立重建，并携带项目许可证和必要资源。"""
    with tarfile.open(source_distribution, "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}

    _check_archive_contents(names)
    _require_suffix(names, "/LICENSE")


def _check_archive_contents(names: set[str]) -> None:
    """验证两种发行格式共享的内容边界。"""
    _require_suffix(names, "/pygame_ui/assets/NotoSansSC-Regular.otf")
    _require_suffix(names, "/pygame_ui/assets/OFL.txt")
    if any(name == "experiments" or "/experiments/" in name for name in names):
        raise RuntimeError("experiments must not be included in distributions")
    if any(name == "tests" or "/tests/" in name for name in names):
        raise RuntimeError("tests must not be included in distributions")


def _require_suffix(names: set[str], suffix: str) -> None:
    """要求归档中恰有一个以指定路径结尾的文件。"""
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one archive entry ending in {suffix!r}: {matches}"
        )


def _check_base_install(
    distribution: Path,
    virtual_environment: Path,
    *,
    root: Path,
    environment: dict[str, str],
) -> None:
    """隔离安装一个发行格式，并验证基础 API、依赖边界和 CLI。"""
    python = _install_distribution(
        str(distribution),
        virtual_environment,
        root=root,
        environment=environment,
    )
    _run(
        str(python),
        "-I",
        "-c",
        _SMOKE_PROGRAM,
        cwd=root,
        environment=environment,
    )
    _run(
        str(_venv_executable(virtual_environment, "quoridor")),
        "--help",
        cwd=root,
        environment=environment,
    )


def _check_pygame_extra(
    wheel: Path,
    virtual_environment: Path,
    *,
    root: Path,
    environment: dict[str, str],
) -> None:
    """通过 wheel extra 安装 Pygame，并确认图形模块可导入。"""
    python = _install_distribution(
        f"{wheel}[pygame]",
        virtual_environment,
        root=root,
        environment=environment,
    )
    _run(
        str(python),
        "-I",
        "-c",
        "import pygame; import quoridor_rl.pygame_ui.app",
        cwd=root,
        environment=environment,
    )


def _install_distribution(
    requirement: str,
    virtual_environment: Path,
    *,
    root: Path,
    environment: dict[str, str],
) -> Path:
    """建立空虚拟环境，安装给定发行包并返回其 Python 路径。"""
    _run(
        "uv",
        "venv",
        str(virtual_environment),
        "--python",
        sys.executable,
        cwd=root,
        environment=environment,
    )
    python = _venv_executable(virtual_environment, "python")
    _run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        requirement,
        cwd=root,
        environment=environment,
    )
    return python


def _venv_executable(virtual_environment: Path, name: str) -> Path:
    """按当前操作系统返回虚拟环境内可执行文件路径。"""
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return virtual_environment / directory / f"{name}{suffix}"


def _run(
    *command: str,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    """在指定目录和环境中运行子进程，非零退出码立即失败。"""
    subprocess.run(command, cwd=cwd, env=environment, check=True)


_SMOKE_PROGRAM = """
import importlib.util

from quoridor_rl import Position, env

position = Position.initial()
assert len(position.legal_actions()) == 131

environment = env()
environment.reset(seed=0)
assert environment.agent_selection == "player_0"

assert importlib.util.find_spec("experiments") is None
assert importlib.util.find_spec("pygame") is None
"""


if __name__ == "__main__":
    main()

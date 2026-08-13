"""Build and smoke-test the distribution in an isolated environment."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()

    with tempfile.TemporaryDirectory(prefix="quoridor-wheel-") as temporary:
        root = Path(temporary)
        distributions = root / "dist"
        virtual_environment = root / "venv"

        _run(
            "uv",
            "build",
            "--out-dir",
            str(distributions),
            cwd=repository,
            environment=environment,
        )
        wheels = tuple(distributions.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {len(wheels)}")

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
            str(wheels[0]),
            cwd=root,
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


def _venv_executable(virtual_environment: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return virtual_environment / directory / f"{name}{suffix}"


def _run(
    *command: str,
    cwd: Path,
    environment: dict[str, str],
) -> None:
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
"""


if __name__ == "__main__":
    main()

"""限时 AlphaZero 验证实验的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.alphazero.experiment import run_experiment, run_smoke


def main() -> None:
    """解析实验参数，执行正式或冒烟运行并打印证据产物路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/alphazero/results/seed-0"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("experiments/alphazero/artifacts/seed-0"),
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=Path("experiments/alphazero/artifacts/smoke"),
    )
    parser.add_argument("--training-minutes", type=float, default=120)
    parser.add_argument("--total-minutes", type=float, default=150)
    parser.add_argument("--validation-games", type=int, default=200)
    parser.add_argument("--final-games", type=int, default=1_000)
    arguments = parser.parse_args()
    device = _device(arguments.device)
    if arguments.smoke:
        artifacts = run_smoke(arguments.smoke_dir, device=device)
    else:
        artifacts = run_experiment(
            arguments.results_dir,
            arguments.artifacts_dir,
            device=device,
            training_seconds=arguments.training_minutes * 60,
            total_seconds=arguments.total_minutes * 60,
            validation_games=arguments.validation_games,
            final_games=arguments.final_games,
        )
    print(f"metrics: {artifacts.metrics_path}")
    print(f"summary: {artifacts.summary_path}")
    print(f"curve: {artifacts.curve_path}")
    print(f"checkpoint: {artifacts.checkpoint_path}")
    if artifacts.comparison_path is not None:
        print(f"comparison: {artifacts.comparison_path}")


def _device(requested: str) -> torch.device:
    """解析设备选项；自动模式在 CUDA 可用时优先使用 GPU。"""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()

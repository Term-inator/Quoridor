"""AlphaZero 限时实验编排、阶段评估与证据产物生成。"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.alphazero.evaluation import EvaluationResult, evaluate
from experiments.alphazero.model import PolicyValueNetwork
from experiments.alphazero.self_play import SelfPlayGame, play_self_game
from experiments.alphazero.training import (
    AlphaZeroConfig,
    PolicyValueUpdater,
    ReplayBuffer,
)


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    """一次实验生成的指标、摘要、曲线、权重、配置及可选对比路径。"""

    metrics_path: Path
    summary_path: Path
    curve_path: Path
    checkpoint_path: Path
    config_path: Path
    comparison_path: Path | None = None


def run_experiment(
    results_directory: Path,
    artifacts_directory: Path,
    *,
    device: torch.device,
    config: AlphaZeroConfig | None = None,
    training_seconds: float = 120 * 60,
    total_seconds: float = 150 * 60,
    checkpoint_seconds: tuple[float, ...] = (15 * 60, 30 * 60, 60 * 60, 120 * 60),
    validation_games: int = 200,
    final_games: int = 1_000,
) -> ExperimentArtifacts:
    """运行受训练/总时限约束的单种子 AlphaZero 实验。

    每轮先产生一局搜索自对弈，仅把有明确胜负的样本加入回放，再执行若干策略—价值
    更新。阶段评估选择最佳检查点，最终生成指标、图表与跨算法对比证据。
    """
    config = AlphaZeroConfig() if config is None else config
    _validate(config, training_seconds, total_seconds)
    _prepare_runtime(config, device)
    results_directory.mkdir(parents=True, exist_ok=True)
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(artifacts_directory / "tensorboard")
    model = PolicyValueNetwork().to(device)
    updater = PolicyValueUpdater(model, config, device)
    replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
    started = time.monotonic()
    deadline = started + total_seconds
    training_elapsed = 0.0
    game_count = 0
    update_count = 0
    checkpoint_index = 0
    history: list[dict[str, float | int]] = []
    best_checkpoint: Path | None = None
    best_validation: EvaluationResult | None = None
    last_game: SelfPlayGame | None = None
    status = "completed"

    try:
        while training_elapsed < training_seconds and time.monotonic() < deadline:
            iteration_started = time.monotonic()
            game = play_self_game(
                model,
                config,
                device,
                game_index=game_count,
            )
            game_count += 1
            last_game = game
            accepted_examples = 0
            if not game.truncated:
                for example in game.examples:
                    replay.add(example)
                    accepted_examples += 1

            totals: dict[str, float] = {}
            updates = 0
            if len(replay) >= max(config.replay_warmup, config.batch_size):
                for _ in range(config.updates_per_game):
                    metrics = updater.update(replay.sample(config.batch_size))
                    updates += 1
                    update_count += 1
                    for name, value in metrics.items():
                        totals[name] = totals.get(name, 0.0) + value
            averaged = {name: value / updates for name, value in totals.items()}
            iteration_seconds = time.monotonic() - iteration_started
            training_elapsed += iteration_seconds
            point: dict[str, float | int] = {
                "game": game_count,
                "training_seconds": training_elapsed,
                "plies": game.plies,
                "truncated": int(game.truncated),
                "examples": len(game.examples),
                "accepted_examples": accepted_examples,
                "replay_size": len(replay),
                "updates": update_count,
                "expanded_nodes": game.expanded_nodes,
                "maximum_search_depth": game.maximum_search_depth,
                "self_play_seconds": game.elapsed_seconds,
                "nodes_per_second": (
                    game.expanded_nodes / game.elapsed_seconds
                    if game.elapsed_seconds
                    else 0.0
                ),
                **averaged,
            }
            history.append(point)
            print(
                f"game={game_count} plies={game.plies} "
                f"result={'unresolved' if game.truncated else f'player_{int(game.winner)}'} "
                f"train_minutes={training_elapsed / 60:.1f} "
                f"replay={len(replay)} updates={update_count} "
                f"nodes_per_second={point['nodes_per_second']:.1f}",
                flush=True,
            )
            for name, value in averaged.items():
                writer.add_scalar(f"train/{name}", value, update_count)
            writer.add_scalar("self_play/plies", game.plies, game_count)
            writer.add_scalar("self_play/truncated", int(game.truncated), game_count)
            writer.add_scalar(
                "self_play/nodes_per_second", point["nodes_per_second"], game_count
            )
            writer.add_scalar("replay/size", len(replay), game_count)

            if checkpoint_index < len(checkpoint_seconds) and (
                training_elapsed >= checkpoint_seconds[checkpoint_index]
            ):
                checkpoint_path = artifacts_directory / (
                    f"checkpoint-{int(checkpoint_seconds[checkpoint_index] // 60):03d}m.pt"
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    updater=updater,
                    config=config,
                    games=game_count,
                    updates=update_count,
                    training_seconds=training_elapsed,
                )
                validation = evaluate(
                    model,
                    games=validation_games,
                    device=device,
                    config=config,
                    seed=20_000 + checkpoint_index * validation_games,
                    deadline=deadline,
                    progress=True,
                )
                point.update(_validation_metrics(validation))
                if validation.complete and _is_better(validation, best_validation):
                    best_validation = validation
                    best_checkpoint = checkpoint_path
                print(
                    f"validation games={validation.games} "
                    f"win_rate={validation.win_rate:.1%} "
                    f"unresolved_rate={validation.truncation_rate:.1%}",
                    flush=True,
                )
                checkpoint_index += 1
                if validation.illegal_actions:
                    status = "illegal-action"
                    break
                if not validation.complete:
                    status = "total-deadline"
                    break

        if last_game is None:
            raise RuntimeError("training ended before one self-play game completed")
        if best_checkpoint is None and time.monotonic() < deadline:
            best_checkpoint = artifacts_directory / "checkpoint-final.pt"
            _save_checkpoint(
                best_checkpoint,
                model=model,
                updater=updater,
                config=config,
                games=game_count,
                updates=update_count,
                training_seconds=training_elapsed,
            )
            validation = evaluate(
                model,
                games=validation_games,
                device=device,
                config=config,
                seed=30_000,
                deadline=deadline,
                progress=True,
            )
            history[-1].update(_validation_metrics(validation))
            if validation.complete:
                best_validation = validation
            else:
                status = "total-deadline"

        if best_checkpoint is None:
            best_checkpoint = artifacts_directory / "checkpoint-deadline.pt"
            _save_checkpoint(
                best_checkpoint,
                model=model,
                updater=updater,
                config=config,
                games=game_count,
                updates=update_count,
                training_seconds=training_elapsed,
            )
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        final_evaluation = evaluate(
            model,
            games=final_games,
            device=device,
            config=config,
            seed=40_000,
            deadline=deadline,
            progress=True,
        )
        if not final_evaluation.complete:
            status = "total-deadline"
        print(
            f"final games={final_evaluation.games} "
            f"win_rate={final_evaluation.win_rate:.1%} "
            f"unresolved_rate={final_evaluation.truncation_rate:.1%}",
            flush=True,
        )
        artifacts = _write_artifacts(
            results_directory,
            config=config,
            device=device,
            history=history,
            last_game=last_game,
            final_evaluation=final_evaluation,
            checkpoint_path=best_checkpoint,
            status=status,
            total_elapsed_seconds=time.monotonic() - started,
            smoke_checks=None,
        )
        comparison_path = _write_comparison(
            results_directory,
            alpha_evaluation=final_evaluation,
        )
        return ExperimentArtifacts(
            metrics_path=artifacts.metrics_path,
            summary_path=artifacts.summary_path,
            curve_path=artifacts.curve_path,
            checkpoint_path=artifacts.checkpoint_path,
            config_path=artifacts.config_path,
            comparison_path=comparison_path,
        )
    finally:
        writer.close()


def run_smoke(
    output_directory: Path,
    *,
    device: torch.device,
) -> ExperimentArtifacts:
    """用极小搜索和回放配置验证 AlphaZero 端到端执行路径。"""
    """Exercise search, self-play, replay, updates, restore, and evaluation."""
    config = AlphaZeroConfig(
        max_plies=128,
        simulations_per_move=8,
        evaluation_simulations=2,
        evaluation_workers=1,
        maximum_search_actions=8,
        pawn_only_curriculum_games=1,
        temperature_plies=10,
        replay_capacity=256,
        replay_warmup=8,
        batch_size=8,
        updates_per_game=12,
        learning_rate=1e-2,
        torch_threads=1,
    )
    _prepare_runtime(config, device)
    output_directory.mkdir(parents=True, exist_ok=True)
    model = PolicyValueNetwork().to(device)
    updater = PolicyValueUpdater(model, config, device)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    game = play_self_game(model, config, device, game_index=0)
    replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
    if game.truncated:
        raise RuntimeError("pawn-only smoke game did not reach a normal terminal")
    for example in game.examples:
        replay.add(example)
    first_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] | None = None
    for _ in range(config.updates_per_game):
        last_metrics = updater.update(replay.sample(config.batch_size))
        if first_metrics is None:
            first_metrics = last_metrics
    assert first_metrics is not None and last_metrics is not None
    checkpoint_path = output_directory / "smoke.pt"
    _save_checkpoint(
        checkpoint_path,
        model=model,
        updater=updater,
        config=config,
        games=1,
        updates=config.updates_per_game,
        training_seconds=game.elapsed_seconds,
    )
    restored = PolicyValueNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    evaluation = evaluate(
        restored,
        games=2,
        device=device,
        config=config,
        seed=10_000,
    )
    smoke_checks = {
        "checkpoint_restored": all(
            torch.equal(value, restored.state_dict()[name])
            for name, value in model.state_dict().items()
        ),
        "legal_search_policy": all(
            example.policy.sum() > 0 and np.isfinite(example.policy).all()
            for example in game.examples
        ),
        "parameters_updated": any(
            not torch.equal(value, model.state_dict()[name])
            for name, value in before.items()
        ),
        "pure_terminal_targets": all(
            abs(example.value) == 1 for example in game.examples
        ),
    }
    if not all(smoke_checks.values()) or evaluation.illegal_actions:
        raise RuntimeError(f"AlphaZero smoke failed: {smoke_checks}")
    history: list[dict[str, float | int]] = [
        {
            "game": 1,
            "training_seconds": game.elapsed_seconds,
            "plies": game.plies,
            "truncated": int(game.truncated),
            "examples": len(game.examples),
            "accepted_examples": len(game.examples),
            "replay_size": len(replay),
            "updates": config.updates_per_game,
            "expanded_nodes": game.expanded_nodes,
            "maximum_search_depth": game.maximum_search_depth,
            "self_play_seconds": game.elapsed_seconds,
            "nodes_per_second": game.expanded_nodes / game.elapsed_seconds,
            **last_metrics,
        }
    ]
    return _write_artifacts(
        output_directory,
        config=config,
        device=device,
        history=history,
        last_game=game,
        final_evaluation=evaluation,
        checkpoint_path=checkpoint_path,
        status="smoke-passed",
        total_elapsed_seconds=game.elapsed_seconds,
        smoke_checks=smoke_checks,
    )


def _prepare_runtime(config: AlphaZeroConfig, device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.torch_threads)


def _validate(
    config: AlphaZeroConfig,
    training_seconds: float,
    total_seconds: float,
) -> None:
    if training_seconds <= 0 or total_seconds < training_seconds:
        raise ValueError("time budgets must be positive and total must cover training")
    if config.max_plies <= 0 or config.temperature_plies < 0:
        raise ValueError("ply limits are invalid")
    if config.maximum_search_actions <= 0 or config.pawn_only_curriculum_games < 0:
        raise ValueError("search candidate and curriculum limits are invalid")
    if not 0 <= config.curriculum_progress_prior <= 1:
        raise ValueError("curriculum_progress_prior must be between zero and one")
    if config.evaluation_workers <= 0:
        raise ValueError("evaluation_workers must be positive")
    if config.replay_warmup > config.replay_capacity:
        raise ValueError("replay warmup cannot exceed capacity")
    if config.batch_size > config.replay_capacity:
        raise ValueError("batch size cannot exceed replay capacity")


def _save_checkpoint(
    path: Path,
    *,
    model: PolicyValueNetwork,
    updater: PolicyValueUpdater,
    config: AlphaZeroConfig,
    games: int,
    updates: int,
    training_seconds: float,
) -> None:
    torch.save(
        {
            "seed": config.seed,
            "model": model.state_dict(),
            "optimizer": updater.optimizer.state_dict(),
            "config": asdict(config),
            "games": games,
            "updates": updates,
            "training_seconds": training_seconds,
        },
        path,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model"].keys() != model.state_dict().keys():
        raise RuntimeError(f"checkpoint could not be restored: {path}")


def _is_better(
    candidate: EvaluationResult,
    incumbent: EvaluationResult | None,
) -> bool:
    if incumbent is None:
        return True
    return (candidate.win_rate, -candidate.truncation_rate) > (
        incumbent.win_rate,
        -incumbent.truncation_rate,
    )


def _validation_metrics(result: EvaluationResult) -> dict[str, float | int]:
    return {
        "validation_games": result.games,
        "validation_win_rate": result.win_rate,
        "validation_truncation_rate": result.truncation_rate,
        "validation_illegal_actions": result.illegal_actions,
    }


def _write_artifacts(
    output_directory: Path,
    *,
    config: AlphaZeroConfig,
    device: torch.device,
    history: list[dict[str, float | int]],
    last_game: SelfPlayGame,
    final_evaluation: EvaluationResult,
    checkpoint_path: Path,
    status: str,
    total_elapsed_seconds: float,
    smoke_checks: dict[str, bool] | None,
) -> ExperimentArtifacts:
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = output_directory / "config.json"
    metrics_path = output_directory / "metrics.json"
    summary_path = output_directory / "summary.md"
    curve_path = output_directory / "learning-curve.png"
    hardware = _hardware(device)
    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics: dict[str, object] = {
        "status": status,
        "total_elapsed_seconds": total_elapsed_seconds,
        "provenance": _provenance(),
        "hardware": hardware,
        "history": history,
        "last_self_play": {
            "plies": last_game.plies,
            "winner": None if last_game.winner is None else int(last_game.winner),
            "truncated": last_game.truncated,
            "expanded_nodes": last_game.expanded_nodes,
        },
        "final_evaluation": _evaluation_dict(final_evaluation),
    }
    if smoke_checks is not None:
        metrics["smoke_checks"] = smoke_checks
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        _summary(
            status, final_evaluation, history, checkpoint_path, total_elapsed_seconds
        ),
        encoding="utf-8",
    )
    _plot_history(history, curve_path)
    return ExperimentArtifacts(
        metrics_path=metrics_path,
        summary_path=summary_path,
        curve_path=curve_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


def _evaluation_dict(result: EvaluationResult) -> dict[str, object]:
    return {
        "requested_games": result.requested_games,
        "games": result.games,
        "complete": result.complete,
        "wins": result.wins,
        "losses": result.losses,
        "unresolved": result.truncated,
        "win_rate": result.win_rate,
        "unresolved_rate": result.truncation_rate,
        "illegal_actions": result.illegal_actions,
        "as_player_0": asdict(result.as_player_0),
        "as_player_1": asdict(result.as_player_1),
    }


def _summary(
    status: str,
    evaluation: EvaluationResult,
    history: list[dict[str, float | int]],
    checkpoint_path: Path,
    elapsed: float,
) -> str:
    target_met = (
        evaluation.complete
        and evaluation.win_rate >= 0.70
        and evaluation.truncation_rate <= 0.05
        and evaluation.illegal_actions == 0
    )
    return (
        "# AlphaZero seed 0 summary\n\n"
        f"- Status: `{status}`\n"
        f"- Training games: {len(history)}\n"
        f"- Total elapsed: {elapsed / 60:.1f} minutes\n"
        f"- Final evaluation: {evaluation.games}/{evaluation.requested_games} games\n"
        f"- Wins/losses/unresolved: {evaluation.wins}/{evaluation.losses}/{evaluation.truncated}\n"
        f"- Win rate: {evaluation.win_rate:.1%}\n"
        f"- Unresolved rate: {evaluation.truncation_rate:.1%}\n"
        f"- Illegal actions: {evaluation.illegal_actions}\n"
        f"- Exploratory target met: {'yes' if target_met else 'no'}\n"
        f"- Selected checkpoint: `{checkpoint_path}`\n"
    )


def _plot_history(history: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib.pyplot as plt

    minutes = [float(point["training_seconds"]) / 60 for point in history]
    losses = [float(point.get("loss", float("nan"))) for point in history]
    unresolved = [float(point["truncated"]) for point in history]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(minutes, losses)
    axes[0].set_ylabel("policy + value loss")
    axes[1].plot(minutes, unresolved, alpha=0.6)
    axes[1].set_ylabel("self-play unresolved")
    axes[1].set_xlabel("training minutes")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _hardware(device: torch.device) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cpu_count": os.cpu_count(),
    }


def _provenance() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "numpy": version("numpy"),
        "pettingzoo": version("pettingzoo"),
    }


def _write_comparison(
    output_directory: Path,
    *,
    alpha_evaluation: EvaluationResult,
) -> Path:
    rows = [("AlphaZero", _evaluation_dict(alpha_evaluation))]
    for name, path in (
        ("PPO", Path("experiments/ppo/results/seed-0/metrics.json")),
        ("DQN", Path("experiments/dqn/results/seed-0/metrics.json")),
    ):
        if path.is_file():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            evaluation = metrics.get("final_evaluation")
            if isinstance(evaluation, dict):
                rows.append((name, evaluation))
    lines = [
        "# Seed 0 exploratory comparison",
        "",
        "| Experiment | Games | Win rate | Unresolved rate | Illegal actions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, evaluation in rows:
        games = evaluation.get("games", 0)
        win_rate = float(evaluation.get("win_rate", 0.0))
        unresolved_rate = float(
            evaluation.get("unresolved_rate", evaluation.get("truncation_rate", 0.0))
        )
        illegal = evaluation.get("illegal_actions", 0)
        lines.append(
            f"| {name} | {games} | {win_rate:.1%} | {unresolved_rate:.1%} | {illegal} |"
        )
    lines.extend(
        [
            "",
            "These are independent single-seed explorations and do not establish algorithm-level superiority.",
            "",
        ]
    )
    path = output_directory / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

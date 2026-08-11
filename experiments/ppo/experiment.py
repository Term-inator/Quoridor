"""Experiment orchestration and result artifact generation."""

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

from experiments.ppo.evaluation import EvaluationResult, evaluate
from experiments.ppo.model import MaskedActorCritic
from experiments.ppo.training import (
    PPOConfig,
    PPOUpdater,
    Rollout,
    collect_rollout,
)


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    metrics_path: Path
    summary_path: Path
    curve_path: Path
    checkpoint_path: Path
    config_path: Path


def run_experiment(
    results_directory: Path,
    artifacts_directory: Path,
    *,
    device: torch.device,
    config: PPOConfig | None = None,
    training_seconds: float = 120 * 60,
    total_seconds: float = 150 * 60,
    checkpoint_seconds: tuple[float, ...] = (15 * 60, 30 * 60, 60 * 60, 120 * 60),
    validation_games: int = 200,
    final_games: int = 1000,
) -> ExperimentArtifacts:
    """Run the bounded single-seed experiment and write its evidence."""
    if config is None:
        config = PPOConfig()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    _seed_everything(config.seed)
    results_directory.mkdir(parents=True, exist_ok=True)
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(artifacts_directory / "tensorboard")
    model = MaskedActorCritic().to(device)
    updater = PPOUpdater(model, config, device)
    started = time.monotonic()
    deadline = started + total_seconds
    training_elapsed = 0.0
    transition_count = 0
    update_number = 0
    checkpoint_index = 0
    best_checkpoint: Path | None = None
    best_win_rate = -1.0
    validation_rates: list[float] = []
    no_normal_evaluations = 0
    stop_reason: str | None = None
    history: list[dict[str, float | int]] = []
    last_rollout: Rollout | None = None

    try:
        while training_elapsed < training_seconds and time.monotonic() < deadline:
            update_started = time.monotonic()
            rollout = collect_rollout(model, config, device)
            update_metrics = updater.update(rollout.transitions)
            update_seconds = time.monotonic() - update_started
            training_elapsed += update_seconds
            transition_count += len(rollout.transitions)
            update_number += 1
            last_rollout = rollout
            point: dict[str, float | int] = {
                "update": update_number,
                "training_seconds": training_elapsed,
                "transitions": transition_count,
                "transitions_per_second": len(rollout.transitions) / update_seconds,
                "terminated_episodes": sum(
                    episode.terminated for episode in rollout.episodes
                ),
                "truncated_episodes": sum(
                    episode.truncated for episode in rollout.episodes
                ),
                **update_metrics,
            }
            history.append(point)
            print(
                f"update={update_number} transitions={transition_count} "
                f"train_minutes={training_elapsed / 60:.1f} "
                f"throughput={point['transitions_per_second']:.1f}/s",
                flush=True,
            )
            for name, value in update_metrics.items():
                writer.add_scalar(f"ppo/{name}", value, transition_count)
            writer.add_scalar(
                "rollout/transitions_per_second",
                point["transitions_per_second"],
                transition_count,
            )

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
                    update_number=update_number,
                    training_seconds=training_elapsed,
                )
                validation = evaluate(
                    model,
                    games=validation_games,
                    device=device,
                    max_plies=config.max_plies,
                    seed=20_000 + checkpoint_index * validation_games,
                    deadline=deadline,
                    progress=True,
                )
                point.update(
                    {
                        "validation_games": validation.games,
                        "validation_win_rate": validation.win_rate,
                        "validation_truncation_rate": validation.truncation_rate,
                        "validation_illegal_actions": validation.illegal_actions,
                    }
                )
                writer.add_scalar(
                    "validation/win_rate",
                    validation.win_rate,
                    transition_count,
                )
                writer.add_scalar(
                    "validation/truncation_rate",
                    validation.truncation_rate,
                    transition_count,
                )
                if validation.complete and validation.win_rate > best_win_rate:
                    best_win_rate = validation.win_rate
                    best_checkpoint = checkpoint_path
                normal_games = validation.wins + validation.losses
                no_normal_evaluations = (
                    no_normal_evaluations + 1 if normal_games == 0 else 0
                )
                validation_rates.append(validation.win_rate)
                print(
                    f"validation games={validation.games} "
                    f"win_rate={validation.win_rate:.1%} "
                    f"truncation_rate={validation.truncation_rate:.1%}",
                    flush=True,
                )
                if validation.illegal_actions:
                    stop_reason = "illegal-action"
                elif no_normal_evaluations >= 2:
                    stop_reason = "no-normal-terminations"
                elif len(validation_rates) >= 3 and (
                    validation_rates[-3] > validation_rates[-2] > validation_rates[-1]
                ):
                    stop_reason = "three-declining-evaluations"
                elif not validation.complete:
                    stop_reason = "total-deadline"
                checkpoint_index += 1
                if stop_reason is not None:
                    break

        if last_rollout is None:
            raise RuntimeError("training ended before one rollout completed")
        if best_checkpoint is None and time.monotonic() < deadline:
            best_checkpoint = artifacts_directory / "checkpoint-final.pt"
            _save_checkpoint(
                best_checkpoint,
                model=model,
                updater=updater,
                config=config,
                update_number=update_number,
                training_seconds=training_elapsed,
            )
            validation = evaluate(
                model,
                games=validation_games,
                device=device,
                max_plies=config.max_plies,
                seed=30_000,
                deadline=deadline,
                progress=True,
            )
            history[-1].update(
                {
                    "validation_games": validation.games,
                    "validation_win_rate": validation.win_rate,
                    "validation_truncation_rate": validation.truncation_rate,
                    "validation_illegal_actions": validation.illegal_actions,
                }
            )

        if best_checkpoint is None:
            best_checkpoint = artifacts_directory / "checkpoint-deadline.pt"
            _save_checkpoint(
                best_checkpoint,
                model=model,
                updater=updater,
                config=config,
                update_number=update_number,
                training_seconds=training_elapsed,
            )
        checkpoint = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
        final_evaluation = evaluate(
            model,
            games=final_games,
            device=device,
            max_plies=config.max_plies,
            seed=40_000,
            deadline=deadline,
            progress=True,
        )
        print(
            f"final games={final_evaluation.games} "
            f"win_rate={final_evaluation.win_rate:.1%} "
            f"truncation_rate={final_evaluation.truncation_rate:.1%}",
            flush=True,
        )
        status = stop_reason or (
            "completed" if final_evaluation.complete else "total-deadline"
        )
        return _write_artifacts(
            results_directory,
            config=config,
            device=device,
            history=history,
            rollout=last_rollout,
            final_evaluation=final_evaluation,
            checkpoint_path=best_checkpoint,
            status=status,
            total_elapsed_seconds=time.monotonic() - started,
        )
    except Exception as error:
        _write_failure(results_directory, config, device, error, history)
        raise
    finally:
        writer.close()


def run_smoke(
    output_directory: Path,
    *,
    device: torch.device,
) -> ExperimentArtifacts:
    """Run one tiny collect/update/reload/evaluate cycle."""
    config = PPOConfig(
        environment_count=1,
        rollout_size=8,
        max_plies=4,
        minibatch_size=4,
        update_epochs=1,
    )
    _seed_everything(config.seed)
    model = MaskedActorCritic().to(device)
    updater = PPOUpdater(model, config, device)
    started = time.monotonic()
    rollout = collect_rollout(model, config, device)
    update_metrics = updater.update(rollout.transitions)
    elapsed = time.monotonic() - started

    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts_directory = output_directory / "artifacts"
    artifacts_directory.mkdir(exist_ok=True)
    checkpoint_path = artifacts_directory / "smoke.pt"
    torch.save(
        {
            "seed": config.seed,
            "model": model.state_dict(),
            "optimizer": updater.optimizer.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )
    restored = MaskedActorCritic().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    evaluation = evaluate(
        restored,
        games=2,
        device=device,
        max_plies=4,
        seed=10_000,
    )
    history = [
        {
            "training_seconds": elapsed,
            "transitions": len(rollout.transitions),
            **update_metrics,
            "validation_win_rate": evaluation.win_rate,
            "validation_truncation_rate": evaluation.truncation_rate,
        }
    ]
    return _write_artifacts(
        output_directory,
        config=config,
        device=device,
        history=history,
        rollout=rollout,
        final_evaluation=evaluation,
        checkpoint_path=checkpoint_path,
        status="smoke-passed",
        total_elapsed_seconds=time.monotonic() - started,
    )


def _write_artifacts(
    output_directory: Path,
    *,
    config: PPOConfig,
    device: torch.device,
    history: list[dict[str, float | int]],
    rollout: Rollout,
    final_evaluation: EvaluationResult,
    checkpoint_path: Path,
    status: str,
    total_elapsed_seconds: float,
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
    metrics = {
        "status": status,
        "total_elapsed_seconds": total_elapsed_seconds,
        "hardware": hardware,
        "history": history,
        "last_rollout": {
            "transitions": len(rollout.transitions),
            "episodes": len(rollout.episodes),
            "terminations": sum(episode.terminated for episode in rollout.episodes),
            "truncations": sum(episode.truncated for episode in rollout.episodes),
        },
        "final_evaluation": _evaluation_dict(final_evaluation),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        _summary(
            status,
            hardware,
            final_evaluation,
            history,
            checkpoint_path,
            total_elapsed_seconds,
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
        "complete": result.complete,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "truncated": result.truncated,
        "illegal_actions": result.illegal_actions,
        "win_rate": result.win_rate,
        "truncation_rate": result.truncation_rate,
        "as_player_0": asdict(result.as_player_0),
        "as_player_1": asdict(result.as_player_1),
    }


def _summary(
    status: str,
    hardware: dict[str, object],
    evaluation: EvaluationResult,
    history: list[dict[str, float | int]],
    checkpoint_path: Path,
    total_elapsed_seconds: float,
) -> str:
    passed = (
        evaluation.win_rate >= 0.70
        and evaluation.truncation_rate <= 0.05
        and evaluation.illegal_actions == 0
    )
    validation = [point for point in history if "validation_win_rate" in point]
    best_validation = max(
        validation,
        key=lambda point: float(point["validation_win_rate"]),
        default=None,
    )
    best_line = (
        "- Best checkpoint validation: unavailable"
        if best_validation is None
        else (
            f"- Best checkpoint validation: "
            f"{float(best_validation['validation_win_rate']):.1%} wins, "
            f"{float(best_validation['validation_truncation_rate']):.1%} unresolved "
            f"at {float(best_validation['training_seconds']) / 60:.1f} minutes"
        )
    )
    return "\n".join(
        (
            "# PPO single-seed validation",
            "",
            f"- Status: `{status}`",
            f"- Device: `{hardware['device']}`",
            f"- Total elapsed: {total_elapsed_seconds / 60:.1f} minutes",
            f"- Selected checkpoint: `{checkpoint_path.name}`",
            best_line,
            f"- Games: {evaluation.games}",
            (
                f"- Wins / losses / unresolved: {evaluation.wins} / "
                f"{evaluation.losses} / {evaluation.truncated}"
            ),
            f"- Win rate: {evaluation.win_rate:.1%}",
            f"- Unresolved rate: {evaluation.truncation_rate:.1%}",
            f"- Illegal actions: {evaluation.illegal_actions}",
            f"- Exploratory target met: {'yes' if passed else 'no'}",
            "",
        )
    )


def _plot_history(
    history: list[dict[str, float | int]],
    path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quoridor-matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    validation = [point for point in history if "validation_win_rate" in point]
    x = [float(point["training_seconds"]) / 60 for point in validation]
    win_rates = [float(point["validation_win_rate"]) for point in validation]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x, win_rates, marker="o", label="vs random")
    axis.axhline(0.70, color="tab:green", linestyle="--", label="target")
    axis.set(xlabel="training minutes", ylabel="win rate", ylim=(0, 1))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_checkpoint(
    path: Path,
    *,
    model: MaskedActorCritic,
    updater: PPOUpdater,
    config: PPOConfig,
    update_number: int,
    training_seconds: float,
) -> None:
    torch.save(
        {
            "seed": config.seed,
            "model": model.state_dict(),
            "optimizer": updater.optimizer.state_dict(),
            "config": asdict(config),
            "update": update_number,
            "training_seconds": training_seconds,
        },
        path,
    )


def _write_failure(
    results_directory: Path,
    config: PPOConfig,
    device: torch.device,
    error: Exception,
    history: list[dict[str, float | int]],
) -> None:
    results_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "config": asdict(config),
        "hardware": _hardware(device),
        "history": history,
    }
    (results_directory / "failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hardware(device: torch.device) -> dict[str, object]:
    git_status = _command_output(("git", "status", "--porcelain"))
    return {
        "git_commit": _command_output(("git", "rev-parse", "HEAD")),
        "git_dirty": bool(git_status),
        "python": platform.python_version(),
        "device": str(device),
        "torch": torch.__version__,
        "torchrl": version("torchrl"),
        "pettingzoo": version("pettingzoo"),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "nvidia_driver": _command_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        ),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def _command_output(command: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

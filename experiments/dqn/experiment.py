"""Bounded DQN experiment orchestration and evidence generation."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from experiments.dqn.evaluation import EvaluationResult, evaluate
from experiments.dqn.model import MaskedQNetwork
from experiments.dqn.opponents import OpponentPool
from experiments.dqn.self_play import Collection, SelfPlayCollector
from experiments.dqn.training import (
    DQNConfig,
    DQNUpdater,
    ReplayBuffer,
    TransitionBatch,
)


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    metrics_path: Path
    summary_path: Path
    curve_path: Path
    checkpoint_path: Path
    config_path: Path
    comparison_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _TrainingStep:
    transition_count: int
    update_count: int
    metrics: dict[str, float]
    target_synced: bool
    opponent_snapshot_added: bool


def run_experiment(
    results_directory: Path,
    artifacts_directory: Path,
    *,
    device: torch.device,
    config: DQNConfig | None = None,
    training_seconds: float = 120 * 60,
    total_seconds: float = 150 * 60,
    checkpoint_seconds: tuple[float, ...] = (15 * 60, 30 * 60, 60 * 60, 120 * 60),
    validation_games: int = 200,
    final_games: int = 1_000,
    ppo_results_directory: Path = Path("experiments/ppo/results/seed-0"),
) -> ExperimentArtifacts:
    """Run the bounded single-seed DQN experiment and write its evidence."""
    if config is None:
        config = DQNConfig()
    _validate_config(config)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    _seed_everything(config.seed)
    results_directory.mkdir(parents=True, exist_ok=True)
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(artifacts_directory / "tensorboard")
    online = MaskedQNetwork().to(device)
    updater = DQNUpdater(online, config, device)
    replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
    opponents = OpponentPool(
        config.opponent_pool_capacity,
        random_probability=config.random_opponent_probability,
        seed=config.seed + 1,
    )
    collector = SelfPlayCollector(online, opponents, config, device)
    started = time.monotonic()
    deadline = started + total_seconds
    training_elapsed = 0.0
    transition_count = 0
    update_count = 0
    checkpoint_index = 0
    best_checkpoint: Path | None = None
    best_validation: EvaluationResult | None = None
    stop_reason: str | None = None
    history: list[dict[str, float | int]] = []
    last_collection: Collection | None = None

    try:
        while training_elapsed < training_seconds and time.monotonic() < deadline:
            iteration_started = time.monotonic()
            epsilon = _epsilon(config, transition_count)
            collection = collector.collect(
                config.collection_size,
                epsilon=epsilon,
            )
            step = _consume_collection(
                collection,
                replay=replay,
                updater=updater,
                opponents=opponents,
                config=config,
                transition_count=transition_count,
                update_count=update_count,
            )
            iteration_seconds = time.monotonic() - iteration_started
            training_elapsed += iteration_seconds
            transition_count = step.transition_count
            update_count = step.update_count
            last_collection = collection
            point: dict[str, float | int] = {
                "iteration": len(history) + 1,
                "training_seconds": training_elapsed,
                "transitions": transition_count,
                "updates": update_count,
                "transitions_per_second": len(collection.transitions)
                / iteration_seconds,
                "epsilon": epsilon,
                "replay_size": len(replay),
                "opponent_pool_size": len(opponents),
                "terminated_episodes": sum(
                    episode.terminated for episode in collection.episodes
                ),
                "truncated_episodes": sum(
                    episode.truncated for episode in collection.episodes
                ),
                **step.metrics,
            }
            history.append(point)
            print(
                f"iteration={len(history)} transitions={transition_count} "
                f"updates={update_count} train_minutes={training_elapsed / 60:.1f} "
                f"epsilon={epsilon:.3f} "
                f"throughput={point['transitions_per_second']:.1f}/s",
                flush=True,
            )
            for name, value in step.metrics.items():
                writer.add_scalar(f"dqn/{name}", value, transition_count)
            writer.add_scalar("dqn/epsilon", epsilon, transition_count)
            writer.add_scalar("replay/size", len(replay), transition_count)
            writer.add_scalar(
                "collection/transitions_per_second",
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
                    updater=updater,
                    config=config,
                    transition_count=transition_count,
                    update_count=update_count,
                    training_seconds=training_elapsed,
                )
                validation = evaluate(
                    online,
                    games=validation_games,
                    device=device,
                    max_plies=config.max_plies,
                    seed=20_000 + checkpoint_index * validation_games,
                    deadline=deadline,
                    progress=True,
                )
                point.update(_validation_metrics(validation))
                writer.add_scalar(
                    "validation/win_rate", validation.win_rate, transition_count
                )
                writer.add_scalar(
                    "validation/truncation_rate",
                    validation.truncation_rate,
                    transition_count,
                )
                if validation.complete and _is_better(validation, best_validation):
                    best_validation = validation
                    best_checkpoint = checkpoint_path
                print(
                    f"validation games={validation.games} "
                    f"win_rate={validation.win_rate:.1%} "
                    f"truncation_rate={validation.truncation_rate:.1%}",
                    flush=True,
                )
                checkpoint_index += 1
                if validation.illegal_actions:
                    stop_reason = "illegal-action"
                    break
                if not validation.complete:
                    stop_reason = "total-deadline"
                    break

        if last_collection is None:
            raise RuntimeError("training ended before one collection completed")
        if best_checkpoint is None and time.monotonic() < deadline:
            best_checkpoint = artifacts_directory / "checkpoint-final.pt"
            _save_checkpoint(
                best_checkpoint,
                updater=updater,
                config=config,
                transition_count=transition_count,
                update_count=update_count,
                training_seconds=training_elapsed,
            )
            validation = evaluate(
                online,
                games=validation_games,
                device=device,
                max_plies=config.max_plies,
                seed=30_000,
                deadline=deadline,
                progress=True,
            )
            history[-1].update(_validation_metrics(validation))
            if validation.complete:
                best_validation = validation

        if best_checkpoint is None:
            best_checkpoint = artifacts_directory / "checkpoint-deadline.pt"
            _save_checkpoint(
                best_checkpoint,
                updater=updater,
                config=config,
                transition_count=transition_count,
                update_count=update_count,
                training_seconds=training_elapsed,
            )
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
        online.load_state_dict(checkpoint["model"])
        final_evaluation = evaluate(
            online,
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
        artifacts = _write_artifacts(
            results_directory,
            config=config,
            device=device,
            history=history,
            collection=last_collection,
            final_evaluation=final_evaluation,
            checkpoint_path=best_checkpoint,
            status=status,
            total_elapsed_seconds=time.monotonic() - started,
            smoke_checks=None,
        )
        comparison_path = _write_comparison(
            results_directory,
            ppo_results_directory=ppo_results_directory,
            dqn_metrics_path=artifacts.metrics_path,
            dqn_config_path=artifacts.config_path,
        )
        return replace(artifacts, comparison_path=comparison_path)
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
    """Exercise every DQN training subsystem in one bounded run."""
    config = DQNConfig(
        environment_count=1,
        max_plies=4,
        collection_size=4,
        replay_capacity=64,
        replay_warmup=8,
        batch_size=4,
        update_interval=1,
        target_sync_interval=4,
        opponent_snapshot_interval=8,
        opponent_pool_capacity=2,
        learning_rate=1e-2,
        epsilon_decay_transitions=16,
    )
    _seed_everything(config.seed)
    online = MaskedQNetwork().to(device)
    updater = DQNUpdater(online, config, device)
    fixed_batch_overfit = _overfit_fixed_batch(updater, device)
    replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
    opponents = OpponentPool(
        config.opponent_pool_capacity,
        random_probability=config.random_opponent_probability,
        seed=config.seed + 1,
    )
    collector = SelfPlayCollector(online, opponents, config, device)
    started = time.monotonic()
    transition_count = 0
    update_count = 0
    target_synced = False
    snapshot_added = False
    collections: list[Collection] = []
    metric_totals: dict[str, float] = {}
    metric_points = 0
    while transition_count < 16:
        collection = collector.collect(
            config.collection_size,
            epsilon=_epsilon(config, transition_count),
        )
        collections.append(collection)
        step = _consume_collection(
            collection,
            replay=replay,
            updater=updater,
            opponents=opponents,
            config=config,
            transition_count=transition_count,
            update_count=update_count,
        )
        transition_count = step.transition_count
        update_count = step.update_count
        target_synced = target_synced or step.target_synced
        snapshot_added = snapshot_added or step.opponent_snapshot_added
        if step.metrics:
            metric_points += 1
            for name, value in step.metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "smoke.pt"
    _save_checkpoint(
        checkpoint_path,
        updater=updater,
        config=config,
        transition_count=transition_count,
        update_count=update_count,
        training_seconds=time.monotonic() - started,
    )
    restored = MaskedQNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    checkpoint_restored = all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in online.state_dict().items()
    )
    evaluation = evaluate(
        restored,
        games=2,
        device=device,
        max_plies=4,
        seed=10_000,
    )
    merged = Collection(
        transitions=[
            transition
            for collection in collections
            for transition in collection.transitions
        ],
        episodes=[
            episode for collection in collections for episode in collection.episodes
        ],
    )
    history = [
        {
            "training_seconds": time.monotonic() - started,
            "transitions": transition_count,
            "updates": update_count,
            "epsilon": _epsilon(config, transition_count),
            "replay_size": len(replay),
            "opponent_pool_size": len(opponents),
            **{name: total / metric_points for name, total in metric_totals.items()},
            **_validation_metrics(evaluation),
        }
    ]
    smoke_checks = {
        "checkpoint_restored": checkpoint_restored,
        "fixed_batch_overfit": fixed_batch_overfit,
        "opponent_snapshot_added": snapshot_added,
        "target_synced": target_synced,
    }
    if not all(smoke_checks.values()) or evaluation.illegal_actions:
        raise RuntimeError(f"DQN smoke checks failed: {smoke_checks}")
    return _write_artifacts(
        output_directory,
        config=config,
        device=device,
        history=history,
        collection=merged,
        final_evaluation=evaluation,
        checkpoint_path=checkpoint_path,
        status="smoke-passed",
        total_elapsed_seconds=time.monotonic() - started,
        smoke_checks=smoke_checks,
    )


def recover_user_stopped_experiment(
    results_directory: Path,
    artifacts_directory: Path,
    *,
    device: torch.device,
    ppo_results_directory: Path = Path("experiments/ppo/results/seed-0"),
) -> ExperimentArtifacts:
    """Recover completed checkpoint evidence after an operator interruption."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    checkpoints = sorted(artifacts_directory.glob("checkpoint-???m.pt"))
    event_files = sorted((artifacts_directory / "tensorboard").glob("events.*"))
    if not checkpoints or not event_files:
        raise FileNotFoundError("stopped run has no checkpoints or TensorBoard events")
    checkpoint_data = {
        path: torch.load(path, map_location="cpu", weights_only=True)
        for path in checkpoints
    }
    config = DQNConfig(**checkpoint_data[checkpoints[-1]]["config"])
    events = EventAccumulator(str(event_files[-1]))
    events.Reload()
    win_events = events.Scalars("validation/win_rate")
    truncation_events = events.Scalars("validation/truncation_rate")
    epsilon_events = events.Scalars("dqn/epsilon")
    if len(win_events) != len(truncation_events):
        raise RuntimeError("validation TensorBoard series have different lengths")
    checkpoints_by_transition = {
        int(data["transitions"]): (path, data) for path, data in checkpoint_data.items()
    }
    history: list[dict[str, float | int]] = []
    best_checkpoint: Path | None = None
    best_score = (-1.0, -1.0)
    for wins, unresolved in zip(win_events, truncation_events, strict=True):
        checkpoint, data = checkpoints_by_transition[wins.step]
        win_rate = float(wins.value)
        truncation_rate = float(unresolved.value)
        score = (win_rate, -truncation_rate)
        if score > best_score:
            best_score = score
            best_checkpoint = checkpoint
        history.append(
            {
                "training_seconds": float(data["training_seconds"]),
                "transitions": int(data["transitions"]),
                "updates": int(data["updates"]),
                "validation_games": 200,
                "validation_win_rate": win_rate,
                "validation_truncation_rate": truncation_rate,
                "validation_illegal_actions": 0,
            }
        )
    assert best_checkpoint is not None
    last_checkpoint_data = checkpoint_data[checkpoints[-1]]
    last_event = epsilon_events[-1]
    checkpoint_event = next(
        event
        for event in epsilon_events
        if event.step == int(last_checkpoint_data["transitions"])
    )
    recovered_training_seconds = float(last_checkpoint_data["training_seconds"]) + (
        last_event.wall_time - checkpoint_event.wall_time
    )
    first_event = epsilon_events[0]
    recovered_total_seconds = (
        last_event.wall_time
        - first_event.wall_time
        + float(history[0]["training_seconds"])
        * first_event.step
        / int(history[0]["transitions"])
    )
    last_updates = (
        last_event.step // config.update_interval
        - (config.replay_warmup + config.update_interval - 1) // config.update_interval
        + 1
    )
    history.append(
        {
            "training_seconds": recovered_training_seconds,
            "transitions": last_event.step,
            "updates": last_updates,
            "epsilon": float(last_event.value),
            "replay_size": min(last_event.step, config.replay_capacity),
            "opponent_pool_size": min(
                last_event.step // config.opponent_snapshot_interval,
                config.opponent_pool_capacity,
            ),
        }
    )
    results_directory.mkdir(parents=True, exist_ok=True)
    config_path = results_directory / "config.json"
    metrics_path = results_directory / "metrics.json"
    summary_path = results_directory / "summary.md"
    curve_path = results_directory / "learning-curve.png"
    hardware = _hardware(device)
    config_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "status": "user-stopped",
        "stop_reason": "operator stopped after the 60-minute validation",
        "recovered_from_tensorboard": True,
        "total_elapsed_seconds": recovered_total_seconds,
        "hardware": hardware,
        "history": history,
        "selected_checkpoint": best_checkpoint.name,
        "final_evaluation": None,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best_point = max(
        history[:-1],
        key=lambda point: (
            float(point["validation_win_rate"]),
            -float(point["validation_truncation_rate"]),
        ),
    )
    summary_path.write_text(
        "\n".join(
            (
                "# Masked Double DQN single-seed validation",
                "",
                "- Status: `user-stopped`",
                "- Stop reason: operator stopped after the 60-minute validation",
                f"- Device: `{hardware['device']}`",
                f"- Total elapsed: {recovered_total_seconds / 60:.1f} minutes",
                f"- Training elapsed: {recovered_training_seconds / 60:.1f} minutes",
                f"- Transitions: {last_event.step}",
                f"- Updates: {last_updates}",
                f"- Selected completed checkpoint: `{best_checkpoint.name}`",
                (
                    "- Best checkpoint validation: "
                    f"{float(best_point['validation_win_rate']):.1%} wins, "
                    f"{float(best_point['validation_truncation_rate']):.1%} unresolved"
                ),
                "- Final 1,000-game evaluation: not run",
                "- Exploratory target met: no",
                "",
            )
        ),
        encoding="utf-8",
    )
    _plot_history(history, curve_path)
    artifacts = ExperimentArtifacts(
        metrics_path=metrics_path,
        summary_path=summary_path,
        curve_path=curve_path,
        checkpoint_path=best_checkpoint,
        config_path=config_path,
    )
    comparison_path = _write_comparison(
        results_directory,
        ppo_results_directory=ppo_results_directory,
        dqn_metrics_path=metrics_path,
        dqn_config_path=config_path,
    )
    return replace(artifacts, comparison_path=comparison_path)


def _consume_collection(
    collection: Collection,
    *,
    replay: ReplayBuffer,
    updater: DQNUpdater,
    opponents: OpponentPool,
    config: DQNConfig,
    transition_count: int,
    update_count: int,
) -> _TrainingStep:
    metric_totals: dict[str, float] = {}
    metric_count = 0
    target_synced = False
    snapshot_added = False
    for transition in collection.transitions:
        replay.add(transition)
        transition_count += 1
        if (
            transition_count >= config.replay_warmup
            and transition_count % config.update_interval == 0
        ):
            metrics = updater.update(replay.sample(config.batch_size))
            metric_count += 1
            update_count += 1
            for name, value in metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
        if transition_count % config.target_sync_interval == 0:
            updater.sync_target()
            target_synced = True
        if transition_count % config.opponent_snapshot_interval == 0:
            opponents.add(updater.online)
            snapshot_added = True
    averaged = {name: total / metric_count for name, total in metric_totals.items()}
    return _TrainingStep(
        transition_count=transition_count,
        update_count=update_count,
        metrics=averaged,
        target_synced=target_synced,
        opponent_snapshot_added=snapshot_added,
    )


def _overfit_fixed_batch(updater: DQNUpdater, device: torch.device) -> bool:
    for parameter in updater.online.parameters():
        parameter.data.zero_()
    updater.sync_target()
    size = 16
    action_masks = torch.zeros((size, 209), dtype=torch.bool)
    action_masks[:, :3] = True
    batch = TransitionBatch(
        observations=torch.zeros((size, 6, 9, 9)),
        action_masks=action_masks,
        actions=torch.ones(size, dtype=torch.int64),
        rewards=torch.ones(size),
        next_observations=torch.zeros((size, 6, 9, 9)),
        next_action_masks=torch.zeros((size, 209), dtype=torch.bool),
        done=torch.ones(size, dtype=torch.bool),
    )
    for _ in range(120):
        updater.update(batch)
    values = updater.online(torch.zeros((1, 6, 9, 9), device=device))[0, :3]
    return bool(values.argmax().item() == 1 and values[1].item() > 0.5)


def _epsilon(config: DQNConfig, transitions: int) -> float:
    fraction = min(1.0, transitions / config.epsilon_decay_transitions)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


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


def _save_checkpoint(
    path: Path,
    *,
    updater: DQNUpdater,
    config: DQNConfig,
    transition_count: int,
    update_count: int,
    training_seconds: float,
) -> None:
    torch.save(
        {
            "seed": config.seed,
            "model": updater.online.state_dict(),
            "target": updater.target.state_dict(),
            "optimizer": updater.optimizer.state_dict(),
            "config": asdict(config),
            "transitions": transition_count,
            "updates": update_count,
            "training_seconds": training_seconds,
        },
        path,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model"].keys() != updater.online.state_dict().keys():
        raise RuntimeError(f"checkpoint could not be restored: {path}")


def _write_artifacts(
    output_directory: Path,
    *,
    config: DQNConfig,
    device: torch.device,
    history: list[dict[str, float | int]],
    collection: Collection,
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
        "hardware": hardware,
        "history": history,
        "last_collection": {
            "transitions": len(collection.transitions),
            "episodes": len(collection.episodes),
            "terminations": sum(episode.terminated for episode in collection.episodes),
            "truncations": sum(episode.truncated for episode in collection.episodes),
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
        key=lambda point: (
            float(point["validation_win_rate"]),
            -float(point["validation_truncation_rate"]),
        ),
        default=None,
    )
    best_line = (
        "- Best checkpoint validation: unavailable"
        if best_validation is None
        else (
            "- Best checkpoint validation: "
            f"{float(best_validation['validation_win_rate']):.1%} wins, "
            f"{float(best_validation['validation_truncation_rate']):.1%} unresolved "
            f"at {float(best_validation['training_seconds']) / 60:.1f} minutes"
        )
    )
    return "\n".join(
        (
            "# Masked Double DQN single-seed validation",
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
    truncation_rates = [
        float(point["validation_truncation_rate"]) for point in validation
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x, win_rates, marker="o", label="win rate vs random")
    axis.plot(x, truncation_rates, marker="o", label="unresolved rate")
    axis.axhline(0.70, color="tab:green", linestyle="--", label="win target")
    axis.axhline(0.05, color="tab:red", linestyle="--", label="unresolved target")
    axis.set(xlabel="training minutes", ylabel="rate", ylim=(0, 1))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_comparison(
    results_directory: Path,
    *,
    ppo_results_directory: Path,
    dqn_metrics_path: Path,
    dqn_config_path: Path,
) -> Path:
    ppo_metrics = json.loads(
        (ppo_results_directory / "metrics.json").read_text(encoding="utf-8")
    )
    ppo_config = json.loads(
        (ppo_results_directory / "config.json").read_text(encoding="utf-8")
    )
    dqn_metrics = json.loads(dqn_metrics_path.read_text(encoding="utf-8"))
    dqn_config = json.loads(dqn_config_path.read_text(encoding="utf-8"))
    path = results_directory / "comparison.md"
    lines = [
        "# PPO and Masked Double DQN seed-0 observations",
        "",
        (
            "This is a simple side-by-side record of two independent seed-0 "
            "experiments, not an algorithm-level or statistically significant conclusion."
        ),
        "",
        "| Metric | PPO | Masked Double DQN |",
        "| --- | ---: | ---: |",
        f"| Learning rate | {ppo_config['learning_rate']} | {dqn_config['learning_rate']} |",
        f"| Gamma | {ppo_config['gamma']} | {dqn_config['gamma']} |",
        (
            f"| Total elapsed (minutes) | "
            f"{float(ppo_metrics['total_elapsed_seconds']) / 60:.1f} | "
            f"{float(dqn_metrics['total_elapsed_seconds']) / 60:.1f} |"
        ),
        (
            f"| Training transitions | {_last_transitions(ppo_metrics)} | "
            f"{_last_transitions(dqn_metrics)} |"
        ),
        (
            f"| Final wins / losses / unresolved | "
            f"{_outcome(ppo_metrics)} | {_outcome(dqn_metrics)} |"
        ),
        (
            f"| Final win rate | {_rate(ppo_metrics, 'win_rate')} | "
            f"{_rate(dqn_metrics, 'win_rate')} |"
        ),
        (
            f"| Final unresolved rate | {_rate(ppo_metrics, 'truncation_rate')} | "
            f"{_rate(dqn_metrics, 'truncation_rate')} |"
        ),
        (
            f"| Illegal actions | {_illegal_actions(ppo_metrics)} | "
            f"{_illegal_actions(dqn_metrics)} |"
        ),
        "",
        "## Checkpoint validation",
        "",
        "| Training minute | PPO win / unresolved | DQN win / unresolved |",
        "| ---: | ---: | ---: |",
    ]
    ppo_validations = _validations(ppo_metrics)
    dqn_validations = _validations(dqn_metrics)
    for minute in (15, 30, 60, 120):
        lines.append(
            f"| {minute} | {_format_validation(ppo_validations.get(minute))} | "
            f"{_format_validation(dqn_validations.get(minute))} |"
        )
    lines.extend(
        (
            "",
            "## Provenance",
            "",
            (
                f"- PPO: commit `{ppo_metrics['hardware']['git_commit']}`, "
                f"dirty `{str(ppo_metrics['hardware']['git_dirty']).lower()}`."
            ),
            (
                f"- DQN: commit `{dqn_metrics['hardware']['git_commit']}`, "
                f"dirty `{str(dqn_metrics['hardware']['git_dirty']).lower()}`."
            ),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validations(metrics: dict[str, object]) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    history = cast(list[dict[str, object]], metrics["history"])
    for point in history:
        if "validation_win_rate" not in point:
            continue
        minute = min(
            (15, 30, 60, 120),
            key=lambda candidate: abs(
                float(cast(int | float, point["training_seconds"])) / 60 - candidate
            ),
        )
        result[minute] = point
    return result


def _format_validation(point: dict[str, object] | None) -> str:
    if point is None:
        return "unavailable"
    return (
        f"{float(cast(int | float, point['validation_win_rate'])):.1%} / "
        f"{float(cast(int | float, point['validation_truncation_rate'])):.1%}"
    )


def _last_transitions(metrics: dict[str, object]) -> int:
    history = cast(list[dict[str, object]], metrics["history"])
    return int(cast(int | float, history[-1]["transitions"]))


def _final(metrics: dict[str, object]) -> dict[str, object] | None:
    final = metrics["final_evaluation"]
    return None if final is None else cast(dict[str, object], final)


def _outcome(metrics: dict[str, object]) -> str:
    final = _final(metrics)
    if final is None:
        return "not run"
    return f"{final['wins']} / {final['losses']} / {final['truncated']}"


def _rate(metrics: dict[str, object], name: str) -> str:
    final = _final(metrics)
    if final is None:
        return "not run"
    return f"{float(cast(int | float, final[name])):.1%}"


def _illegal_actions(metrics: dict[str, object]) -> object:
    final = _final(metrics)
    return "not run" if final is None else final["illegal_actions"]


def _write_failure(
    results_directory: Path,
    config: DQNConfig,
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


def _validate_config(config: DQNConfig) -> None:
    if config.replay_warmup < config.batch_size:
        raise ValueError("replay warm-up must be at least the batch size")
    for value in (
        config.collection_size,
        config.update_interval,
        config.target_sync_interval,
        config.opponent_snapshot_interval,
        config.epsilon_decay_transitions,
    ):
        if value <= 0:
            raise ValueError("DQN interval values must be positive")

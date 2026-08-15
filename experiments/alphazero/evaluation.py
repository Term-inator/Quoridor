"""策略—价值 MCTS 对战随机参与者的平衡角色评估。"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import TypeAlias

import numpy as np
import torch

from experiments.alphazero.mcts import PUCTSearch
from experiments.alphazero.model import PolicyValueNetwork
from experiments.alphazero.training import AlphaZeroConfig
from quoridor_rl.codec import ActionCodec
from quoridor_rl.game import IllegalActionError, Player, Position

_WorkerResult: TypeAlias = tuple[int, int | None, bool, int] | None
_worker_model: PolicyValueNetwork | None = None
_worker_config: AlphaZeroConfig | None = None


@dataclass(frozen=True, slots=True)
class RoleResult:
    """搜索策略担任某个固定角色时的结果计数。"""

    games: int
    wins: int
    losses: int
    truncated: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """汇总先后手、非法动作及计划局数完成情况。"""

    as_player_0: RoleResult
    as_player_1: RoleResult
    illegal_actions: int
    requested_games: int

    @property
    def games(self) -> int:
        return self.as_player_0.games + self.as_player_1.games

    @property
    def wins(self) -> int:
        return self.as_player_0.wins + self.as_player_1.wins

    @property
    def losses(self) -> int:
        return self.as_player_0.losses + self.as_player_1.losses

    @property
    def truncated(self) -> int:
        return self.as_player_0.truncated + self.as_player_1.truncated

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def truncation_rate(self) -> float:
        return self.truncated / self.games if self.games else 0.0

    @property
    def complete(self) -> bool:
        return self.games == self.requested_games


def evaluate(
    model: PolicyValueNetwork,
    *,
    games: int,
    device: torch.device,
    config: AlphaZeroConfig,
    seed: int = 10_000,
    deadline: float | None = None,
    progress: bool = False,
) -> EvaluationResult:
    """以固定模拟次数、零温度 MCTS 平衡担任先后手进行评估。

    CPU 且任务足够多时使用 ``spawn`` 多进程并行；其他设备走当前进程，避免跨进程
    复制加速器状态。
    """
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even integer")
    counts = {
        Player.PLAYER_0: {"games": 0, "wins": 0, "losses": 0, "truncated": 0},
        Player.PLAYER_1: {"games": 0, "wins": 0, "losses": 0, "truncated": 0},
    }
    if device.type == "cpu" and config.evaluation_workers > 1 and games > 2:
        state = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
        with ProcessPoolExecutor(
            max_workers=config.evaluation_workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(state, config),
        ) as executor:
            results = executor.map(
                _run_worker_game,
                ((game_index, seed, deadline) for game_index in range(games)),
                chunksize=1,
            )
            worker_results = results
            for completed_index, result in enumerate(worker_results, start=1):
                if result is None:
                    continue
                _record_result(counts, result)
                if progress and completed_index % 20 == 0:
                    _print_progress(counts, games)
    else:
        model.eval()
        for game_index in range(games):
            if deadline is not None and time.monotonic() >= deadline:
                break
            result = _play_game(
                model,
                config,
                device,
                game_index=game_index,
                seed=seed,
            )
            _record_result(counts, result)
            if progress and (game_index + 1) % 20 == 0:
                _print_progress(counts, games)

    illegal_actions = sum(counts[role].pop("illegal_actions", 0) for role in Player)
    return EvaluationResult(
        as_player_0=RoleResult(**counts[Player.PLAYER_0]),
        as_player_1=RoleResult(**counts[Player.PLAYER_1]),
        illegal_actions=illegal_actions,
        requested_games=games,
    )


def _initialize_worker(
    state: dict[str, torch.Tensor],
    config: AlphaZeroConfig,
) -> None:
    global _worker_model, _worker_config
    torch.set_num_threads(1)
    _worker_model = PolicyValueNetwork()
    _worker_model.load_state_dict(state)
    _worker_model.eval()
    _worker_config = config


def _run_worker_game(arguments: tuple[int, int, float | None]) -> _WorkerResult:
    game_index, seed, deadline = arguments
    if deadline is not None and time.monotonic() >= deadline:
        return None
    if _worker_model is None or _worker_config is None:
        raise RuntimeError("evaluation worker was not initialized")
    return _play_game(
        _worker_model,
        _worker_config,
        torch.device("cpu"),
        game_index=game_index,
        seed=seed,
    )


def _play_game(
    model: PolicyValueNetwork,
    config: AlphaZeroConfig,
    device: torch.device,
    *,
    game_index: int,
    seed: int,
) -> _WorkerResult:
    trained_player = Player(game_index % 2)
    random = np.random.default_rng(seed + game_index)
    position = Position.initial()
    codec = ActionCodec()
    illegal_actions = 0
    search = PUCTSearch(
        model,
        device=device,
        simulations=config.evaluation_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        root_noise_fraction=config.root_noise_fraction,
        maximum_actions=config.maximum_search_actions,
        allow_walls=True,
        progress_prior_fraction=0.0,
        seed=seed + game_index,
    )
    for ply in range(config.max_plies):
        player = position.to_move
        if player is None:
            break
        if player is trained_player:
            result = search.run(
                position,
                remaining_plies=config.max_plies - ply,
                add_root_noise=False,
                temperature=0.0,
            )
            action_id = result.action_id
        else:
            legal_actions = position.legal_actions()
            action = legal_actions[int(random.integers(len(legal_actions)))]
            action_id = codec.encode(action, player)
        try:
            position = position.play(codec.decode(action_id, player))
        except IllegalActionError:
            illegal_actions += 1
            break
        search.advance(action_id, position)
        if position.winner is not None:
            break
    winner = None if position.winner is None else int(position.winner)
    return int(trained_player), winner, position.winner is None, illegal_actions


def _record_result(
    counts: dict[Player, dict[str, int]],
    result: _WorkerResult,
) -> None:
    if result is None:
        return
    trained_id, winner_id, truncated, illegal_actions = result
    trained_player = Player(trained_id)
    counts[trained_player]["games"] += 1
    counts[trained_player]["illegal_actions"] = (
        counts[trained_player].get("illegal_actions", 0) + illegal_actions
    )
    if truncated:
        counts[trained_player]["truncated"] += 1
    elif winner_id == trained_id:
        counts[trained_player]["wins"] += 1
    else:
        counts[trained_player]["losses"] += 1


def _print_progress(counts: dict[Player, dict[str, int]], games: int) -> None:
    completed = sum(counts[role]["games"] for role in Player)
    wins = sum(counts[role]["wins"] for role in Player)
    unresolved = sum(counts[role]["truncated"] for role in Player)
    print(
        f"evaluation {completed}/{games} wins={wins} unresolved={unresolved}",
        flush=True,
    )

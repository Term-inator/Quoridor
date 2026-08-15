"""AlphaZero 自我对弈训练数据生成。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from experiments.alphazero.mcts import PUCTSearch
from experiments.alphazero.model import PolicyValueNetwork
from experiments.alphazero.training import AlphaZeroConfig, TrainingExample
from quoridor_rl.codec import ActionCodec, ObservationCodec
from quoridor_rl.game import Player, Position


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    """一局搜索自对弈产生的样本、结果与搜索性能统计。"""

    examples: list[TrainingExample]
    plies: int
    winner: Player | None
    truncated: bool
    expanded_nodes: int
    maximum_search_depth: int
    elapsed_seconds: float


def play_self_game(
    model: PolicyValueNetwork,
    config: AlphaZeroConfig,
    device: torch.device,
    *,
    game_index: int,
) -> SelfPlayGame:
    """完成一局带根噪声的搜索自对弈，并用最终胜负标注所有历史视角。

    课程阶段只允许移动棋子，并混入偏向缩短路径的先验；正式阶段恢复放墙。开局使用
    温度采样增加多样性，之后转为确定性选择。
    """
    started = time.monotonic()
    position = Position.initial()
    action_codec = ActionCodec()
    observation_codec = ObservationCodec()
    in_curriculum = game_index < config.pawn_only_curriculum_games
    search = PUCTSearch(
        model,
        device=device,
        simulations=config.simulations_per_move,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        root_noise_fraction=config.root_noise_fraction,
        maximum_actions=config.maximum_search_actions,
        allow_walls=not in_curriculum,
        progress_prior_fraction=(
            config.curriculum_progress_prior if in_curriculum else 0.0
        ),
        seed=config.seed + game_index * 10_003,
    )
    records: list[tuple[np.ndarray, np.ndarray, Player]] = []
    expanded_nodes = 0
    maximum_search_depth = 0

    for ply in range(config.max_plies):
        player = position.to_move
        if player is None:
            break
        result = search.run(
            position,
            remaining_plies=config.max_plies - ply,
            add_root_noise=True,
            temperature=1.0 if ply < config.temperature_plies else 0.0,
        )
        records.append(
            (
                observation_codec.encode(position, player),
                result.policy,
                player,
            )
        )
        expanded_nodes += result.expanded_nodes
        maximum_search_depth = max(maximum_search_depth, result.maximum_depth)
        action = action_codec.decode(result.action_id, player)
        position = position.play(action)
        search.advance(result.action_id, position)
        if position.winner is not None:
            break

    winner = position.winner
    truncated = winner is None
    examples = [
        TrainingExample(
            observation=observation,
            policy=policy,
            value=(0.0 if winner is None else 1.0 if winner is player else -1.0),
        )
        for observation, policy, player in records
    ]
    return SelfPlayGame(
        examples=examples,
        plies=len(records),
        winner=winner,
        truncated=truncated,
        expanded_nodes=expanded_nodes,
        maximum_search_depth=maximum_search_depth,
        elapsed_seconds=time.monotonic() - started,
    )

import json

import numpy as np
import torch

from experiments.alphazero.evaluation import EvaluationResult, RoleResult, evaluate
from experiments.alphazero.experiment import _write_comparison, run_smoke
from experiments.alphazero.mcts import PUCTSearch
from experiments.alphazero.model import PolicyValueNetwork
from experiments.alphazero.self_play import play_self_game
from experiments.alphazero.training import (
    AlphaZeroConfig,
    PolicyValueUpdater,
    ReplayBuffer,
    TrainingExample,
)
from quoridor_rl.codec import ActionCodec, ObservationCodec
from quoridor_rl.env import env
from quoridor_rl.game import Player, Position


def test_observation_codec_matches_the_pettingzoo_adapter() -> None:
    environment = env()
    environment.reset()
    encoded = ObservationCodec().encode(Position.initial(), Player.PLAYER_0)
    assert np.array_equal(encoded, environment.observe("player_0")["observation"])


def test_policy_value_network_has_bounded_value_and_fixed_policy_shape() -> None:
    logits, values = PolicyValueNetwork()(torch.zeros((3, 6, 9, 9)))
    assert logits.shape == (3, 209)
    assert values.shape == (3,)
    assert (values >= -1).all() and (values <= 1).all()


def test_puct_search_visits_only_legal_actions() -> None:
    position = Position.initial()
    search = PUCTSearch(
        PolicyValueNetwork(),
        device=torch.device("cpu"),
        simulations=4,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        root_noise_fraction=0.25,
        maximum_actions=16,
        allow_walls=True,
        progress_prior_fraction=0.0,
        seed=0,
    )
    result = search.run(
        position,
        remaining_plies=8,
        add_root_noise=True,
        temperature=1.0,
    )
    legal_ids = {
        ActionCodec().encode(action, Player.PLAYER_0)
        for action in position.legal_actions()
    }
    visited_ids = set(np.flatnonzero(result.policy))
    assert result.action_id in legal_ids
    assert visited_ids <= legal_ids
    assert np.isclose(result.policy.sum(), 1.0)
    assert len(visited_ids) <= 4


def test_candidate_limit_lets_a_small_search_revisit_deeper_branches() -> None:
    search = PUCTSearch(
        PolicyValueNetwork(),
        device=torch.device("cpu"),
        simulations=16,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        root_noise_fraction=0.25,
        maximum_actions=4,
        allow_walls=True,
        progress_prior_fraction=0.0,
        seed=1,
    )
    result = search.run(
        Position.initial(),
        remaining_plies=16,
        add_root_noise=False,
        temperature=0.0,
    )
    assert np.count_nonzero(result.policy) <= 4
    assert result.maximum_depth >= 2


def test_truncated_self_play_reports_zero_targets_for_exclusion_from_replay() -> None:
    config = AlphaZeroConfig(
        max_plies=2,
        simulations_per_move=1,
        evaluation_simulations=1,
        evaluation_workers=1,
        temperature_plies=1,
        torch_threads=1,
    )
    game = play_self_game(
        PolicyValueNetwork(),
        config,
        torch.device("cpu"),
        game_index=0,
    )
    assert game.truncated
    assert game.plies == 2
    assert all(example.value == 0 for example in game.examples)
    assert all(np.isclose(example.policy.sum(), 1.0) for example in game.examples)


def test_policy_value_update_changes_parameters_with_finite_metrics() -> None:
    config = AlphaZeroConfig(batch_size=4, replay_capacity=8, torch_threads=1)
    replay = ReplayBuffer(config.replay_capacity, seed=0)
    for action in range(4):
        policy = np.zeros(209, dtype=np.float32)
        policy[action] = 1
        replay.add(
            TrainingExample(
                observation=np.zeros((6, 9, 9), dtype=np.float32),
                policy=policy,
                value=1.0,
            )
        )
    model = PolicyValueNetwork()
    updater = PolicyValueUpdater(model, config, torch.device("cpu"))
    before = model.policy_head.weight.detach().clone()
    metrics = updater.update(replay.sample(4))
    assert not torch.equal(before, model.policy_head.weight)
    assert all(np.isfinite(value) for value in metrics.values())


def test_evaluation_balances_player_identities_and_never_plays_illegally() -> None:
    config = AlphaZeroConfig(
        max_plies=2,
        simulations_per_move=1,
        evaluation_simulations=1,
        evaluation_workers=1,
        torch_threads=1,
    )
    result = evaluate(
        PolicyValueNetwork(),
        games=2,
        device=torch.device("cpu"),
        config=config,
    )
    assert result.games == 2
    assert result.as_player_0.games == 1
    assert result.as_player_1.games == 1
    assert result.illegal_actions == 0


def test_cpu_evaluation_can_run_games_in_parallel() -> None:
    config = AlphaZeroConfig(
        max_plies=2,
        simulations_per_move=1,
        evaluation_simulations=1,
        evaluation_workers=2,
        torch_threads=1,
    )
    result = evaluate(
        PolicyValueNetwork(),
        games=4,
        device=torch.device("cpu"),
        config=config,
    )
    assert result.games == 4
    assert result.as_player_0.games == 2
    assert result.as_player_1.games == 2
    assert result.illegal_actions == 0


def test_alphazero_smoke_writes_reloadable_evidence(tmp_path) -> None:
    result = run_smoke(tmp_path, device=torch.device("cpu"))
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert metrics["status"] == "smoke-passed"
    assert all(metrics["smoke_checks"].values())
    assert metrics["final_evaluation"]["illegal_actions"] == 0
    assert checkpoint["model"]
    assert checkpoint["optimizer"]
    assert result.summary_path.is_file()
    assert result.curve_path.is_file()


def test_comparison_skips_an_experiment_without_a_final_evaluation(
    tmp_path, monkeypatch
) -> None:
    dqn_directory = tmp_path / "experiments/dqn/results/seed-0"
    dqn_directory.mkdir(parents=True)
    (dqn_directory / "metrics.json").write_text(
        json.dumps({"final_evaluation": None}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = EvaluationResult(
        as_player_0=RoleResult(games=1, wins=1, losses=0, truncated=0),
        as_player_1=RoleResult(games=1, wins=0, losses=1, truncated=0),
        illegal_actions=0,
        requested_games=2,
    )
    output = tmp_path / "results"
    output.mkdir()
    path = _write_comparison(output, alpha_evaluation=result)
    assert "AlphaZero" in path.read_text(encoding="utf-8")
    assert "DQN" not in path.read_text(encoding="utf-8")

import json

import torch

from experiments.dqn.evaluation import evaluate
from experiments.dqn.experiment import run_smoke
from experiments.dqn.model import MaskedQNetwork
from experiments.dqn.opponents import OpponentPool
from experiments.dqn.self_play import SelfPlayCollector
from experiments.dqn.training import (
    DQNConfig,
    DQNUpdater,
    ReplayBuffer,
    Transition,
    double_dqn_targets,
)


def test_masked_q_network_only_selects_legal_actions() -> None:
    model = MaskedQNetwork()
    observation = torch.zeros((16, 6, 9, 9))
    action_mask = torch.zeros((16, 209), dtype=torch.bool)
    action_mask[:, 17] = True

    q_values = model(observation)
    greedy = model.select_actions(observation, action_mask, epsilon=0.0)
    exploratory = model.select_actions(observation, action_mask, epsilon=1.0)

    assert q_values.shape == (16, 209)
    assert greedy.tolist() == [17] * 16
    assert exploratory.tolist() == [17] * 16


def test_double_dqn_targets_use_online_legal_choice_and_target_value() -> None:
    online = MaskedQNetwork(action_count=3)
    target = MaskedQNetwork(action_count=3)
    for parameter in online.parameters():
        parameter.data.zero_()
    for parameter in target.parameters():
        parameter.data.zero_()
    online.q_head.bias.data.copy_(torch.tensor([1.0, 5.0, 100.0]))
    target.q_head.bias.data.copy_(torch.tensor([3.0, 7.0, 200.0]))
    next_observation = torch.zeros((2, 6, 9, 9))
    next_action_mask = torch.tensor(
        [[True, True, False], [False, False, False]],
    )

    targets = double_dqn_targets(
        online,
        target,
        next_observation,
        next_action_mask,
        rewards=torch.tensor([2.0, -1.0]),
        done=torch.tensor([False, True]),
        gamma=0.5,
    )

    assert torch.equal(targets, torch.tensor([5.5, -1.0]))


def test_replay_buffer_keeps_latest_transitions_and_samples_real_batches() -> None:
    replay = ReplayBuffer(capacity=2, seed=0)
    for action in (3, 7, 11):
        replay.add(
            Transition(
                observation=torch.full((6, 9, 9), float(action)),
                action_mask=torch.ones(209, dtype=torch.bool),
                action=action,
                reward=float(action),
                next_observation=torch.zeros((6, 9, 9)),
                next_action_mask=torch.ones(209, dtype=torch.bool),
                done=False,
            )
        )

    batch = replay.sample(2)

    assert len(replay) == 2
    assert set(batch.actions.tolist()) == {7, 11}
    assert set(batch.rewards.tolist()) == {7.0, 11.0}


def test_opponent_pool_is_fifo_and_keeps_a_random_opponent_anchor() -> None:
    pool = OpponentPool(capacity=2, random_probability=0.2, seed=0)
    for marker in (1.0, 2.0, 3.0):
        model = MaskedQNetwork()
        model.q_head.bias.data.fill_(marker)
        pool.add(model)

    sampled = [pool.sample() for _ in range(2_000)]
    snapshots = [model for model in sampled if model is not None]
    random_count = sum(model is None for model in sampled)

    assert len(pool) == 2
    assert 300 <= random_count <= 500
    assert snapshots
    assert {float(model.q_head.bias[0].item()) for model in snapshots} == {2.0, 3.0}


def test_dqn_update_changes_online_values_and_target_sync_is_exact() -> None:
    replay = ReplayBuffer(capacity=4, seed=0)
    for action in range(4):
        mask = torch.zeros(209, dtype=torch.bool)
        mask[action] = True
        replay.add(
            Transition(
                observation=torch.zeros((6, 9, 9)),
                action_mask=mask,
                action=action,
                reward=1.0,
                next_observation=torch.zeros((6, 9, 9)),
                next_action_mask=torch.zeros(209, dtype=torch.bool),
                done=True,
            )
        )
    model = MaskedQNetwork()
    updater = DQNUpdater(model, DQNConfig(batch_size=4), torch.device("cpu"))
    before = model.q_head.weight.detach().clone()

    metrics = updater.update(replay.sample(4))

    assert not torch.equal(before, model.q_head.weight)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(model.q_head.weight, updater.target.q_head.weight)
    updater.sync_target()
    assert torch.equal(model.q_head.weight, updater.target.q_head.weight)


def test_self_play_collects_only_learner_decisions_across_opponent_turns() -> None:
    config = DQNConfig(environment_count=1, max_plies=4)
    model = MaskedQNetwork()
    opponents = OpponentPool(capacity=2, random_probability=0.2, seed=1)
    collector = SelfPlayCollector(
        model,
        opponents,
        config,
        torch.device("cpu"),
    )

    collection = collector.collect(20, epsilon=1.0)

    learners = {
        episode.episode: episode.learner_agent for episode in collection.episodes
    }
    transition_counts: dict[int, int] = {}
    for transition in collection.transitions:
        transition_counts[transition.episode] = (
            transition_counts.get(transition.episode, 0) + 1
        )
        assert transition.agent == learners[transition.episode]
        assert transition.action_mask[transition.action]
    assert set(learners.values()) == {"player_0", "player_1"}
    assert set(transition_counts.values()) == {2}
    assert all(episode.truncated for episode in collection.episodes)


def test_dqn_evaluation_swaps_roles_and_reports_game_outcomes() -> None:
    result = evaluate(
        MaskedQNetwork(),
        games=2,
        device=torch.device("cpu"),
        max_plies=2,
        seed=10_000,
    )

    assert result.games == 2
    assert result.as_player_0.games == 1
    assert result.as_player_1.games == 1
    assert result.wins + result.losses + result.truncated == 2
    assert result.illegal_actions == 0


def test_dqn_smoke_exercises_training_and_writes_reloadable_evidence(
    tmp_path,
) -> None:
    result = run_smoke(tmp_path, device=torch.device("cpu"))

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert metrics["status"] == "smoke-passed"
    assert metrics["smoke_checks"] == {
        "checkpoint_restored": True,
        "fixed_batch_overfit": True,
        "opponent_snapshot_added": True,
        "target_synced": True,
    }
    assert metrics["history"][-1]["updates"] > 0
    assert metrics["final_evaluation"]["illegal_actions"] == 0
    assert checkpoint["model"]
    assert checkpoint["target"]
    assert result.summary_path.is_file()
    assert result.curve_path.is_file()

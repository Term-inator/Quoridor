import numpy as np
import pytest
import torch

from experiments.ppo.evaluation import evaluate
from experiments.ppo.experiment import run_smoke
from experiments.ppo.model import MaskedActorCritic
from experiments.ppo.training import PPOConfig, PPOUpdater, collect_rollout
from experiments.reward import PotentialRewardWrapper
from quoridor_rl.codec import ActionCodec
from quoridor_rl.env import env
from quoridor_rl.game import MovePawn, Player, Square


def test_masked_actor_critic_preserves_batch_and_action_shapes() -> None:
    model = MaskedActorCritic()
    observation = torch.zeros((2, 6, 9, 9))
    action_mask = torch.zeros((2, 209), dtype=torch.bool)
    action_mask[:, :3] = True

    logits, values = model(observation, action_mask)

    assert logits.shape == (2, 209)
    assert values.shape == (2,)
    assert torch.isneginf(logits[:, 3:]).all()
    assert torch.isfinite(logits[:, :3]).all()


def test_masked_actor_critic_never_samples_an_illegal_action() -> None:
    model = MaskedActorCritic()
    observation = torch.zeros((32, 6, 9, 9))
    action_mask = torch.zeros((32, 209), dtype=torch.bool)
    action_mask[:, 17] = True

    actions, log_probabilities, entropy, values = model.action_and_value(
        observation,
        action_mask,
    )

    assert actions.tolist() == [17] * 32
    assert log_probabilities.shape == (32,)
    assert entropy.shape == (32,)
    assert values.shape == (32,)


def test_masked_actor_critic_checkpoint_round_trip_is_exact(tmp_path) -> None:
    original = MaskedActorCritic()
    checkpoint = tmp_path / "model.pt"
    torch.save(original.state_dict(), checkpoint)
    restored = MaskedActorCritic()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    observation = torch.rand((2, 6, 9, 9))
    action_mask = torch.ones((2, 209), dtype=torch.bool)

    original_logits, original_values = original(observation, action_mask)
    restored_logits, restored_values = restored(observation, action_mask)

    assert torch.equal(original_logits, restored_logits)
    assert torch.equal(original_values, restored_values)


def test_rollout_collection_closes_both_player_identity_trajectories() -> None:
    model = MaskedActorCritic()
    config = PPOConfig(environment_count=1, rollout_size=4, max_plies=4)

    rollout = collect_rollout(model, config, torch.device("cpu"))

    assert len(rollout.transitions) >= 4
    assert {transition.agent for transition in rollout.transitions} == {
        "player_0",
        "player_1",
    }
    assert all(
        transition.action_mask[transition.action] for transition in rollout.transitions
    )
    assert rollout.episodes
    assert all(episode.truncated for episode in rollout.episodes)


def test_ppo_update_changes_shared_parameters_with_finite_metrics() -> None:
    model = MaskedActorCritic()
    config = PPOConfig(
        environment_count=1,
        rollout_size=8,
        max_plies=4,
        minibatch_size=4,
        update_epochs=1,
    )
    rollout = collect_rollout(model, config, torch.device("cpu"))
    before = model.policy_head.weight.detach().clone()
    updater = PPOUpdater(model, config, torch.device("cpu"))

    metrics = updater.update(rollout.transitions)

    assert not torch.equal(before, model.policy_head.weight)
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0 <= metrics["clip_fraction"] <= 1


def test_evaluation_swaps_roles_and_reports_real_game_outcomes() -> None:
    result = evaluate(
        MaskedActorCritic(),
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


def test_smoke_run_writes_reproducible_metrics_and_reloadable_checkpoint(
    tmp_path,
) -> None:
    result = run_smoke(tmp_path, device=torch.device("cpu"))

    assert result.metrics_path.is_file()
    assert result.summary_path.is_file()
    assert result.curve_path.is_file()
    assert result.checkpoint_path.is_file()
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["seed"] == 0
    assert checkpoint["model"]


def test_potential_reward_is_dense_and_zero_sum() -> None:
    environment = PotentialRewardWrapper(env())
    environment.reset()
    action = ActionCodec().encode(MovePawn(Square(7, 4)), Player.PLAYER_0)

    environment.step(action)

    assert environment.rewards["player_0"] == pytest.approx(0.0099)
    assert environment.rewards["player_1"] == pytest.approx(-0.0099)
    assert sum(environment.rewards.values()) == pytest.approx(0.0)
    assert environment.last_shaping_rewards == environment.rewards


def test_potential_reward_clips_each_players_shaping_symmetrically() -> None:
    environment = PotentialRewardWrapper(env(), scale=1.0)
    environment.reset()
    action = ActionCodec().encode(MovePawn(Square(7, 4)), Player.PLAYER_0)

    environment.step(np.int64(action))  # type: ignore[arg-type]

    assert environment.last_shaping_rewards == {
        "player_0": pytest.approx(0.05),
        "player_1": pytest.approx(-0.05),
    }


def test_potential_reward_keeps_a_truncation_zero_sum_and_zero_reward() -> None:
    environment = PotentialRewardWrapper(env(max_plies=1))
    environment.reset()
    action = ActionCodec().encode(MovePawn(Square(7, 4)), Player.PLAYER_0)

    environment.step(action)

    assert environment.truncations == {"player_0": True, "player_1": True}
    assert environment.rewards == {"player_0": 0.0, "player_1": 0.0}


def test_potential_reward_stops_on_an_illegal_action_without_shaping_it() -> None:
    environment = PotentialRewardWrapper(env())
    environment.reset()

    with pytest.raises(RuntimeError, match="illegal action"):
        environment.step(0)

    assert environment.rewards == {"player_0": -1.0, "player_1": 1.0}
    assert environment.last_shaping_rewards == {"player_0": 0.0, "player_1": 0.0}


def test_potential_reward_reset_restores_the_initial_potential() -> None:
    environment = PotentialRewardWrapper(env())
    action = ActionCodec().encode(MovePawn(Square(7, 4)), Player.PLAYER_0)
    environment.reset()
    environment.step(action)

    environment.reset()
    environment.step(action)

    assert environment.last_shaping_rewards["player_0"] == pytest.approx(0.0099)


def test_potential_reward_preserves_the_base_terminal_win_and_loss() -> None:
    environment = PotentialRewardWrapper(env())
    environment.reset()
    codec = ActionCodec()
    for target in (
        Square(7, 4),
        Square(0, 3),
        Square(6, 4),
        Square(0, 4),
        Square(5, 4),
        Square(0, 3),
        Square(4, 4),
        Square(0, 4),
        Square(3, 4),
        Square(0, 3),
        Square(2, 4),
        Square(0, 4),
        Square(1, 4),
        Square(0, 3),
        Square(0, 4),
    ):
        player = (
            Player.PLAYER_0
            if environment.agent_selection == "player_0"
            else Player.PLAYER_1
        )
        environment.step(codec.encode(MovePawn(target), player))

    shaping = environment.last_shaping_rewards
    assert environment.rewards["player_0"] - shaping["player_0"] == pytest.approx(1)
    assert environment.rewards["player_1"] - shaping["player_1"] == pytest.approx(-1)
    assert sum(environment.rewards.values()) == pytest.approx(0)

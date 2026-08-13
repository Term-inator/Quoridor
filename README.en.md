# Quoridor RL

English | [中文](README.md)

A Python implementation of official two-player Quoridor with:

- an immutable, hashable rules core independent of training frameworks;
- a standard PettingZoo AEC environment;
- a fixed 209-action discrete action space and legal-action mask;
- canonical observations suitable for a network shared by both players;
- human-vs-human and human-vs-random terminal games;
- an optional bilingual Pygame desktop interface for local games and agent spectating.

The first release uses a fixed 9×9 board with 10 walls per player. It does not include four-player games, variable board sizes, training algorithms, or reward shaping. The project has not yet been published to PyPI.

## Development environment

The project uses uv. Development uses Python 3.14, and the package supports Python 3.11–3.14.

```bash
uv sync --python 3.14
uv run pytest tests/package
```

The base installation does not install Pygame. To use the graphical interface:

```bash
uv sync --extra pygame
uv run quoridor-pygame
```

## PettingZoo AEC environment

AEC (Agent Environment Cycle) means that agents act in sequence: only the player named by `agent_selection` calls `step()` during a turn. Terminated or truncated agents must complete a dead step with `None`.

```python
import numpy as np

from quoridor_rl import env

environment = env(max_plies=512)
environment.reset(seed=0)

for agent in environment.agent_iter():
    observation, reward, terminated, truncated, info = environment.last()
    if terminated or truncated:
        action = None
    else:
        legal_ids = np.flatnonzero(observation["action_mask"])
        action = int(np.random.choice(legal_ids))
    environment.step(action)
```

For an English ANSI board, select the language explicitly:

```python
from quoridor_rl import Language, env

environment = env(render_mode="ansi", language=Language.ENGLISH)
```

### Action space

The action space is `Discrete(209)` and always uses the current player's canonical perspective:

| ID | Meaning |
| --- | --- |
| 0–80 | 9×9 pawn target squares, row-major |
| 81–144 | 8×8 horizontal-wall anchors, row-major |
| 145–208 | 8×8 vertical-wall anchors, row-major |

The absolute board for `player_1` is rotated 180°, so both players move upward from the bottom in their observations. Actions, observations, and masks use the same perspective; callers do not need to rotate them.

### Observation

Each observation is a dictionary:

- `observation`: `float32`, shape `(6, 9, 9)`;
- `action_mask`: `int8`, shape `(209,)`; only the active player has legal actions, while the other player's mask is all zeros.

The six planes contain the current pawn, opposing pawn, horizontal-wall anchors, vertical-wall anchors, current player's remaining-wall count, and opposing player's remaining-wall count. Wall counts are divided by 10 and broadcast across the 9×9 plane. Wall-anchor planes use the upper-left 8×8 area, with zeros in the final row and column.

### Rewards and endings

- Normal win: `+1` for the winner and `-1` for the loser; this is a termination.
- Other legal actions: `0` for both players.
- Reaching `max_plies`: `0` for both players; this is a truncation.
- Illegal action: the acting player immediately loses with `-1`, and the opponent receives `+1`.

The sparse reward is the environment's standard definition of the real zero-sum objective, not a claim that it is easiest for every training algorithm. It makes algorithm results comparable and directly supports terminal value targets for MCTS and AlphaZero. Experiments that need dense shaping should add it explicitly in a separate wrapper instead of changing the base task.

The current immutable rules position is available read-only through `environment.unwrapped.position`, allowing training wrappers and search algorithms to reuse the rules core. Assigning to that property fails.

## Using the rules core directly

MCTS implementations, bots, and tests can bypass RL action IDs and work with semantic actions:

```python
from quoridor_rl import MovePawn, Player, Position, Square

position = Position.initial()
assert MovePawn(Square(7, 4)) in position.legal_actions()

next_position = position.play(MovePawn(Square(7, 4)))
assert position != next_position  # The original state is unchanged.
assert position.shortest_path_length(Player.PLAYER_0) == 8
```

Absolute coordinates use `row=0..8` from top to bottom and `col=0..8` from left to right. An illegal but structurally valid action raises `IllegalActionError` with a stable `reason`.

`shortest_path_length()` returns the minimum number of steps to the player's target row while considering walls and ignoring pawn occupancy. It reuses the same pathfinding rules as wall validation.

## Local PPO learning validation

The repository includes an exploratory masked PPO self-play experiment that is excluded from the distribution wheel. It validates the observation, action mask, and reward signal before release. The base environment retains sparse terminal zero-sum rewards; dense potential rewards exist only in the local experiment's training wrapper.

```bash
uv sync --group train

# Short CPU/CUDA pipeline validation
uv run --group train python -m experiments.ppo.train --smoke --device cuda

# Seed 0, up to 120 minutes of training and 150 minutes overall
uv run --group train python -m experiments.ppo.train --device cuda
```

The formal experiment evaluates 200 balanced first/second-player games after 15, 30, 60, and 120 minutes, then evaluates the best checkpoint against a random agent for 1,000 games. Committable metrics, summaries, and plots are written to `experiments/ppo/results/seed-0/`. Checkpoints and TensorBoard logs go to the Git-ignored `experiments/ppo/artifacts/seed-0/`. Passing with one seed only justifies multi-seed validation; it is not a PyPI release conclusion.

## Local Masked Double DQN learning validation

The repository also includes a single-seed Masked Double DQN experiment parallel to PPO. The online network controls a random player identity each game, stores only its own decisions in uniform replay, and plays against a random agent or recent frozen policy snapshots. It uses the same observation, legal-action mask, CNN capacity, potential reward, two-hour training budget, and random-agent evaluation protocol as PPO. Results are presented side by side and do not claim one algorithm is superior.

```bash
# CUDA smoke gate
uv run --group train python -m experiments.dqn.train --smoke --device cuda

# Seed 0, up to 120 minutes of training and 150 minutes overall
uv run --group train python -m experiments.dqn.train --device cuda
```

Committable results are written to `experiments/dqn/results/seed-0/`; checkpoints, TensorBoard logs, and smoke artifacts go to the Git-ignored `experiments/dqn/artifacts/`.

## Local AlphaZero learning validation

The third single-seed exploration uses a shared policy-value network and PUCT MCTS with a fixed simulation count. It starts from a random network and uses MCTS root visit counts as policy targets and normal terminal outcomes as value targets. Undecided games do not enter replay, and the experiment does not use PPO/DQN stepwise reward shaping.

```bash
# Search, self-play, update, checkpoint, and short evaluation smoke gate
uv run --group train python -m experiments.alphazero.train --smoke --device cpu

# Seed 0, up to 120 minutes of training and 150 minutes overall
uv run --group train python -m experiments.alphazero.train --device cpu
```

Training uses 32 MCTS simulations per step. Evaluation uses 8 fixed simulations per step and runs four CPU games in parallel. Each node retains every pawn move and the highest-prior wall actions, up to 16 candidates. The first 32 games use a pawn-only curriculum to establish a terminal signal; afterward, all actions are restored and the progress prior is removed. Self-play adds exploration noise at the root, while formal evaluation disables noise and deterministically chooses the most visited action. Committable results are written to `experiments/alphazero/results/seed-0/`; large artifacts go to the Git-ignored `experiments/alphazero/artifacts/`.

## Playing in the terminal

```bash
# Two human players
uv run quoridor --opponent human

# Play against a random agent with a fixed seed
uv run quoridor --opponent random --seed 42

# Use the English terminal interface
uv run quoridor --language en --opponent random
```

Board input uses human-readable coordinates with `a1` in the lower-left corner:

```text
move e2
wall d4 horizontal
wall d4 v
quit
```

## Using the Pygame interface

The start screen switches between 中文 and English; Chinese is the default, and the selected language is retained for the current application process. It offers Human vs Human, Human vs Random Agent, and Random Agent vs Random Agent modes. In Human vs Random Agent mode, the human can choose to play first or second. Modes involving random agents accept an integer seed and offer playback-speed controls.

- Move: click a displayed legal target directly.
- Place a wall: choose Horizontal or Vertical; the pointer snaps to the nearest wall anchor, and a click confirms placement.
- Player-colored walls: placed walls use the same identity color as their player's pawn; a legal preview uses the current player's color.
- Wall inventory: two player cards display the exact remaining count and a ten-segment wall bar; the active player is highlighted.
- Move history: the sidebar preserves the complete history with the newest action first and each player's actions in their identity color. Use the mouse wheel when records exceed the visible area.
- End-game replay: after a normal ending, open the replay and click any action or the initial position to inspect that board state without changing the final result.
- Illegal wall: a red preview, cross mark, and localized reason appear together, and the turn does not advance.
- `Esc`: return to Move mode.
- Space: pause or resume an agent game.
- Right Arrow: advance one agent action while paused.

If no player wins after 512 plies, the interface reports the game as undecided rather than a draw under the Quoridor rules. The graphical interface has only been verified on Linux. Although pygame-ce publishes wheels for other major platforms, this project does not yet claim to have verified them.

The package includes Noto Sans SC Regular and its SIL Open Font License 1.1 text in the graphical assets package. The font is included in the base wheel, while pygame-ce remains available only through the `pygame` extra.

## Verification

The distribution package, optional graphical interface, and local training experiments are verified separately:

```bash
# Base distribution package
uv run pytest tests/package

# Optional Pygame interface
uv run --extra pygame pytest tests/pygame

# Build a wheel and verify installation, rules, AEC, CLI, and package contents
uv run python scripts/check-wheel.py

# Training experiments excluded from the wheel
uv run --group train pytest tests/experiments

# Run all tests after installing every dependency
uv run --group train --extra pygame pytest

uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

`tests/package/` covers ordinary movement, straight jumps, diagonal jumps when a wall or boundary blocks the square behind a pawn, wall conflicts, preservation of both players' paths, terminal states, action-codec round trips, canonical observations, the AEC lifecycle, complete random games, and the CLI. This layer also runs PettingZoo's official `api_test`. `tests/pygame/` verifies the optional graphical interface, while `tests/experiments/` verifies training code excluded from the package. Experimental win rates are not a release gate.

## License

[MIT](LICENSE)

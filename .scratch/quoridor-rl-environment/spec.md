# 双人围墙棋 RL 环境规格

Status: implemented

## Problem Statement

用户希望把围墙棋（Quoridor）做成一个其他人可以直接使用的强化学习环境，而不是只为某个训练脚本编写一套临时游戏逻辑。当前仓库没有可运行代码、构建配置、规则实现、测试或文档，因此首先需要建立一个规则可信、接口稳定、可测试且能端到端运行的最小产品。

这个产品必须把官方双人围墙棋规则与具体训练算法分离。环境应为多智能体顺序行动提供标准接口，同时允许 CLI、随机智能体和未来的 MCTS 等调用者复用同一个规则实现。它还必须明确动作编号、观察编码、合法动作 mask、奖励、终止和截断语义，避免每个使用者重复解释规则或自行拼接不兼容的编码。

## Solution

构建一个 Python 双人围墙棋环境库，核心是纯内存、确定性、不可变的规则模块。规则模块使用语义动作和绝对棋盘坐标，完整处理移动、直跳、受阻斜跳、墙冲突、路径保留、回合推进和胜负判断。它不依赖 NumPy、PettingZoo 或训练框架。

在规则模块之上构建 PettingZoo AEC adapter。adapter 为两个玩家提供统一的当前玩家视角、固定 209 动作离散空间、合法动作 mask、六通道空间 observation、稀疏终局奖励、非法动作处理和最大行动数截断。另提供 ASCII CLI 与随机智能体，使两个人类玩家以及人类与随机智能体可以在没有训练模型时完成对局。

首版只实现官方标准双人制，不实现四人制、可变棋盘、训练算法或奖励塑形。开发按测试驱动的纵向切片推进，先让规则核心工作，再叠加编码、环境和 CLI。

## User Stories

1. As an RL researcher, I want to instantiate a standard two-player Quoridor environment, so that I can evaluate an algorithm without implementing the game rules myself.
2. As an RL researcher, I want the environment to use the PettingZoo AEC interaction model, so that each player is represented as an explicit sequentially acting agent.
3. As an RL researcher, I want a fixed discrete action space, so that I can attach a policy head with a stable output dimension.
4. As an RL researcher, I want an action mask in every observation, so that my policy can avoid sampling illegal moves.
5. As an RL researcher, I want non-current agents to receive an all-zero action mask, so that turn ownership is unambiguous.
6. As an RL researcher, I want observations to contain the complete game state, so that the environment remains a perfect-information game.
7. As an RL researcher, I want both players to observe the board from the same canonical perspective, so that they can share one policy network.
8. As an RL researcher, I want action coordinates to use the same canonical perspective as observations, so that masks, logits and executed actions cannot become misaligned.
9. As an RL researcher, I want remaining wall counts encoded in a neural-network-friendly range, so that basic preprocessing does not need to know game constants.
10. As an RL researcher, I want terminal win and loss rewards to be exactly +1 and -1, so that the environment expresses the true zero-sum objective.
11. As an RL researcher, I want all non-terminal rewards to be zero, so that environment results are comparable and free of an unvalidated heuristic objective.
12. As an RL researcher, I want an episode-length limit reported as truncation rather than a game result, so that cyclic play cannot hang training while the official winner rule remains intact.
13. As an RL researcher, I want deterministic initial conditions and a fixed first-player identity, so that experiments are reproducible.
14. As an RL researcher, I want to exchange policies between the two agent identities during evaluation, so that first-player advantage can be measured rather than hidden by randomization.
15. As an RL researcher, I want invalid action semantics to be documented, so that integration errors have predictable outcomes.
16. As an RL researcher, I want the environment to pass PettingZoo's conformance test, so that downstream tooling can rely on the AEC contract.
17. As an AlphaZero researcher, I want terminal outcomes exposed without reward shaping, so that I can train value targets from actual game results.
18. As an AlphaZero researcher, I want semantic game states and legal actions outside the AEC lifecycle, so that MCTS can explore branches directly.
19. As an MCTS implementer, I want each applied action to return a new immutable position, so that tree branches cannot accidentally mutate one another.
20. As an MCTS implementer, I want positions to support value equality and in-process hashing, so that I can use a transposition table.
21. As an MCTS implementer, I want repeated access to legal actions to be deterministic, so that searches and diagnostics are reproducible.
22. As a bot author, I want legal actions represented by domain concepts, so that my bot does not need to understand RL integer encodings.
23. As a bot author, I want to select a target square for every pawn move, so that normal moves, straight jumps and diagonal jumps share one action concept.
24. As a bot author, I want wall placement represented by an anchor and orientation, so that wall geometry is explicit.
25. As a bot author, I want illegal actions to raise a structured error in the rule module, so that mistakes can be diagnosed programmatically.
26. As a bot author, I want the rule module to expose all and only legal actions, so that I do not need separate move and wall validators.
27. As a maintainer, I want one authoritative implementation of every game rule, so that the AEC environment, CLI and bots cannot drift apart.
28. As a maintainer, I want path search and wall geometry hidden behind the rule interface, so that their implementation can change without affecting callers.
29. As a maintainer, I want rule state to use absolute coordinates internally, so that logs, tests and human rendering remain easy to reason about.
30. As a maintainer, I want RL rotation and integer encoding isolated in the adapter, so that changes to model representation do not modify the game rules.
31. As a maintainer, I want the initial state to enforce the official 9×9 board, two players and ten walls per player, so that unsupported variants cannot enter accidentally.
32. As a maintainer, I want wall placements to reject crossing, overlap and shared wall segments, so that all stored positions remain physically valid.
33. As a maintainer, I want every accepted wall placement to leave a path for both players, so that the defining Quoridor constraint is always preserved.
34. As a maintainer, I want ordinary movement, straight jumping and blocked straight-jump diagonal movement tested separately, so that rare movement rules do not regress.
35. As a maintainer, I want the board boundary treated like an obstruction behind an adjacent opponent, so that edge diagonal jumping has a deterministic digital interpretation.
36. As a maintainer, I want the core rule module to depend only on the Python standard library, so that rule testing and reuse stay lightweight.
37. As a maintainer, I want the project managed with uv and a project-local virtual environment, so that development dependencies and commands are reproducible.
38. As a maintainer, I want development to use Python 3.14 while the library remains compatible with Python 3.11 and later, so that modern tooling does not unnecessarily exclude users.
39. As a maintainer, I want property-based legal-play tests, so that state invariants are checked across many action sequences rather than only hand-selected examples.
40. As a maintainer, I want numeric action codec round-trip tests, so that all 209 action identifiers remain stable and reversible.
41. As a maintainer, I want perspective round-trip tests for both players, so that square and wall rotations cannot silently corrupt actions.
42. As a maintainer, I want random complete-game smoke tests, so that lifecycle, termination and truncation bugs appear before release.
43. As a human player, I want an ASCII rendering of the board, so that I can inspect a game without installing graphical dependencies.
44. As a human player, I want readable commands for pawn movement and wall placement, so that I do not need to enter raw action IDs.
45. As a human player, I want a local human-versus-human mode, so that two people can verify and enjoy the implementation.
46. As a human player, I want a human-versus-random mode, so that I can play before any trained model exists.
47. As a CLI user, I want illegal commands and moves to produce understandable errors, so that I can correct input without inspecting code.
48. As a library user, I want type information for public domain values and interfaces, so that editors and static analysis can detect misuse.
49. As a library user, I want the project licensed under MIT, so that I can reuse and modify the implementation under clear permissive terms.
50. As a future contributor, I want the specification and behavioral tests to define the intended environment, so that internal implementation changes do not require rediscovering design decisions.

## Implementation Decisions

- The product is a reusable RL environment library, not a bundled training platform or a single trained agent.
- The first implementation supports only official standard two-player Quoridor on a fixed 9×9 board. Each player starts with ten walls. Four-player rules, three-player rules, variable board sizes and custom variants are not generalized in advance.
- Player 0 begins at the center of the bottom row, moves toward the top row and always acts first. Player 1 begins at the center of the top row and moves toward the bottom row.
- A player wins immediately upon reaching any square of the opposite goal row.
- Pawn movement supports one-square orthogonal movement, straight jumping over an adjacent opponent and diagonal movement around that opponent only when the straight landing square is obstructed by a wall or board boundary. A wall between the pawns prevents the jump interaction.
- The board boundary is treated as an obstruction behind an adjacent pawn for purposes of diagonal jumping. This resolves an ambiguity in the physical rule text and is part of the digital rules contract.
- A wall occupies two adjacent edges. Walls may meet end-to-end but may not cross, overlap completely, share only one already occupied segment or extend beyond the 8×8 wall-anchor grid.
- A wall placement is legal only if both players retain at least one path to their respective goal row. Path existence ignores current pawn occupancy and considers wall geometry.
- The rule module is a deep, deterministic, in-process module. Its public interface centers on an immutable Position that can create the official initial state, expose stable read-only domain facts, enumerate legal semantic actions and produce a new Position by playing an action.
- A Position exposes the two pawn squares, both remaining-wall counts, placed walls, the player to move and the winner. It does not expose pathfinder, graph, bitboard, cache or validation implementation details.
- Position equality and hashing include all rule-relevant state: pawn locations, placed walls, remaining-wall counts and player to move. They exclude object identity, caches, reward, episode length, observation perspective and truncation state.
- The rule module uses absolute coordinates. Its action union consists of a pawn move targeting a board square and a wall placement targeting an anchor with horizontal or vertical orientation.
- The rule module performs complete validation inside the play operation even if the caller previously enumerated legal actions. Invalid value objects fail at construction; well-formed but illegal actions raise a structured IllegalActionError with a stable reason.
- The rule module has no reward, maximum-ply logic, random number generator, rendering, NumPy dependency, PettingZoo dependency or numeric action IDs.
- The PettingZoo layer is an AEC adapter over the rule module. The two agents are explicitly named player 0 and player 1, and only the selected agent can act.
- The numeric action space is Discrete(209). IDs 0 through 80 select a pawn destination square in row-major current-player coordinates. IDs 81 through 144 select a horizontal wall anchor in 8×8 row-major order. IDs 145 through 208 select a vertical wall anchor in the same order.
- Current-player square coordinates rotate by 180 degrees for player 1. A board square maps by reversing both 9×9 axes; a wall anchor maps by reversing both 8×8 axes. The adapter owns both encoding and decoding, so callers never rotate observations, masks or actions themselves.
- The observation is a dictionary containing a float32 tensor with shape 6×9×9 and a 209-element binary action mask.
- Observation planes represent, in order, the observing player's pawn, the opponent pawn, horizontal wall anchors, vertical wall anchors, the observing player's remaining walls and the opponent's remaining walls.
- Wall-anchor planes use their 8×8 upper-left region and pad the final row and column with zero. Wall-count planes broadcast the remaining count divided by ten across all 81 cells.
- Every agent receives a complete observation from its own canonical perspective. The current agent receives a mask for all and only legal actions; every non-current or finished agent receives an all-zero mask.
- The adapter returns +1 to the winner, -1 to the loser and zero for every other legal transition. The base environment contains no distance, progress, wall or heuristic reward shaping.
- A normal win is a PettingZoo termination. An episode reaching the configurable maximum of 512 plies is a truncation with zero reward to both players. The rule module does not treat truncation as a draw or store the ply counter.
- Player 0 always acts first. Fair evaluation swaps the two tested policies across agent identities and reports first-player, second-player and combined results rather than randomizing the environment's first mover.
- The public rule interface raises IllegalActionError for illegal actions. The PettingZoo environment converts an illegal selected action into immediate loss for the acting agent. Normal training is expected to honor the action mask.
- The base environment is deterministic. Reset accepts the standard seed argument for framework compatibility, although the official initial position itself contains no randomness.
- Reward shaping is deferred. If later training evidence demonstrates a need, it will be added as a separate, explicitly documented environment wrapper rather than a reward callback or alternate default objective.
- The environment supports ANSI rendering. The ASCII renderer is shared with a CLI that supports local human-versus-human and human-versus-random games using readable square and wall commands.
- The bundled random agent samples only from legal actions and owns its own seeded random-number generator. Randomness does not enter the rule module.
- The implementation uses a uv-managed library project with a project-local virtual environment. Development is pinned to CPython 3.14; package metadata declares Python 3.11 or later, with validation across 3.11 through 3.14.
- The source is released under the MIT License. Choosing and publishing a PyPI distribution name is deferred and does not block local implementation.
- Correctness and interface clarity take priority over speculative low-level optimization. Immutable state, straightforward graph search and complete legal-action generation are implemented first; bitboards, structural sharing, native extensions or specialized caches require profiling evidence.

## Testing Decisions

- Tests verify observable behavior through confirmed public seams. They do not call private pathfinding, wall-bit, graph, cache, rotation helper or validation implementation details.
- Development follows vertical red-green TDD slices: one failing behavioral test, the minimum implementation that passes it, and then the next behavior. Tests are not written as one horizontal batch ahead of implementation.
- The Position seam verifies the official initial state, immutable transitions, deterministic legal-action enumeration, all movement variants, winning, wall inventory, wall conflicts, path preservation, terminal behavior, structured errors, equality and hashing.
- The ActionCodec seam verifies all 209 IDs in both directions, stable row-major numbering, player-1 square rotation, player-1 wall-anchor rotation and rejection of out-of-range IDs.
- The PettingZoo environment seam verifies reset, observe, last, step, reward accumulation, terminations, truncations, dead-agent steps, selected-agent order, observation spaces, action spaces, active masks, inactive masks and illegal-action behavior.
- The CLI seam verifies human-readable parsing, ASCII rendering, human-versus-human completion, human-versus-random completion and understandable illegal-input feedback.
- Official rule examples are treated as independent expected behavior for ordinary movement, straight jumping, blocked diagonal jumping, goal detection and wall path preservation.
- Explicit boundary cases cover diagonal jumping at a board edge, walls touching end-to-end, complete overlap, one-segment overlap, crossing, out-of-bounds anchors, no walls remaining and attempted total path blockage.
- Property-based tests generate legal action sequences and continuously assert invariants: distinct pawn squares, valid wall counts, wall inventory conservation, legal wall geometry, path existence for both non-terminal players and immutability of every parent Position.
- Codec tests verify encode-decode and decode-encode round trips independently of internal storage. Observation tests verify that canonical views for opposing players rotate to consistent semantic states.
- PettingZoo's official API test is required to pass. Random legal policies also run repeated games until normal termination or the 512-ply truncation without hangs or inconsistent lifecycle state.
- Tests verify that a built distribution can be installed in a clean environment and that its basic rule, PettingZoo and CLI examples run. Actual publication to an external package index is not part of this specification.
- The compatibility matrix covers Python 3.11, 3.12, 3.13 and 3.14. The normal project virtual environment uses Python 3.14.
- There is no prior test suite in the repository. The authoritative prior art is the official Quoridor rulebook for game behavior and PettingZoo's conformance utilities and classic turn-based environments for AEC behavior.

## Out of Scope

- Four-player, three-player or arbitrary-player Quoridor.
- Variable board sizes, configurable starting walls or custom rule variants.
- PPO, DQN, AlphaZero, MCTS or any other training/search implementation.
- A trained policy, model checkpoint, model loader or general agent plug-in protocol.
- Dense reward shaping, curriculum learning or injectable reward functions.
- Gymnasium single-agent wrappers with a built-in opponent.
- PettingZoo Parallel environments; Quoridor is sequential and uses AEC only.
- Web UI, Pygame, graphical windows, animations and RGB-array rendering.
- A formal benchmark suite, frozen leaderboards, Elo infrastructure or baseline training scores.
- Four-player reward vectors, kingmaking evaluation or multi-player search semantics.
- Serialization formats, saved-game compatibility, arbitrary-position import or network play.
- Native extensions, Rust/C++ bindings, batch vector environments and unproven performance optimizations.
- Publishing to PyPI, reserving a distribution name or configuring external release automation.
- Backward-compatibility layers for designs that have not yet been released.

## Further Notes

- The official physical game supports both two- and four-player modes, but this specification deliberately narrows the first product to the simpler two-player zero-sum game.
- The standard fixed board and equal wall resources are geometrically symmetric, but the first move may still affect win rate. Fairness belongs in the evaluation protocol through role swapping, not through modifying game rules.
- Sparse reward may make model-free PPO training difficult. That is an algorithmic limitation rather than a reason to redefine the base task. AlphaZero-style systems can use terminal outcomes together with MCTS policy targets.
- Stable-Baselines3 does not directly consume arbitrary PettingZoo multi-agent environments without adaptation. Supporting a particular trainer is intentionally deferred until the environment itself is correct.
- The four confirmed test seams are Position, ActionCodec, PettingZoo env and the CLI command. These are also the principal caller seams of the implementation.
- The local issue tracker convention records this specification as ready for agent implementation. No separate implementation tickets are created by this skill.

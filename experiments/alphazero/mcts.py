"""由策略—价值网络引导的 PUCT 蒙特卡洛树搜索。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from experiments.alphazero.model import PolicyValueNetwork
from quoridor_rl.codec import ActionCodec, ObservationCodec
from quoridor_rl.game import Action, MovePawn, Position


@dataclass(slots=True)
class _Edge:
    """父节点上的动作边，保存先验、访问统计及可复用子树。"""

    action: Action
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: _Node | None = None

    @property
    def mean_value(self) -> float:
        """返回从父节点行动方视角累计的平均价值。"""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(slots=True)
class _Node:
    """一个规则局面及其惰性展开的出边。"""

    position: Position
    children: dict[int, _Edge] = field(default_factory=dict)
    expanded: bool = False
    noise_applied: bool = False


@dataclass(frozen=True, slots=True)
class SearchResult:
    """一次根搜索产生的动作、访问策略和性能诊断。"""

    action_id: int
    policy: np.ndarray
    root_value: float
    expanded_nodes: int
    maximum_depth: int


class PUCTSearch:
    """单局内可随真实动作向下复用的搜索树。"""

    def __init__(
        self,
        model: PolicyValueNetwork,
        *,
        device: torch.device,
        simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        root_noise_fraction: float,
        maximum_actions: int,
        allow_walls: bool,
        progress_prior_fraction: float,
        seed: int,
    ) -> None:
        """校验搜索、探索噪声、分支裁剪和课程先验参数。"""
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        if c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= root_noise_fraction <= 1:
            raise ValueError("root_noise_fraction must be between zero and one")
        if maximum_actions <= 0:
            raise ValueError("maximum_actions must be positive")
        if not 0 <= progress_prior_fraction <= 1:
            raise ValueError("progress_prior_fraction must be between zero and one")
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise_fraction = root_noise_fraction
        self.maximum_actions = maximum_actions
        self.allow_walls = allow_walls
        self.progress_prior_fraction = progress_prior_fraction
        self.random = np.random.default_rng(seed)
        self.action_codec = ActionCodec()
        self.observation_codec = ObservationCodec()
        self.root: _Node | None = None

    def run(
        self,
        position: Position,
        *,
        remaining_plies: int,
        add_root_noise: bool,
        temperature: float,
    ) -> SearchResult:
        """搜索 ``position``，根据根节点访问次数策略选择动作。

        每次模拟依次执行 PUCT 选择、首次叶节点展开和符号交替的价值回传；达到剩余
        手数上限按和局价值零处理。
        """
        if position.to_move is None:
            raise ValueError("cannot search a terminal position")
        if remaining_plies <= 0:
            raise ValueError("remaining_plies must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.root is None or self.root.position != position:
            self.root = _Node(position)
        root = self.root
        expanded_nodes = 0
        if not root.expanded:
            root_value = self._expand(root)
            expanded_nodes += 1
        else:
            root_value = _node_mean_value(root)
        if not root.children:
            raise RuntimeError("a non-terminal position has no legal actions")
        if add_root_noise and not root.noise_applied:
            self._add_root_noise(root)

        maximum_depth = 0
        for _ in range(self.simulations):
            node = root
            path: list[_Edge] = []
            depth = 0
            while True:
                if depth >= remaining_plies:
                    leaf_value = 0.0
                    break
                edge = self._select(node)
                path.append(edge)
                depth += 1
                if edge.child is None:
                    edge.child = _Node(node.position.play(edge.action))
                node = edge.child
                if node.position.winner is not None:
                    leaf_value = -1.0
                    break
                if depth >= remaining_plies:
                    leaf_value = 0.0
                    break
                if not node.expanded:
                    leaf_value = self._expand(node)
                    expanded_nodes += 1
                    break
            maximum_depth = max(maximum_depth, depth)
            _backpropagate(path, leaf_value)

        policy = _visit_policy(root)
        action_id = _select_action(policy, temperature, self.random)
        return SearchResult(
            action_id=action_id,
            policy=policy,
            root_value=root_value,
            expanded_nodes=expanded_nodes,
            maximum_depth=maximum_depth,
        )

    def advance(self, action_id: int, position: Position) -> None:
        """真实动作与预测子节点一致时复用子树，否则从新局面重建根。"""
        child = None
        if self.root is not None:
            edge = self.root.children.get(action_id)
            child = None if edge is None else edge.child
        self.root = (
            child
            if child is not None and child.position == position
            else _Node(position)
        )

    @torch.no_grad()
    def _expand(self, node: _Node) -> float:
        """用网络展开节点，并按配置裁剪候选墙、混合课程进度先验。"""
        player = node.position.to_move
        if player is None:
            raise ValueError("cannot expand a terminal node")
        actions = node.position.legal_actions()
        if not self.allow_walls:
            actions = tuple(
                action for action in actions if isinstance(action, MovePawn)
            )
        action_ids = [self.action_codec.encode(action, player) for action in actions]
        observation = (
            torch.from_numpy(self.observation_codec.encode(node.position, player))
            .unsqueeze(0)
            .to(self.device)
        )
        self.model.eval()
        logits, value = self.model(observation)
        if self.allow_walls and len(actions) > self.maximum_actions:
            pawn_indices = [
                index
                for index, action in enumerate(actions)
                if isinstance(action, MovePawn)
            ]
            wall_indices = [
                index
                for index, action in enumerate(actions)
                if not isinstance(action, MovePawn)
            ]
            wall_budget = max(0, self.maximum_actions - len(pawn_indices))
            wall_logits = logits[0, [action_ids[index] for index in wall_indices]]
            selected_walls = torch.topk(
                wall_logits,
                k=min(wall_budget, len(wall_indices)),
            ).indices.tolist()
            kept_indices = pawn_indices + [
                wall_indices[index] for index in selected_walls
            ]
            actions = tuple(actions[index] for index in kept_indices)
            action_ids = [action_ids[index] for index in kept_indices]
        legal_logits = logits[0, action_ids]
        priors = torch.softmax(legal_logits, dim=0).cpu().numpy()
        if self.progress_prior_fraction:
            progress = np.asarray(
                [
                    math.exp(
                        -2 * node.position.play(action).shortest_path_length(player)
                    )
                    for action in actions
                ],
                dtype=np.float64,
            )
            progress /= progress.sum()
            fraction = self.progress_prior_fraction
            priors = (1 - fraction) * priors + fraction * progress
        if not np.isfinite(priors).all() or not math.isfinite(float(value.item())):
            raise FloatingPointError("policy-value network produced NaN or infinity")
        node.children = {
            action_id: _Edge(action=action, prior=float(prior))
            for action_id, action, prior in zip(
                action_ids, actions, priors, strict=True
            )
        }
        node.expanded = True
        return float(value.item())

    def _add_root_noise(self, root: _Node) -> None:
        """把 Dirichlet 噪声混入根先验，增加自我对弈探索多样性。"""
        edges = list(root.children.values())
        noise = self.random.dirichlet([self.dirichlet_alpha] * len(edges))
        fraction = self.root_noise_fraction
        for edge, sample in zip(edges, noise, strict=True):
            edge.prior = (1 - fraction) * edge.prior + fraction * float(sample)
        root.noise_applied = True

    def _select(self, node: _Node) -> _Edge:
        """按平均价值与先验探索奖励之和选择 PUCT 出边。"""
        total_visits = sum(edge.visit_count for edge in node.children.values())
        scale = math.sqrt(max(1, total_visits))
        return max(
            node.children.values(),
            key=lambda edge: (
                edge.mean_value
                + self.c_puct * edge.prior * scale / (1 + edge.visit_count),
                edge.prior,
            ),
        )


def _backpropagate(path: list[_Edge], leaf_value: float) -> None:
    """沿路径反向累计叶价值；每过一手切换玩家视角并反号。"""
    value = leaf_value
    for edge in reversed(path):
        value = -value
        edge.visit_count += 1
        edge.value_sum += value


def _visit_policy(root: _Node) -> np.ndarray:
    """把根出边访问次数归一化到固定 209 维策略空间。"""
    policy = np.zeros(ActionCodec.action_count, dtype=np.float32)
    total = sum(edge.visit_count for edge in root.children.values())
    if total <= 0:
        raise RuntimeError("search produced no root visits")
    for action_id, edge in root.children.items():
        policy[action_id] = edge.visit_count / total
    return policy


def _select_action(
    policy: np.ndarray,
    temperature: float,
    random: np.random.Generator,
) -> int:
    """按温度变换后的访问策略采样；零温度直接取众数动作。"""
    if temperature == 0:
        return int(policy.argmax())
    weights = np.power(policy, 1 / temperature, dtype=np.float64)
    weights /= weights.sum()
    return int(random.choice(len(weights), p=weights))


def _node_mean_value(node: _Node) -> float:
    """聚合节点所有出边的平均搜索价值。"""
    visits = sum(edge.visit_count for edge in node.children.values())
    if not visits:
        return 0.0
    return sum(edge.value_sum for edge in node.children.values()) / visits

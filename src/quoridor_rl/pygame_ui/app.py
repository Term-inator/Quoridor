"""Pygame desktop application for local Quoridor games."""

from __future__ import annotations

import io
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Protocol

import pygame

from quoridor_rl.agents import RandomAgent
from quoridor_rl.game import (
    Action,
    IllegalActionError,
    IllegalActionReason,
    MovePawn,
    Orientation,
    PlaceWall,
    Player,
    Position,
    Square,
    WallAnchor,
)

WINDOW_INITIAL = (1280, 800)
WINDOW_MIN = (960, 640)

COLORS = {
    "paper": pygame.Color("#f4f0e7"),
    "panel": pygame.Color("#fffdf8"),
    "ink": pygame.Color("#17211c"),
    "muted": pygame.Color("#607069"),
    "line": pygame.Color("#d7d0c2"),
    "board": pygame.Color("#c99b62"),
    "board_dark": pygame.Color("#8c6038"),
    "p0": pygame.Color("#176b55"),
    "p1": pygame.Color("#a44437"),
    "valid": pygame.Color("#238b61"),
    "invalid": pygame.Color("#c83933"),
    "focus": pygame.Color("#f0a33a"),
    "selected": pygame.Color("#16231e"),
    "white": pygame.Color("#ffffff"),
}

REASON_TEXT = {
    IllegalActionReason.GAME_OVER: "对局已经结束",
    IllegalActionReason.ILLEGAL_PAWN_MOVE: "棋子不能移动到该格",
    IllegalActionReason.NO_WALLS_REMAINING: "当前行动方已经没有墙",
    IllegalActionReason.WALL_CONFLICT: "与已有墙重叠或交叉",
    IllegalActionReason.WALL_BLOCKS_PATH: "会堵死至少一方的所有路径",
}


class ApplicationScreen(Enum):
    """User-visible screens in the desktop application."""

    START = "start"
    PLAYING = "playing"
    RESULT = "result"
    ERROR = "error"


class GameMode(Enum):
    """Supported combinations of local participants."""

    HUMAN_HUMAN = "human_human"
    HUMAN_RANDOM = "human_random"
    RANDOM_RANDOM = "random_random"


class Control(Enum):
    """Semantic controls exposed for UI automation at the app seam."""

    MODE_HUMAN_HUMAN = "mode_human_human"
    MODE_HUMAN_RANDOM = "mode_human_random"
    MODE_RANDOM_RANDOM = "mode_random_random"
    HUMAN_PLAYER_0 = "human_player_0"
    HUMAN_PLAYER_1 = "human_player_1"
    SEED_INPUT = "seed_input"
    START_GAME = "start_game"
    MOVE = "move"
    HORIZONTAL_WALL = "horizontal_wall"
    VERTICAL_WALL = "vertical_wall"
    RESTART_GAME = "restart_game"
    RETURN_TO_START = "return_to_start"
    EXIT = "exit"
    PAUSE_RESUME = "pause_resume"
    STEP = "step"
    SPEED_SLOW = "speed_slow"
    SPEED_NORMAL = "speed_normal"
    SPEED_FAST = "speed_fast"


class InputMode(Enum):
    """Current human action intent."""

    MOVE = "move"
    HORIZONTAL_WALL = "horizontal_wall"
    VERTICAL_WALL = "vertical_wall"


@dataclass(frozen=True, slots=True)
class ApplicationSnapshot:
    """Observable state exposed at the application test seam."""

    screen: ApplicationScreen
    feedback: str
    position: Position | None
    plies: int
    input_mode: InputMode
    preview_wall: PlaceWall | None
    preview_reason: IllegalActionReason | None
    paused: bool
    game_mode: GameMode | None
    human_player: Player
    seed: int | None
    winner: Player | None
    font_resource: str
    agent_delay_ms: int
    last_action: Action | None
    last_move_start: Square | None
    window_size: tuple[int, int]
    status_lines: tuple[str, ...]


class ActionChoosingAgent(Protocol):
    """Minimal semantic interface shared by bundled and future agents."""

    def choose_action(self, position: Position) -> Action: ...


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    """Map semantic board locations into the current visible board."""

    rect: pygame.Rect
    cell: float
    gap: float
    origin_x: float
    origin_y: float

    @classmethod
    def from_rect(cls, rect: pygame.Rect) -> BoardGeometry:
        inset = max(18.0, rect.width * 0.047)
        usable = rect.width - inset * 2
        gap_ratio = 0.18
        cell = usable / (9 + 8 * gap_ratio)
        return cls(rect, cell, cell * gap_ratio, rect.left + inset, rect.top + inset)

    @property
    def pitch(self) -> float:
        return self.cell + self.gap

    def square_rect(self, square: Square) -> pygame.Rect:
        return pygame.Rect(
            round(self.origin_x + square.col * self.pitch),
            round(self.origin_y + square.row * self.pitch),
            round(self.cell),
            round(self.cell),
        )

    def anchor_center(self, anchor: WallAnchor) -> tuple[float, float]:
        return (
            self.origin_x + anchor.col * self.pitch + self.cell + self.gap / 2,
            self.origin_y + anchor.row * self.pitch + self.cell + self.gap / 2,
        )

    def wall_rect(self, wall: PlaceWall) -> pygame.Rect:
        if wall.orientation is Orientation.HORIZONTAL:
            return pygame.Rect(
                round(self.origin_x + wall.anchor.col * self.pitch),
                round(self.origin_y + wall.anchor.row * self.pitch + self.cell),
                round(self.cell * 2 + self.gap),
                max(5, round(self.gap)),
            )
        return pygame.Rect(
            round(self.origin_x + wall.anchor.col * self.pitch + self.cell),
            round(self.origin_y + wall.anchor.row * self.pitch),
            max(5, round(self.gap)),
            round(self.cell * 2 + self.gap),
        )

    def nearest_anchor(self, point: tuple[int, int]) -> WallAnchor | None:
        interactive = self.rect.inflate(
            -round(self.rect.width * 0.05),
            -round(self.rect.height * 0.05),
        )
        if not interactive.collidepoint(point):
            return None
        col = round((point[0] - self.origin_x - self.cell - self.gap / 2) / self.pitch)
        row = round((point[1] - self.origin_y - self.cell - self.gap / 2) / self.pitch)
        return WallAnchor(max(0, min(7, row)), max(0, min(7, col)))


class PygameApplication:
    """Own the desktop interaction and rendering lifecycle."""

    def __init__(
        self,
        *,
        agent_factory: Callable[[int | None], ActionChoosingAgent] = RandomAgent,
        max_plies: int = 512,
    ) -> None:
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        pygame.font.init()
        self._max_plies = max_plies
        self._font_resource = "NotoSansSC-Regular.otf"
        self._font_bytes = (
            resources.files("quoridor_rl.pygame_ui.assets")
            .joinpath(self._font_resource)
            .read_bytes()
        )
        self._screen = ApplicationScreen.START
        self._feedback = "请选择对局模式并开始。"
        self._position: Position | None = None
        self._legal_actions: tuple[Action, ...] = ()
        self._preview_reason_cache: dict[PlaceWall, IllegalActionReason] = {}
        self._plies = 0
        self._agent_factory = agent_factory
        self._selected_game_mode: GameMode | None = None
        self._game_mode: GameMode | None = None
        self._human_player = Player.PLAYER_0
        self._seed_text = ""
        self._seed: int | None = None
        self._seed_focused = False
        self._result_winner: Player | None = None
        self._last_action: Action | None = None
        self._last_move_start: Square | None = None
        self._quit_requested = False
        self._agents: dict[Player, ActionChoosingAgent] = {}
        self._paused = False
        self._agent_elapsed_ms = 0
        self._agent_delay_ms = 500
        self._input_mode = InputMode.MOVE
        self._preview_wall: PlaceWall | None = None
        self._preview_reason: IllegalActionReason | None = None
        self._surface_size = WINDOW_INITIAL
        self._controls: dict[Control, pygame.Rect] = {}
        self._player_status_cards: dict[Player, pygame.Rect] = {}
        self._wall_inventory_segments: dict[
            Player,
            tuple[pygame.Rect, ...],
        ] = {}
        self._board: BoardGeometry | None = None
        self._fonts: dict[int, pygame.font.Font] = {}

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return ApplicationSnapshot(
            screen=self._screen,
            feedback=self._feedback,
            position=self._position,
            plies=self._plies,
            input_mode=self._input_mode,
            preview_wall=self._preview_wall,
            preview_reason=self._preview_reason,
            paused=self._paused,
            game_mode=self._game_mode,
            human_player=self._human_player,
            seed=self._seed,
            winner=self._result_winner,
            font_resource=self._font_resource,
            agent_delay_ms=self._agent_delay_ms,
            last_action=self._last_action,
            last_move_start=self._last_move_start,
            window_size=self._surface_size,
            status_lines=self._status_lines(),
        )

    def control_rect(self, control: Control) -> pygame.Rect:
        """Return the current visible rectangle for a semantic control."""
        return self._controls[control].copy()

    def square_rect(self, square: Square) -> pygame.Rect:
        """Return the current visible rectangle for a semantic square."""
        if self._board is None:
            raise RuntimeError("the playing board has not been drawn")
        return self._board.square_rect(square)

    def wall_anchor_point(self, anchor: WallAnchor) -> tuple[int, int]:
        """Return the visible snap point for a semantic wall anchor."""
        if self._board is None:
            raise RuntimeError("the playing board has not been drawn")
        x, y = self._board.anchor_center(anchor)
        return round(x), round(y)

    def player_status_rect(self, player: Player) -> pygame.Rect:
        """Return the visible status-card rectangle for a player identity."""
        return self._player_status_cards[player].copy()

    def wall_inventory_segment_rect(
        self,
        player: Player,
        index: int,
    ) -> pygame.Rect:
        """Return one visible segment from a player's ten-wall inventory."""
        return self._wall_inventory_segments[player][index].copy()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle one real Pygame event and report whether to continue."""
        if event.type == pygame.QUIT:
            return False
        if event.type in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED):
            requested = event.size if hasattr(event, "size") else (event.x, event.y)
            self._surface_size = (
                max(WINDOW_MIN[0], requested[0]),
                max(WINDOW_MIN[1], requested[1]),
            )
            return True
        if (
            event.type == pygame.KEYDOWN
            and self._screen is ApplicationScreen.START
            and self._seed_focused
        ):
            self._handle_seed_key(event)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)
            return not self._quit_requested
        elif event.type == pygame.MOUSEMOTION:
            self._update_wall_preview(event.pos)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._set_input_mode(InputMode.MOVE)
        elif (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_SPACE
            and self._game_mode is GameMode.RANDOM_RANDOM
            and self._screen is ApplicationScreen.PLAYING
        ):
            self._paused = not self._paused
            self._feedback = "已暂停。" if self._paused else "已继续自动播放。"
        elif (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_RIGHT
            and self._paused
        ):
            self._perform_agent_action()
        return True

    def update(self, elapsed_ms: int) -> None:
        """Advance time-dependent behavior."""
        if self._screen is not ApplicationScreen.PLAYING or self._paused:
            return
        if not self._is_agent_turn():
            return
        self._agent_elapsed_ms += max(0, elapsed_ms)
        if self._agent_elapsed_ms >= self._agent_delay_ms:
            self._agent_elapsed_ms = 0
            self._perform_agent_action()

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the current visible state to a real Pygame surface."""
        self._surface_size = surface.get_size()
        self._player_status_cards = {}
        self._wall_inventory_segments = {}
        surface.fill(COLORS["paper"])
        if self._screen is ApplicationScreen.START:
            self._draw_start(surface)
        else:
            self._draw_game(surface)
            if self._screen in (ApplicationScreen.RESULT, ApplicationScreen.ERROR):
                self._draw_result(surface)

    def _handle_click(self, point: tuple[int, int]) -> None:
        if self._screen is ApplicationScreen.START:
            self._seed_focused = False
            if self._controls.get(
                Control.MODE_HUMAN_HUMAN, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._selected_game_mode = GameMode.HUMAN_HUMAN
                self._feedback = "已选择：人类 vs 人类。"
            elif self._controls.get(
                Control.MODE_HUMAN_RANDOM, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._selected_game_mode = GameMode.HUMAN_RANDOM
                self._feedback = "已选择：人类 vs 随机智能体。"
            elif self._controls.get(
                Control.MODE_RANDOM_RANDOM, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._selected_game_mode = GameMode.RANDOM_RANDOM
                self._feedback = "已选择：随机智能体 vs 随机智能体。"
            elif self._controls.get(
                Control.HUMAN_PLAYER_0, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._human_player = Player.PLAYER_0
                self._feedback = "人类控制先手 player_0。"
            elif self._controls.get(
                Control.HUMAN_PLAYER_1, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._human_player = Player.PLAYER_1
                self._feedback = "人类控制后手 player_1。"
            elif self._controls.get(
                Control.SEED_INPUT, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._seed_focused = True
                self._feedback = "输入整数随机种子，留空则每局不同。"
            elif self._speed_control_at(point) is not None:
                self._set_speed(self._speed_control_at(point))
            elif (
                self._controls.get(
                    Control.START_GAME, pygame.Rect(0, 0, 0, 0)
                ).collidepoint(point)
                and self._selected_game_mode is not None
            ):
                try:
                    self._seed = int(self._seed_text) if self._seed_text else None
                except ValueError:
                    self._feedback = "随机种子必须是整数或留空。"
                else:
                    self._start_game()
            return

        if self._screen in (ApplicationScreen.RESULT, ApplicationScreen.ERROR):
            restart = self._controls.get(Control.RESTART_GAME)
            if restart is not None and restart.collidepoint(point):
                self._start_game()
                return
            start = self._controls.get(Control.RETURN_TO_START)
            if start is not None and start.collidepoint(point):
                self._screen = ApplicationScreen.START
                self._position = None
                self._game_mode = None
                self._feedback = "请选择对局模式并开始。"
                return
            exit_button = self._controls.get(Control.EXIT)
            if exit_button is not None and exit_button.collidepoint(point):
                self._quit_requested = True
                return

        if self._game_mode is GameMode.RANDOM_RANDOM:
            pause = self._controls.get(Control.PAUSE_RESUME)
            if pause is not None and pause.collidepoint(point):
                self._paused = not self._paused
                self._feedback = "已暂停。" if self._paused else "已继续自动播放。"
                return
            step = self._controls.get(Control.STEP)
            if step is not None and step.collidepoint(point):
                if self._paused:
                    self._perform_agent_action()
                return
            speed_control = self._speed_control_at(point)
            if speed_control is not None:
                self._set_speed(speed_control)
                return

        if (
            self._position is None
            or self._position.to_move is None
            or self._board is None
        ):
            return
        for control, mode in (
            (Control.MOVE, InputMode.MOVE),
            (Control.HORIZONTAL_WALL, InputMode.HORIZONTAL_WALL),
            (Control.VERTICAL_WALL, InputMode.VERTICAL_WALL),
        ):
            rect = self._controls.get(control)
            if rect is not None and rect.collidepoint(point):
                self._set_input_mode(mode)
                return

        if self._is_agent_turn():
            return

        if self._input_mode is not InputMode.MOVE:
            self._update_wall_preview(point)
            if self._preview_wall is not None:
                self._attempt_action(self._preview_wall)
            return

        for action in self._legal_actions:
            if isinstance(action, MovePawn) and self._board.square_rect(
                action.target
            ).collidepoint(point):
                self._attempt_action(action)
                return

    def _start_game(self) -> None:
        assert self._selected_game_mode is not None
        self._screen = ApplicationScreen.PLAYING
        self._game_mode = self._selected_game_mode
        self._position = Position.initial()
        self._legal_actions = self._position.legal_actions()
        self._preview_reason_cache = {}
        self._plies = 0
        self._input_mode = InputMode.MOVE
        self._paused = False
        self._agent_elapsed_ms = 0
        self._result_winner = None
        self._last_action = None
        self._last_move_start = None
        self._agents = {}
        if self._game_mode is GameMode.HUMAN_RANDOM:
            agent_player = Player(1 - self._human_player)
            self._agents = {agent_player: self._agent_factory(self._seed)}
        elif self._game_mode is GameMode.RANDOM_RANDOM:
            seeds = self._agent_seeds()
            self._agents = {
                Player.PLAYER_0: self._agent_factory(seeds[0]),
                Player.PLAYER_1: self._agent_factory(seeds[1]),
            }
        self._feedback = "对局开始：player_0 行动。"

    def _agent_seeds(self) -> tuple[int | None, int | None]:
        if self._seed is None:
            return None, None
        generator = random.Random(self._seed)
        return generator.randrange(2**63), generator.randrange(2**63)

    def _handle_seed_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_BACKSPACE:
            self._seed_text = self._seed_text[:-1]
        elif (
            event.unicode and event.unicode.isprintable() and len(self._seed_text) < 20
        ):
            self._seed_text += event.unicode

    def _speed_control_at(self, point: tuple[int, int]) -> Control | None:
        for control in (
            Control.SPEED_SLOW,
            Control.SPEED_NORMAL,
            Control.SPEED_FAST,
        ):
            rect = self._controls.get(control)
            if rect is not None and rect.collidepoint(point):
                return control
        return None

    def _set_speed(self, control: Control | None) -> None:
        speeds = {
            Control.SPEED_SLOW: (1000, "慢速"),
            Control.SPEED_NORMAL: (500, "正常"),
            Control.SPEED_FAST: (200, "快速"),
        }
        if control not in speeds:
            return
        self._agent_delay_ms, label = speeds[control]
        self._agent_elapsed_ms = 0
        self._feedback = f"智能体速度：{label}。"

    def _is_agent_turn(self) -> bool:
        return (
            self._position is not None
            and self._position.to_move is not None
            and self._position.to_move in self._agents
        )

    def _status_lines(self) -> tuple[str, ...]:
        if self._position is None:
            return (self._feedback,)
        input_labels = {
            InputMode.MOVE: "移动",
            InputMode.HORIZONTAL_WALL: "横墙",
            InputMode.VERTICAL_WALL: "竖墙",
        }
        return (
            f"已行动：{self._plies} 手",
            f"当前操作：{input_labels[self._input_mode]}",
            self._feedback,
        )

    def _perform_agent_action(self) -> None:
        if not self._is_agent_turn():
            return
        assert self._position is not None
        assert self._position.to_move is not None
        agent_player = self._position.to_move
        agent = self._agents[agent_player]
        try:
            action = agent.choose_action(self._position)
        except Exception as error:  # noqa: BLE001 - isolate participant failures
            self._screen = ApplicationScreen.ERROR
            self._result_winner = None
            self._feedback = f"智能体运行失败：{error}"
            return
        self._attempt_action(
            action,
            agent_player=agent_player,
        )

    def _set_input_mode(self, mode: InputMode) -> None:
        self._input_mode = mode
        self._preview_wall = None
        self._preview_reason = None
        labels = {
            InputMode.MOVE: "移动",
            InputMode.HORIZONTAL_WALL: "横墙",
            InputMode.VERTICAL_WALL: "竖墙",
        }
        self._feedback = f"当前操作：{labels[mode]}。"

    def _update_wall_preview(self, point: tuple[int, int]) -> None:
        self._preview_wall = None
        self._preview_reason = None
        if (
            self._screen is not ApplicationScreen.PLAYING
            or self._position is None
            or self._board is None
            or self._input_mode is InputMode.MOVE
        ):
            return
        anchor = self._board.nearest_anchor(point)
        if anchor is None:
            return
        orientation = (
            Orientation.HORIZONTAL
            if self._input_mode is InputMode.HORIZONTAL_WALL
            else Orientation.VERTICAL
        )
        wall = PlaceWall(anchor, orientation)
        self._preview_wall = wall
        if wall in self._legal_actions:
            return
        cached_reason = self._preview_reason_cache.get(wall)
        if cached_reason is not None:
            self._preview_reason = cached_reason
            return
        try:
            self._position.play(wall)
        except IllegalActionError as error:
            self._preview_reason = error.reason
            self._preview_reason_cache[wall] = error.reason

    def _attempt_action(
        self,
        action: Action,
        *,
        agent_player: Player | None = None,
    ) -> None:
        assert self._position is not None
        mover = self._position.to_move
        move_start = (
            self._position.pawns[mover]
            if mover is not None and isinstance(action, MovePawn)
            else None
        )
        try:
            next_position = self._position.play(action)
        except IllegalActionError as error:
            if agent_player is not None:
                self._result_winner = Player(1 - agent_player)
                self._screen = ApplicationScreen.RESULT
                self._feedback = (
                    f"player_{int(agent_player)} 提交非法动作，"
                    f"player_{int(self._result_winner)} 获胜。"
                )
                return
            self._preview_reason = error.reason
            self._feedback = f"未执行：{REASON_TEXT[error.reason]}。回合没有推进。"
            return

        self._position = next_position
        self._legal_actions = self._position.legal_actions()
        self._preview_reason_cache = {}
        self._plies += 1
        self._last_action = action
        self._last_move_start = move_start
        self._input_mode = InputMode.MOVE
        self._preview_wall = None
        self._preview_reason = None
        if isinstance(action, MovePawn):
            self._feedback = (
                f"已移动：{_human_square(action.target)}。下一回合已回到移动模式。"
            )
        else:
            label = "横墙" if action.orientation is Orientation.HORIZONTAL else "竖墙"
            self._feedback = (
                f"已放置{label}：{_human_anchor(action.anchor)}。"
                "下一回合已回到移动模式。"
            )
        if self._position.winner is not None:
            self._result_winner = self._position.winner
            self._screen = ApplicationScreen.RESULT
            self._feedback = f"player_{int(self._position.winner)} 获胜。"
        elif self._plies >= self._max_plies:
            self._screen = ApplicationScreen.RESULT
            self._feedback = f"达到 {self._max_plies} 手行动上限，本局未决。"

    def _draw_start(self, surface: pygame.Surface) -> None:
        width, height = surface.get_size()
        panel = pygame.Rect(0, 0, min(760, width - 60), min(700, height - 40))
        panel.center = (width // 2, height // 2)
        pygame.draw.rect(surface, COLORS["panel"], panel, border_radius=22)
        pygame.draw.rect(surface, COLORS["line"], panel, width=2, border_radius=22)

        title = self._font(42).render("围墙棋", True, COLORS["ink"])
        surface.blit(title, (panel.centerx - title.get_width() // 2, panel.top + 48))
        subtitle = self._font(20).render("选择本地对局模式", True, COLORS["muted"])
        surface.blit(
            subtitle,
            (panel.centerx - subtitle.get_width() // 2, panel.top + 108),
        )

        mode_gap = 12
        mode_width = (panel.width - 100 - mode_gap * 2) // 3
        human_mode = pygame.Rect(panel.left + 50, panel.top + 150, mode_width, 58)
        human_random_mode = human_mode.move(mode_width + mode_gap, 0)
        random_mode = human_random_mode.move(mode_width + mode_gap, 0)
        player_0 = pygame.Rect(panel.left + 120, panel.top + 238, 220, 48)
        player_1 = pygame.Rect(panel.right - 340, panel.top + 238, 220, 48)
        seed_input = pygame.Rect(
            panel.left + 170, panel.top + 312, panel.width - 340, 48
        )
        speed_y = panel.top + 386
        speed_width = 140
        speed_gap = 16
        speed_left = panel.centerx - (speed_width * 3 + speed_gap * 2) // 2
        speed_rects = {
            Control.SPEED_SLOW: pygame.Rect(speed_left, speed_y, speed_width, 46),
            Control.SPEED_NORMAL: pygame.Rect(
                speed_left + speed_width + speed_gap,
                speed_y,
                speed_width,
                46,
            ),
            Control.SPEED_FAST: pygame.Rect(
                speed_left + (speed_width + speed_gap) * 2,
                speed_y,
                speed_width,
                46,
            ),
        }
        start = pygame.Rect(panel.left + 150, panel.bottom - 105, panel.width - 300, 56)
        self._controls = {
            Control.MODE_HUMAN_HUMAN: human_mode,
            Control.MODE_HUMAN_RANDOM: human_random_mode,
            Control.MODE_RANDOM_RANDOM: random_mode,
            Control.HUMAN_PLAYER_0: player_0,
            Control.HUMAN_PLAYER_1: player_1,
            Control.SEED_INPUT: seed_input,
            Control.START_GAME: start,
            **speed_rects,
        }
        self._draw_button(
            surface,
            human_mode,
            "人类 vs 人类",
            selected=self._selected_game_mode is GameMode.HUMAN_HUMAN,
        )
        self._draw_button(
            surface,
            human_random_mode,
            "人类 vs 随机",
            selected=self._selected_game_mode is GameMode.HUMAN_RANDOM,
        )
        self._draw_button(
            surface,
            random_mode,
            "随机 vs 随机",
            selected=self._selected_game_mode is GameMode.RANDOM_RANDOM,
        )
        self._draw_button(
            surface,
            player_0,
            "人类先手",
            selected=self._human_player is Player.PLAYER_0,
        )
        self._draw_button(
            surface,
            player_1,
            "人类后手",
            selected=self._human_player is Player.PLAYER_1,
        )
        pygame.draw.rect(surface, COLORS["panel"], seed_input, border_radius=10)
        pygame.draw.rect(
            surface,
            COLORS["selected"] if self._seed_focused else COLORS["line"],
            seed_input,
            width=2,
            border_radius=10,
        )
        seed_label = self._font(18).render(
            self._seed_text or "随机种子（可留空）",
            True,
            COLORS["ink"] if self._seed_text else COLORS["muted"],
        )
        surface.blit(
            seed_label,
            (seed_input.left + 14, seed_input.centery - seed_label.get_height() // 2),
        )
        for control, label, delay in (
            (Control.SPEED_SLOW, "慢速", 1000),
            (Control.SPEED_NORMAL, "正常", 500),
            (Control.SPEED_FAST, "快速", 200),
        ):
            self._draw_button(
                surface,
                speed_rects[control],
                label,
                selected=self._agent_delay_ms == delay,
            )
        self._draw_button(surface, start, "开始对局", selected=False)
        feedback = self._font(17).render(self._feedback, True, COLORS["muted"])
        surface.blit(feedback, (panel.left + 70, panel.bottom - 155))

    def _draw_game(self, surface: pygame.Surface) -> None:
        assert self._position is not None
        width, height = surface.get_size()
        margin = max(16, round(min(width, height) * 0.025))
        header_height = max(56, round(height * 0.08))
        content_height = height - header_height - margin * 2
        board_side = min(round((width - margin * 3) * 0.73), content_height)
        board_rect = pygame.Rect(margin, header_height + margin, board_side, board_side)
        self._board = BoardGeometry.from_rect(board_rect)

        title = self._font(30).render("围墙棋", True, COLORS["ink"])
        surface.blit(title, (margin, max(8, (header_height - title.get_height()) // 2)))
        pygame.draw.rect(surface, COLORS["board_dark"], board_rect, border_radius=18)

        legal_targets = {
            action.target
            for action in self._legal_actions
            if isinstance(action, MovePawn)
        }
        for row in range(9):
            for col in range(9):
                square = Square(row, col)
                rect = self._board.square_rect(square)
                pygame.draw.rect(surface, COLORS["board"], rect, border_radius=5)
                if square in legal_targets:
                    pygame.draw.circle(
                        surface,
                        COLORS["valid"],
                        rect.center,
                        max(5, round(rect.width * 0.13)),
                    )

        coordinate_font = self._font(max(13, round(self._board.cell * 0.25)))
        for col, label in enumerate("abcdefghi"):
            text = coordinate_font.render(label, True, COLORS["panel"])
            cell = self._board.square_rect(Square(8, col))
            surface.blit(
                text,
                (cell.centerx - text.get_width() // 2, cell.bottom + 3),
            )
        for row in range(9):
            text = coordinate_font.render(str(9 - row), True, COLORS["panel"])
            cell = self._board.square_rect(Square(row, 0))
            surface.blit(
                text,
                (
                    cell.left - text.get_width() - 5,
                    cell.centery - text.get_height() // 2,
                ),
            )

        if (
            isinstance(self._last_action, MovePawn)
            and self._last_move_start is not None
        ):
            start_rect = self._board.square_rect(self._last_move_start)
            target_rect = self._board.square_rect(self._last_action.target)
            pygame.draw.line(
                surface,
                COLORS["focus"],
                start_rect.center,
                target_rect.center,
                width=max(3, round(self._board.gap * 0.45)),
            )
            pygame.draw.circle(
                surface,
                COLORS["focus"],
                start_rect.center,
                max(6, round(start_rect.width * 0.18)),
                width=3,
            )
            pygame.draw.circle(
                surface,
                COLORS["focus"],
                target_rect.center,
                max(8, round(target_rect.width * 0.38)),
                width=3,
            )

        for player in Player:
            square = self._position.pawns[player]
            rect = self._board.square_rect(square)
            pygame.draw.circle(
                surface,
                _player_color(player),
                rect.center,
                max(8, round(rect.width * 0.31)),
            )
        for player in Player:
            for wall in self._position.placed_walls_by_player[player]:
                wall_rect = self._board.wall_rect(wall)
                pygame.draw.rect(
                    surface,
                    _player_color(player),
                    wall_rect,
                    border_radius=3,
                )
                if wall == self._last_action:
                    pygame.draw.rect(
                        surface,
                        COLORS["focus"],
                        wall_rect.inflate(6, 6),
                        width=3,
                        border_radius=5,
                    )
        if self._preview_wall is not None:
            preview = self._board.wall_rect(self._preview_wall)
            preview_color = (
                _player_color(self._position.to_move)
                if self._preview_reason is None and self._position.to_move is not None
                else COLORS["invalid"]
            )
            pygame.draw.rect(surface, preview_color, preview, border_radius=3)
            if self._preview_reason is None:
                pygame.draw.rect(
                    surface,
                    COLORS["white"],
                    preview,
                    width=2,
                    border_radius=3,
                )
            else:
                pygame.draw.line(
                    surface,
                    COLORS["white"],
                    preview.topleft,
                    preview.bottomright,
                    width=max(2, round(self._board.gap * 0.3)),
                )
                pygame.draw.line(
                    surface,
                    COLORS["white"],
                    preview.topright,
                    preview.bottomleft,
                    width=max(2, round(self._board.gap * 0.3)),
                )

        sidebar = pygame.Rect(
            board_rect.right + margin,
            board_rect.top,
            width - board_rect.right - margin * 2,
            board_rect.height,
        )
        pygame.draw.rect(surface, COLORS["panel"], sidebar, border_radius=16)
        cards_bottom = self._draw_player_status_cards(surface, sidebar)
        button_gap = 8
        button_width = max(76, (sidebar.width - 44 - button_gap * 2) // 3)
        button_y = cards_bottom + 22
        self._controls = {}
        if self._game_mode is GameMode.RANDOM_RANDOM:
            controls_bottom = self._draw_playback_controls(
                surface,
                sidebar,
                button_y,
            )
        else:
            mode_controls = (
                (Control.MOVE, "移动", InputMode.MOVE),
                (Control.HORIZONTAL_WALL, "横墙", InputMode.HORIZONTAL_WALL),
                (Control.VERTICAL_WALL, "竖墙", InputMode.VERTICAL_WALL),
            )
            for index, (control, label, mode) in enumerate(mode_controls):
                rect = pygame.Rect(
                    sidebar.left + 22 + index * (button_width + button_gap),
                    button_y,
                    button_width,
                    46,
                )
                self._controls[control] = rect
                self._draw_button(
                    surface,
                    rect,
                    label,
                    selected=self._input_mode is mode,
                )
            controls_bottom = button_y + 46
        status_top = controls_bottom + 20
        for index, line in enumerate(self._status_lines()):
            text = self._font(16 if index < 2 else 15).render(
                line,
                True,
                COLORS["ink"] if index < 2 else COLORS["muted"],
            )
            surface.blit(text, (sidebar.left + 22, status_top + index * 31))

    def _draw_player_status_cards(
        self,
        surface: pygame.Surface,
        sidebar: pygame.Rect,
    ) -> int:
        assert self._position is not None
        gap = 10
        left = sidebar.left + 22
        top = sidebar.top + 22
        card_width = (sidebar.width - 44 - gap) // 2
        card_height = 122

        for player in Player:
            card = pygame.Rect(
                left + int(player) * (card_width + gap),
                top,
                card_width,
                card_height,
            )
            self._player_status_cards[player] = card
            player_color = _player_color(player)
            is_active = self._position.to_move is player
            pygame.draw.rect(surface, COLORS["paper"], card, border_radius=12)
            pygame.draw.rect(
                surface,
                player_color if is_active else COLORS["line"],
                card,
                width=3 if is_active else 1,
                border_radius=12,
            )

            marker_center = (card.left + 13, card.top + 17)
            pygame.draw.circle(surface, player_color, marker_center, 5)
            identity = self._font(15).render(
                f"player_{int(player)}",
                True,
                COLORS["ink"],
            )
            surface.blit(identity, (card.left + 23, card.top + 7))
            if is_active:
                active = self._font(12).render("行动中", True, player_color)
                surface.blit(
                    active, (card.right - active.get_width() - 10, card.top + 9)
                )

            label = self._font(13).render("剩余", True, COLORS["muted"])
            surface.blit(label, (card.left + 11, card.top + 43))
            remaining = self._position.walls_remaining[player]
            count = self._font(22).render(
                f"{remaining} / 10",
                True,
                player_color,
            )
            surface.blit(count, (card.right - count.get_width() - 10, card.top + 34))

            segment_gap = 2
            segment_left = card.left + 10
            segment_width = max(
                3,
                (card.width - 20 - segment_gap * 9) // 10,
            )
            segment_top = card.bottom - 28
            segments = tuple(
                pygame.Rect(
                    segment_left + index * (segment_width + segment_gap),
                    segment_top,
                    segment_width,
                    14,
                )
                for index in range(10)
            )
            self._wall_inventory_segments[player] = segments
            for index, segment in enumerate(segments):
                if index < remaining:
                    pygame.draw.rect(
                        surface,
                        player_color,
                        segment,
                        border_radius=2,
                    )
                else:
                    pygame.draw.rect(
                        surface,
                        COLORS["line"],
                        segment,
                        width=1,
                        border_radius=2,
                    )

        return top + card_height

    def _draw_playback_controls(
        self,
        surface: pygame.Surface,
        sidebar: pygame.Rect,
        top: int,
    ) -> int:
        gap = 10
        primary_width = (sidebar.width - 54) // 2
        pause = pygame.Rect(sidebar.left + 22, top, primary_width, 46)
        step = pause.move(primary_width + gap, 0)
        self._controls[Control.PAUSE_RESUME] = pause
        self._controls[Control.STEP] = step
        self._draw_button(
            surface,
            pause,
            "继续" if self._paused else "暂停",
            selected=self._paused,
        )
        self._draw_button(surface, step, "单步", selected=False)

        speed_width = (sidebar.width - 64) // 3
        speed_top = top + 60
        for index, (control, label, delay) in enumerate(
            (
                (Control.SPEED_SLOW, "慢", 1000),
                (Control.SPEED_NORMAL, "正常", 500),
                (Control.SPEED_FAST, "快", 200),
            )
        ):
            rect = pygame.Rect(
                sidebar.left + 22 + index * (speed_width + gap),
                speed_top,
                speed_width,
                42,
            )
            self._controls[control] = rect
            self._draw_button(
                surface,
                rect,
                label,
                selected=self._agent_delay_ms == delay,
            )
        return speed_top + 42

    def _draw_button(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        *,
        selected: bool,
    ) -> None:
        color = COLORS["selected"] if selected else COLORS["panel"]
        text_color = COLORS["white"] if selected else COLORS["ink"]
        pygame.draw.rect(surface, color, rect, border_radius=12)
        pygame.draw.rect(surface, COLORS["line"], rect, width=2, border_radius=12)
        text = self._font(20).render(label, True, text_color)
        surface.blit(
            text,
            (
                rect.centerx - text.get_width() // 2,
                rect.centery - text.get_height() // 2,
            ),
        )

    def _draw_result(self, surface: pygame.Surface) -> None:
        width, height = surface.get_size()
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((23, 33, 28, 150))
        surface.blit(overlay, (0, 0))
        panel = pygame.Rect(0, 0, min(560, width - 80), min(330, height - 80))
        panel.center = (width // 2, height // 2)
        pygame.draw.rect(surface, COLORS["panel"], panel, border_radius=20)
        heading = "对局中止" if self._screen is ApplicationScreen.ERROR else "对局结束"
        title = self._font(34).render(heading, True, COLORS["ink"])
        surface.blit(title, (panel.centerx - title.get_width() // 2, panel.top + 40))
        result = self._font(22).render(self._feedback, True, COLORS["muted"])
        surface.blit(result, (panel.centerx - result.get_width() // 2, panel.top + 94))

        button_width = 150
        gap = 14
        total = button_width * 3 + gap * 2
        left = panel.centerx - total // 2
        controls = (
            (Control.RESTART_GAME, "再来一局"),
            (Control.RETURN_TO_START, "返回开始"),
            (Control.EXIT, "退出"),
        )
        self._controls = {}
        for index, (control, label) in enumerate(controls):
            rect = pygame.Rect(
                left + index * (button_width + gap),
                panel.bottom - 100,
                button_width,
                52,
            )
            self._controls[control] = rect
            self._draw_button(surface, rect, label, selected=False)

    def _font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(io.BytesIO(self._font_bytes), size)
        return self._fonts[size]


def _player_color(player: Player) -> pygame.Color:
    return COLORS["p0"] if player is Player.PLAYER_0 else COLORS["p1"]


def _human_square(square: Square) -> str:
    return f"{chr(ord('a') + square.col)}{9 - square.row}"


def _human_anchor(anchor: WallAnchor) -> str:
    return f"{chr(ord('a') + anchor.col)}{8 - anchor.row}"


def run() -> int:
    """Run the interactive desktop event loop."""
    pygame.init()
    try:
        surface = pygame.display.set_mode(WINDOW_INITIAL, pygame.RESIZABLE)
        pygame.display.set_caption("围墙棋")
        clock = pygame.time.Clock()
        application = PygameApplication()
        running = True
        while running:
            elapsed_ms = clock.tick(60)
            for event in pygame.event.get():
                running = application.handle_event(event) and running
            if application.snapshot.window_size != surface.get_size():
                surface = pygame.display.set_mode(
                    application.snapshot.window_size,
                    pygame.RESIZABLE,
                )
            application.update(elapsed_ms)
            application.draw(surface)
            pygame.display.flip()
        return 0
    finally:
        pygame.quit()

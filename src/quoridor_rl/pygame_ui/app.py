"""用于本地围墙棋对局的 Pygame 桌面应用。

应用把规则状态、输入状态和绘制几何分开：所有规则判断委托给 ``Position``，本模块
只维护页面切换、参与者调度、历史回放以及响应式绘制。公开快照和语义控件矩形构成
自动化测试边界，测试无需依赖像素颜色猜测应用状态。
"""

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
from quoridor_rl.language import Language

WINDOW_INITIAL = (1280, 800)
WINDOW_MIN = (960, 640)
HISTORY_ROW_HEIGHT = 34

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

PYGAME_TEXT = {
    Language.CHINESE: {
        "window_title": "围墙棋",
        "start_prompt": "请选择对局模式并开始。",
        "paused": "已暂停。",
        "resumed": "已继续自动播放。",
        "selected_human_human": "已选择：人类 vs 人类。",
        "selected_human_random": "已选择：人类 vs 随机智能体。",
        "selected_random_random": "已选择：随机智能体 vs 随机智能体。",
        "human_first": "人类控制先手 player_0。",
        "human_second": "人类控制后手 player_1。",
        "seed_help": "输入整数随机种子，留空则每局不同。",
        "seed_error": "随机种子必须是整数或留空。",
        "game_started": "对局开始：player_0 行动。",
        "speed_slow": "慢速",
        "speed_normal": "正常",
        "speed_fast": "快速",
        "speed_feedback": "智能体速度：{speed}。",
        "history_status": "历史回放：第 {ply} / {plies} 手",
        "history_hint": "滚轮浏览，点击记录切换局面。",
        "mode_move": "移动",
        "mode_horizontal_wall": "横墙",
        "mode_vertical_wall": "竖墙",
        "plies": "已行动：{plies} 手",
        "current_operation": "当前操作：{mode}",
        "agent_error": "智能体运行失败：{error}",
        "illegal_agent": "{player} 提交非法动作，{winner} 获胜。",
        "illegal_human": "未执行：{reason}。回合没有推进。",
        "moved": "已移动：{target}。下一回合已回到移动模式。",
        "horizontal_wall": "横墙",
        "vertical_wall": "竖墙",
        "placed_wall": "已放置{wall}：{anchor}。下一回合已回到移动模式。",
        "winner": "{player} 获胜。",
        "limit": "达到 {max_plies} 手行动上限，本局未决。",
        "title": "围墙棋",
        "subtitle": "选择本地对局模式",
        "human_human": "人类 vs 人类",
        "human_random": "人类 vs 随机",
        "random_random": "随机 vs 随机",
        "human_first_label": "人类先手",
        "human_second_label": "人类后手",
        "seed_placeholder": "随机种子（可留空）",
        "slow_label": "慢速",
        "normal_label": "正常",
        "fast_label": "快速",
        "start_game": "开始对局",
        "return_result": "返回结果",
        "active": "行动中",
        "remaining": "剩余",
        "history_heading": "行动记录（最新在上）",
        "resume": "继续",
        "pause": "暂停",
        "step": "单步",
        "slow_short": "慢",
        "normal_short": "正常",
        "fast_short": "快",
        "game_aborted": "对局中止",
        "game_over": "对局结束",
        "view_history": "查看回放",
        "restart": "再来一局",
        "return_start": "返回开始",
        "exit": "退出",
        "history_initial": "0. 初始局面",
        "history_move": "{ply}. {player} 移动 {start} → {target}",
        "history_wall": "{ply}. {player} 放置{wall} {anchor}",
        "reason_game_over": "对局已经结束",
        "reason_illegal_pawn_move": "棋子不能移动到该格",
        "reason_no_walls_remaining": "当前行动方已经没有墙",
        "reason_wall_conflict": "与已有墙重叠或交叉",
        "reason_wall_blocks_path": "会堵死至少一方的所有路径",
    },
    Language.ENGLISH: {
        "window_title": "Quoridor",
        "start_prompt": "Choose a game mode to begin.",
        "paused": "Paused.",
        "resumed": "Automatic playback resumed.",
        "selected_human_human": "Selected: Human vs Human.",
        "selected_human_random": "Selected: Human vs Random Agent.",
        "selected_random_random": "Selected: Random Agent vs Random Agent.",
        "human_first": "Human controls the first player, player_0.",
        "human_second": "Human controls the second player, player_1.",
        "seed_help": "Enter an integer seed, or leave blank for a different game.",
        "seed_error": "The random seed must be an integer or blank.",
        "game_started": "Game started: player_0 to move.",
        "speed_slow": "Slow",
        "speed_normal": "Normal",
        "speed_fast": "Fast",
        "speed_feedback": "Agent speed: {speed}.",
        "history_status": "History: ply {ply} / {plies}",
        "history_hint": "Scroll to browse; click an entry to view it.",
        "mode_move": "Move",
        "mode_horizontal_wall": "Horizontal",
        "mode_vertical_wall": "Vertical",
        "plies": "Plies: {plies}",
        "current_operation": "Action: {mode}",
        "agent_error": "Agent failed: {error}",
        "illegal_agent": "{player} submitted an illegal action; {winner} wins.",
        "illegal_human": "Not played: {reason}. The turn did not advance.",
        "moved": "Moved to {target}. Move mode restored for the next turn.",
        "horizontal_wall": "horizontal wall",
        "vertical_wall": "vertical wall",
        "placed_wall": "Placed {wall} at {anchor}. Move mode restored for the next turn.",
        "winner": "{player} wins.",
        "limit": "Reached the {max_plies}-ply limit; game undecided.",
        "title": "Quoridor",
        "subtitle": "Choose a local game mode",
        "human_human": "Human vs Human",
        "human_random": "Human vs Random",
        "random_random": "Random vs Random",
        "human_first_label": "Human First",
        "human_second_label": "Human Second",
        "seed_placeholder": "Random seed (optional)",
        "slow_label": "Slow",
        "normal_label": "Normal",
        "fast_label": "Fast",
        "start_game": "Start Game",
        "return_result": "Back to Result",
        "active": "Active",
        "remaining": "Walls",
        "history_heading": "Move history (newest first)",
        "resume": "Resume",
        "pause": "Pause",
        "step": "Step",
        "slow_short": "Slow",
        "normal_short": "Normal",
        "fast_short": "Fast",
        "game_aborted": "Game Aborted",
        "game_over": "Game Over",
        "view_history": "View History",
        "restart": "Play Again",
        "return_start": "Start Screen",
        "exit": "Exit",
        "history_initial": "0. Initial position",
        "history_move": "{ply}. {player} moved {start} → {target}",
        "history_wall": "{ply}. {player} placed {wall} at {anchor}",
        "reason_game_over": "the game has already ended",
        "reason_illegal_pawn_move": "the pawn cannot move to that square",
        "reason_no_walls_remaining": "the current player has no walls remaining",
        "reason_wall_conflict": "the wall overlaps or crosses an existing wall",
        "reason_wall_blocks_path": "the wall would block every path for a player",
    },
}

assert PYGAME_TEXT[Language.CHINESE].keys() == PYGAME_TEXT[Language.ENGLISH].keys()

REASON_KEYS = {
    IllegalActionReason.GAME_OVER: "reason_game_over",
    IllegalActionReason.ILLEGAL_PAWN_MOVE: "reason_illegal_pawn_move",
    IllegalActionReason.NO_WALLS_REMAINING: "reason_no_walls_remaining",
    IllegalActionReason.WALL_CONFLICT: "reason_wall_conflict",
    IllegalActionReason.WALL_BLOCKS_PATH: "reason_wall_blocks_path",
}


class ApplicationScreen(Enum):
    """桌面应用对用户可见的顶层页面。"""

    START = "start"
    PLAYING = "playing"
    RESULT = "result"
    ERROR = "error"


class GameMode(Enum):
    """本地参与者的受支持组合。"""

    HUMAN_HUMAN = "human_human"
    HUMAN_RANDOM = "human_random"
    RANDOM_RANDOM = "random_random"


class Control(Enum):
    """在应用测试边界暴露的语义控件标识。"""

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
    VIEW_HISTORY = "view_history"
    RETURN_TO_RESULT = "return_to_result"
    LANGUAGE_CHINESE = "language_chinese"
    LANGUAGE_ENGLISH = "language_english"


class InputMode(Enum):
    """当前人类玩家准备执行的动作类型。"""

    MOVE = "move"
    HORIZONTAL_WALL = "horizontal_wall"
    VERTICAL_WALL = "vertical_wall"


@dataclass(frozen=True, slots=True)
class ActionHistoryEntry:
    """桌面对局的一条历史记录；第零条表示无动作的初始局面。"""

    ply: int
    player: Player | None
    action: Action | None
    resulting_position: Position
    move_start: Square | None


@dataclass(frozen=True, slots=True)
class ApplicationSnapshot:
    """通过应用测试边界暴露的只读可观察状态。"""

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
    action_history: tuple[ActionHistoryEntry, ...]
    reviewing_history: bool
    reviewed_ply: int | None
    displayed_position: Position | None
    language: Language
    window_title: str


class ActionChoosingAgent(Protocol):
    """内置及未来智能体共同遵守的最小语义接口。"""

    def choose_action(self, position: Position) -> Action:
        """根据当前不可变局面选择一个语义动作。"""
        ...


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    """把规则层棋盘坐标映射到当前窗口中的像素几何。"""

    rect: pygame.Rect
    cell: float
    gap: float
    origin_x: float
    origin_y: float

    @classmethod
    def from_rect(cls, rect: pygame.Rect) -> BoardGeometry:
        """根据可用正方形区域计算格子、间隙和原点尺寸。"""
        inset = max(18.0, rect.width * 0.047)
        usable = rect.width - inset * 2
        gap_ratio = 0.18
        cell = usable / (9 + 8 * gap_ratio)
        return cls(rect, cell, cell * gap_ratio, rect.left + inset, rect.top + inset)

    @property
    def pitch(self) -> float:
        """返回相邻格左上角之间的像素距离。"""
        return self.cell + self.gap

    def square_rect(self, square: Square) -> pygame.Rect:
        """返回指定棋格的像素矩形。"""
        return pygame.Rect(
            round(self.origin_x + square.col * self.pitch),
            round(self.origin_y + square.row * self.pitch),
            round(self.cell),
            round(self.cell),
        )

    def anchor_center(self, anchor: WallAnchor) -> tuple[float, float]:
        """返回墙锚点在格间交叉处的像素中心。"""
        return (
            self.origin_x + anchor.col * self.pitch + self.cell + self.gap / 2,
            self.origin_y + anchor.row * self.pitch + self.cell + self.gap / 2,
        )

    def wall_rect(self, wall: PlaceWall) -> pygame.Rect:
        """返回一堵横墙或竖墙跨越两个格边的像素矩形。"""
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
        """把鼠标位置吸附到最近墙锚点；棋盘交互区外返回 ``None``。"""
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
    """管理桌面应用的交互状态、智能体调度与绘制生命周期。"""

    def __init__(
        self,
        *,
        agent_factory: Callable[[int | None], ActionChoosingAgent] = RandomAgent,
        max_plies: int = 512,
        language: Language = Language.CHINESE,
    ) -> None:
        """加载字体资源并初始化尚未开始对局的应用状态。"""
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        if not isinstance(language, Language):
            raise TypeError("language must be a Language value")
        pygame.font.init()
        self._language = language
        self._max_plies = max_plies
        self._font_resource = "NotoSansSC-Regular.otf"
        self._font_bytes = (
            resources.files("quoridor_rl.pygame_ui.assets")
            .joinpath(self._font_resource)
            .read_bytes()
        )
        self._screen = ApplicationScreen.START
        self._feedback = self._text("start_prompt")
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
        self._action_history: list[ActionHistoryEntry] = []
        self._reviewing_history = False
        self._reviewed_ply: int | None = None
        self._history_scroll_rows = 0
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
        self._history_entry_rects: dict[int, pygame.Rect] = {}
        self._history_visible_rows = 0

    @property
    def snapshot(self) -> ApplicationSnapshot:
        """返回不暴露内部可变容器的完整应用快照。"""
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
            action_history=tuple(self._action_history),
            reviewing_history=self._reviewing_history,
            reviewed_ply=self._reviewed_ply,
            displayed_position=self._displayed_position(),
            language=self._language,
            window_title=self._text("window_title"),
        )

    def _text(self, key: str, **values: object) -> str:
        """Return one localized Pygame message."""
        return PYGAME_TEXT[self._language][key].format(**values)

    def control_rect(self, control: Control) -> pygame.Rect:
        """返回语义控件当前可见矩形的副本。"""
        return self._controls[control].copy()

    def square_rect(self, square: Square) -> pygame.Rect:
        """返回语义棋格当前可见的矩形。"""
        if self._board is None:
            raise RuntimeError("the playing board has not been drawn")
        return self._board.square_rect(square)

    def wall_anchor_point(self, anchor: WallAnchor) -> tuple[int, int]:
        """返回语义墙锚点在屏幕上的吸附点。"""
        if self._board is None:
            raise RuntimeError("the playing board has not been drawn")
        x, y = self._board.anchor_center(anchor)
        return round(x), round(y)

    def player_status_rect(self, player: Player) -> pygame.Rect:
        """返回指定玩家的可见状态卡矩形。"""
        return self._player_status_cards[player].copy()

    def wall_inventory_segment_rect(
        self,
        player: Player,
        index: int,
    ) -> pygame.Rect:
        """返回玩家十段剩余墙库存中的一个可见矩形。"""
        return self._wall_inventory_segments[player][index].copy()

    def history_entry_rect(self, ply: int) -> pygame.Rect:
        """返回指定手数历史记录的可见矩形。"""
        return self._history_entry_rects[ply].copy()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """处理一个真实 Pygame 事件，并返回主循环是否应继续。"""
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
        if event.type == pygame.MOUSEWHEEL:
            self._scroll_history(event.y)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)
            return not self._quit_requested
        elif event.type == pygame.MOUSEMOTION:
            self._update_wall_preview(event.pos)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self._reviewing_history:
                self._leave_history_review()
            else:
                self._set_input_mode(InputMode.MOVE)
        elif (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_SPACE
            and self._game_mode is GameMode.RANDOM_RANDOM
            and self._screen is ApplicationScreen.PLAYING
        ):
            self._paused = not self._paused
            self._feedback = self._text("paused" if self._paused else "resumed")
        elif (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_RIGHT
            and self._paused
        ):
            self._perform_agent_action()
        return True

    def update(self, elapsed_ms: int) -> None:
        """推进与时间有关的行为，在延迟到期时触发智能体行动。"""
        if self._screen is not ApplicationScreen.PLAYING or self._paused:
            return
        if not self._is_agent_turn():
            return
        self._agent_elapsed_ms += max(0, elapsed_ms)
        if self._agent_elapsed_ms >= self._agent_delay_ms:
            self._agent_elapsed_ms = 0
            self._perform_agent_action()

    def draw(self, surface: pygame.Surface) -> None:
        """把当前应用状态绘制到真实 Pygame 画布，并刷新命中测试几何。"""
        self._surface_size = surface.get_size()
        self._player_status_cards = {}
        self._wall_inventory_segments = {}
        self._history_entry_rects = {}
        self._history_visible_rows = 0
        surface.fill(COLORS["paper"])
        if self._screen is ApplicationScreen.START:
            self._draw_start(surface)
        else:
            self._draw_game(surface)
            if (
                self._screen in (ApplicationScreen.RESULT, ApplicationScreen.ERROR)
                and not self._reviewing_history
            ):
                self._draw_result(surface)

    def _handle_click(self, point: tuple[int, int]) -> None:
        """按当前页面和交互优先级分派一次左键点击。"""
        if self._screen is ApplicationScreen.START:
            self._seed_focused = False
            if self._controls.get(
                Control.LANGUAGE_CHINESE, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._set_language(Language.CHINESE)
            elif self._controls.get(
                Control.LANGUAGE_ENGLISH, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._set_language(Language.ENGLISH)
            elif self._controls.get(
                Control.MODE_HUMAN_HUMAN, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._selected_game_mode = GameMode.HUMAN_HUMAN
                self._feedback = self._text("selected_human_human")
            elif self._controls.get(
                Control.MODE_HUMAN_RANDOM, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._selected_game_mode = GameMode.HUMAN_RANDOM
                self._feedback = self._text("selected_human_random")
            elif self._controls.get(
                Control.MODE_RANDOM_RANDOM, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._selected_game_mode = GameMode.RANDOM_RANDOM
                self._feedback = self._text("selected_random_random")
            elif self._controls.get(
                Control.HUMAN_PLAYER_0, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._human_player = Player.PLAYER_0
                self._feedback = self._text("human_first")
            elif self._controls.get(
                Control.HUMAN_PLAYER_1, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._human_player = Player.PLAYER_1
                self._feedback = self._text("human_second")
            elif self._controls.get(
                Control.SEED_INPUT, pygame.Rect(0, 0, 0, 0)
            ).collidepoint(point):
                self._seed_focused = True
                self._feedback = self._text("seed_help")
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
                    self._feedback = self._text("seed_error")
                else:
                    self._start_game()
            return

        if self._reviewing_history:
            return_to_result = self._controls.get(Control.RETURN_TO_RESULT)
            if return_to_result is not None and return_to_result.collidepoint(point):
                self._leave_history_review()
                return
            for ply, history_rect in self._history_entry_rects.items():
                if history_rect.collidepoint(point):
                    self._reviewed_ply = ply
                    return
            return

        if self._screen in (ApplicationScreen.RESULT, ApplicationScreen.ERROR):
            view_history = self._controls.get(Control.VIEW_HISTORY)
            if (
                self._screen is ApplicationScreen.RESULT
                and view_history is not None
                and view_history.collidepoint(point)
            ):
                self._reviewing_history = True
                self._reviewed_ply = self._plies
                self._history_scroll_rows = 0
                return
            restart = self._controls.get(Control.RESTART_GAME)
            if restart is not None and restart.collidepoint(point):
                self._start_game()
                return
            start = self._controls.get(Control.RETURN_TO_START)
            if start is not None and start.collidepoint(point):
                self._screen = ApplicationScreen.START
                self._position = None
                self._game_mode = None
                self._action_history = []
                self._reviewing_history = False
                self._reviewed_ply = None
                self._history_scroll_rows = 0
                self._feedback = self._text("start_prompt")
                return
            exit_button = self._controls.get(Control.EXIT)
            if exit_button is not None and exit_button.collidepoint(point):
                self._quit_requested = True
                return

        if self._game_mode is GameMode.RANDOM_RANDOM:
            pause = self._controls.get(Control.PAUSE_RESUME)
            if pause is not None and pause.collidepoint(point):
                self._paused = not self._paused
                self._feedback = self._text("paused" if self._paused else "resumed")
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
            control_rect = self._controls.get(control)
            if control_rect is not None and control_rect.collidepoint(point):
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
        """按已选模式创建全新局面、参与智能体和第零条历史记录。"""
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
        self._action_history = [
            ActionHistoryEntry(
                ply=0,
                player=None,
                action=None,
                resulting_position=self._position,
                move_start=None,
            )
        ]
        self._reviewing_history = False
        self._reviewed_ply = None
        self._history_scroll_rows = 0
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
        self._feedback = self._text("game_started")

    def _set_language(self, language: Language) -> None:
        """Switch the start screen language without changing game settings."""
        self._language = language
        self._feedback = self._text("start_prompt")

    def _agent_seeds(self) -> tuple[int | None, int | None]:
        """从局级种子稳定派生两个互不相同的智能体种子。"""
        if self._seed is None:
            return None, None
        generator = random.Random(self._seed)
        return generator.randrange(2**63), generator.randrange(2**63)

    def _handle_seed_key(self, event: pygame.event.Event) -> None:
        """编辑开始页的种子文本，最终整数校验留给开始按钮。"""
        if event.key == pygame.K_BACKSPACE:
            self._seed_text = self._seed_text[:-1]
        elif (
            event.unicode and event.unicode.isprintable() and len(self._seed_text) < 20
        ):
            self._seed_text += event.unicode

    def _speed_control_at(self, point: tuple[int, int]) -> Control | None:
        """返回鼠标位置命中的自动播放速度控件。"""
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
        """设置自动智能体的行动间隔并重置当前计时。"""
        speeds = {
            Control.SPEED_SLOW: (1000, "speed_slow"),
            Control.SPEED_NORMAL: (500, "speed_normal"),
            Control.SPEED_FAST: (200, "speed_fast"),
        }
        if control not in speeds:
            return
        self._agent_delay_ms, label_key = speeds[control]
        self._agent_elapsed_ms = 0
        self._feedback = self._text(
            "speed_feedback",
            speed=self._text(label_key),
        )

    def _is_agent_turn(self) -> bool:
        """判断当前行动玩家是否由已注册智能体控制。"""
        return (
            self._position is not None
            and self._position.to_move is not None
            and self._position.to_move in self._agents
        )

    def _status_lines(self) -> tuple[str, ...]:
        """根据普通对局或历史回放状态生成状态栏文本。"""
        if self._position is None:
            return (self._feedback,)
        if self._reviewing_history:
            entry = self._reviewed_history_entry()
            assert entry is not None
            return (
                self._text("history_status", ply=entry.ply, plies=self._plies),
                _action_history_text(entry, language=self._language),
                self._text("history_hint"),
            )
        input_labels = {
            InputMode.MOVE: self._text("mode_move"),
            InputMode.HORIZONTAL_WALL: self._text("mode_horizontal_wall"),
            InputMode.VERTICAL_WALL: self._text("mode_vertical_wall"),
        }
        return (
            self._text("plies", plies=self._plies),
            self._text("current_operation", mode=input_labels[self._input_mode]),
            self._feedback,
        )

    def _reviewed_history_entry(self) -> ActionHistoryEntry | None:
        """返回回放游标指向的历史项；未回放时返回 ``None``。"""
        if not self._reviewing_history or self._reviewed_ply is None:
            return None
        return self._action_history[self._reviewed_ply]

    def _displayed_position(self) -> Position | None:
        """返回应绘制的实时局面或历史局面。"""
        entry = self._reviewed_history_entry()
        return entry.resulting_position if entry is not None else self._position

    def _leave_history_review(self) -> None:
        """退出历史回放并清除回放游标和滚动量。"""
        self._reviewing_history = False
        self._reviewed_ply = None
        self._history_scroll_rows = 0

    def _scroll_history(self, wheel_y: int) -> None:
        """在可见历史范围内处理鼠标滚轮位移。"""
        if not (self._screen is ApplicationScreen.PLAYING or self._reviewing_history):
            return
        maximum = max(0, len(self._action_history) - self._history_visible_rows)
        self._history_scroll_rows = max(
            0,
            min(maximum, self._history_scroll_rows - wheel_y),
        )

    def _perform_agent_action(self) -> None:
        """隔离智能体异常并尝试提交其选择的语义动作。"""
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
            self._feedback = self._text("agent_error", error=error)
            return
        self._attempt_action(
            action,
            agent_player=agent_player,
        )

    def _set_input_mode(self, mode: InputMode) -> None:
        """切换人类输入意图，同时清除旧的放墙预览。"""
        self._input_mode = mode
        self._preview_wall = None
        self._preview_reason = None
        labels = {
            InputMode.MOVE: self._text("mode_move"),
            InputMode.HORIZONTAL_WALL: self._text("mode_horizontal_wall"),
            InputMode.VERTICAL_WALL: self._text("mode_vertical_wall"),
        }
        punctuation = "。" if self._language is Language.CHINESE else "."
        self._feedback = (
            self._text("current_operation", mode=labels[mode]) + punctuation
        )

    def _update_wall_preview(self, point: tuple[int, int]) -> None:
        """更新吸附墙预览，并缓存非法原因以避免每帧重复寻路。"""
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
        """执行人类或智能体动作，并原子地推进局面、历史和结果状态。"""
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
                self._feedback = self._text(
                    "illegal_agent",
                    player=f"player_{int(agent_player)}",
                    winner=f"player_{int(self._result_winner)}",
                )
                return
            self._preview_reason = error.reason
            self._feedback = self._text(
                "illegal_human",
                reason=self._text(REASON_KEYS[error.reason]),
            )
            return

        self._position = next_position
        self._legal_actions = self._position.legal_actions()
        self._preview_reason_cache = {}
        self._plies += 1
        self._last_action = action
        self._last_move_start = move_start
        assert mover is not None
        self._action_history.append(
            ActionHistoryEntry(
                ply=self._plies,
                player=mover,
                action=action,
                resulting_position=self._position,
                move_start=move_start,
            )
        )
        self._history_scroll_rows = 0
        self._input_mode = InputMode.MOVE
        self._preview_wall = None
        self._preview_reason = None
        if isinstance(action, MovePawn):
            self._feedback = self._text(
                "moved",
                target=_human_square(action.target),
            )
        else:
            label = self._text(
                "horizontal_wall"
                if action.orientation is Orientation.HORIZONTAL
                else "vertical_wall"
            )
            self._feedback = self._text(
                "placed_wall",
                wall=label,
                anchor=_human_anchor(action.anchor),
            )
        if self._position.winner is not None:
            self._result_winner = self._position.winner
            self._screen = ApplicationScreen.RESULT
            self._feedback = self._text(
                "winner",
                player=f"player_{int(self._position.winner)}",
            )
        elif self._plies >= self._max_plies:
            self._screen = ApplicationScreen.RESULT
            self._feedback = self._text("limit", max_plies=self._max_plies)

    def _draw_start(self, surface: pygame.Surface) -> None:
        """绘制模式、执子方、种子和速度选项组成的开始页。"""
        width, height = surface.get_size()
        panel = pygame.Rect(0, 0, min(760, width - 60), min(700, height - 40))
        panel.center = (width // 2, height // 2)
        pygame.draw.rect(surface, COLORS["panel"], panel, border_radius=22)
        pygame.draw.rect(surface, COLORS["line"], panel, width=2, border_radius=22)

        language_width = 82
        language_gap = 8
        language_y = panel.top + 22
        english_language = pygame.Rect(
            panel.right - 32 - language_width,
            language_y,
            language_width,
            34,
        )
        chinese_language = english_language.move(-(language_width + language_gap), 0)

        title = self._font(42).render(self._text("title"), True, COLORS["ink"])
        surface.blit(title, (panel.centerx - title.get_width() // 2, panel.top + 48))
        subtitle = self._font(20).render(
            self._text("subtitle"),
            True,
            COLORS["muted"],
        )
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
            Control.LANGUAGE_CHINESE: chinese_language,
            Control.LANGUAGE_ENGLISH: english_language,
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
            chinese_language,
            "中文",
            selected=self._language is Language.CHINESE,
            font_size=15,
        )
        self._draw_button(
            surface,
            english_language,
            "English",
            selected=self._language is Language.ENGLISH,
            font_size=15,
        )
        self._draw_button(
            surface,
            human_mode,
            self._text("human_human"),
            selected=self._selected_game_mode is GameMode.HUMAN_HUMAN,
            font_size=17,
        )
        self._draw_button(
            surface,
            human_random_mode,
            self._text("human_random"),
            selected=self._selected_game_mode is GameMode.HUMAN_RANDOM,
            font_size=17,
        )
        self._draw_button(
            surface,
            random_mode,
            self._text("random_random"),
            selected=self._selected_game_mode is GameMode.RANDOM_RANDOM,
            font_size=17,
        )
        self._draw_button(
            surface,
            player_0,
            self._text("human_first_label"),
            selected=self._human_player is Player.PLAYER_0,
        )
        self._draw_button(
            surface,
            player_1,
            self._text("human_second_label"),
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
            self._seed_text or self._text("seed_placeholder"),
            True,
            COLORS["ink"] if self._seed_text else COLORS["muted"],
        )
        surface.blit(
            seed_label,
            (seed_input.left + 14, seed_input.centery - seed_label.get_height() // 2),
        )
        for control, label, delay in (
            (Control.SPEED_SLOW, self._text("slow_label"), 1000),
            (Control.SPEED_NORMAL, self._text("normal_label"), 500),
            (Control.SPEED_FAST, self._text("fast_label"), 200),
        ):
            self._draw_button(
                surface,
                speed_rects[control],
                label,
                selected=self._agent_delay_ms == delay,
            )
        self._draw_button(
            surface,
            start,
            self._text("start_game"),
            selected=False,
        )
        feedback = self._font(17).render(self._feedback, True, COLORS["muted"])
        surface.blit(feedback, (panel.left + 70, panel.bottom - 155))

    def _draw_game(self, surface: pygame.Surface) -> None:
        """绘制对局页整体布局，并根据窗口尺寸重算响应式分栏。"""
        position = self._displayed_position()
        assert position is not None
        history_entry = self._reviewed_history_entry()
        last_action = (
            history_entry.action if history_entry is not None else self._last_action
        )
        last_move_start = (
            history_entry.move_start
            if history_entry is not None
            else self._last_move_start
        )
        width, height = surface.get_size()
        margin = max(16, round(min(width, height) * 0.025))
        header_height = max(56, round(height * 0.08))
        content_height = height - header_height - margin * 2
        board_side = min(round((width - margin * 3) * 0.73), content_height)
        board_rect = pygame.Rect(margin, header_height + margin, board_side, board_side)
        self._board = BoardGeometry.from_rect(board_rect)

        title = self._font(30).render(self._text("title"), True, COLORS["ink"])
        surface.blit(title, (margin, max(8, (header_height - title.get_height()) // 2)))
        pygame.draw.rect(surface, COLORS["board_dark"], board_rect, border_radius=18)

        legal_targets = (
            set()
            if self._reviewing_history
            else {
                action.target
                for action in self._legal_actions
                if isinstance(action, MovePawn)
            }
        )
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

        if isinstance(last_action, MovePawn) and last_move_start is not None:
            start_rect = self._board.square_rect(last_move_start)
            target_rect = self._board.square_rect(last_action.target)
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
            square = position.pawns[player]
            rect = self._board.square_rect(square)
            pygame.draw.circle(
                surface,
                _player_color(player),
                rect.center,
                max(8, round(rect.width * 0.31)),
            )
        for player in Player:
            for wall in position.placed_walls_by_player[player]:
                wall_rect = self._board.wall_rect(wall)
                pygame.draw.rect(
                    surface,
                    _player_color(player),
                    wall_rect,
                    border_radius=3,
                )
                if wall == last_action:
                    pygame.draw.rect(
                        surface,
                        COLORS["focus"],
                        wall_rect.inflate(6, 6),
                        width=3,
                        border_radius=5,
                    )
        if self._preview_wall is not None and not self._reviewing_history:
            preview = self._board.wall_rect(self._preview_wall)
            preview_color = (
                _player_color(position.to_move)
                if self._preview_reason is None and position.to_move is not None
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
        cards_bottom = self._draw_player_status_cards(surface, sidebar, position)
        button_gap = 8
        button_width = max(76, (sidebar.width - 44 - button_gap * 2) // 3)
        button_y = cards_bottom + 22
        self._controls = {}
        if self._reviewing_history:
            return_to_result = pygame.Rect(
                sidebar.left + 22,
                button_y,
                sidebar.width - 44,
                46,
            )
            self._controls[Control.RETURN_TO_RESULT] = return_to_result
            self._draw_button(
                surface,
                return_to_result,
                self._text("return_result"),
                selected=False,
            )
            controls_bottom = return_to_result.bottom
        elif self._game_mode is GameMode.RANDOM_RANDOM:
            controls_bottom = self._draw_playback_controls(
                surface,
                sidebar,
                button_y,
            )
        else:
            mode_controls = (
                (Control.MOVE, self._text("mode_move"), InputMode.MOVE),
                (
                    Control.HORIZONTAL_WALL,
                    self._text("mode_horizontal_wall"),
                    InputMode.HORIZONTAL_WALL,
                ),
                (
                    Control.VERTICAL_WALL,
                    self._text("mode_vertical_wall"),
                    InputMode.VERTICAL_WALL,
                ),
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
                    font_size=16,
                )
            controls_bottom = button_y + 46
        status_top = controls_bottom + 20
        status_lines = self._status_lines()
        next_line_top = status_top
        for index, line in enumerate(status_lines):
            font = self._font(16 if index < 2 else 15)
            color = COLORS["ink"] if index < 2 else COLORS["muted"]
            for wrapped_line in _wrap_text(line, font, sidebar.width - 44):
                text = font.render(wrapped_line, True, color)
                surface.blit(text, (sidebar.left + 22, next_line_top))
                next_line_top += 24
            next_line_top += 7
        history_top = next_line_top + 6
        self._draw_action_history(surface, sidebar, history_top)

    def _draw_player_status_cards(
        self,
        surface: pygame.Surface,
        sidebar: pygame.Rect,
        position: Position,
    ) -> int:
        """绘制双方身份、当前回合以及剩余墙库存，并返回内容底边。"""
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
            is_active = position.to_move is player
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
                active = self._font(12).render(
                    self._text("active"),
                    True,
                    player_color,
                )
                surface.blit(
                    active, (card.right - active.get_width() - 10, card.top + 9)
                )

            label = self._font(13).render(
                self._text("remaining"),
                True,
                COLORS["muted"],
            )
            surface.blit(label, (card.left + 11, card.top + 43))
            remaining = position.walls_remaining[player]
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

    def _draw_action_history(
        self,
        surface: pygame.Surface,
        sidebar: pygame.Rect,
        top: int,
    ) -> None:
        """绘制可滚动的动作历史，并记录各行的点击矩形。"""
        heading = self._font(16).render(
            self._text("history_heading"),
            True,
            COLORS["ink"],
        )
        surface.blit(heading, (sidebar.left + 22, top))

        viewport = pygame.Rect(
            sidebar.left + 18,
            top + heading.get_height() + 8,
            sidebar.width - 36,
            max(0, sidebar.bottom - top - heading.get_height() - 26),
        )
        if viewport.height < HISTORY_ROW_HEIGHT:
            return

        pygame.draw.rect(surface, COLORS["paper"], viewport, border_radius=8)
        pygame.draw.rect(
            surface,
            COLORS["line"],
            viewport,
            width=1,
            border_radius=8,
        )
        visible_rows = max(1, viewport.height // HISTORY_ROW_HEIGHT)
        self._history_visible_rows = visible_rows
        maximum = max(0, len(self._action_history) - visible_rows)
        self._history_scroll_rows = min(self._history_scroll_rows, maximum)
        newest_first = tuple(reversed(self._action_history))
        visible_entries = newest_first[
            self._history_scroll_rows : self._history_scroll_rows + visible_rows
        ]

        previous_clip = surface.get_clip()
        surface.set_clip(viewport)
        row_width = viewport.width - 14
        for index, entry in enumerate(visible_entries):
            row = pygame.Rect(
                viewport.left + 4,
                viewport.top + index * HISTORY_ROW_HEIGHT + 2,
                row_width,
                HISTORY_ROW_HEIGHT - 4,
            )
            self._history_entry_rects[entry.ply] = row
            selected = self._reviewing_history and self._reviewed_ply == entry.ply
            pygame.draw.rect(surface, COLORS["panel"], row, border_radius=6)
            if selected:
                pygame.draw.rect(
                    surface,
                    COLORS["focus"],
                    row,
                    width=2,
                    border_radius=6,
                )
            color = (
                _player_color(entry.player)
                if entry.player is not None
                else COLORS["muted"]
            )
            pygame.draw.circle(surface, color, (row.left + 11, row.centery), 5)
            label = self._font(14).render(
                _action_history_text(entry, language=self._language),
                True,
                color,
            )
            surface.blit(
                label,
                (row.left + 23, row.centery - label.get_height() // 2),
            )
        surface.set_clip(previous_clip)

        if maximum == 0:
            return
        track = pygame.Rect(
            viewport.right - 7, viewport.top + 5, 3, viewport.height - 10
        )
        pygame.draw.rect(surface, COLORS["line"], track, border_radius=2)
        thumb_height = max(18, round(track.height * visible_rows / len(newest_first)))
        thumb_travel = track.height - thumb_height
        thumb_top = track.top + round(
            thumb_travel * self._history_scroll_rows / maximum
        )
        thumb = pygame.Rect(track.left, thumb_top, track.width, thumb_height)
        pygame.draw.rect(surface, COLORS["muted"], thumb, border_radius=2)

    def _draw_playback_controls(
        self,
        surface: pygame.Surface,
        sidebar: pygame.Rect,
        top: int,
    ) -> int:
        """绘制暂停、单步和速度控件，并返回控件区底边。"""
        gap = 10
        primary_width = (sidebar.width - 54) // 2
        pause = pygame.Rect(sidebar.left + 22, top, primary_width, 46)
        step = pause.move(primary_width + gap, 0)
        self._controls[Control.PAUSE_RESUME] = pause
        self._controls[Control.STEP] = step
        self._draw_button(
            surface,
            pause,
            self._text("resume" if self._paused else "pause"),
            selected=self._paused,
        )
        self._draw_button(surface, step, self._text("step"), selected=False)

        speed_width = (sidebar.width - 64) // 3
        speed_top = top + 60
        for index, (control, label, delay) in enumerate(
            (
                (Control.SPEED_SLOW, self._text("slow_short"), 1000),
                (Control.SPEED_NORMAL, self._text("normal_short"), 500),
                (Control.SPEED_FAST, self._text("fast_short"), 200),
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
                font_size=16,
            )
        return speed_top + 42

    def _draw_button(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        *,
        selected: bool,
        font_size: int = 20,
    ) -> None:
        """按统一视觉样式绘制按钮及其选中状态。"""
        color = COLORS["selected"] if selected else COLORS["panel"]
        text_color = COLORS["white"] if selected else COLORS["ink"]
        pygame.draw.rect(surface, color, rect, border_radius=12)
        pygame.draw.rect(surface, COLORS["line"], rect, width=2, border_radius=12)
        text = self._font(font_size).render(label, True, text_color)
        surface.blit(
            text,
            (
                rect.centerx - text.get_width() // 2,
                rect.centery - text.get_height() // 2,
            ),
        )

    def _draw_result(self, surface: pygame.Surface) -> None:
        """在棋盘上方绘制对局结果或参与者错误对话框。"""
        width, height = surface.get_size()
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((23, 33, 28, 150))
        surface.blit(overlay, (0, 0))
        panel = pygame.Rect(0, 0, min(560, width - 80), min(330, height - 80))
        panel.center = (width // 2, height // 2)
        pygame.draw.rect(surface, COLORS["panel"], panel, border_radius=20)
        heading = self._text(
            "game_aborted" if self._screen is ApplicationScreen.ERROR else "game_over"
        )
        title = self._font(34).render(heading, True, COLORS["ink"])
        surface.blit(title, (panel.centerx - title.get_width() // 2, panel.top + 40))
        result = self._font(22).render(self._feedback, True, COLORS["muted"])
        surface.blit(result, (panel.centerx - result.get_width() // 2, panel.top + 94))

        controls = (
            (
                (Control.VIEW_HISTORY, self._text("view_history")),
                (Control.RESTART_GAME, self._text("restart")),
                (Control.RETURN_TO_START, self._text("return_start")),
                (Control.EXIT, self._text("exit")),
            )
            if self._screen is ApplicationScreen.RESULT
            else (
                (Control.RESTART_GAME, self._text("restart")),
                (Control.RETURN_TO_START, self._text("return_start")),
                (Control.EXIT, self._text("exit")),
            )
        )
        gap = 10
        available_width = panel.width - 60
        button_width = (available_width - gap * (len(controls) - 1)) // len(controls)
        total = button_width * len(controls) + gap * (len(controls) - 1)
        left = panel.centerx - total // 2
        self._controls = {}
        for index, (control, label) in enumerate(controls):
            rect = pygame.Rect(
                left + index * (button_width + gap),
                panel.bottom - 100,
                button_width,
                52,
            )
            self._controls[control] = rect
            self._draw_button(
                surface,
                rect,
                label,
                selected=False,
                font_size=16,
            )

    def _font(self, size: int) -> pygame.font.Font:
        """按字号缓存从内置字节资源加载的中文字体。"""
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(io.BytesIO(self._font_bytes), size)
        return self._fonts[size]


def _player_color(player: Player) -> pygame.Color:
    """返回玩家在整个界面中稳定使用的主题色。"""
    return COLORS["p0"] if player is Player.PLAYER_0 else COLORS["p1"]


def _human_square(square: Square) -> str:
    """把内部棋格坐标转换为 ``a1``～``i9``。"""
    return f"{chr(ord('a') + square.col)}{9 - square.row}"


def _human_anchor(anchor: WallAnchor) -> str:
    """把内部墙锚点转换为 ``a1``～``h8``。"""
    return f"{chr(ord('a') + anchor.col)}{8 - anchor.row}"


def _action_history_text(
    entry: ActionHistoryEntry,
    *,
    language: Language = Language.CHINESE,
) -> str:
    """生成一条适合历史面板显示的中文动作说明。"""
    if entry.player is None:
        return PYGAME_TEXT[language]["history_initial"]
    assert entry.action is not None
    player = f"player_{int(entry.player)}"
    if isinstance(entry.action, MovePawn):
        assert entry.move_start is not None
        return PYGAME_TEXT[language]["history_move"].format(
            ply=entry.ply,
            player=player,
            start=_human_square(entry.move_start),
            target=_human_square(entry.action.target),
        )
    orientation = (
        PYGAME_TEXT[language]["horizontal_wall"]
        if entry.action.orientation is Orientation.HORIZONTAL
        else PYGAME_TEXT[language]["vertical_wall"]
    )
    return PYGAME_TEXT[language]["history_wall"].format(
        ply=entry.ply,
        player=player,
        wall=orientation,
        anchor=_human_anchor(entry.action.anchor),
    )


def _wrap_text(
    value: str,
    font: pygame.font.Font,
    maximum_width: int,
) -> tuple[str, ...]:
    """Wrap localized UI text so it remains visible in the narrow sidebar."""
    if font.size(value)[0] <= maximum_width:
        return (value,)
    units = value.split(" ") if " " in value else list(value)
    separator = " " if " " in value else ""
    lines: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else current + separator + unit
        if current and font.size(candidate)[0] > maximum_width:
            lines.append(current)
            current = unit
        else:
            current = candidate
    if current:
        lines.append(current)
    return tuple(lines)


def run() -> int:
    """创建可缩放窗口并运行事件、更新、绘制主循环。"""
    """Run the interactive desktop event loop."""
    pygame.init()
    try:
        surface = pygame.display.set_mode(WINDOW_INITIAL, pygame.RESIZABLE)
        pygame.display.set_caption(PYGAME_TEXT[Language.CHINESE]["window_title"])
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
            pygame.display.set_caption(application.snapshot.window_title)
            pygame.display.flip()
        return 0
    finally:
        pygame.quit()

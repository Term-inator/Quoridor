"""PROTOTYPE — throwaway Pygame UI for validating Quoridor interaction.

Question: Which of three left-board/right-sidebar layouts makes board scale,
wall-anchor snapping, legal/illegal feedback, and resized-window readability
feel best in a real desktop window?

This file deliberately drives the production Position API directly while keeping
all presentation and interaction code local to the prototype.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    import pygame
except ImportError as error:  # pragma: no cover - convenience for a manual prototype
    raise SystemExit(
        "未安装 Pygame。请在仓库根目录运行：\n"
        "UV_CACHE_DIR=/tmp/quoridor-uv-cache uv run --with 'pygame-ce>=2.5.8,<3' "
        "python .scratch/quoridor-pygame-prototype/pygame_ui_prototype.py"
    ) from error

from quoridor_rl.game import (  # noqa: I001 - follows the optional Pygame import guard
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


WINDOW_MIN = (960, 640)
WINDOW_INITIAL = (1280, 800)
VARIANTS = ("A", "B", "C")
VARIANT_NAMES = {
    "A": "棋盘优先",
    "B": "均衡工作台",
    "C": "规则调试台",
}

COLORS = {
    "ink": pygame.Color("#17211c"),
    "muted": pygame.Color("#607069"),
    "paper": pygame.Color("#f4f0e7"),
    "panel": pygame.Color("#fffdf8"),
    "line": pygame.Color("#d7d0c2"),
    "board": pygame.Color("#c99b62"),
    "board_dark": pygame.Color("#8c6038"),
    "p0": pygame.Color("#176b55"),
    "p1": pygame.Color("#a44437"),
    "valid": pygame.Color("#238b61"),
    "invalid": pygame.Color("#c83933"),
    "focus": pygame.Color("#f0a33a"),
    "switcher": pygame.Color("#16231e"),
    "white": pygame.Color("#ffffff"),
    "shadow": pygame.Color(0, 0, 0, 26),
    "wash_green": pygame.Color("#e1f2e9"),
    "wash_red": pygame.Color("#f9e4e1"),
    "wash_amber": pygame.Color("#fff0ce"),
}

FONT_CANDIDATES = (
    Path(__file__).parents[2] / "src/quoridor_rl/assets/NotoSansSC-Regular.otf",
    Path("/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc"),
    Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"),
)

REASON_TEXT = {
    IllegalActionReason.GAME_OVER: "对局已经结束",
    IllegalActionReason.ILLEGAL_PAWN_MOVE: "棋子不能移动到该格",
    IllegalActionReason.NO_WALLS_REMAINING: "当前行动方已经没有墙",
    IllegalActionReason.WALL_CONFLICT: "与已有墙重叠或交叉",
    IllegalActionReason.WALL_BLOCKS_PATH: "会堵死至少一方的所有路径",
}


@dataclass(frozen=True)
class BoardGeometry:
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
        anchor = wall.anchor
        if wall.orientation is Orientation.HORIZONTAL:
            return pygame.Rect(
                round(self.origin_x + anchor.col * self.pitch),
                round(self.origin_y + anchor.row * self.pitch + self.cell),
                round(self.cell * 2 + self.gap),
                max(5, round(self.gap)),
            )
        return pygame.Rect(
            round(self.origin_x + anchor.col * self.pitch + self.cell),
            round(self.origin_y + anchor.row * self.pitch),
            max(5, round(self.gap)),
            round(self.cell * 2 + self.gap),
        )

    def nearest_anchor(self, point: tuple[int, int]) -> WallAnchor | None:
        if not self.rect.inflate(
            -self.rect.width * 0.05, -self.rect.height * 0.05
        ).collidepoint(point):
            return None
        col = round((point[0] - self.origin_x - self.cell - self.gap / 2) / self.pitch)
        row = round((point[1] - self.origin_y - self.cell - self.gap / 2) / self.pitch)
        return WallAnchor(max(0, min(7, row)), max(0, min(7, col)))


@dataclass
class HitAreas:
    board: BoardGeometry
    modes: dict[str, pygame.Rect]
    reset: pygame.Rect
    variants: dict[str, pygame.Rect]


class Prototype:
    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.position = Position.initial()
        self.mode = "move"
        self.feedback = "原型就绪：点击高亮目标移动，或切换放墙模式。"
        self.feedback_is_error = False
        self.last_action: Action | None = None
        self.last_move_start: Square | None = None
        self.plies = 0
        self.hover_anchor: WallAnchor | None = None
        self.hover_wall: PlaceWall | None = None
        self.hover_reason: IllegalActionReason | None = None
        self.hit_areas: HitAreas | None = None
        self.font_path = next((path for path in FONT_CANDIDATES if path.exists()), None)
        self._fonts: dict[tuple[int, bool], pygame.font.Font] = {}

    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (max(12, size), bold)
        if key not in self._fonts:
            font = pygame.font.Font(
                str(self.font_path) if self.font_path else None, key[0]
            )
            font.set_bold(bold)
            self._fonts[key] = font
        return self._fonts[key]

    def reset(self) -> None:
        self.position = Position.initial()
        self.mode = "move"
        self.feedback = "已回到初始局面。"
        self.feedback_is_error = False
        self.last_action = None
        self.last_move_start = None
        self.plies = 0
        self.clear_hover()

    def clear_hover(self) -> None:
        self.hover_anchor = None
        self.hover_wall = None
        self.hover_reason = None

    def set_variant(self, variant: str) -> None:
        self.variant = variant
        self.feedback = (
            f"已切换到方案 {variant}：{VARIANT_NAMES[variant]}。对局状态保持不变。"
        )
        self.feedback_is_error = False

    def cycle_variant(self, delta: int) -> None:
        current = VARIANTS.index(self.variant)
        self.set_variant(VARIANTS[(current + delta) % len(VARIANTS)])

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.clear_hover()
        labels = {"move": "移动", "horizontal": "横墙", "vertical": "竖墙"}
        self.feedback = f"当前操作：{labels[mode]}。"
        self.feedback_is_error = False

    def update_hover(self, point: tuple[int, int]) -> None:
        self.clear_hover()
        if self.mode == "move" or self.hit_areas is None:
            return
        anchor = self.hit_areas.board.nearest_anchor(point)
        if anchor is None:
            return
        orientation = (
            Orientation.HORIZONTAL
            if self.mode == "horizontal"
            else Orientation.VERTICAL
        )
        wall = PlaceWall(anchor, orientation)
        self.hover_anchor = anchor
        self.hover_wall = wall
        if wall not in self.position.legal_actions():
            try:
                self.position.play(wall)
            except IllegalActionError as error:
                self.hover_reason = error.reason

    def click(self, point: tuple[int, int]) -> None:
        if self.hit_areas is None:
            return
        for variant, rect in self.hit_areas.variants.items():
            if rect.collidepoint(point):
                self.set_variant(variant)
                return
        for mode, rect in self.hit_areas.modes.items():
            if rect.collidepoint(point):
                self.set_mode(mode)
                return
        if self.hit_areas.reset.collidepoint(point):
            self.reset()
            return
        if self.mode == "move":
            self._click_move(point)
        elif self.hover_wall is not None:
            self._play(self.hover_wall)

    def _click_move(self, point: tuple[int, int]) -> None:
        assert self.hit_areas is not None
        legal_moves = [
            action
            for action in self.position.legal_actions()
            if isinstance(action, MovePawn)
        ]
        for move in legal_moves:
            if self.hit_areas.board.square_rect(move.target).collidepoint(point):
                self._play(move)
                return

    def _play(self, action: Action) -> None:
        mover = self.position.to_move
        if mover is None:
            return
        start = self.position.pawns[mover] if isinstance(action, MovePawn) else None
        try:
            next_position = self.position.play(action)
        except IllegalActionError as error:
            self.feedback = f"未执行：{REASON_TEXT[error.reason]}。回合没有推进。"
            self.feedback_is_error = True
            self.hover_reason = error.reason
            return

        self.position = next_position
        self.last_action = action
        self.last_move_start = start
        self.plies += 1
        self.mode = "move"
        self.feedback_is_error = False
        if isinstance(action, MovePawn):
            self.feedback = (
                f"已移动：{human_square(action.target)}。下一回合已回到移动模式。"
            )
        else:
            orientation = (
                "横墙" if action.orientation is Orientation.HORIZONTAL else "竖墙"
            )
            self.feedback = f"已放置{orientation}：{human_anchor(action.anchor)}。下一回合已回到移动模式。"
        self.clear_hover()

    def layout(
        self, size: tuple[int, int]
    ) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        width, height = size
        header_h = max(58, round(height * 0.082))
        switcher_h = 58
        margin = max(16, round(min(width, height) * 0.025))
        content = pygame.Rect(
            margin,
            header_h + 6,
            width - margin * 2,
            height - header_h - switcher_h - margin,
        )
        ratios = {"A": 0.73, "B": 0.65, "C": 0.57}
        height_scales = {"A": 1.0, "B": 0.90, "C": 0.82}
        desired_board_width = round(content.width * ratios[self.variant])
        board_side = min(
            desired_board_width, round(content.height * height_scales[self.variant])
        )
        board = pygame.Rect(content.left, content.top, board_side, board_side)
        sidebar_left = board.right + margin
        sidebar = pygame.Rect(
            sidebar_left, content.top, content.right - sidebar_left, content.height
        )
        header = pygame.Rect(margin, 0, width - margin * 2, header_h)
        return header, board, sidebar

    def draw(
        self, surface: pygame.Surface, mouse: tuple[int, int] | None = None
    ) -> None:
        surface.fill(COLORS["paper"])
        header, board_rect, sidebar = self.layout(surface.get_size())
        board_geometry = BoardGeometry.from_rect(board_rect)
        self._draw_header(surface, header)
        self._draw_board(surface, board_geometry)
        modes, reset = self._draw_sidebar(surface, sidebar)
        variants = self._draw_switcher(surface)
        self.hit_areas = HitAreas(board_geometry, modes, reset, variants)
        if mouse is not None:
            self.update_hover(mouse)
            self._draw_wall_preview(surface, board_geometry)

    def _draw_header(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        title_size = max(20, round(rect.height * 0.40))
        title = self.font(title_size, True).render(
            "围墙棋 · 交互原型", True, COLORS["ink"]
        )
        surface.blit(title, (rect.left, rect.centery - title.get_height() // 2))
        subtitle = self.font(max(13, round(title_size * 0.58))).render(
            f"方案 {self.variant} · {VARIANT_NAMES[self.variant]}  |  PROTOTYPE — 不进入正式实现",
            True,
            COLORS["muted"],
        )
        surface.blit(
            subtitle,
            (
                rect.right - subtitle.get_width(),
                rect.centery - subtitle.get_height() // 2,
            ),
        )

    def _draw_board(self, surface: pygame.Surface, geometry: BoardGeometry) -> None:
        shadow = geometry.rect.move(0, 5)
        pygame.draw.rect(surface, COLORS["shadow"], shadow, border_radius=18)
        pygame.draw.rect(surface, COLORS["board_dark"], geometry.rect, border_radius=18)
        inner = geometry.rect.inflate(-10, -10)
        pygame.draw.rect(surface, COLORS["board"], inner, border_radius=14)

        legal_targets = {
            action.target
            for action in self.position.legal_actions()
            if isinstance(action, MovePawn)
        }
        for row in range(9):
            for col in range(9):
                square = Square(row, col)
                rect = geometry.square_rect(square)
                color = (
                    pygame.Color("#ead1a7")
                    if (row + col) % 2 == 0
                    else pygame.Color("#e2c393")
                )
                pygame.draw.rect(
                    surface,
                    color,
                    rect,
                    border_radius=max(2, round(geometry.cell * 0.08)),
                )
                pygame.draw.rect(
                    surface, pygame.Color("#b38351"), rect, 1, border_radius=3
                )
                if self.mode == "move" and square in legal_targets:
                    radius = max(5, round(geometry.cell * 0.12))
                    pygame.draw.circle(surface, COLORS["valid"], rect.center, radius)
                    pygame.draw.circle(
                        surface, COLORS["white"], rect.center, max(2, radius // 3)
                    )

        if isinstance(self.last_action, MovePawn) and self.last_move_start is not None:
            start = geometry.square_rect(self.last_move_start)
            end = geometry.square_rect(self.last_action.target)
            pygame.draw.circle(
                surface,
                COLORS["focus"],
                start.center,
                max(6, round(geometry.cell * 0.17)),
                3,
            )
            pygame.draw.line(
                surface,
                COLORS["focus"],
                start.center,
                end.center,
                max(3, round(geometry.cell * 0.06)),
            )
            pygame.draw.circle(
                surface,
                COLORS["focus"],
                end.center,
                max(8, round(geometry.cell * 0.25)),
                3,
            )

        for wall in self.position.placed_walls:
            rect = geometry.wall_rect(wall)
            color = (
                COLORS["focus"] if wall == self.last_action else pygame.Color("#5c3925")
            )
            pygame.draw.rect(
                surface, color, rect, border_radius=max(2, rect.height // 3)
            )
            pygame.draw.rect(surface, pygame.Color("#3b2418"), rect, 2, border_radius=3)

        for player, pawn in zip(Player, self.position.pawns, strict=True):
            rect = geometry.square_rect(pawn)
            radius = max(12, round(geometry.cell * 0.33))
            pygame.draw.circle(
                surface, COLORS["shadow"], (rect.centerx, rect.centery + 3), radius
            )
            pygame.draw.circle(
                surface,
                COLORS["p0"] if player is Player.PLAYER_0 else COLORS["p1"],
                rect.center,
                radius,
            )
            pygame.draw.circle(
                surface, COLORS["white"], rect.center, radius, max(2, radius // 8)
            )
            label = self.font(max(13, round(geometry.cell * 0.28)), True).render(
                "0" if player is Player.PLAYER_0 else "1", True, COLORS["white"]
            )
            surface.blit(label, label.get_rect(center=rect.center))

        coord_font = self.font(max(12, round(geometry.cell * 0.25)), True)
        for col, letter in enumerate("abcdefghi"):
            cell = geometry.square_rect(Square(8, col))
            text = coord_font.render(letter, True, COLORS["paper"])
            surface.blit(
                text,
                (
                    cell.centerx - text.get_width() // 2,
                    geometry.rect.bottom - text.get_height() - 2,
                ),
            )
        for row in range(9):
            cell = geometry.square_rect(Square(row, 0))
            text = coord_font.render(str(9 - row), True, COLORS["paper"])
            surface.blit(
                text, (geometry.rect.left + 4, cell.centery - text.get_height() // 2)
            )

        self._draw_wall_preview(surface, geometry)

    def _draw_wall_preview(
        self, surface: pygame.Surface, geometry: BoardGeometry
    ) -> None:
        if self.hover_wall is None:
            return
        rect = geometry.wall_rect(self.hover_wall)
        valid = self.hover_reason is None
        color = COLORS["valid"] if valid else COLORS["invalid"]
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((*color[:3], 205))
        surface.blit(overlay, rect)
        pygame.draw.rect(surface, COLORS["white"], rect, 2, border_radius=3)
        if not valid:
            radius = max(9, round(geometry.cell * 0.20))
            pygame.draw.circle(surface, COLORS["invalid"], rect.center, radius + 3)
            pygame.draw.circle(surface, COLORS["white"], rect.center, radius + 3, 2)
            pygame.draw.line(
                surface,
                COLORS["white"],
                (rect.centerx - radius, rect.centery - radius),
                (rect.centerx + radius, rect.centery + radius),
                3,
            )
            pygame.draw.line(
                surface,
                COLORS["white"],
                (rect.centerx + radius, rect.centery - radius),
                (rect.centerx - radius, rect.centery + radius),
                3,
            )

    def _draw_sidebar(
        self, surface: pygame.Surface, rect: pygame.Rect
    ) -> tuple[dict[str, pygame.Rect], pygame.Rect]:
        pygame.draw.rect(surface, COLORS["panel"], rect, border_radius=16)
        pygame.draw.rect(surface, COLORS["line"], rect, 1, border_radius=16)
        padding = max(14, round(rect.width * 0.055))
        inner = rect.inflate(-padding * 2, -padding * 2)
        if self.variant == "A":
            return self._sidebar_a(surface, inner)
        if self.variant == "B":
            return self._sidebar_b(surface, inner)
        return self._sidebar_c(surface, inner)

    def _sidebar_a(
        self, surface: pygame.Surface, rect: pygame.Rect
    ) -> tuple[dict[str, pygame.Rect], pygame.Rect]:
        compact = rect.height < 540
        y = rect.top
        y = self._section_title(surface, "本局状态", rect.left, y, rect.width)
        y = self._draw_turn_card(
            surface, pygame.Rect(rect.left, y, rect.width, 62 if compact else 74)
        ) + (9 if compact else 14)
        y = self._section_title(surface, "选择操作", rect.left, y, rect.width)
        modes, y = self._draw_mode_buttons(
            surface, rect.left, y, rect.width, stacked=not compact
        )
        y += 9 if compact else 14
        y = self._section_title(surface, "最近反馈", rect.left, y, rect.width)
        feedback_height = 72 if compact else min(112, max(76, rect.bottom - y - 190))
        y = self._draw_feedback(
            surface, pygame.Rect(rect.left, y, rect.width, feedback_height)
        ) + (9 if compact else 14)
        y = self._section_title(surface, "完整交互状态", rect.left, y, rect.width)
        self._draw_state(
            surface,
            pygame.Rect(rect.left, y, rect.width, max(80, rect.bottom - y - 42)),
        )
        reset = pygame.Rect(rect.left, rect.bottom - 34, rect.width, 34)
        self._button(surface, reset, "重置原型", selected=False, compact=True)
        return modes, reset

    def _sidebar_b(
        self, surface: pygame.Surface, rect: pygame.Rect
    ) -> tuple[dict[str, pygame.Rect], pygame.Rect]:
        compact = rect.height < 540
        y = rect.top
        y = self._section_title(surface, "操作工作台", rect.left, y, rect.width)
        modes, y = self._draw_mode_buttons(
            surface, rect.left, y, rect.width, stacked=False
        )
        y += 8 if compact else 12
        feedback_height = (
            72 if compact else min(142, max(92, round(rect.height * 0.24)))
        )
        y = self._draw_feedback(
            surface, pygame.Rect(rect.left, y, rect.width, feedback_height)
        ) + (9 if compact else 14)
        y = self._section_title(surface, "行动方与资源", rect.left, y, rect.width)
        y = self._draw_turn_card(
            surface, pygame.Rect(rect.left, y, rect.width, 62 if compact else 84)
        ) + (9 if compact else 14)
        y = self._section_title(surface, "状态快照", rect.left, y, rect.width)
        self._draw_state(
            surface,
            pygame.Rect(rect.left, y, rect.width, max(80, rect.bottom - y - 42)),
        )
        reset = pygame.Rect(rect.left, rect.bottom - 34, rect.width, 34)
        self._button(surface, reset, "重新开始", selected=False, compact=True)
        return modes, reset

    def _sidebar_c(
        self, surface: pygame.Surface, rect: pygame.Rect
    ) -> tuple[dict[str, pygame.Rect], pygame.Rect]:
        compact = rect.height < 540
        y = rect.top
        label = self.font(15, True).render("规则调试台 / LIVE", True, COLORS["invalid"])
        surface.blit(label, (rect.left, y))
        y += label.get_height() + (7 if compact else 10)
        modes, y = self._draw_mode_buttons(
            surface, rect.left, y, rect.width, stacked=False
        )
        y += 7 if compact else 10
        state_height = 130 if compact else min(210, max(150, round(rect.height * 0.34)))
        self._draw_state(
            surface, pygame.Rect(rect.left, y, rect.width, state_height), mono=True
        )
        y += state_height + (8 if compact else 12)
        y = self._section_title(surface, "规则核心返回", rect.left, y, rect.width)
        feedback_height = 72 if compact else min(132, max(84, rect.bottom - y - 145))
        y = self._draw_feedback(
            surface, pygame.Rect(rect.left, y, rect.width, feedback_height)
        ) + (8 if compact else 12)
        y = self._section_title(surface, "阅读性检查", rect.left, y, rect.width)
        note = "棋盘 / 坐标 / 按钮 / 反馈文字\n拖动窗口边缘，逐项确认是否仍清楚。"
        self._paragraph(
            surface,
            note,
            pygame.Rect(rect.left, y, rect.width, 60),
            COLORS["muted"],
            14,
        )
        reset = pygame.Rect(rect.left, rect.bottom - 34, rect.width, 34)
        self._button(surface, reset, "清空并重置", selected=False, compact=True)
        return modes, reset

    def _section_title(
        self, surface: pygame.Surface, text: str, x: int, y: int, width: int
    ) -> int:
        font = self.font(max(14, min(18, round(width * 0.06))), True)
        rendered = font.render(text, True, COLORS["ink"])
        surface.blit(rendered, (x, y))
        return y + rendered.get_height() + 7

    def _draw_turn_card(self, surface: pygame.Surface, rect: pygame.Rect) -> int:
        pygame.draw.rect(surface, COLORS["wash_amber"], rect, border_radius=10)
        player = self.position.to_move
        turn = "对局结束" if player is None else f"player_{int(player)} 行动"
        font = self.font(max(16, min(22, round(rect.width * 0.07))), True)
        surface.blit(
            font.render(turn, True, COLORS["ink"]), (rect.left + 12, rect.top + 9)
        )
        sub = f"墙：P0 {self.position.walls_remaining[0]}  ·  P1 {self.position.walls_remaining[1]}     手数：{self.plies}"
        surface.blit(
            self.font(14).render(sub, True, COLORS["muted"]),
            (rect.left + 12, rect.bottom - 27),
        )
        return rect.bottom

    def _draw_mode_buttons(
        self, surface: pygame.Surface, x: int, y: int, width: int, *, stacked: bool
    ) -> tuple[dict[str, pygame.Rect], int]:
        labels = (("move", "移动"), ("horizontal", "横墙"), ("vertical", "竖墙"))
        result: dict[str, pygame.Rect] = {}
        if stacked:
            button_h = 36
            for mode, label in labels:
                rect = pygame.Rect(x, y, width, button_h)
                self._button(surface, rect, label, selected=self.mode == mode)
                result[mode] = rect
                y += button_h + 7
            return result, y
        gap = 7
        button_width = (width - gap * 2) // 3
        for index, (mode, label) in enumerate(labels):
            rect = pygame.Rect(x + index * (button_width + gap), y, button_width, 38)
            self._button(surface, rect, label, selected=self.mode == mode)
            result[mode] = rect
        return result, y + 38

    def _button(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        *,
        selected: bool,
        compact: bool = False,
    ) -> None:
        color = COLORS["ink"] if selected else COLORS["paper"]
        pygame.draw.rect(surface, color, rect, border_radius=8)
        pygame.draw.rect(
            surface,
            COLORS["focus"] if selected else COLORS["line"],
            rect,
            2 if selected else 1,
            border_radius=8,
        )
        text_color = COLORS["white"] if selected else COLORS["ink"]
        text = self.font(13 if compact else 15, selected).render(
            label, True, text_color
        )
        surface.blit(text, text.get_rect(center=rect.center))

    def _draw_feedback(self, surface: pygame.Surface, rect: pygame.Rect) -> int:
        background = (
            COLORS["wash_red"] if self.feedback_is_error else COLORS["wash_green"]
        )
        accent = COLORS["invalid"] if self.feedback_is_error else COLORS["valid"]
        pygame.draw.rect(surface, background, rect, border_radius=10)
        pygame.draw.rect(surface, accent, rect, 2, border_radius=10)
        icon = "×" if self.feedback_is_error else "✓"
        icon_text = self.font(22, True).render(icon, True, accent)
        surface.blit(icon_text, (rect.left + 11, rect.top + 8))
        text_rect = pygame.Rect(
            rect.left + 40, rect.top + 10, rect.width - 52, rect.height - 20
        )
        self._paragraph(surface, self.feedback, text_rect, COLORS["ink"], 15)
        return rect.bottom

    def _draw_state(
        self, surface: pygame.Surface, rect: pygame.Rect, *, mono: bool = False
    ) -> None:
        pygame.draw.rect(
            surface,
            pygame.Color("#eef0eb") if mono else COLORS["paper"],
            rect,
            border_radius=9,
        )
        pygame.draw.rect(surface, COLORS["line"], rect, 1, border_radius=9)
        hover = "—"
        if self.hover_wall is not None:
            orientation = (
                "H" if self.hover_wall.orientation is Orientation.HORIZONTAL else "V"
            )
            verdict = (
                "合法"
                if self.hover_reason is None
                else f"非法/{self.hover_reason.value}"
            )
            hover = f"{human_anchor(self.hover_wall.anchor)} {orientation} · {verdict}"
        last = action_text(self.last_action) if self.last_action is not None else "—"
        player = (
            "—"
            if self.position.to_move is None
            else f"player_{int(self.position.to_move)}"
        )
        window = (
            f"{pygame.display.get_surface().get_width()}×"
            f"{pygame.display.get_surface().get_height()}"
        )
        if rect.height < 100:
            lines = (
                f"turn {player} · mode {self.mode} · ply {self.plies}",
                f"pawns P0 {human_square(self.position.pawns[0])} · P1 {human_square(self.position.pawns[1])} · walls {self.position.walls_remaining[0]}/{self.position.walls_remaining[1]}",
                f"hover {hover} · last {last} · placed {len(self.position.placed_walls)} · {window}",
            )
        elif rect.height < 170:
            lines = (
                f"turn {player} · mode {self.mode} · ply {self.plies}",
                f"pawns P0 {human_square(self.position.pawns[0])} · P1 {human_square(self.position.pawns[1])} · walls {self.position.walls_remaining[0]}/{self.position.walls_remaining[1]}",
                f"hover {hover}",
                f"last {last} · placed {len(self.position.placed_walls)} · window {window}",
            )
        else:
            lines = (
                f"to_move    {player}",
                f"pawns      P0 {human_square(self.position.pawns[0])} / P1 {human_square(self.position.pawns[1])}",
                f"walls      {self.position.walls_remaining[0]} / {self.position.walls_remaining[1]}",
                f"mode       {self.mode}",
                f"hover      {hover}",
                f"last       {last}",
                f"placed     {len(self.position.placed_walls)}",
                f"window     {window}",
            )
        font_size = max(12, min(14, round(rect.width * 0.045)))
        font = self.font(font_size)
        y = rect.top + 9
        for line in lines:
            rendered = font.render(line, True, COLORS["ink"])
            clipped = rendered.subsurface(
                (
                    0,
                    0,
                    min(rendered.get_width(), rect.width - 16),
                    rendered.get_height(),
                )
            )
            surface.blit(clipped, (rect.left + 8, y))
            y += font.get_linesize()
            if y + font.get_linesize() > rect.bottom:
                break

    def _paragraph(
        self,
        surface: pygame.Surface,
        text: str,
        rect: pygame.Rect,
        color: pygame.Color,
        size: int,
    ) -> None:
        font = self.font(size)
        y = rect.top
        for paragraph in text.splitlines():
            for line in wrap_text(paragraph, font, rect.width):
                rendered = font.render(line, True, color)
                if y + rendered.get_height() > rect.bottom:
                    return
                surface.blit(rendered, (rect.left, y))
                y += font.get_linesize()

    def _draw_switcher(self, surface: pygame.Surface) -> dict[str, pygame.Rect]:
        width, height = surface.get_size()
        bar_width = min(510, width - 32)
        bar = pygame.Rect((width - bar_width) // 2, height - 50, bar_width, 40)
        pygame.draw.rect(surface, COLORS["switcher"], bar.move(0, 3), border_radius=20)
        pygame.draw.rect(surface, COLORS["switcher"], bar, border_radius=20)
        label = self.font(14, True).render(
            f"原型方案 {self.variant} — {VARIANT_NAMES[self.variant]}",
            True,
            COLORS["white"],
        )
        surface.blit(label, label.get_rect(center=bar.center))
        areas: dict[str, pygame.Rect] = {}
        x = bar.left + 8
        for variant in VARIANTS:
            rect = pygame.Rect(x, bar.top + 6, 32, 28)
            selected = variant == self.variant
            pygame.draw.rect(
                surface,
                COLORS["focus"] if selected else pygame.Color("#31433b"),
                rect,
                border_radius=14,
            )
            text = self.font(13, True).render(
                variant, True, COLORS["ink"] if selected else COLORS["white"]
            )
            surface.blit(text, text.get_rect(center=rect.center))
            areas[variant] = rect
            x += 38
        hint = self.font(12).render("← → 切换", True, pygame.Color("#b9c7c0"))
        surface.blit(
            hint,
            (bar.right - hint.get_width() - 13, bar.centery - hint.get_height() // 2),
        )
        return areas


def wrap_text(text: str, font: pygame.font.Font, width: int) -> Iterable[str]:
    if not text:
        yield ""
        return
    line = ""
    for character in text:
        candidate = line + character
        if line and font.size(candidate)[0] > width:
            yield line
            line = character
        else:
            line = candidate
    if line:
        yield line


def human_square(square: Square) -> str:
    return f"{'abcdefghi'[square.col]}{9 - square.row}"


def human_anchor(anchor: WallAnchor) -> str:
    return f"{'abcdefgh'[anchor.col]}{8 - anchor.row}"


def action_text(action: Action) -> str:
    if isinstance(action, MovePawn):
        return f"move {human_square(action.target)}"
    orientation = "H" if action.orientation is Orientation.HORIZONTAL else "V"
    return f"wall {human_anchor(action.anchor)} {orientation}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Throwaway Quoridor Pygame UI prototype"
    )
    parser.add_argument("--variant", choices=VARIANTS, default="A")
    parser.add_argument(
        "--screenshots",
        type=Path,
        metavar="DIR",
        help="render all variants at representative sizes, save PNGs, then exit",
    )
    return parser.parse_args()


def render_screenshots(prototype: Prototype, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = ((960, 640), (1280, 800), (1600, 950))
    for variant in VARIANTS:
        prototype.set_variant(variant)
        for width, height in sizes:
            surface = pygame.display.set_mode((width, height))
            prototype.draw(surface, (0, 0))
            pygame.image.save(
                surface, output_dir / f"variant-{variant}-{width}x{height}.png"
            )


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.display.set_caption("围墙棋交互原型 — PROTOTYPE")
    prototype = Prototype(args.variant)

    if args.screenshots:
        render_screenshots(prototype, args.screenshots)
        pygame.quit()
        return

    screen = pygame.display.set_mode(WINDOW_INITIAL, pygame.RESIZABLE)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                size = (max(WINDOW_MIN[0], event.w), max(WINDOW_MIN[1], event.h))
                screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    prototype.set_mode("move")
                elif event.key == pygame.K_LEFT:
                    prototype.cycle_variant(-1)
                elif event.key == pygame.K_RIGHT:
                    prototype.cycle_variant(1)
                elif event.key == pygame.K_r:
                    prototype.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                prototype.click(event.pos)

        mouse = pygame.mouse.get_pos()
        prototype.update_hover(mouse)
        prototype.draw(screen, mouse)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

import pytest


def test_pygame_command_explains_how_to_install_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from quoridor_rl.pygame_ui import entry

    monkeypatch.setattr(entry.importlib.util, "find_spec", lambda name: None)

    assert entry.main() == 2
    captured = capsys.readouterr()
    assert 'pip install "quoridor-rl[pygame]"' in captured.err
    assert "Traceback" not in captured.err


def test_pygame_application_draws_start_screen_and_handles_quit() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl.pygame_ui.app import ApplicationScreen, PygameApplication

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))

    application.draw(surface)

    assert application.snapshot.screen is ApplicationScreen.START
    assert application.snapshot.feedback == "请选择对局模式并开始。"
    assert application.handle_event(pygame.event.Event(pygame.QUIT)) is False


def test_humans_can_start_a_game_and_move_to_a_legal_target() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import MovePawn, Position, Square
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        PygameApplication,
    )

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)

    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.MODE_HUMAN_HUMAN).center,
        )
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.START_GAME).center,
        )
    )
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.square_rect(Square(7, 4)).center,
        )
    )

    assert application.snapshot.screen is ApplicationScreen.PLAYING
    assert application.snapshot.position == Position.initial().play(
        MovePawn(Square(7, 4))
    )
    assert application.snapshot.plies == 1
    assert application.snapshot.feedback == "已移动：e2。下一回合已回到移动模式。"


def test_legal_move_targets_follow_the_current_player_color() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Square
    from quoridor_rl.pygame_ui.app import COLORS, Control, PygameApplication

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )

    application.draw(surface)
    player_0_target = Square(7, 4)
    assert (
        surface.get_at(application.square_rect(player_0_target).center)
        == COLORS["p0"]
    )

    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.square_rect(player_0_target).center,
        )
    )
    application.draw(surface)
    player_1_target = Square(1, 4)
    assert (
        surface.get_at(application.square_rect(player_1_target).center)
        == COLORS["p1"]
    )


def test_human_wall_preview_uses_rules_and_illegal_click_does_not_advance() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Orientation, PlaceWall, Position, WallAnchor
    from quoridor_rl.pygame_ui.app import (
        COLORS,
        Control,
        InputMode,
        PygameApplication,
    )

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.draw(surface)

    wall = PlaceWall(WallAnchor(3, 3), Orientation.HORIZONTAL)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.HORIZONTAL_WALL).center,
        )
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEMOTION,
            pos=application.wall_anchor_point(wall.anchor),
        )
    )
    assert application.snapshot.preview_wall == wall
    assert application.snapshot.preview_reason is None
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.wall_anchor_point(wall.anchor),
        )
    )

    expected = Position.initial().play(wall)
    assert application.snapshot.position == expected
    assert application.snapshot.plies == 1
    assert application.snapshot.input_mode is InputMode.MOVE
    assert len(application.snapshot.action_history) == 2

    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.HORIZONTAL_WALL).center,
        )
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEMOTION,
            pos=application.wall_anchor_point(wall.anchor),
        )
    )
    application.draw(surface)
    invalid_center = application.wall_anchor_point(wall.anchor)
    assert surface.get_at(invalid_center) == COLORS["white"]
    assert (
        surface.get_at((invalid_center[0], invalid_center[1] + 4)) == COLORS["invalid"]
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.wall_anchor_point(wall.anchor),
        )
    )

    assert application.snapshot.position == expected
    assert application.snapshot.position.to_move == expected.to_move
    assert application.snapshot.plies == 1
    assert len(application.snapshot.action_history) == 2
    assert application.snapshot.feedback == (
        "未执行：与已有墙重叠或交叉。回合没有推进。"
    )


def test_wall_colors_and_inventory_cards_follow_player_identity() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Orientation, PlaceWall, Player, WallAnchor
    from quoridor_rl.pygame_ui.app import COLORS, Control, PygameApplication

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.draw(surface)

    player_0_wall = PlaceWall(WallAnchor(3, 3), Orientation.HORIZONTAL)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.HORIZONTAL_WALL).center,
        )
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEMOTION,
            pos=application.wall_anchor_point(player_0_wall.anchor),
        )
    )
    application.draw(surface)
    assert (
        surface.get_at(application.wall_anchor_point(player_0_wall.anchor))
        == (COLORS["p0"])
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.wall_anchor_point(player_0_wall.anchor),
        )
    )
    application.draw(surface)

    assert (
        surface.get_at(application.player_status_rect(Player.PLAYER_1).midtop)
        == (COLORS["p1"])
    )
    assert (
        surface.get_at(
            application.wall_inventory_segment_rect(Player.PLAYER_0, 8).center
        )
        == COLORS["p0"]
    )
    assert (
        surface.get_at(
            application.wall_inventory_segment_rect(Player.PLAYER_0, 9).center
        )
        == COLORS["paper"]
    )
    assert (
        surface.get_at(
            application.wall_inventory_segment_rect(Player.PLAYER_1, 9).center
        )
        == COLORS["p1"]
    )

    player_1_wall = PlaceWall(WallAnchor(5, 5), Orientation.VERTICAL)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.VERTICAL_WALL).center,
        )
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEMOTION,
            pos=application.wall_anchor_point(player_1_wall.anchor),
        )
    )
    application.draw(surface)
    assert (
        surface.get_at(application.wall_anchor_point(player_1_wall.anchor))
        == (COLORS["p1"])
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.wall_anchor_point(player_1_wall.anchor),
        )
    )
    application.draw(surface)

    assert (
        surface.get_at(application.wall_anchor_point(player_0_wall.anchor))
        == (COLORS["p0"])
    )
    assert (
        surface.get_at(application.wall_anchor_point(player_1_wall.anchor))
        == (COLORS["p1"])
    )
    assert (
        surface.get_at(application.player_status_rect(Player.PLAYER_0).midtop)
        == (COLORS["p0"])
    )
    assert (
        surface.get_at(
            application.wall_inventory_segment_rect(Player.PLAYER_1, 9).center
        )
        == COLORS["paper"]
    )
    assert application.snapshot.position is not None
    assert application.snapshot.position.walls_remaining == (9, 9)


def test_agent_game_waits_pauses_and_advances_exactly_one_step() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import MovePawn, Position, Square
    from quoridor_rl.pygame_ui.app import Control, PygameApplication

    class FirstLegalAgent:
        def choose_action(self, position: Position):
            return position.legal_actions()[0]

    application = PygameApplication(agent_factory=lambda seed: FirstLegalAgent())
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.MODE_RANDOM_RANDOM).center,
        )
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.START_GAME).center,
        )
    )

    application.update(499)
    assert application.snapshot.position == Position.initial()
    application.update(1)
    after_first = Position.initial().play(MovePawn(Square(7, 4)))
    assert application.snapshot.position == after_first
    assert application.snapshot.plies == 1

    application.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    application.update(500)
    assert application.snapshot.position == after_first
    application.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    assert application.snapshot.plies == 2
    assert application.snapshot.paused is True


def test_human_can_choose_second_player_identity_against_seeded_agent() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Player, Position
    from quoridor_rl.pygame_ui.app import Control, GameMode, PygameApplication

    class FirstLegalAgent:
        def choose_action(self, position: Position):
            return position.legal_actions()[0]

    application = PygameApplication(agent_factory=lambda seed: FirstLegalAgent())
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (
        Control.MODE_HUMAN_RANDOM,
        Control.HUMAN_PLAYER_1,
        Control.SEED_INPUT,
    ):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    for character in "42":
        application.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                key=ord(character),
                unicode=character,
            )
        )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.START_GAME).center,
        )
    )

    assert application.snapshot.game_mode is GameMode.HUMAN_RANDOM
    assert application.snapshot.human_player is Player.PLAYER_1
    assert application.snapshot.seed == 42
    application.update(500)
    assert application.snapshot.plies == 1
    assert application.snapshot.position is not None
    assert application.snapshot.position.to_move is Player.PLAYER_1
    application.update(500)
    assert application.snapshot.plies == 1


def test_completed_game_shows_result_and_can_restart_with_same_mode() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Player, Position, Square
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        GameMode,
        PygameApplication,
    )

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )

    targets = (
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
    )
    for target in targets:
        application.draw(surface)
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.square_rect(target).center,
            )
        )

    assert application.snapshot.screen is ApplicationScreen.RESULT
    assert application.snapshot.position is not None
    assert application.snapshot.position.winner is Player.PLAYER_0
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.RESTART_GAME).center,
        )
    )

    assert application.snapshot.screen is ApplicationScreen.PLAYING
    assert application.snapshot.game_mode is GameMode.HUMAN_HUMAN
    assert application.snapshot.position == Position.initial()
    assert application.snapshot.plies == 0
    assert tuple(entry.ply for entry in application.snapshot.action_history) == (0,)


def test_action_limit_ends_the_game_as_undecided_not_a_draw() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Square
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        PygameApplication,
    )

    application = PygameApplication(max_plies=1)
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.square_rect(Square(7, 4)).center,
        )
    )

    assert application.snapshot.screen is ApplicationScreen.RESULT
    assert application.snapshot.position is not None
    assert application.snapshot.position.winner is None
    assert application.snapshot.feedback == "达到 1 手行动上限，本局未决。"
    assert "平局" not in application.snapshot.feedback


def test_agent_that_submits_illegal_action_loses_without_changing_position() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import MovePawn, Player, Position, Square
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        PygameApplication,
    )

    class IllegalAgent:
        def choose_action(self, position: Position):
            return MovePawn(Square(0, 0))

    application = PygameApplication(agent_factory=lambda seed: IllegalAgent())
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_RANDOM_RANDOM, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.update(500)

    assert application.snapshot.screen is ApplicationScreen.RESULT
    assert application.snapshot.position == Position.initial()
    assert application.snapshot.winner is Player.PLAYER_1
    assert application.snapshot.feedback == ("player_0 提交非法动作，player_1 获胜。")


def test_agent_exception_aborts_without_creating_a_winner() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Position
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        PygameApplication,
    )

    class BrokenAgent:
        def choose_action(self, position: Position):
            raise RuntimeError("模型损坏")

    application = PygameApplication(agent_factory=lambda seed: BrokenAgent())
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_RANDOM_RANDOM, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.update(500)

    assert application.snapshot.screen is ApplicationScreen.ERROR
    assert application.snapshot.winner is None
    assert application.snapshot.position == Position.initial()
    assert application.snapshot.feedback == "智能体运行失败：模型损坏"


def test_application_uses_packaged_chinese_font_with_its_license() -> None:
    pygame = pytest.importorskip("pygame")
    from importlib import resources

    from quoridor_rl.pygame_ui.app import PygameApplication

    application = PygameApplication()
    assets = resources.files("quoridor_rl.pygame_ui.assets")
    font_resource = assets.joinpath(application.snapshot.font_resource)
    license_resource = assets.joinpath("OFL.txt")
    font = pygame.font.Font(font_resource.open("rb"), 32)
    first = pygame.image.tobytes(font.render("围", True, "white"), "RGBA")
    second = pygame.image.tobytes(font.render("墙", True, "white"), "RGBA")

    assert first != second
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_resource.read_text(
        encoding="utf-8"
    )


def test_agent_playback_controls_select_speed_pause_and_step() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Position
    from quoridor_rl.pygame_ui.app import Control, PygameApplication

    class FirstLegalAgent:
        def choose_action(self, position: Position):
            return position.legal_actions()[0]

    application = PygameApplication(agent_factory=lambda seed: FirstLegalAgent())
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (
        Control.MODE_RANDOM_RANDOM,
        Control.SPEED_FAST,
        Control.START_GAME,
    ):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.update(199)
    assert application.snapshot.plies == 0
    application.update(1)
    assert application.snapshot.plies == 1

    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.PAUSE_RESUME).center,
        )
    )
    application.update(200)
    assert application.snapshot.plies == 1
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.STEP).center,
        )
    )
    assert application.snapshot.plies == 2
    assert application.snapshot.agent_delay_ms == 200


def test_last_move_remains_visible_until_another_legal_action() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import MovePawn, Square
    from quoridor_rl.pygame_ui.app import Control, PygameApplication

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.square_rect(Square(7, 4)).center,
        )
    )
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0))
    )

    assert application.snapshot.last_action == MovePawn(Square(7, 4))
    assert application.snapshot.last_move_start == Square(8, 4)


def test_resizing_keeps_board_and_controls_visible_above_minimum_size() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Player, Square
    from quoridor_rl.pygame_ui.app import (
        WINDOW_MIN,
        Control,
        PygameApplication,
    )

    application = PygameApplication()
    initial = pygame.Surface((1280, 800))
    application.draw(initial)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.handle_event(
        pygame.event.Event(pygame.VIDEORESIZE, size=(700, 400), w=700, h=400)
    )
    assert application.snapshot.window_size == WINDOW_MIN

    for size in (WINDOW_MIN, (1280, 800), (1600, 950)):
        surface = pygame.Surface(size)
        application.draw(surface)
        visible = surface.get_rect()
        assert visible.contains(application.square_rect(Square(0, 0)))
        assert visible.contains(application.square_rect(Square(8, 8)))
        assert visible.contains(application.control_rect(Control.HORIZONTAL_WALL))
        for player in Player:
            assert visible.contains(application.player_status_rect(player))
            for index in range(10):
                assert visible.contains(
                    application.wall_inventory_segment_rect(player, index)
                )


def test_playing_sidebar_exposes_match_details_below_inventory_cards() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl.pygame_ui.app import Control, PygameApplication

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.draw(surface)

    assert application.snapshot.status_lines == (
        "已行动：0 手",
        "当前操作：移动",
        "对局开始：player_0 行动。",
    )


def test_result_exit_button_stops_the_application() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Square
    from quoridor_rl.pygame_ui.app import Control, PygameApplication

    application = PygameApplication(max_plies=1)
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.square_rect(Square(7, 4)).center,
        )
    )
    application.draw(surface)

    should_continue = application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.EXIT).center,
        )
    )

    assert should_continue is False


def test_invalid_seed_stays_on_start_screen_with_explanation() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        PygameApplication,
    )

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_RANDOM, Control.SEED_INPUT):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.handle_event(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, unicode="x")
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.START_GAME).center,
        )
    )

    assert application.snapshot.screen is ApplicationScreen.START
    assert application.snapshot.feedback == "随机种子必须是整数或留空。"


def test_action_history_records_and_colors_successful_actions_newest_first() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import (
        MovePawn,
        Orientation,
        PlaceWall,
        Player,
        Position,
        Square,
        WallAnchor,
    )
    from quoridor_rl.pygame_ui.app import (
        COLORS,
        Control,
        PygameApplication,
        _action_history_text,
    )

    application = PygameApplication()
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.square_rect(Square(7, 4)).center,
        )
    )
    application.draw(surface)
    wall = PlaceWall(WallAnchor(3, 3), Orientation.HORIZONTAL)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.HORIZONTAL_WALL).center,
        )
    )
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.wall_anchor_point(wall.anchor),
        )
    )
    application.draw(surface)

    history = application.snapshot.action_history
    assert tuple(entry.ply for entry in history) == (0, 1, 2)
    assert history[0].player is None
    assert history[0].action is None
    assert history[0].resulting_position == Position.initial()
    assert history[1].player is Player.PLAYER_0
    assert history[1].action == MovePawn(Square(7, 4))
    assert history[1].move_start == Square(8, 4)
    assert history[2].player is Player.PLAYER_1
    assert history[2].action == wall
    assert history[2].resulting_position == history[1].resulting_position.play(wall)
    assert _action_history_text(history[0]) == "0. 初始局面"
    assert _action_history_text(history[1]) == "1. player_0 移动 e1 → e2"
    assert _action_history_text(history[2]) == "2. player_1 放置横墙 d5"

    newest = application.history_entry_rect(2)
    previous = application.history_entry_rect(1)
    initial = application.history_entry_rect(0)
    assert newest.top < previous.top < initial.top
    assert surface.get_at((newest.left + 11, newest.centery)) == COLORS["p1"]
    assert surface.get_at((previous.left + 11, previous.centery)) == COLORS["p0"]
    assert surface.get_at((initial.left + 11, initial.centery)) == COLORS["muted"]


def test_live_history_is_read_only_and_scrolls_to_the_initial_position() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Position
    from quoridor_rl.pygame_ui.app import WINDOW_MIN, Control, PygameApplication

    class FirstLegalAgent:
        def choose_action(self, position: Position):
            return position.legal_actions()[0]

    application = PygameApplication(agent_factory=lambda seed: FirstLegalAgent())
    surface = pygame.Surface(WINDOW_MIN)
    application.draw(surface)
    for control in (Control.MODE_RANDOM_RANDOM, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    for _ in range(6):
        application.update(500)
    application.draw(surface)

    live_position = application.snapshot.position
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.history_entry_rect(4).center,
        )
    )
    assert application.snapshot.reviewing_history is False
    assert application.snapshot.displayed_position == live_position

    application.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-100, x=0))
    application.draw(surface)
    assert application.history_entry_rect(0)
    application.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=100, x=0))
    application.draw(surface)
    assert application.history_entry_rect(6)


def test_completed_game_can_replay_history_without_changing_final_position() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Position, Square
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        PygameApplication,
    )

    application = PygameApplication(max_plies=2)
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    for target in (Square(7, 4), Square(1, 4)):
        application.draw(surface)
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.square_rect(target).center,
            )
        )

    assert application.snapshot.screen is ApplicationScreen.RESULT
    final_position = application.snapshot.position
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.VIEW_HISTORY).center,
        )
    )
    assert application.snapshot.reviewing_history is True
    assert application.snapshot.reviewed_ply == 2
    assert application.snapshot.displayed_position == final_position

    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.history_entry_rect(0).center,
        )
    )
    assert application.snapshot.position == final_position
    assert application.snapshot.reviewed_ply == 0
    assert application.snapshot.displayed_position == Position.initial()
    assert application.snapshot.status_lines[0] == "历史回放：第 0 / 2 手"

    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.RETURN_TO_RESULT).center,
        )
    )
    assert application.snapshot.reviewing_history is False
    assert application.snapshot.reviewed_ply is None
    assert application.snapshot.displayed_position == final_position


def test_pygame_can_switch_to_english_and_keeps_it_across_navigation() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Language, Square
    from quoridor_rl.pygame_ui.app import (
        ApplicationScreen,
        Control,
        PygameApplication,
    )

    application = PygameApplication(max_plies=1)
    surface = pygame.Surface((1280, 800))
    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.LANGUAGE_ENGLISH).center,
        )
    )

    assert application.snapshot.language is Language.ENGLISH
    assert application.snapshot.window_title == "Quoridor"
    assert application.snapshot.feedback == "Choose a game mode to begin."

    application.draw(surface)
    for control in (Control.MODE_HUMAN_HUMAN, Control.START_GAME):
        application.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=application.control_rect(control).center,
            )
        )
    assert application.snapshot.status_lines == (
        "Plies: 0",
        "Action: Move",
        "Game started: player_0 to move.",
    )

    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.square_rect(Square(7, 4)).center,
        )
    )
    assert application.snapshot.screen is ApplicationScreen.RESULT
    assert application.snapshot.feedback == ("Reached the 1-ply limit; game undecided.")

    application.draw(surface)
    application.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=application.control_rect(Control.RETURN_TO_START).center,
        )
    )
    assert application.snapshot.screen is ApplicationScreen.START
    assert application.snapshot.language is Language.ENGLISH
    assert application.snapshot.feedback == "Choose a game mode to begin."


def test_english_pygame_catalog_history_and_supported_sizes_are_complete() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Language, MovePawn, Player, Position, Square
    from quoridor_rl.pygame_ui.app import (
        PYGAME_TEXT,
        WINDOW_MIN,
        ActionHistoryEntry,
        Control,
        PygameApplication,
        _action_history_text,
    )

    assert PYGAME_TEXT[Language.CHINESE].keys() == PYGAME_TEXT[Language.ENGLISH].keys()
    entry = ActionHistoryEntry(
        ply=1,
        player=Player.PLAYER_0,
        action=MovePawn(Square(7, 4)),
        resulting_position=Position.initial().play(MovePawn(Square(7, 4))),
        move_start=Square(8, 4),
    )
    assert _action_history_text(entry, language=Language.ENGLISH) == (
        "1. player_0 moved e1 → e2"
    )

    application = PygameApplication(language=Language.ENGLISH)
    for size in (WINDOW_MIN, (1280, 800), (1600, 950)):
        surface = pygame.Surface(size)
        application.draw(surface)
        visible = surface.get_rect()
        for control in (
            Control.LANGUAGE_CHINESE,
            Control.LANGUAGE_ENGLISH,
            Control.MODE_HUMAN_HUMAN,
            Control.MODE_HUMAN_RANDOM,
            Control.MODE_RANDOM_RANDOM,
            Control.START_GAME,
        ):
            assert visible.contains(application.control_rect(control))

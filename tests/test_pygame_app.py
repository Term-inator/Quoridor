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


def test_human_wall_preview_uses_rules_and_illegal_click_does_not_advance() -> None:
    pygame = pytest.importorskip("pygame")
    from quoridor_rl import Orientation, PlaceWall, Position, WallAnchor
    from quoridor_rl.pygame_ui.app import Control, InputMode, PygameApplication

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
    assert application.snapshot.feedback == (
        "未执行：与已有墙重叠或交叉。回合没有推进。"
    )


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
    from quoridor_rl import Square
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


def test_playing_sidebar_exposes_complete_match_status() -> None:
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
        "当前行动方：player_0",
        "剩余墙：player_0 10 · player_1 10",
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

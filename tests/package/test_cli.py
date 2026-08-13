import subprocess


def test_console_command_completes_a_human_vs_human_game() -> None:
    moves = (
        "move e2\nmove d9\nmove e3\nmove e9\nmove e4\nmove d9\n"
        "move e5\nmove e9\nmove e6\nmove d9\nmove e7\nmove e9\n"
        "move e8\nmove d9\nmove e9"
    )

    completed = subprocess.run(
        ["quoridor", "--opponent", "human"],
        input=f"{moves}\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "玩家 1 获胜" in completed.stdout


def test_console_command_runs_human_vs_random_and_reports_truncation() -> None:
    completed = subprocess.run(
        ["quoridor", "--opponent", "random", "--seed", "3", "--max-plies", "1"],
        input="move e2\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "达到 1 手上限，对局截断" in completed.stdout


def test_console_command_explains_bad_input_and_illegal_moves() -> None:
    completed = subprocess.run(
        ["quoridor"],
        input="move z0\nmove e3\nquit\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "输入错误：格子必须为 a1 到 i9" in completed.stdout
    assert "不合法动作：illegal_pawn_move" in completed.stdout


def test_console_command_supports_complete_english_output() -> None:
    completed = subprocess.run(
        ["quoridor", "--language", "en", "--max-plies", "1"],
        input="move z0\nmove e2\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Commands: move e2" in completed.stdout
    assert "Input error: square must be between a1 and i9" in completed.stdout
    assert "Walls: player_0 = 10, player_1 = 10" in completed.stdout
    assert "Reached the 1-ply limit; game truncated." in completed.stdout
    assert "玩家" not in completed.stdout


def test_console_help_uses_the_selected_language() -> None:
    completed = subprocess.run(
        ["quoridor", "--language", "en", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Play two-player Quoridor in the terminal" in completed.stdout
    assert "interface language (default: zh)" in completed.stdout
    assert "玩家" not in completed.stdout

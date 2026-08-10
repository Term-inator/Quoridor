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

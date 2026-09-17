from __future__ import annotations

import subprocess
import sys


def test_utterrender_imports_against_current_utterplan() -> None:
    from utterplan import UtterancePlan, UtterancePlanner

    import utterrender

    assert utterrender is not None
    assert UtterancePlan is not None
    assert UtterancePlanner is not None


def test_module_cli_help_starts() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "utterrender", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_script_cli_help_starts() -> None:
    result = subprocess.run(
        ["utterrender", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()

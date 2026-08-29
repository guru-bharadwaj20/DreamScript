"""Phase 0.2.5 acceptance test — the task runner covers every target and stays in sync.

`make` is not installed on the Windows dev machine, so `tasks.ps1` is what actually runs
there while the Makefile remains the reference for Linux and CI. The two must not drift.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
TASKS_PS1 = ROOT / "tasks.ps1"

# Targets that exist only for make (environment bootstrap, unix housekeeping).
MAKE_ONLY = {"help", "env", "clean", "clean-experiments", "verify-classical", "test-fast"}


def make_targets() -> set[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^([a-z][a-z0-9-]*):.*?##", text, re.MULTILINE)}


def ps1_tasks() -> set[str]:
    text = TASKS_PS1.read_text(encoding="utf-8")
    block = text.split("$Tasks = [ordered]@{", 1)[1]
    return {m.group(1) for m in re.finditer(r'^\s*"([a-z][a-z0-9-]*)"\s*=', block, re.MULTILINE)}


def test_makefile_and_task_runner_agree():
    missing = (make_targets() - MAKE_ONLY) - ps1_tasks()
    assert not missing, f"targets in Makefile but not tasks.ps1: {sorted(missing)}"


def test_every_pipeline_stage_has_a_target():
    targets = ps1_tasks()
    for expected in ["data", "preprocess", "features", "train-clf", "eval", "app", "serve"]:
        assert expected in targets


@pytest.mark.parametrize("task", sorted(ps1_tasks()))
def test_task_dispatches(task: str):
    """Every task must resolve to a runnable command line (checked with -DryRun)."""
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(TASKS_PS1),
            task,
            "-DryRun",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "python.exe" in result.stdout


def test_unknown_task_fails_loudly():
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(TASKS_PS1),
            "no-such-task",
            "-DryRun",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "unknown task" in (result.stderr + result.stdout)

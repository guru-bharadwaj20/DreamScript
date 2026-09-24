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
    # Three tasks are the venv rather than a command inside it: `env` builds it with `uv`,
    # `clean` and `clean-experiments` remove directories. They still have to print what they
    # would do, which is what -DryRun is for.
    if task in {"env", "clean", "clean-experiments"}:
        assert result.stdout.strip(), f"{task} -DryRun printed nothing"
    else:
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


# --- one for one with the Makefile, as the docstring claims (audit 34) ------------------------


def _task_names() -> set[str]:
    text = (ROOT / "tasks.ps1").read_text(encoding="utf-8")
    block = text.split("$Tasks = [ordered]@{", 1)[1].split("\n}", 1)[0]
    return set(re.findall(r'^\s*"([a-z][a-z0-9-]*)"\s*=', block, re.M))


def _make_names() -> set[str]:
    return {
        match.group(1)
        for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^([A-Za-z][A-Za-z0-9_-]*)\s*:(?!=)", line))
    }


def test_tasks_ps1_mirrors_every_makefile_target():
    """The docstring says "mirrors every Makefile target one-for-one". It was missing env,
    clean, clean-experiments, verify-classical and stages, so the claim was false for five."""
    missing = _make_names() - _task_names() - {"help"}
    assert not missing, sorted(missing)


def test_no_task_exists_that_the_makefile_does_not_have():
    extra = _task_names() - _make_names() - {"help"}
    assert not extra, sorted(extra)


def test_lint_runs_all_three_checkers_because_ci_does():
    """It ran ruff alone, so the Windows lint path was weaker than the gate it stands in for:
    a commit could pass `./tasks.ps1 lint` and fail CI on black or isort."""
    text = (ROOT / "tasks.ps1").read_text(encoding="utf-8")
    block = text.split('"lint"', 1)[1].split('"format"', 1)[0]
    for tool in ("ruff", "black", "isort"):
        assert tool in block, f"lint does not run {tool}"


def test_format_applies_all_three():
    text = (ROOT / "tasks.ps1").read_text(encoding="utf-8")
    block = text.split('"format"', 1)[1].split('"repro"', 1)[0]
    for tool in ("isort", "black", "ruff"):
        assert tool in block


def test_repro_prepends_the_venv_to_path():
    """Without it `dvc repro` spawns `python`, Windows resolves it to the Store stub and every
    stage exits 9009 - the trap dvc.yaml documents and `make repro` guards against."""
    text = (ROOT / "tasks.ps1").read_text(encoding="utf-8")
    block = text.split('"repro"', 1)[1].split('"dag"', 1)[0]
    assert "PrependPath" in block
    assert "$env:PATH" in text


def test_no_task_still_invokes_a_stage_entry_point_with_only_a_config():
    text = (ROOT / "tasks.ps1").read_text(encoding="utf-8")
    assert "configs/$configName.yaml" not in text
    assert "uvicorn" in text

"""Phase 0.2.3 - the package entry points route to the work in the package.

Eleven `__main__.py` files shipped a `run()` that raised `StageNotImplemented`, so
`python -m src.detect` printed a resolved config and exited 2 while `src/detect/` held sixteen
modules with working `main()` functions, reachable only by knowing their names. The contract was
not wrong, only empty: `--config` and `--print-config` still go where they always went.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.utils.stage import dispatch, packages, runnable

STAGES = (
    "src.ingest",
    "src.preprocess",
    "src.features",
    "src.classify",
    "src.detect",
    "src.ocr",
    "src.parse",
    "src.rl",
    "src.synth",
    "src.eval",
    "src.serve",
)


@pytest.mark.parametrize("package", STAGES)
def test_every_stage_entry_point_offers_at_least_one_command(package):
    commands = runnable(package)
    assert commands, f"{package} has a __main__.py and nothing to run"
    assert all(summary for summary in commands.values()), "a command with no docstring summary"


@pytest.mark.parametrize("package", STAGES)
def test_no_stage_entry_point_still_raises_stage_not_implemented(package):
    from pathlib import Path

    from src.utils.config import ROOT

    text = Path(ROOT, *package.split("."), "__main__.py").read_text(encoding="utf-8")
    assert "StageNotImplemented" not in text.split('"""')[-1]
    assert "dispatch(" in text


def test_listing_is_the_answer_to_no_arguments(capsys):
    assert dispatch("src.synth", []) == 0
    printed = capsys.readouterr().out
    assert "graphs" in printed
    assert "python -m src.synth" in printed


def test_an_unknown_command_is_exit_2_and_names_what_there_is(capsys):
    assert dispatch("src.synth", ["nosuch"]) == 2
    assert "no command 'nosuch'" in capsys.readouterr().err


def test_discovery_reads_rather_than_imports():
    """Importing sixteen modules to print a list of names means importing torch, ultralytics and
    OpenCV in order to answer `--help`."""
    import inspect

    from src.utils import stage

    source = inspect.getsource(stage.runnable)
    assert "ast.parse" in source
    assert "import_module" not in source


def test_a_named_command_actually_runs():
    """End to end through a real interpreter, because the thing under test is `python -m`."""
    result = subprocess.run(
        [sys.executable, "-m", "src.synth", "graphs", "--summary"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "flowchart" in result.stdout


def test_the_config_contract_is_untouched():
    result = subprocess.run(
        [sys.executable, "-m", "src.ingest", "--print-config"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "project: dreamscript" in result.stdout


def test_packages_finds_every_package_with_a_main():
    assert set(STAGES) <= set(packages())

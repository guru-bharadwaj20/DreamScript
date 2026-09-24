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


# --- a module reachable by name is not an orphan (audit 42) -----------------------------------

#: Packages that gained an entry point for their own sake rather than as a pipeline stage.
LIBRARY_PACKAGES = (
    "src.assemble",
    "src.cluster",
    "src.codegen",
    "src.embed",
    "src.ir",
    "src.llm",
    "src.mlops",
)

#: Of those, the ones with no `configs/<name>.yaml`, which do not want one. `src.llm` is the
#: exception and keeps the config contract: `configs/llm.yaml` drives `src.llm.run`, and it is
#: one of the two configs in the repo that steers real code.
NO_CONFIG = tuple(p for p in LIBRARY_PACKAGES if p != "src.llm")

#: The five modules the audit found imported by nothing and named in no config, doc or workflow,
#: and the command each is reachable as now. Named rather than counted, because the point is that
#: a specific module stopped being unreachable.
FORMER_ORPHANS = {
    "src.assemble": "labels",
    "src.llm": "sweep",
    "src.ingest": "balance",
    "src.ocr": "linewise",
}


@pytest.mark.parametrize("package", LIBRARY_PACKAGES)
def test_every_package_with_runnable_modules_has_an_entry_point(package):
    from pathlib import Path

    from src.utils.config import ROOT

    assert runnable(package), f"{package} was given an entry point and has nothing to run"
    assert Path(ROOT, *package.split("."), "__main__.py").is_file()


@pytest.mark.parametrize(("package", "command"), sorted(FORMER_ORPHANS.items()))
def test_a_former_orphan_is_reachable_by_name(package, command):
    assert command in runnable(package)


def test_the_hidden_table_is_the_only_thing_keeping_a_package_without_an_entry_point():
    """`src.utils.cli.main(package, run, argv)` is the contract `dispatch` calls, not a command.
    Detecting it as one gave `src.utils` an entry point whose only offer was the machinery behind
    every other entry point."""
    from src.utils.stage import HIDDEN

    assert "cli" in HIDDEN
    assert not runnable("src.utils")


def test_no_package_with_runnable_modules_is_left_without_one():

    from src.utils.config import ROOT

    missing = []
    for directory in sorted((ROOT / "src").iterdir()):
        if not directory.is_dir() or directory.name.startswith(("_", ".")):
            continue
        package = f"src.{directory.name}"
        if runnable(package) and not (directory / "__main__.py").is_file():
            missing.append(package)
    assert not missing, missing


@pytest.mark.parametrize("package", NO_CONFIG)
def test_a_library_package_does_not_advertise_a_config_it_has_not_got(package, capsys):
    from src.utils.stage import has_config

    assert not has_config(package)
    dispatch(package, [])
    assert "--print-config" not in capsys.readouterr().out
    assert dispatch(package, ["--print-config"]) == 2


# --- a producer says it is a producer (audit 43) ----------------------------------------------

#: The six modules whose only importer was their own test - 2,182 lines that read as dead from
#: the outside. Each is a producer or a report generator, which is a different thing from dead,
#: and each now says which in its first lines and names the command that runs it.
PRODUCERS = {
    "src/ir/targets.py": "src.ir targets",
    "src/ir/agreement.py": "src.ir agreement",
    "src/preprocess/failures.py": "src.preprocess failures",
    "src/ir/convert/didi.py": "src.ir convert.didi",
    "src/ir/convert/sketch2code.py": "src.ir convert.sketch2code",
    "src/ingest/audit.py": "src.ingest audit",
}


@pytest.mark.parametrize(("path", "command"), sorted(PRODUCERS.items()))
def test_each_producer_names_the_command_that_runs_it(path: str, command: str):
    from src.utils.config import ROOT

    head = (ROOT / path).read_text(encoding="utf-8").split('"""')[1]
    assert f"python -m {command}" in head, f"{path} does not say how to run it"


@pytest.mark.parametrize(("path", "command"), sorted(PRODUCERS.items()))
def test_each_producer_is_reachable_by_that_command(path: str, command: str):
    package, name = command.rsplit(" ", 1)
    assert name in runnable(package)

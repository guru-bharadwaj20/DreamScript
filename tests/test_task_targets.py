"""Phase 0.2.5 - every Makefile pipeline target names work that exists.

Fourteen of the twenty-four targets ran `python -m src.<package> --config ...` and every one of
those raised `StageNotImplemented` and exited 2. `make data`, `make features`, `make detect`,
`make eval`, `make serve` - all of them did nothing, for as long as the stub entry points did
nothing, and nothing in the repo noticed because nothing checked.

This checks. It reads the Makefile, pulls out every `python -m` invocation, and asserts the
module or the package command on the other end is real - without running any of them, because
`make detect` is a GPU training run.
"""

from __future__ import annotations

import re

import pytest

from src.utils.config import ROOT
from src.utils.stage import runnable

MAKEFILE = ROOT / "Makefile"

#: `python -m <thing>` invocations that are not this project's code.
EXTERNAL = {"pytest", "ruff", "black", "isort", "dvc", "uvicorn"}


def invocations() -> list[tuple[str, str]]:
    """`(module, first argument)` for every `$(PY) -m ...` line in the Makefile."""
    out = []
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        for match in re.finditer(r"\$\(PY\) -m ([\w.]+)(?:\s+([\w.]+))?", line):
            module = match.group(1)
            # `make stages` loops `-m src.$$pkg` over the package names; the loop body is not a
            # target to resolve, and the packages it names are covered by test_utils_stage.
            if module.endswith("."):
                continue
            out.append((module, match.group(2) or ""))
    return out


def test_the_makefile_still_invokes_things():
    assert len(invocations()) >= 20


@pytest.mark.parametrize(("module", "argument"), invocations())
def test_every_python_m_target_resolves(module: str, argument: str):
    if module in EXTERNAL:
        return
    assert module.startswith("src."), module
    path = ROOT / module.replace(".", "/")
    if (path / "__main__.py").is_file():
        # A package entry point. If the line names a command, the package must have it.
        if argument and not argument.startswith("$"):
            commands = runnable(module)
            assert (
                argument in commands
            ), f"make target runs `{module} {argument}`, which is not a command"
        return
    assert path.with_suffix(".py").is_file(), f"{module} is neither a package nor a module"


def test_no_target_still_points_at_a_stage_that_raises():
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "--config configs/detect.yaml" not in text
    assert "-m src.serve --config" not in text
    # The real server, the one the Dockerfile and the CI image job run.
    assert "uvicorn src.serve.api:app" in text

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


# --- .dvcignore is not the stock template (audit 23) ------------------------------------------


def test_dvcignore_excludes_the_trees_dvc_could_never_track():
    """It shipped as three comment lines and no patterns, so every `dvc status` walked `.venv/`,
    `mlruns/`, every `__pycache__` and 1.3 GB of checkpoints."""
    text = (ROOT / ".dvcignore").read_text(encoding="utf-8")
    patterns = {
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    }
    assert patterns, ".dvcignore is still the stock template"
    assert {".venv/", "__pycache__/", "mlruns/"} <= patterns


def test_dvcignore_never_hides_a_file_inside_a_tracked_output():
    """Ignoring a file inside a directory DVC tracks removes it from that directory's hash, which
    reports the output as modified and re-pushes it - the opposite of what this file is for.
    Three stages declare directories under `experiments/` that all contain `weights/last.pt`."""
    text = (ROOT / ".dvcignore").read_text(encoding="utf-8")
    patterns = [
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    tracked_roots = (
        "experiments/detect",
        "experiments/ocr",
        "experiments/llm",
        "data/raw",
        "data/processed",
        "data/features",
    )
    for pattern in patterns:
        assert not any(pattern.startswith(root) for root in tracked_roots), pattern


# --- the app boundary is stated, and the dependency it declares is satisfiable (audit 25) -----


def _compose() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_app_is_a_readme_not_two_invisible_placeholders():
    """`app/backend/.gitkeep` and `app/frontend/.gitkeep` said nothing about what was missing."""
    assert not (ROOT / "app" / "backend").exists()
    assert not (ROOT / "app" / "frontend").exists()
    text = (ROOT / "app" / "README.md").read_text(encoding="utf-8")
    assert "There is no code here yet" in text
    assert "uvicorn src.serve.api:app" in text


def test_a_service_depended_on_for_health_declares_a_healthcheck():
    """`docker compose config` parses, it does not resolve conditions - so an `app` service
    depending on `model` with `condition: service_healthy`, against a `model` with no
    `healthcheck`, would have failed the first time anyone dropped the profile. CI's compose step
    passed throughout."""
    services = _compose()["services"]
    for name, service in services.items():
        for target, spec in (service.get("depends_on") or {}).items():
            if isinstance(spec, dict) and spec.get("condition") == "service_healthy":
                assert (
                    "healthcheck" in services[target]
                ), f"{name} waits for {target} to be healthy and {target} has no healthcheck"


def test_the_model_healthcheck_uses_the_probe_that_does_not_load_the_model():
    check = " ".join(_compose()["services"]["model"]["healthcheck"]["test"])
    assert "/health" in check

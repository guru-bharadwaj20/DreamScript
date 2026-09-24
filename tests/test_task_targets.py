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
BACKSLASH = chr(92)

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


# --- .PHONY and the rules are the same set (audit 33) -----------------------------------------


def _makefile_targets() -> set[str]:
    return {
        match.group(1)
        for line in MAKEFILE.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^([A-Za-z][A-Za-z0-9_-]*)\s*:(?!=)", line))
    }


def _phony() -> set[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    block = text.split(".PHONY:", 1)[1]
    lines = []
    for line in block.splitlines():
        lines.append(line.rstrip(BACKSLASH + " "))
        if not line.rstrip().endswith(BACKSLASH):
            break
    return set(" ".join(lines).split())


def test_phony_declares_every_target_and_no_others():
    """It had drifted both ways: `verify-classical` was declared with no rule behind it, so
    `make verify-classical` answered "No rule to make target", and `dummy-run` had a rule and was
    not declared, so a file named `dummy-run` would have shadowed it."""
    assert _phony() == _makefile_targets()


def test_verify_classical_has_the_rule_its_declaration_promised():
    assert "verify-classical" in _makefile_targets()
    assert "--only classical" in MAKEFILE.read_text(encoding="utf-8")


# --- pytest's totals line survives the project's own defaults (audit 41a) ---------------------


def test_addopts_does_not_stack_a_second_quiet_flag():
    """`addopts = "-q"` plus a command-line `-q` is `-qq`, which suppresses the totals line
    entirely: no pass count, no fail count, no skip count, no duration. Both CI test steps pass
    `-q`, so CI's own logs ended at the `short test summary info` block."""
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = config["tool"]["pytest"]["ini_options"].get("addopts", "")
    assert "-q" not in addopts.split(), addopts
    assert "--quiet" not in addopts.split(), addopts


def test_no_ci_step_passes_a_flag_that_would_stack_with_addopts():
    import re
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = set(config["tool"]["pytest"]["ini_options"].get("addopts", "").split())
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for line in re.findall(r"^.*python -m pytest.*$", workflow, re.M):
        passed = [token for token in line.split() if token.startswith("-")]
        clash = addopts.intersection(passed)
        assert not clash, f"{line.strip()!r} repeats {sorted(clash)} from addopts"


# --- mypy is in a gate (audit 38) -------------------------------------------------------------


def test_mypy_is_configured_and_ignores_third_party_stubs():
    """661 errors, 318 of them `Library stubs not installed for "cv2"`-shaped reports about
    third-party packages, which say nothing about this repo's code."""
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = config["tool"]["mypy"]
    assert mypy["ignore_missing_imports"] is True
    assert mypy["files"] == ["src"]


def test_mypy_runs_in_make_lint_and_in_ci():
    """It was in no gate at all - not `make lint`, not pre-commit, not CI - while a
    `.mypy_cache/` sat in the tree proving it was being run by hand and ignored."""
    makefile = MAKEFILE.read_text(encoding="utf-8")
    lint = makefile.split("lint:", 1)[1].split("\n\n", 1)[0]
    assert "typecheck.py" in lint
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/typecheck.py" in workflow
    assert "mypy==" in workflow


def test_the_baseline_exists_and_is_per_error_code():
    """A single total makes a regression illegible: "3 new arg-type" is a review comment,
    "344 errors" is not."""
    import json

    baseline = json.loads((ROOT / "reports" / "mypy_baseline.json").read_text(encoding="utf-8"))
    assert baseline["total"] == sum(baseline["by_code"].values())
    assert baseline["by_code"], "the baseline records no error codes"


def test_mypy_is_declared_where_the_thing_that_runs_it_is_installed():
    assert "mypy" in (ROOT / "requirements" / "dev.txt").read_text(encoding="utf-8")


# --- line endings are the repository's decision, not the clone's (audit 53) -------------------


def test_gitattributes_exists_and_normalises_text():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text


def test_the_two_file_types_windows_needs_crlf_for_are_pinned():
    """`.ps1` and `.bat` will not run with LF endings, and `tasks.ps1` is how this project is
    driven on the machine it is developed on."""
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in ("*.ps1 text eol=crlf", "*.bat text eol=crlf"):
        assert pattern in text


def test_every_binary_extension_in_the_tree_is_declared():
    """Git's text/binary heuristic guessing wrong on one file corrupts it silently, and this
    repo tracks .pt checkpoints, .parquet tables and .npy matrices."""
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    declared = {line.split()[0].lstrip("*") for line in text.splitlines() if " binary" in line}
    for extension in (".png", ".pt", ".npy", ".parquet", ".joblib", ".pdf"):
        assert extension in declared, extension


# --- every tracked text file ends with a newline, and stays that way (audit 54) ---------------

#: Extensions whose bytes are their content. Kept in step with `.gitattributes`.
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf", ".pt", ".pth",
    ".onnx", ".npy", ".npz", ".parquet", ".joblib", ".pkl", ".7z", ".zip", ".gz",
    ".ico", ".ttf", ".woff", ".woff2",
}  # fmt: skip


def _tracked_text_files():
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.split()
    for name in listed:
        path = ROOT / name
        if path.suffix.lower() not in BINARY_SUFFIXES and path.is_file():
            yield path


def test_every_tracked_text_file_ends_with_a_newline():
    """36 did not - `master_results.json`, `reproducibility.json`, `drift.json`,
    `model_registry.json` and 32 more. A file with no final newline makes `cat` run two files
    together, makes the last line invisible to line-oriented tools, and shows as a spurious
    change in every diff that touches it."""
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in _tracked_text_files()
        if (data := path.read_bytes()) and not data.endswith(b"\n")
    ]
    assert not missing, missing


#: The two characters a source file contains when it appends a newline: backslash, n.
NEWLINE_LITERAL = '+ "' + chr(92) + 'n"'


def test_no_json_writer_omits_the_final_newline():
    """Fixing the 36 files without fixing the 103 writers is a fix that lasts until the next
    run. Every `write_text(json.dumps(...))` in src/ and scripts/ appends one."""
    import re

    offenders = []
    for base in ("src", "scripts"):
        for path in sorted((ROOT / base).rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r"write_text\(\s*json\.dumps\(", text):
                tail = text[match.start() : match.start() + 400]
                if NEWLINE_LITERAL not in tail.split("encoding=")[0]:
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}")
    assert not offenders, sorted(set(offenders))


# --- the s5 sweep's scratch output is not a report (audit 51) ---------------------------------

#: `reports/s5_*.json` that something actually names. The rest were `python -m src.assemble.s5
#: --out ...` writing one arm of a sweep to a new filename each time - nineteen files of ~146 KB
#: each, within bytes of their neighbours, ~2.6 MB, referenced by nothing.
KEPT_S5_REPORTS = {
    "s5_graph_ged.json",  # the val default `s5.main` writes
    "s5_graph_ged_test.json",  # and the test one
    "s5_direction_ablation.json",
    "s5_direction_fitted.json",
    "s5_test_large.json",
    "s5_test_posem.json",
    "s5_test_ranker.json",
    "s5_val_ranker.json",
}

#: The two `s5.main` writes by default, which nothing has to name.
DEFAULT_S5_REPORTS = {"s5_graph_ged.json", "s5_graph_ged_test.json"}


def test_every_s5_report_is_one_something_names():
    import subprocess

    present = {path.name for path in (ROOT / "reports").glob("s5_*.json")}
    assert present == KEPT_S5_REPORTS, present ^ KEPT_S5_REPORTS

    for name in present - DEFAULT_S5_REPORTS:
        hits = subprocess.run(
            ["git", "grep", "-l", name, "--", "src", "tests", "scripts", "docs", "*.md", "*.yaml"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        assert hits.strip(), f"{name} is kept and referenced by nothing"

"""Phase 15.9 - every third-party import is declared, in the layer that actually installs it.

This test exists because the same defect happened twice in one session, in both directions.

First: six distributions - `jsonschema`, `referencing`, `pydantic`, `joblib`, `hypothesis`,
`psutil` - were imported across `src/` and `tests/` and declared in no requirements file at all.
On the development machine every one arrived as a transitive install, so nothing was visibly
wrong; on a runner installing from `requirements/`, `jsonschema` alone interrupted collection
with 34 errors.

Second, and the reason this file checks *layers* rather than just "declared somewhere":
`piexif` was declared - in `dev.txt`. CI installs the runtime layers, not the developer
tooling, so a test dependency parked in `dev.txt` is a test that never runs on a runner. A
check that only asked "is it in requirements/ anywhere" would have passed it.

Nothing here imports the packages it reasons about; it reads source and requirements as text, so
it gives the same answer on a machine with everything installed and on one with nothing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements"

#: The layers CI installs for the test job. `dev.txt` is deliberately absent - that is the whole
#: point of the second failure above.
CI_LAYERS = ("base.txt", "classical.txt", "cv.txt", "serve.txt", "genai.txt", "torch.txt")

#: Import name -> distribution name, for the cases where they differ. Kept explicit rather than
#: resolved through importlib.metadata so the test does not depend on what happens to be
#: installed in the environment running it.
IMPORT_TO_DIST = {
    "cv2": "opencv-contrib-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "win32api": "pywin32",
    "win32con": "pywin32",
    "win32job": "pywin32",
    "win32process": "pywin32",
    "win32security": "pywin32",
    "pythoncom": "pywin32",
    "pywintypes": "pywin32",
    "fitz": "pymupdf",
    "dateutil": "python-dateutil",
    "multipart": "python-multipart",
    "pdf2image": "pdf2image",
    "umap": "umap-learn",
    "imblearn": "imbalanced-learn",
    "py7zr": "py7zr",
    "OpenSSL": "pyopenssl",
}

#: Imported by `src/` but genuinely developer-only, and never reached by a test CI runs.
#: `src/mlops/registry.py` talks to MLflow, which is a 200 MB install whose absence costs the
#: test job nothing - so it stays in `dev.txt` and is named here rather than silently tolerated.
DEV_ONLY_OK = {"mlflow", "dvc"}

#: First-party and tooling packages that are not distributions to install.
LOCAL = {"src", "tests", "scripts", "app", "conftest"}


def _declared(files: tuple[str, ...]) -> set[str]:
    names: set[str] = set()
    for name in files:
        path = REQUIREMENTS / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            names.add(_normalise(line.split("==")[0].split(">=")[0].split(";")[0].strip()))
    return names


def _imports() -> dict[str, set[str]]:
    """Top-level module name -> the files that import it, across src/, tests/ and scripts/."""
    found: dict[str, set[str]] = {}
    for base in ("src", "tests", "scripts"):
        for path in (ROOT / base).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    top = module.split(".")[0]
                    found.setdefault(top, set()).add(str(path.relative_to(ROOT)))
    return found


def _third_party() -> dict[str, set[str]]:
    return {
        module: files
        for module, files in _imports().items()
        if module not in sys.stdlib_module_names and module not in LOCAL
    }


def _normalise(name: str) -> str:
    """PEP 503 normalisation. `huggingface_hub` is imported with an underscore and declared with
    a hyphen, and the two are the same distribution."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution(module: str) -> str:
    return _normalise(IMPORT_TO_DIST.get(module, module))


def test_every_third_party_import_is_declared_in_some_requirements_file():
    declared = _declared(tuple(p.name for p in REQUIREMENTS.glob("*.txt")))
    undeclared = {
        module: sorted(files)[0]
        for module, files in sorted(_third_party().items())
        if _distribution(module) not in declared
    }
    assert not undeclared, (
        "imported but declared in no requirements file - this is the shape of defect that is "
        "invisible on a machine where the package arrived transitively:\n"
        + "\n".join(f"  {m} (e.g. {f})" for m, f in undeclared.items())
    )


def test_every_import_ci_needs_is_in_a_layer_ci_installs():
    """Declared somewhere is not enough: it has to be in a layer the test job installs."""
    ci_declared = _declared(CI_LAYERS)
    missing = {
        module: sorted(files)[0]
        for module, files in sorted(_third_party().items())
        if _distribution(module) not in ci_declared and module not in DEV_ONLY_OK
    }
    assert not missing, (
        "declared only outside the layers CI installs "
        f"({', '.join(CI_LAYERS)}), so these never run on a runner:\n"
        + "\n".join(f"  {m} (e.g. {f})" for m, f in missing.items())
    )


@pytest.mark.parametrize("layer", CI_LAYERS)
def test_every_ci_layer_exists_and_pins_exactly(layer: str):
    """A layer CI installs from must exist, and every line in it must be `==` pinned - a range
    is how CI came to test torch 2.14.0 against a pinned 2.5.1."""
    path = REQUIREMENTS / layer
    assert path.exists(), f"{layer} is installed by CI but does not exist"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        assert "==" in line, f"{layer}: {line!r} is not pinned with =="


# ---------------------------------------------------------------------------------------------
# Phase 15.9, second half (audit 31/41) - the four install paths agree with each other.
#
# The test above asks "is every import declared in a layer CI installs". That is one of the two
# questions, and the other one had three live answers: `make env` installed base, torch, cv,
# classical, genai; `env.yml` installed the same five; the Dockerfile installs five *different*
# ones; CI installs six. Nobody following the project's own setup instructions got the
# environment CI tests against, and `make serve`, `make lint`, `make repro` and `make dag` all
# failed on a clean `make env` for that reason.
#
# Parsed as text, like everything else here, so this answers the same on a machine with
# everything installed and on one with nothing.
# ---------------------------------------------------------------------------------------------

MAKEFILE = ROOT / "Makefile"
ENV_YML = ROOT / "env.yml"
DOCKERFILE = ROOT / "Dockerfile"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: What the *serving* image needs, which is deliberately not what a test runner needs.
#: `genai.txt` is the QLoRA training stack - bitsandbytes, peft, trl - and a container that
#: answers `/predict` from the emitter never loads any of it. Named here so the Dockerfile
#: installing fewer layers than CI is a decision this file records rather than a drift it misses.
SERVE_LAYERS = ("base.txt", "torch.txt", "classical.txt", "cv.txt", "serve.txt")


def _layers(text: str) -> set[str]:
    """Every `requirements/<name>.txt` mentioned in a block of text."""
    return set(re.findall(r"requirements/([a-z]+\.txt)", text))


def _make_target(name: str) -> str:
    """One Makefile target's recipe."""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}:"))
    body = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(("\t", " ", "#")):
            break
        body.append(line)
    return "\n".join(body)


def test_make_env_installs_every_layer_ci_installs():
    """A developer who runs the documented setup must end up with at least what CI tests."""
    installed = _layers(_make_target("env"))
    assert set(CI_LAYERS) <= installed, sorted(set(CI_LAYERS) - installed)


def test_make_env_also_installs_the_developer_tooling():
    """`make lint`, `make repro` and `make dag` run ruff, black, isort and dvc, all of which live
    in `dev.txt`. CI does not install it, deliberately; a developer machine must."""
    assert "dev.txt" in _layers(_make_target("env"))


def test_env_yml_installs_exactly_what_make_env_installs():
    """Two documented setup paths that disagree is one of them being wrong for somebody."""
    assert _layers(ENV_YML.read_text(encoding="utf-8")) == _layers(_make_target("env"))


def test_the_dockerfile_installs_the_serve_layers_and_says_so():
    """Fewer than CI, on purpose: a container that answers /predict never loads the QLoRA stack.
    The set is named in this file so it is a recorded decision, not an unnoticed difference."""
    assert _layers(DOCKERFILE.read_text(encoding="utf-8")) == set(SERVE_LAYERS)
    assert set(SERVE_LAYERS) < set(CI_LAYERS)


def test_the_ci_test_job_installs_exactly_ci_layers():
    """`CI_LAYERS` is this file's claim about the workflow; the workflow is the fact."""
    assert _layers(WORKFLOW.read_text(encoding="utf-8")) == set(CI_LAYERS)


@pytest.mark.parametrize("layer", CI_LAYERS)
def test_no_layer_is_installed_by_a_path_that_does_not_declare_it(layer: str):
    """Every layer CI installs exists in `requirements/` and is reachable from a setup path."""
    assert (REQUIREMENTS / layer).is_file()
    reachable = _layers(_make_target("env")) | _layers(ENV_YML.read_text(encoding="utf-8"))
    assert layer in reachable

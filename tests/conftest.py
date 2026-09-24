"""Phase 0.2.7 — shared pytest fixtures.

Five tiny synthetic sketches live in tests/fixtures/, one per diagram type the pipeline
handles. They are committed (about 15 KB in total) so the suite runs with no dataset
present, and regenerated deterministically by scripts/make_fixtures.py.

    def test_something(sample_image):          # parametrized over all five
        ...

    def test_flowchart_only(flowchart_image):  # one specific type
        ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"

# Make `import src...` work no matter where pytest is invoked from.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIAGRAM_TYPES = ["flowchart", "wireframe", "state_machine", "er_diagram", "circuit"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "gpu: requires a CUDA device")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip what this machine cannot run: GPU tests with no CUDA, corpus tests with no payload.

    One hook, deliberately. pytest calls a plugin hook once per module, so a second
    `pytest_collection_modifyitems` defined lower in this file would *replace* this one rather
    than run alongside it - and the GPU skip would silently stop happening.
    """
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        no_gpu = pytest.mark.skip(reason="no CUDA device available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(no_gpu)

    if sys.platform != "win32":
        not_windows = pytest.mark.skip(
            reason="Windows-only: drives tasks.ps1 through powershell.exe, or a Win32 job object"
        )
        for item in items:
            name = item.name.split("[")[0]
            if Path(item.fspath).stem in NEEDS_WINDOWS or name in NEEDS_WINDOWS_TESTS:
                item.add_marker(not_windows)

    if not payload_present():
        no_data = pytest.mark.skip(
            reason="needs the DVC payload (data/processed, data/features); run `dvc pull`"
        )
        for item in items:
            if Path(item.fspath).stem in NEEDS_PAYLOAD:
                item.add_marker(no_data)


@pytest.fixture(autouse=True)
def _torch_global_flags_are_restored():
    """Undo torch's process-global switches after every test that changes them.

    `src.utils.seed.set_seed(deterministic=True)` sets `cudnn.deterministic`, `cudnn.benchmark`
    and `torch.use_deterministic_algorithms` on the *process*, not on a scope, and several tests
    call it. The consequence was one order-dependent failure that looked like a bug in the thing
    it pointed at: `test_captured_update_is_bit_identical_to_the_eager_update` passed alone and
    failed in the suite with "CUDA error: operation failed due to a previous error during
    capture", because CUDA graph capture cannot run under deterministic algorithms - and the run
    that turned them on was `tests/test_determinism.py`, ten modules earlier.

    Snapshot and restore rather than force a value: a test that wants deterministic kernels still
    gets them, and only stops handing them to the next test. Skipped entirely when torch has not
    been imported, so a suite of pure-python tests pays nothing.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        yield
        return
    before = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        after = (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        if after != before:
            torch.backends.cudnn.deterministic = before[0]
            torch.backends.cudnn.benchmark = before[1]
            torch.use_deterministic_algorithms(before[2], warn_only=before[3])


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    assert FIXTURES.is_dir(), "tests/fixtures missing - run scripts/make_fixtures.py"
    return FIXTURES


@pytest.fixture(scope="session")
def fixture_manifest(fixtures_dir: Path) -> list[dict]:
    return json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(params=DIAGRAM_TYPES)
def sample_image_path(request: pytest.FixtureRequest, fixtures_dir: Path) -> Path:
    """Parametrized: every test using this runs once per diagram type."""
    return fixtures_dir / f"{request.param}.png"


@pytest.fixture
def sample_image(sample_image_path: Path) -> np.ndarray:
    img = cv2.imread(str(sample_image_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"could not read {sample_image_path}"
    return img


def _single(name: str):
    @pytest.fixture
    def _fixture(fixtures_dir: Path) -> np.ndarray:
        return cv2.imread(str(fixtures_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)

    return _fixture


flowchart_image = _single("flowchart")
wireframe_image = _single("wireframe")
state_machine_image = _single("state_machine")
er_diagram_image = _single("er_diagram")
circuit_image = _single("circuit")


@pytest.fixture
def run_dir(tmp_path: Path):
    """An isolated experiments root, so tests never write into the real one."""
    d = tmp_path / "experiments"
    d.mkdir()
    return d


@pytest.fixture
def seeded():
    """Deterministic RNG state for any test that samples."""
    from src.utils.seed import set_seed

    return set_seed(42, deterministic=True)


# --- the DVC payload, and what cannot be tested without it -----------------------------------
#
# `data/` holds `.dvc` pointers in git and its payload in a DVC remote, so a fresh clone - CI
# above all - has the pointers and none of the content. Measured on a detached worktree with
# exactly that shape, **40 tests across 16 modules failed for want of the payload rather than
# because anything was wrong**, which is the difference between a suite that reports a real
# regression and one nobody can read.
#
# Those tests skip here instead. The modules are listed by name rather than detected, because a
# test that quietly stops running is worse than one that fails: a name in this tuple is a
# deliberate statement that the module needs the corpus, and it is greppable.
#
# This is not a way to make CI green. The 20 failures that remain with the payload present are
# still failures and CI still reports them; what this removes is only the noise of asking a
# machine to check something it has not been given.

#: Presence of any one of these means the payload was pulled.
PAYLOAD = (
    ROOT / "data" / "processed" / "manifest.parquet",
    ROOT / "data" / "features" / "handcrafted.parquet",
)

#: Modules that cannot run without it, measured rather than guessed.
NEEDS_PAYLOAD = {
    "test_assemble_corpus",
    "test_assemble_irdiff",
    "test_assemble_nodes",
    "test_assemble_propagate",
    # Added in 15.9's second pass. Missed the first time because this machine's payload is
    # *partially* present - `data/processed/ir` exists but its didi artefacts do not - so the
    # module failed on a content assertion rather than on an empty corpus, which reads like a
    # data bug and not like a missing payload. Its `CORPUS_FILES` is `data/processed/ir`, so on
    # a runner with no payload at all it asserts five sources against none.
    "test_assemble_serialise",
    "test_embed_backbone",
    "test_eval_master",
    "test_eval_robust",
    "test_eval_stagewise",
    "test_mlops_dag",
    "test_ocr_s3",
    "test_pipeline_golden",
    "test_pipeline_integration",
    "test_preprocess_binarize",
    "test_rl_curriculum",
}


#: Modules that are Windows-only by construction rather than by accident. `tasks.ps1` is a
#: PowerShell task runner and `test_task_runner` invokes it through `powershell.exe`, which a
#: Linux runner does not have - there is no cross-platform behaviour being skipped here, only a
#: shell that does not exist. Named rather than detected, for the same reason NEEDS_PAYLOAD is.
NEEDS_WINDOWS = {
    "test_task_runner",
}

#: Individual tests whose *module* is cross-platform but which are not. The sandbox reports
#: `peak_memory_bytes` from a Win32 job object; on POSIX the containment is real but the
#: accounting is not, so the assertion is None-against-not-None rather than a missed limit.
NEEDS_WINDOWS_TESTS = {
    "test_runaway_allocation_is_contained",
}


def payload_present() -> bool:
    return any(path.exists() for path in PAYLOAD)

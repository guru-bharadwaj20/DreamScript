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
    """Skip GPU-marked tests when no CUDA device is present, rather than failing them."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


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

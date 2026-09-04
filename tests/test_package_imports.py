"""Phase 0.2.2 acceptance test — every pipeline subpackage imports cleanly.

A package that fails to import is a broken dependency or a circular import, and it is far
cheaper to catch that here than three phases later inside a training run.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import src

PACKAGES = [
    "src.ingest",
    "src.ir",
    "src.preprocess",
    "src.features",
    "src.classify",
    "src.embed",
    "src.cluster",
    "src.detect",
    "src.ocr",
    "src.parse",
    "src.rl",
    "src.synth",
    "src.eval",
    "src.serve",
    "src.utils",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    mod = importlib.import_module(name)
    assert mod.__doc__, f"{name} has no module docstring"


def test_no_package_is_missing_from_this_list() -> None:
    """The test list must stay in sync with what actually exists under src/."""
    found = {f"src.{m.name}" for m in pkgutil.iter_modules(src.__path__) if m.ispkg}
    assert found == set(PACKAGES), f"src/ packages and PACKAGES disagree: {found ^ set(PACKAGES)}"


def test_every_package_declares_its_phase() -> None:
    for name in PACKAGES:
        mod = importlib.import_module(name)
        if name == "src.utils":  # cross-cutting: seeding, config, logging
            continue
        assert getattr(mod, "PHASE", None), f"{name} does not declare PHASE"

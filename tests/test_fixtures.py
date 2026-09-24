"""Phase 0.2.7 acceptance test — the committed fixtures load and carry real geometry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from tests.conftest import DIAGRAM_TYPES, ROOT


def test_five_fixtures_exist(fixtures_dir: Path):
    pngs = sorted(p.stem for p in fixtures_dir.glob("*.png"))
    assert pngs == sorted(DIAGRAM_TYPES)


def test_manifest_matches_files(fixture_manifest: list[dict], fixtures_dir: Path):
    assert {m["id"] for m in fixture_manifest} == set(DIAGRAM_TYPES)
    for entry in fixture_manifest:
        assert (fixtures_dir / entry["file"]).is_file()


def test_fixtures_stay_small(fixtures_dir: Path):
    """Committed test data must not bloat the repository."""
    total = sum(p.stat().st_size for p in fixtures_dir.glob("*.png"))
    assert total < 64 * 1024, f"fixtures grew to {total} bytes"


def test_sample_image_loads(sample_image: np.ndarray, sample_image_path: Path):
    assert sample_image.shape == (240, 320), sample_image_path
    assert sample_image.dtype == np.uint8


def test_sample_image_has_ink(sample_image: np.ndarray):
    """Each sketch must contain strokes: a small but non-trivial fraction of dark pixels."""
    ink = float((sample_image < 128).mean())
    assert 0.005 < ink < 0.30


def test_flowchart_has_a_diamond(flowchart_image: np.ndarray):
    """The decision node is what distinguishes a flowchart, so it must survive in the fixture."""
    binary = (flowchart_image < 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = 0
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            quads += 1
    assert quads >= 1


def test_wireframe_is_grid_like(wireframe_image: np.ndarray):
    """Wireframes are dominated by axis-aligned rectangles: expect long straight runs."""
    binary = (wireframe_image < 128).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    )
    assert horizontal.sum() > 0


def test_state_machine_has_circles(state_machine_image: np.ndarray):
    circles = cv2.HoughCircles(
        state_machine_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=120,
        param2=30,
        minRadius=15,
        maxRadius=40,
    )
    assert circles is not None and len(circles[0]) >= 2


def test_fixture_generation_is_deterministic(tmp_path: Path):
    """Regenerating the fixtures must reproduce the committed bytes exactly."""
    before = {p.name: p.read_bytes() for p in (ROOT / "tests" / "fixtures").glob("*.png")}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_fixtures.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    after = {p.name: p.read_bytes() for p in (ROOT / "tests" / "fixtures").glob("*.png")}
    assert before == after

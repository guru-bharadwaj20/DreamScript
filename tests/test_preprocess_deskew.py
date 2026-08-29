"""Phase 3.1.7 - deskew."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import deskew as ds


def ruled_page(size=(700, 900)) -> np.ndarray:
    """A page whose dominant direction is unambiguous: ruled lines plus a few boxes."""
    mask = np.zeros(size, bool)
    canvas = np.zeros(size, np.uint8)
    for y in range(60, size[0] - 40, 40):
        cv2.line(canvas, (30, y), (size[1] - 30, y), 255, 2)
    cv2.rectangle(canvas, (120, 150), (400, 300), 255, 3)
    return mask | (canvas > 0)


@pytest.mark.parametrize("angle", [-7.0, -3.0, -1.0, 1.0, 3.0, 7.0])
def test_known_tilt_is_recovered(angle):
    tilted = ds.rotate(ruled_page(), angle)
    # `rotate` and `estimate` use opposite sign conventions; see the module docstring.
    assert abs(ds.estimate(tilted) + angle) < 1.0


@pytest.mark.parametrize("angle", [-6.0, -2.0, 2.0, 6.0])
def test_deskew_leaves_under_one_degree(angle):
    tilted = ds.rotate(ruled_page(), angle)
    straightened, applied = ds.deskew(tilted, mask=tilted)
    assert applied != 0.0
    assert abs(ds.estimate(straightened)) < 1.0


@pytest.mark.parametrize(
    ("raw", "folded"), [(0, 0), (89, -1), (91, 1), (-89, 1), (45, -45), (180, 0)]
)
def test_fold(raw, folded):
    assert ds.fold(raw) == pytest.approx(folded, abs=1e-6)


def test_a_page_with_no_dominant_direction_is_declined():
    """Circles and freehand arrows have no skew to measure; rotating by noise is worse."""
    canvas = np.zeros((600, 600), np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(14):
        centre = tuple(int(v) for v in rng.integers(80, 520, 2))
        cv2.circle(canvas, centre, int(rng.integers(30, 70)), 255, 3)
    assert ds.estimate(canvas > 0) == 0.0


def test_large_estimates_are_refused():
    """A 40-degree "skew" is a diagonal stroke, not a tilted page."""
    image = np.full((300, 300), 240, np.uint8)
    straightened, applied = ds.deskew(image, mask=np.zeros((300, 300), bool))
    assert applied == 0.0
    assert np.array_equal(straightened, image)


def test_rotation_expands_the_canvas_so_nothing_is_lost():
    mask = ruled_page()
    rotated = ds.rotate(mask, 10.0)
    assert rotated.shape[0] > mask.shape[0]
    assert rotated.sum() > 0.95 * mask.sum()


def test_masks_stay_boolean_through_rotation():
    assert ds.rotate(ruled_page(), 5.0).dtype == bool


def test_corpus_evaluation_meets_the_plans_bar():
    result = ds.evaluate(8)
    if not result["trials"]:
        pytest.skip("flowchartseg images not present")
    assert result["median_skew_left_after_correction_deg"] < 1.0

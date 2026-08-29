"""Phase 3.1.9 - Zhang-Suen thinning.

The two properties that matter are stated as tests: the skeleton is one pixel wide, and it has
the same connectivity as what it came from.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import thinning


def thick_line(width=9) -> np.ndarray:
    canvas = np.zeros((120, 300), np.uint8)
    cv2.line(canvas, (20, 60), (280, 60), 255, width)
    return canvas > 0


def test_a_thick_line_becomes_one_pixel_wide():
    skeleton = thinning.thin(thick_line())
    columns = skeleton[:, 40:260]
    assert columns.any()
    # Every column that has ink must have exactly one skeleton pixel.
    per_column = columns.sum(axis=0)
    assert set(np.unique(per_column[per_column > 0])) == {1}


def test_the_skeleton_never_leaves_the_stroke():
    mask = thick_line()
    assert np.logical_and(thinning.thin(mask), mask).sum() == thinning.thin(mask).sum()


def test_connectivity_is_preserved():
    canvas = np.zeros((200, 200), np.uint8)
    cv2.rectangle(canvas, (40, 40), (160, 160), 255, 7)
    mask = canvas > 0
    before = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]
    after = cv2.connectedComponents(thinning.thin(mask).astype(np.uint8), connectivity=8)[0]
    assert before == after == 2  # background plus one ring


def test_a_stroke_is_not_broken_in_the_middle():
    """The failure a naive erosion produces."""
    skeleton = thinning.thin(thick_line())
    count = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0] - 1
    assert count == 1


def test_thinning_is_idempotent():
    once = thinning.thin(thick_line())
    assert np.array_equal(thinning.thin(once), once)


def test_empty_and_single_pixel_inputs_are_safe():
    assert not thinning.thin(np.zeros((20, 20), bool)).any()
    dot = np.zeros((20, 20), bool)
    dot[10, 10] = True
    assert thinning.thin(dot).sum() == 1


def test_endpoints_and_branches_are_found():
    canvas = np.zeros((200, 200), np.uint8)
    cv2.line(canvas, (100, 20), (100, 180), 255, 5)  # stem
    cv2.line(canvas, (100, 100), (180, 100), 255, 5)  # branch
    skeleton = thinning.thin(canvas > 0)
    assert thinning.branch_points(skeleton).any()
    assert 2 <= thinning.end_points(skeleton).sum() <= 6


def test_width_profile_orders_strokes_correctly():
    """It is a relative measure, not an absolute one: see the docstring for the +2px offset."""
    measured = [thinning.width_profile(thick_line(w)) for w in (5, 9, 13)]
    assert measured == sorted(measured)
    for nominal, value in zip((5, 9, 13), measured, strict=True):
        assert 0 < value - nominal < 4, f"width {nominal} measured as {value}"


def test_width_profile_offset_is_constant_not_a_scale_error():
    offsets = [thinning.width_profile(thick_line(w)) - w for w in (5, 9, 13)]
    assert max(offsets) - min(offsets) < 1.5


def test_corpus_evaluation():
    result = thinning.evaluate(6)
    if not result["items"]:
        pytest.skip("FA database not present")
    assert result["components_preserved"] >= 0.9
    assert result["skeleton_inside_the_stroke"] > 0.999

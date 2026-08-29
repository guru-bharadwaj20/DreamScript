"""Phase 3.2.6 - arrowhead detection.

The mechanism is tested on drawn arrows where the answer is known; the *corpus* precision is
measured in `evaluate` and reported honestly in the module docstring, where it does not meet
the plan's bar.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.preprocess.primitives import arrowheads as ah


def arrow(direction_deg: float, size: int = 500, barb: int = 40) -> np.ndarray:
    """An arrow pointing in `direction_deg`, drawn free-standing."""
    canvas = np.zeros((size, size), np.uint8)
    cx, cy = size // 2, size // 2
    shaft = 160
    sx = int(cx - shaft * math.cos(math.radians(direction_deg)))
    sy = int(cy - shaft * math.sin(math.radians(direction_deg)))
    cv2.line(canvas, (sx, sy), (cx, cy), 255, 3)
    for offset in (145, -145):
        bx = int(cx + barb * math.cos(math.radians(direction_deg + offset)))
        by = int(cy + barb * math.sin(math.radians(direction_deg + offset)))
        cv2.line(canvas, (cx, cy), (bx, by), 255, 3)
    return canvas > 0


@pytest.mark.parametrize("direction", [0.0, 90.0, 180.0, -90.0, 35.0])
def test_a_drawn_arrow_is_found_pointing_the_right_way(direction):
    found = ah.detect(arrow(direction))
    assert len(found) == 1
    got = found[0].shaft_angle
    difference = abs((got - direction + 180.0) % 360.0 - 180.0)
    assert difference < 25.0, f"expected {direction}, got {got}"


def test_the_tip_is_where_the_barbs_meet():
    found = ah.detect(arrow(0.0))[0]
    assert abs(found.x - 250) < 12 and abs(found.y - 250) < 12


def test_a_plain_corner_is_not_an_arrowhead():
    canvas = np.zeros((400, 400), np.uint8)
    cv2.line(canvas, (50, 200), (200, 200), 255, 3)
    cv2.line(canvas, (200, 200), (200, 350), 255, 3)
    assert ah.detect(canvas > 0) == []


def test_a_t_junction_is_not_an_arrowhead():
    """A line meeting a box edge: three branches, but the barbs are 180 degrees apart."""
    canvas = np.zeros((400, 400), np.uint8)
    cv2.line(canvas, (200, 40), (200, 360), 255, 3)  # the edge
    cv2.line(canvas, (200, 200), (360, 200), 255, 3)  # the line arriving
    assert ah.detect(canvas > 0) == []


def test_a_crossing_is_not_an_arrowhead():
    canvas = np.zeros((400, 400), np.uint8)
    cv2.line(canvas, (40, 200), (360, 200), 255, 3)
    cv2.line(canvas, (200, 40), (200, 360), 255, 3)
    assert ah.detect(canvas > 0) == []


def test_both_barbs_on_one_side_is_rejected():
    canvas = np.zeros((400, 400), np.uint8)
    cv2.line(canvas, (40, 200), (200, 200), 255, 3)
    cv2.line(canvas, (200, 200), (250, 240), 255, 3)
    cv2.line(canvas, (200, 200), (250, 270), 255, 3)
    assert ah.detect(canvas > 0) == []


def test_junction_clustering_merges_the_blob():
    """Zhang-Suen leaves several branch pixels at one meeting; they are one junction."""
    from src.preprocess.thinning import thin

    skeleton = thin(arrow(0.0))
    clusters = ah._junction_clusters(skeleton)
    assert len(clusters) == 1
    assert len(clusters[0]) > 1, "the blob should be more than one pixel"


def test_empty_input_is_safe():
    assert ah.detect(np.zeros((50, 50), bool)) == []


def test_corpus_evaluation_runs_and_reports_honestly():
    """The bar is not met; what is asserted is that the number is produced and is real."""
    result = ah.evaluate(4)
    if not result["pages"]:
        pytest.skip("hdBPMN not present")
    assert 0.0 <= result["precision"] <= 1.0
    assert result["annotated_arrows"] > 0

"""Phase 3.2.3 - polygon approximation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess.primitives import polygons as poly


def outline(kind: str) -> np.ndarray:
    canvas = np.zeros((400, 400), np.uint8)
    if kind == "box":
        cv2.rectangle(canvas, (60, 80), (340, 300), 255, 3)
    elif kind == "diamond":
        cv2.polylines(
            canvas, [np.array([[200, 40], [360, 200], [200, 360], [40, 200]])], True, 255, 3
        )
    elif kind == "circle":
        cv2.circle(canvas, (200, 200), 150, 255, 3)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def test_a_box_has_four_corners():
    assert poly.vertex_count(outline("box")) == 4


def test_a_diamond_has_four_corners():
    assert poly.vertex_count(outline("diamond")) == 4


def test_a_circle_has_many():
    assert poly.vertex_count(outline("circle")) > 5


def test_tolerance_trades_corners_for_smoothness():
    counts = [poly.vertex_count(outline("circle"), e) for e in (0.005, 0.02, 0.08)]
    assert counts == sorted(counts, reverse=True), counts


def test_interior_angles_of_a_square_are_right_angles():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    angles = poly.interior_angles(square)
    assert np.allclose(angles, 90.0, atol=1.0)


def test_interior_angles_of_a_diamond_are_still_right_angles_but_the_corners_moved():
    """Shape and orientation are different questions: a diamond is a rotated square."""
    diamond = np.array([[50, 0], [100, 50], [50, 100], [0, 50]])
    assert np.allclose(poly.interior_angles(diamond), 90.0, atol=1.0)


def test_degenerate_polygons_are_safe():
    assert len(poly.interior_angles(np.array([[0, 0], [1, 1]]))) == 0


def test_sweep_counts_cover_every_tolerance():
    from src.preprocess.primitives.contours import Contour

    contour = Contour(0, outline("box"), 0, -1, 1.0, 1.0, True, [0, 0, 1, 1])
    counts = poly.sweep_counts([contour])
    assert set(counts) == set(poly.SWEEP)


def test_histogram_is_written():
    contours = poly.collect(4)
    if not contours:
        pytest.skip("hdBPMN not present")
    path = poly.figure(poly.sweep_counts(contours), len(contours))
    assert path.is_file()

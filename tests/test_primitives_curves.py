"""Phase 3.2.5 - curvature and corners."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess.primitives import curves


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
    elif kind == "ellipse":
        cv2.ellipse(canvas, (200, 200), (170, 90), 0, 0, 360, 255, 3)
    found, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(found, key=cv2.contourArea).reshape(-1, 2)


@pytest.mark.parametrize("kind", ["box", "diamond"])
def test_polygons_have_four_corners(kind):
    assert curves.describe(outline(kind))["corners"] == 4


@pytest.mark.parametrize("kind", ["circle", "ellipse"])
def test_smooth_shapes_have_none(kind):
    assert curves.describe(outline(kind))["corners"] == 0


def test_polygons_are_straighter_than_curves():
    box = curves.describe(outline("box"))["straightness"]
    circle = curves.describe(outline("circle"))["straightness"]
    assert box > 0.6 > circle


def test_line_vs_curve_ratio_separates_the_two_families():
    assert curves.line_vs_curve_ratio(outline("box")) > 3
    assert curves.line_vs_curve_ratio(outline("circle")) < 1


def test_mean_curvature_alone_does_not_separate_anything():
    """A closed outline turns 360 degrees whatever its shape - pinned so nobody uses it as a
    feature by mistake."""
    means = [curves.describe(outline(k))["mean_curvature"] for k in ("box", "circle", "ellipse")]
    assert max(means) - min(means) < 1.0


def test_turning_is_measured_over_a_window_not_between_pixels():
    """Pixel-to-pixel turning on a rasterised line is 45-degree staircase noise."""
    line = np.array([[i, i // 2] for i in range(200)])
    assert float(np.mean(curves.turning(line, closed=False))) < 5.0


def test_an_open_path_does_not_wrap_around():
    """Treating a stroke as a loop invents a sharp turn between its two ends."""
    line = np.array([[i, 0] for i in range(200)])
    assert curves.turning(line, closed=False).max() == pytest.approx(0.0)
    assert curves.turning(line, closed=True).max() > 90.0


def test_corners_are_merged_not_reported_three_times():
    found = curves.corners(outline("box"))
    assert len(found) == 4
    assert len(set(found.tolist())) == 4


def test_short_contours_are_safe():
    assert len(curves.turning(np.array([[0, 0], [1, 1]]))) == 2
    assert curves.describe(np.array([[0, 0], [1, 1]]))["corners"] == 0


def test_skeleton_statistics_on_a_junction():
    canvas = np.zeros((200, 200), np.uint8)
    cv2.line(canvas, (100, 20), (100, 180), 255, 5)
    cv2.line(canvas, (100, 100), (180, 100), 255, 5)
    stats = curves.skeleton_curvature(canvas > 0)
    assert stats["branch_points"] >= 1
    assert stats["end_points"] >= 3

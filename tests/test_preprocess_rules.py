"""Phase 3.1.8 - ruled-paper suppression."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import rules


def drawing(shape=(500, 600)) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    cv2.rectangle(canvas, (80, 80), (260, 200), 255, 3)
    cv2.circle(canvas, (420, 150), 60, 255, 3)
    cv2.line(canvas, (260, 140), (360, 150), 255, 3)
    return canvas > 0


def test_grid_is_found_and_the_drawing_is_not():
    grid = rules.draw_grid((500, 600))
    found = rules.find_rules(np.logical_or(drawing(), grid))
    caught = float(np.logical_and(found, grid).sum()) / grid.sum()
    collateral = float(np.logical_and(found, drawing() & ~grid).sum()) / (drawing() & ~grid).sum()
    assert caught > 0.95, "the grid was not found"
    assert collateral < 0.15, "too much of the drawing was called a rule"


def test_suppression_recovers_the_drawing():
    ink = drawing()
    ruled = np.logical_or(ink, rules.draw_grid(ink.shape))
    cleaned = rules.suppress(ruled)
    kept = float(np.logical_and(cleaned, ink).sum()) / ink.sum()
    assert kept > 0.90
    assert cleaned.sum() < ruled.sum() / 2


def test_a_box_edge_is_not_a_rule():
    """The failure this module has to avoid: deleting the diagram along with the paper."""
    ink = drawing()
    assert not rules.find_rules(ink).any()


def test_a_page_with_no_rules_is_returned_unchanged():
    ink = drawing()
    assert np.array_equal(rules.suppress(ink), ink)


def test_span_threshold_is_what_separates_rules_from_edges():
    ink = drawing()
    ruled = np.logical_or(ink, rules.draw_grid(ink.shape))
    # Demanding a full-width span still finds the grid; demanding almost nothing eats the box.
    assert rules.find_rules(ruled, min_span_frac=0.95).any()
    greedy = rules.find_rules(ruled, min_span_frac=0.05)
    assert np.logical_and(greedy, ink & ~rules.draw_grid(ink.shape)).sum() > 0


def test_repair_rejoins_a_stroke_cut_by_a_rule():
    """A vertical stroke crossed by a horizontal rule must not end up in two pieces."""
    canvas = np.zeros((200, 200), np.uint8)
    cv2.line(canvas, (100, 20), (100, 180), 255, 3)
    cv2.line(canvas, (0, 100), (199, 100), 255, 1)
    cleaned = rules.suppress(canvas > 0)
    count, _, _, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    assert count - 1 == 1, "the stroke was left in pieces"


def test_measured_improvement_holds():
    result = rules.evaluate(8)
    if not result["items"]:
        pytest.skip("FA database not present")
    assert result["f1_after"] > result["f1_before"]
    assert result["grid_pixels_removed"] > 0.90
    assert result["true_ink_recall_after"] > 0.90

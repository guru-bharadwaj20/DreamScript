"""Phase 4.1.8 - directionality."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx
from src.features import direction as di


def boxes_joined(vertical: bool) -> np.ndarray:
    """Two shapes with a long connector between them, one way up or the other."""
    if vertical:
        image = np.zeros((900, 400), np.uint8)
        cv2.rectangle(image, (120, 40), (280, 180), 255, 3)
        cv2.rectangle(image, (120, 700), (280, 840), 255, 3)
        cv2.line(image, (200, 180), (200, 700), 255, 3)
    else:
        image = np.zeros((400, 900), np.uint8)
        cv2.rectangle(image, (40, 120), (180, 280), 255, 3)
        cv2.rectangle(image, (700, 120), (840, 280), 255, 3)
        cv2.line(image, (180, 200), (700, 200), 255, 3)
    return image > 0


def test_the_three_names_are_produced():
    assert set(di.extract(ctx.from_masks(boxes_joined(True), page_id="v"))) == set(di.NAMES)


def test_the_axis_is_positive_for_vertical_and_negative_for_horizontal():
    assert di.extract(ctx.from_masks(boxes_joined(True), page_id="v"))["dir_flow_axis"] > 0.5
    assert di.extract(ctx.from_masks(boxes_joined(False), page_id="h"))["dir_flow_axis"] < -0.5


def test_the_axis_cannot_tell_down_from_up():
    """A line has no arrowhead; the feature is an axis and the test says so."""
    down = ctx.from_masks(boxes_joined(True), page_id="down")
    flipped = ctx.from_masks(np.flipud(boxes_joined(True)), page_id="up")
    assert di.extract(down)["dir_flow_axis"] == pytest.approx(
        di.extract(flipped)["dir_flow_axis"], abs=0.05
    )


def test_entropy_is_low_for_one_direction_and_high_for_many():
    angles = np.zeros(20)
    lengths = np.ones(20)
    assert di.entropy(angles, lengths) == pytest.approx(0.0, abs=1e-9)
    spread = np.linspace(0, 179, 180)
    assert di.entropy(spread, np.ones(180)) > 0.95


def test_entropy_is_length_weighted_not_fragment_counted():
    """One long arrow must outweigh a shower of short fragments from binarization."""
    angles = np.array([90.0, 0.0, 0.0, 0.0])
    counted = di.entropy(angles, np.ones(4))
    weighted = di.entropy(angles, np.array([300.0, 5.0, 5.0, 5.0]))
    assert weighted < counted


def test_the_shapes_own_outlines_are_not_counted_as_connectors():
    """Without the padding, the outer edge of every box is a connector and the axis flips."""
    image = np.zeros((400, 400), np.uint8)
    cv2.rectangle(image, (100, 100), (300, 300), 255, 3)
    assert ctx.from_masks(image > 0, page_id="lonely").connector_segments == []


def test_axis_aligned_is_one_for_a_grid_and_lower_for_diagonals():
    image = np.zeros((600, 600), np.uint8)
    cv2.rectangle(image, (60, 60), (200, 200), 255, 3)
    cv2.rectangle(image, (400, 400), (540, 540), 255, 3)
    cv2.line(image, (200, 200), (400, 400), 255, 3)
    diagonal = di.extract(ctx.from_masks(image > 0, page_id="diag"))["dir_axis_aligned"]
    straight = di.extract(ctx.from_masks(boxes_joined(False), page_id="h"))["dir_axis_aligned"]
    assert straight > diagonal


def test_a_page_with_no_connectors_is_nan():
    image = np.zeros((400, 400), np.uint8)
    cv2.rectangle(image, (100, 100), (300, 300), 255, 3)
    features = di.extract(ctx.from_masks(image > 0, page_id="lonely"))
    assert all(np.isnan(v) for v in features.values())

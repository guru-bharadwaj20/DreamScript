"""Phase 4.1.5 - global geometry."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx
from src.features import geometry as ge


def page(height=400, width=800, draw=True) -> np.ndarray:
    image = np.zeros((height, width), np.uint8)
    if draw:
        cv2.rectangle(image, (100, 100), (300, 300), 255, 3)
    return image > 0


def test_the_three_names_are_produced():
    assert set(ge.extract(ctx.from_masks(page(), page_id="p"))) == set(ge.NAMES)


def test_aspect_is_width_over_height():
    assert ge.extract(ctx.from_masks(page(400, 800), page_id="p"))["global_aspect"] == 2.0
    assert ge.extract(ctx.from_masks(page(800, 400), page_id="p"))["global_aspect"] == 0.5


def test_ink_coverage_is_the_share_of_the_page_that_is_ink():
    mask = np.zeros((100, 100), bool)
    mask[:10, :] = True
    assert ge.extract(ctx.from_masks(mask, page_id="p"))["global_ink_coverage"] == pytest.approx(
        0.1
    )


def test_bbox_fill_is_the_drawing_not_the_ink():
    """A hollow box fills its own bounding box completely, however little ink it holds."""
    features = ge.extract(ctx.from_masks(page(400, 800), page_id="p"))
    assert features["global_bbox_fill"] > features["global_ink_coverage"]
    # the 200x200 outline sits in a 400x800 page, drawn with a 3px pen
    assert features["global_bbox_fill"] == pytest.approx((203 * 203) / (400 * 800), abs=0.01)


def test_a_blank_page_has_coverage_zero_but_no_bbox_fill():
    features = ge.extract(ctx.from_masks(page(draw=False), page_id="blank"))
    assert features["global_ink_coverage"] == 0.0
    assert np.isnan(features["global_bbox_fill"])


def test_ink_bbox_is_none_when_there_is_no_ink():
    assert ge.ink_bbox(np.zeros((10, 10), bool)) is None


def test_ink_bbox_is_inclusive_of_the_last_ink_pixel():
    mask = np.zeros((10, 10), bool)
    mask[2:5, 3:8] = True
    assert ge.ink_bbox(mask) == (3, 2, 5, 3)


def test_these_features_survive_a_page_where_nothing_was_detected():
    """The floor under the vector: no regions, no segments, still three real numbers."""
    speckle = np.zeros((300, 300), bool)
    speckle[150, 150] = True
    context = ctx.from_masks(speckle, page_id="speck")
    assert not context.regions
    features = ge.extract(context)
    assert np.isfinite(features["global_aspect"])
    assert np.isfinite(features["global_ink_coverage"])

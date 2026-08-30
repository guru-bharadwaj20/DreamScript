"""Phase 4.1.9 - containment."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import containment as co
from src.features import context as ctx
from tests.test_features_context import nested_panel
from tests.test_features_structural import chain


def three_deep() -> np.ndarray:
    image = np.zeros((800, 800), np.uint8)
    cv2.rectangle(image, (40, 40), (760, 760), 255, 3)
    cv2.rectangle(image, (120, 120), (680, 680), 255, 3)
    cv2.rectangle(image, (240, 240), (560, 560), 255, 3)
    return image > 0


def test_the_three_names_are_produced():
    assert set(co.extract(ctx.from_masks(nested_panel(), page_id="n"))) == set(co.NAMES)


def test_a_flat_row_of_boxes_does_not_nest():
    features = co.extract(ctx.from_masks(chain(4), page_id="flat"))
    assert features["contain_nested_count"] == 0.0
    assert features["contain_max_depth"] == 0.0
    assert features["contain_nested_share"] == 0.0


def test_a_panel_holding_two_boxes_nests_both():
    features = co.extract(ctx.from_masks(nested_panel(), page_id="panel"))
    assert features["contain_nested_count"] == 2.0
    assert features["contain_max_depth"] == 1.0
    assert features["contain_nested_share"] == pytest.approx(2 / 3)


def test_depth_counts_every_level():
    features = co.extract(ctx.from_masks(three_deep(), page_id="deep"))
    assert features["contain_max_depth"] == 2.0
    assert features["contain_nested_count"] == 2.0


def test_writing_inside_a_box_is_not_nesting():
    """4.1.6 measures labels in shapes; this measures shapes in shapes, and 3.2.8 keeps them
    apart."""
    image = np.zeros((500, 500), np.uint8)
    cv2.rectangle(image, (60, 60), (440, 440), 255, 3)
    cv2.putText(image, "step", (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
    assert co.extract(ctx.from_masks(image > 0, page_id="labelled"))["contain_nested_count"] == 0.0


def test_an_empty_page_is_nan_not_zero():
    features = co.extract(ctx.from_masks(np.zeros((300, 300), bool), page_id="blank"))
    assert all(np.isnan(v) for v in features.values())


def test_nesting_is_read_from_the_recomputed_depth_not_the_contour_tree():
    """Two boxes welded by a connector are siblings; the dropped fusion shell is not a parent."""
    image = np.zeros((400, 700), np.uint8)
    cv2.rectangle(image, (40, 120), (200, 260), 255, 3)
    cv2.rectangle(image, (460, 120), (620, 260), 255, 3)
    cv2.line(image, (200, 190), (460, 190), 255, 3)
    assert co.extract(ctx.from_masks(image > 0, page_id="welded"))["contain_max_depth"] == 0.0

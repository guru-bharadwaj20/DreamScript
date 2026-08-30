"""Phase 4.1.6 - text statistics."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx
from src.features import textstats as ts


def labelled_box(inside_text: bool = True) -> np.ndarray:
    """One drawn box with a word either inside it or well away from it."""
    image = np.zeros((500, 700), np.uint8)
    cv2.rectangle(image, (60, 60), (400, 300), 255, 3)
    x = 140 if inside_text else 470
    y = 180 if inside_text else 430
    cv2.putText(image, "state", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    return image > 0


def test_the_three_names_are_produced():
    assert set(ts.extract(ctx.from_masks(labelled_box(), page_id="p"))) == set(ts.NAMES)


def test_a_page_with_no_writing_has_zero_area_but_no_label_statistics():
    """Zero writing is a fact; the mean length of no labels is not."""
    image = np.zeros((400, 400), np.uint8)
    cv2.rectangle(image, (60, 60), (340, 340), 255, 3)
    features = ts.extract(ctx.from_masks(image > 0, page_id="unlabelled"))
    assert features["text_area_frac"] == 0.0
    assert np.isnan(features["text_label_length"])
    assert np.isnan(features["text_inside_share"])


def test_an_empty_page_has_no_text_fraction_at_all():
    features = ts.extract(ctx.from_masks(np.zeros((200, 200), bool), page_id="blank"))
    assert all(np.isnan(v) for v in features.values())


def test_a_label_inside_a_box_scores_higher_than_one_outside():
    within = ts.extract(ctx.from_masks(labelled_box(True), page_id="in"))["text_inside_share"]
    beyond = ts.extract(ctx.from_masks(labelled_box(False), page_id="out"))["text_inside_share"]
    assert within > beyond


def test_text_area_fraction_is_a_share_of_ink_not_of_the_page():
    context = ctx.from_masks(labelled_box(), page_id="p")
    expected = context.text_mask.sum() / context.mask.sum()
    assert ts.extract(context)["text_area_frac"] == pytest.approx(expected)
    assert 0.0 <= ts.extract(context)["text_area_frac"] <= 1.0


def test_label_length_is_measured_in_page_widths():
    """A width in pixels would change with the camera; this must not."""
    small = ts.extract(ctx.from_masks(labelled_box(), page_id="small"))["text_label_length"]
    big = cv2.resize(
        labelled_box().astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST
    )
    large = ts.extract(ctx.from_masks(big > 0, page_id="large"))["text_label_length"]
    if np.isfinite(small) and np.isfinite(large):
        assert small == pytest.approx(large, abs=0.02)


def test_blocks_come_from_the_text_layer_not_a_second_proposal():
    context = ctx.from_masks(labelled_box(), page_id="p")
    for x, y, w, h in ts.blocks(context):
        assert context.text_mask[y : y + h, x : x + w].any()

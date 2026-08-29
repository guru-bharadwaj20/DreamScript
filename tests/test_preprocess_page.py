"""Phase 3.1.2 - page detection.

Synthetic pages first, because they let a test state exactly where the answer should be; then
the corpus, where the criterion is hdBPMN's own annotations.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import page


def desk_photo(corners=None, size=(700, 900), background=70, paper=225) -> np.ndarray:
    """A bright quadrilateral page on a darker desk, with some ink on it."""
    image = np.full((*size, 3), background, np.uint8)
    corners = (
        corners
        if corners is not None
        else np.array([[120, 90], [780, 120], [760, 610], [140, 580]], np.float32)
    )
    cv2.fillConvexPoly(image, corners.astype(np.int32), (paper, paper, paper))
    cv2.rectangle(image, (280, 240), (520, 400), (30, 30, 30), 4)
    cv2.line(image, (300, 460), (560, 460), (30, 30, 30), 3)
    return image


def test_finds_a_page_on_a_desk():
    corners = np.array([[120, 90], [780, 120], [760, 610], [140, 580]], np.float32)
    found = page.detect(desk_photo(corners))
    assert found is not None
    # Detection grows the quad outward on purpose, so compare loosely and check it never cuts in.
    assert page.contains(found, [[280, 240, 240, 160]])
    assert np.abs(found - page.order_corners(corners)).max() < 60


def test_declines_when_the_page_fills_the_frame():
    """A scan has no page boundary; inventing one can only lose ink."""
    image = np.full((600, 800, 3), 235, np.uint8)
    cv2.rectangle(image, (200, 200), (500, 400), (30, 30, 30), 4)
    assert page.detect(image) is None


def test_declines_on_a_patch_of_paper_surrounded_by_paper():
    """The false positive that survived every geometric test: a quad cutting across a sheet."""
    image = np.full((600, 800, 3), 235, np.uint8)
    cv2.rectangle(image, (100, 100), (400, 300), (215, 215, 215), 2)  # a faint interior box
    assert page.detect(image) is None


@pytest.mark.parametrize(
    "quad",
    [
        [[10, 300], [400, 10], [790, 300], [400, 290]],  # near-triangle: one side collapsed
        [[20, 20], [780, 20], [400, 580], [380, 575]],  # acute wedge
        [[10, 10], [790, 20], [780, 120], [20, 600]],  # sliver: opposite sides 5:1
    ],
)
def test_rejects_degenerate_quads(quad):
    """Canny along the ruled lines of squared paper produced shapes like these, and they were
    accepted as pages until the angle, side-ratio and rectangularity bounds went in."""
    assert not page._plausible_page(np.array(quad, np.float32))


def test_plausibility_alone_does_not_catch_everything():
    """An honest limit: a quad can be perfectly parallelogram-shaped and still not be the page.
    That is why `_brighter_than_surround` and `_edge_support` exist as well."""
    plausible_but_wrong = np.array([[16, 232], [300, 10], [770, 300], [300, 560]], np.float32)
    assert page._plausible_page(plausible_but_wrong)


def test_accepts_a_page_seen_at_an_angle():
    tilted = np.array([[100, 60], [720, 130], [690, 560], [130, 500]], np.float32)
    assert page._plausible_page(tilted)


def test_corners_are_ordered_regardless_of_input_order():
    corners = np.array([[760, 610], [120, 90], [140, 580], [780, 120]], np.float32)
    ordered = page.order_corners(corners)
    assert ordered[0].tolist() == [120, 90]  # top-left
    assert ordered[2].tolist() == [760, 610]  # bottom-right


def test_contains_rejects_a_box_outside_the_quad():
    quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    assert page.contains(quad, [[10, 10, 50, 50]])
    assert not page.contains(quad, [[90, 90, 50, 50]])


def test_detection_never_returns_a_non_convex_quad():
    found = page.detect(desk_photo())
    assert found is not None
    assert cv2.isContourConvex(found.reshape(-1, 1, 2).astype(np.int32))


def test_corpus_evaluation_meets_the_bar():
    """hdBPMN's own annotations are the criterion: a crop is correct iff it keeps every shape."""
    results = page.evaluate(60)
    if not results["total"]:
        pytest.skip("hdBPMN IR not present")
    handled = (results["correct"] + results["no_page"]) / results["total"]
    assert handled >= 0.90, results

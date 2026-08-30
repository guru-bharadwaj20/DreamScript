"""Phase 4.1.2 - ratios."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx
from src.features import ratios as ra
from tests.test_features_structural import chain


def circles(count: int = 3, size: int = 600) -> np.ndarray:
    image = np.zeros((size, size), np.uint8)
    for i in range(count):
        cv2.circle(image, (110 + 190 * (i % 3), 150 + 190 * (i // 3)), 70, 255, 3)
    return image > 0


def test_the_four_names_are_all_produced():
    features = ra.extract(ctx.from_masks(chain(3), page_id="chain"))
    assert set(features) == set(ra.NAMES)


def test_an_undefined_ratio_is_nan_not_zero():
    """A page with no shapes cannot answer "per node"; 4.2.3 needs to be able to tell."""
    blank = ctx.from_masks(np.zeros((300, 300), bool), page_id="blank")
    features = ra.extract(blank)
    assert all(np.isnan(v) for v in features.values())


def test_ratios_are_never_infinite():
    """`inf` in one row would destroy the scaler fitted in 4.2.4."""
    assert ra.ratio(1.0, 0.0) != np.inf
    assert ra.ratio(5.0, 1e-12) == ra.MAX_RATIO
    for value in ra.extract(ctx.from_masks(chain(4), page_id="chain")).values():
        assert not np.isinf(value)


def test_ratios_do_not_move_when_the_drawing_gets_bigger():
    """The point of a ratio: same diagram, more boxes, same numbers."""
    small = ra.extract(ctx.from_masks(chain(3), page_id="small"))
    large = ra.extract(ctx.from_masks(chain(6), page_id="large"))
    assert small["edges_per_node"] == pytest.approx(large["edges_per_node"], abs=0.2)


def test_edges_per_node_counts_what_4_1_1_counted():
    context = ctx.from_masks(chain(5), page_id="chain5")
    assert ra.extract(context)["edges_per_node"] == pytest.approx(4 / 5)


def test_line_to_curve_separates_boxes_from_circles():
    boxes = ra.extract(ctx.from_masks(chain(3), page_id="boxes"))["line_to_curve"]
    round_ = ra.extract(ctx.from_masks(circles(3), page_id="circles"))["line_to_curve"]
    assert boxes > 1.0 > round_


def test_straight_share_is_weighted_by_perimeter():
    """One big box among small circles is a mostly-straight page, not a mostly-curved one."""
    image = np.zeros((700, 700), np.uint8)
    cv2.rectangle(image, (40, 40), (660, 500), 255, 3)
    for i in range(4):
        cv2.circle(image, (120 + 140 * i, 600), 40, 255, 3)
    weighted = ra.straight_share(ctx.from_masks(image > 0, page_id="mixed"))
    assert weighted > 0.4


def test_straight_share_is_nan_without_shapes():
    assert np.isnan(ra.straight_share(ctx.from_masks(np.zeros((200, 200), bool))))

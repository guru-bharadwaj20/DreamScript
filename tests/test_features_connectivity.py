"""Phase 4.1.7 - connectivity."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import connectivity as cn
from src.features import context as ctx
from tests.test_features_structural import chain


def two_boxes(joined: bool) -> np.ndarray:
    image = np.zeros((400, 700), np.uint8)
    cv2.rectangle(image, (40, 120), (200, 260), 255, 3)
    cv2.rectangle(image, (460, 120), (620, 260), 255, 3)
    if joined:
        cv2.line(image, (200, 190), (460, 190), 255, 3)
    return image > 0


def test_the_four_names_are_produced():
    assert set(cn.extract(ctx.from_masks(chain(3), page_id="c"))) == set(cn.NAMES)


def test_a_chain_of_boxes_has_degree_two_minus_the_ends():
    """n boxes and n-1 connectors: mean degree 2(n-1)/n."""
    for nodes in (3, 5):
        features = cn.extract(ctx.from_masks(chain(nodes), page_id=f"c{nodes}"))
        assert features["conn_mean_degree"] == pytest.approx(2 * (nodes - 1) / nodes)
        assert features["conn_components"] == 1.0
        assert features["conn_cycle_count"] == 0.0


def test_unconnected_boxes_are_separate_components():
    features = cn.extract(ctx.from_masks(two_boxes(joined=False), page_id="apart"))
    assert features["conn_components"] == 2.0
    assert features["conn_mean_degree"] == 0.0


def test_joining_them_makes_one_component():
    features = cn.extract(ctx.from_masks(two_boxes(joined=True), page_id="joined"))
    assert features["conn_components"] == 1.0
    assert features["conn_mean_degree"] == pytest.approx(1.0)


def test_a_drawn_loop_back_is_swallowed_by_its_own_enclosed_area():
    """The documented limitation: a cycle in a *drawing* encloses area, and enclosed area is
    what `context` calls a shape. The loop becomes a region and its connectors vanish inside
    it, so `conn_cycle_count` cannot see the cycle it was built to count."""
    image = np.zeros((600, 900), np.uint8)
    for i in range(3):
        cv2.rectangle(image, (60 + 280 * i, 120), (220 + 280 * i, 260), 255, 3)
        if i:
            cv2.line(image, (60 + 280 * i - 120, 190), (60 + 280 * i, 190), 255, 3)
    cv2.line(image, (140, 260), (140, 420), 255, 3)
    cv2.line(image, (140, 420), (700, 420), 255, 3)
    cv2.line(image, (700, 420), (700, 260), 255, 3)
    features = cn.extract(ctx.from_masks(image > 0, page_id="loop"))
    assert features["conn_cycle_count"] == 0.0


def test_the_cyclomatic_formula_is_right_where_the_graph_is_known():
    """edges - nodes + components, checked on graphs rather than on pixels."""
    assert cn.components(3, [(0, 1), (1, 2), (0, 2)]) == 1
    edges = [(0, 1), (1, 2), (0, 2)]
    assert len(edges) - 3 + cn.components(3, edges) == 1
    tree = [(0, 1), (1, 2)]
    assert len(tree) - 3 + cn.components(3, tree) == 0


def test_cycle_count_is_never_negative():
    features = cn.extract(ctx.from_masks(two_boxes(joined=False), page_id="apart"))
    assert features["conn_cycle_count"] == 0.0


def test_a_stub_of_ink_is_not_a_self_loop():
    """Without the length floor every broken arrow end would become a loop."""
    image = np.zeros((500, 500), np.uint8)
    cv2.rectangle(image, (150, 150), (350, 350), 255, 3)
    cv2.line(image, (360, 250), (368, 250), 255, 2)
    assert cn.extract(ctx.from_masks(image > 0, page_id="stub"))["conn_self_loops"] == 0.0


def test_components_counts_isolated_nodes():
    assert cn.components(4, [(0, 1)]) == 3
    assert cn.components(4, [(0, 1), (1, 2), (2, 3)]) == 1
    assert cn.components(0, []) == 0


def test_an_empty_page_is_nan_everywhere():
    features = cn.extract(ctx.from_masks(np.zeros((300, 300), bool), page_id="blank"))
    assert all(np.isnan(v) for v in features.values())

"""Phase 4.1.1 - structural counts."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx
from src.features import structural as st


def chain(nodes: int = 4) -> np.ndarray:
    """`nodes` boxes in a row, joined left to right by a straight connector."""
    image = np.zeros((400, 260 * nodes + 60), np.uint8)
    for i in range(nodes):
        x = 60 + 260 * i
        cv2.rectangle(image, (x, 120), (x + 160, 260), 255, 3)
        if i:
            cv2.line(image, (x - 100, 190), (x, 190), 255, 3)
    return image > 0


def test_the_four_names_are_all_produced():
    features = st.extract(ctx.from_masks(chain(), page_id="chain"))
    assert set(features) == set(st.NAMES)
    assert all(isinstance(v, float) for v in features.values())


def test_node_count_matches_the_boxes_drawn():
    for nodes in (2, 3, 5):
        context = ctx.from_masks(chain(nodes), page_id=f"chain{nodes}")
        assert st.extract(context)["node_count"] == nodes


def test_edge_count_matches_the_connectors_drawn():
    for nodes in (2, 3, 5):
        context = ctx.from_masks(chain(nodes), page_id=f"chain{nodes}")
        assert st.extract(context)["edge_count"] == nodes - 1


def test_a_page_with_no_connectors_has_no_edges():
    image = np.zeros((400, 400), np.uint8)
    cv2.rectangle(image, (40, 40), (180, 180), 255, 3)
    cv2.rectangle(image, (220, 220), (360, 360), 255, 3)
    context = ctx.from_masks(image > 0, page_id="apart")
    assert st.extract(context)["edge_count"] == 0
    assert st.extract(context)["node_count"] == 2


def test_short_fragments_are_not_edges():
    """A speck of loose ink is not a connector, whatever the binarizer left behind."""
    image = np.zeros((400, 700), np.uint8)
    cv2.rectangle(image, (40, 120), (200, 260), 255, 3)
    cv2.rectangle(image, (460, 120), (620, 260), 255, 3)
    cv2.line(image, (300, 320), (303, 323), 255, 1)  # 4px on a 700px page: below the floor
    context = ctx.from_masks(image > 0, page_id="fragment")
    assert st.extract(context)["edge_count"] == 0


def test_the_strand_floor_is_a_fraction_of_the_page_not_a_pixel_count():
    """The same drawing at twice the size must give the same counts."""
    small = st.extract(ctx.from_masks(chain(3), page_id="small"))
    big = np.zeros((800, 1620), np.uint8)
    for i in range(3):
        x = 120 + 520 * i
        cv2.rectangle(big, (x, 240), (x + 320, 520), 255, 6)
        if i:
            cv2.line(big, (x - 200, 380), (x, 380), 255, 6)
    large = st.extract(ctx.from_masks(big > 0, page_id="big"))
    assert small["node_count"] == large["node_count"] == 3
    assert small["edge_count"] == large["edge_count"] == 2


def test_counts_are_finite_on_an_empty_page():
    context = ctx.from_masks(np.zeros((300, 300), bool), page_id="blank")
    features = st.extract(context)
    assert all(np.isfinite(v) for v in features.values())
    assert features["node_count"] == 0
    assert features["edge_count"] == 0


def test_text_blocks_count_writing_not_drawing(flowchart_image):
    context = ctx.from_masks(flowchart_image < 128, flowchart_image, page_id="fixture")
    assert st.extract(context)["text_block_count"] == len(st.text_blocks(context))


def test_agreement_is_computed_over_the_signed_error():
    rows = [
        {"nodes": 5, "true_nodes": 5},
        {"nodes": 4, "true_nodes": 5},
        {"nodes": 8, "true_nodes": 5},
    ]
    scored = st._agreement(rows, "nodes")
    assert scored["exact"] == pytest.approx(1 / 3, abs=1e-3)
    assert scored["within_one"] == pytest.approx(2 / 3, abs=1e-3)
    assert scored["mean_signed_error"] == round((0 - 1 + 3) / 3, 2)

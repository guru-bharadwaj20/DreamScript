"""Label-crop geometry: what claims a line, what a container may not claim, and stacking."""

from __future__ import annotations

import numpy as np

from src.ocr import labelcrops as lc


def test_external_classes_are_the_ones_whose_label_sits_outside():
    assert lc.is_external("Event_1vhsd5l")
    assert lc.is_external("DataObjectReference_02")
    assert lc.is_external("startEvent_3")
    assert not lc.is_external("Activity_0973sxi")
    assert not lc.is_external("task_7")


def test_union_covers_every_line():
    boxes = [[10, 10, 50, 20], [12, 34, 60, 18]]
    assert lc.union(boxes) == [10, 10, 62, 42]


def test_inside_is_inclusive_of_a_small_tolerance():
    assert lc.inside([0, 0, 100, 100], [10, 10, 20, 20])
    assert not lc.inside([0, 0, 100, 100], [90, 90, 40, 40])


def test_polyline_gap_measures_the_whole_connector_not_its_midpoint():
    """An edge label beside the far end of a long arrow must not read as distant."""
    poly = np.array([[0.0, 0.0], [400.0, 0.0]])
    near_end = lc.polyline_gap(poly, [380, 10, 20, 10])
    midpoint_only = float(np.hypot(390 - 200, 15 - 0))
    assert near_end < 20
    assert near_end < midpoint_only


def test_own_lines_lets_one_element_win_several_stacked_lines():
    element = {"id": "Event_1", "bbox": [100, 0, 40, 40], "kind": "node"}
    lines = [[100, 60, 80, 20], [100, 85, 80, 20]]
    won = lc.own_lines([element], lines, page_w=1000)
    assert len(won["Event_1"]) == 2


def test_own_lines_caps_how_many_lines_one_element_may_claim():
    element = {"id": "Event_1", "bbox": [100, 0, 40, 40], "kind": "node"}
    lines = [[100, 50 + 25 * i, 80, 20] for i in range(6)]
    won = lc.own_lines([element], lines, page_w=1000)
    assert len(won["Event_1"]) <= 3


def test_own_lines_ignores_a_line_beyond_reach():
    element = {"id": "Event_1", "bbox": [0, 0, 20, 20], "kind": "node"}
    won = lc.own_lines([element], [[900, 900, 50, 20]], page_w=1000)
    assert won == {}


def test_stack_will_not_chain_a_whole_column_into_one_label():
    """The unbounded version produced a 554x527 'label'; the bound is the fix."""
    lines = [[0, 40 * i, 100, 30] for i in range(10)]
    merged = lc.stack(lines)
    assert all(b[3] < 200 for b in merged), "a merged block swallowed the column"


def test_inset_box_drops_the_drawn_outline():
    box = lc.inset_box({"bbox": [0, 0, 100, 100]}, inset=0.10)
    assert box == [10, 10, 80, 80]


def test_crop_returns_none_for_a_degenerate_region():
    image = np.full((200, 200), 255, np.uint8)
    assert lc.crop(image, [0, 0, 50, 2]) is None


def test_crop_normalises_to_the_shared_line_height():
    image = np.full((300, 300), 255, np.uint8)
    patch = lc.crop(image, [10, 10, 120, 60])
    assert patch is not None and patch.shape[0] == lc.HEIGHT


def test_elements_of_gives_an_edge_its_polyline_and_label():
    diagram = {
        "nodes": [{"id": "Activity_1", "bbox": [0, 0, 10, 10], "text": "a"}],
        "edges": [{"id": "Flow_1", "polyline": [[0, 0], [10, 10]], "label": "yes"}],
    }
    edge = next(e for e in lc.elements_of(diagram) if e["kind"] == "edge")
    assert edge["text"] == "yes"
    assert edge["polyline"] is not None

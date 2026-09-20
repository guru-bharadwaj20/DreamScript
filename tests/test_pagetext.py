"""`src.assemble.pagetext` - the shape-to-rule translation, without a recogniser.

The ownership rules and the recogniser are both already tested where they live. What is new
here is the adapter: `src.ocr.ownership` decides which rule applies by reading a BPMN element
id, and a *predicted* node has no such id - it has the detector's shape class. This file pins
that translation, and one bug that translation is easy to get wrong.

`read_page` is never called. These tests load no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.assemble import pagetext


def _node(node_id, bbox, shape="rectangle"):
    return {"id": node_id, "bbox": bbox, "shape": shape}


def test_a_circle_is_an_event_because_bpmn_writes_its_label_outside():
    elements, back = pagetext.elements_for(
        {"nodes": [_node("d0", [10, 10, 30, 30], "circle")]}, (1000, 1000)
    )
    assert elements[0]["id"].startswith("Event_")
    assert back[elements[0]["id"]] == "d0"


def test_a_double_circle_and_a_diamond_are_also_external():
    for shape in ("double-circle", "diamond"):
        elements, _ = pagetext.elements_for(
            {"nodes": [_node("d0", [10, 10, 30, 30], shape)]}, (1000, 1000)
        )
        assert elements[0]["id"].startswith("Event_"), shape


def test_a_rectangle_is_a_task_because_its_label_sits_inside():
    elements, _ = pagetext.elements_for(
        {"nodes": [_node("d1", [10, 10, 120, 60], "rectangle")]}, (1000, 1000)
    )
    assert elements[0]["id"].startswith("Task_")


def test_a_node_covering_much_of_the_page_is_a_container():
    """A pool or lane writes its title in a rotated header strip, not beside the shape."""
    elements, _ = pagetext.elements_for(
        {"nodes": [_node("d2", [0, 0, 900, 400], "rectangle")]}, (1000, 1000)
    )
    assert elements[0]["id"].startswith("Participant_")


def test_the_container_rule_is_area_based_not_shape_based():
    """Same shape, different size, different rule - which is the whole point of the threshold."""
    small, _ = pagetext.elements_for(
        {"nodes": [_node("d3", [0, 0, 50, 50], "rectangle")]}, (1000, 1000)
    )
    big, _ = pagetext.elements_for(
        {"nodes": [_node("d3", [0, 0, 900, 900], "rectangle")]}, (1000, 1000)
    )
    assert small[0]["id"].startswith("Task_")
    assert big[0]["id"].startswith("Participant_")


def test_a_node_without_a_box_is_skipped_rather_than_crashing():
    elements, back = pagetext.elements_for(
        {"nodes": [{"id": "d4", "shape": "circle"}, _node("d5", [1, 1, 5, 5])]}, (100, 100)
    )
    assert [back[e["id"]] for e in elements] == ["d5"]


def test_ids_stay_distinct_so_the_map_back_is_not_lossy():
    """Two nodes of the same kind must not collide into one ownership element."""
    elements, back = pagetext.elements_for(
        {
            "nodes": [
                _node("d6", [10, 10, 40, 40], "circle"),
                _node("d7", [80, 80, 40, 40], "circle"),
            ]
        },
        (1000, 1000),
    )
    assert len({e["id"] for e in elements}) == 2
    assert sorted(back.values()) == ["d6", "d7"]


def test_boxes_are_used_as_given_and_not_rescaled():
    """The coordinate space is the caller's problem, and getting it wrong is silent.

    This cost a wrong measurement: ground-truth boxes live in the IR's native space (2480x1612
    on the page tested) while the stored image is 1280x832, so feeding annotation boxes straight
    in put every crop outside the image and the reader scored 0 of 51 rather than 0.373. The
    module does not guess a scale - a predicted diagram is already in image space - so the test
    states that plainly rather than letting the next caller rediscover it.
    """
    boxes = [10.0, 20.0, 30.0, 40.0]
    elements, _ = pagetext.elements_for({"nodes": [_node("d8", boxes)]}, (1000, 1000))
    assert elements[0]["bbox"] == boxes


def test_apply_writes_text_onto_dict_nodes(monkeypatch):
    diagram = {"nodes": [_node("d9", [1, 1, 2, 2]), _node("d10", [3, 3, 2, 2])]}
    page = SimpleNamespace(image="x.png", source="hdbpmn", id="p", name="p")
    monkeypatch.setattr(pagetext, "cached", lambda _p: {"d9": "assess risk", "d10": ""})
    assert pagetext.apply(page, diagram) == 1
    assert diagram["nodes"][0]["text"] == "assess risk"
    assert "text" not in diagram["nodes"][1] or diagram["nodes"][1].get("text") == ""


def test_apply_is_a_no_op_when_nothing_was_read(monkeypatch):
    diagram = {"nodes": [_node("d11", [1, 1, 2, 2])]}
    page = SimpleNamespace(image="x.png", source="hdbpmn", id="p", name="p")
    monkeypatch.setattr(pagetext, "cached", lambda _p: {})
    assert pagetext.apply(page, diagram) == 0


def test_fa_bresler_is_deliberately_not_in_sources():
    """Its state names are inside the circle, where `statelabels` reads them at 94.4%."""
    assert "hdbpmn" in pagetext.SOURCES
    assert "fa_bresler" not in pagetext.SOURCES


@pytest.mark.parametrize("area", [0.0, 0.05, 0.11])
def test_nodes_below_the_container_threshold_are_not_containers(area):
    side = (area * 1000 * 1000) ** 0.5
    elements, _ = pagetext.elements_for(
        {"nodes": [_node("d12", [0, 0, side, side], "rectangle")]}, (1000, 1000)
    )
    assert not elements[0]["id"].startswith("Participant_")

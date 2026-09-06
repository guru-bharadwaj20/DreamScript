"""Phase 10.1.1 - `build()`'s three cuts (arrowheads, low score, duplicates), tested in isolation.

Detections are faked directly rather than read through the cache, so these tests exercise
`build()`'s own logic and not 9.1's detector or the corpus's file layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.assemble import corpus, nodes


@dataclass(frozen=True)
class FakePage:
    name: str = "p1"
    id: str = "p1"
    source: str = "hdbpmn"
    split: str = "test"
    scale: float = 1.0


def fake_detections(rows):
    """Patch `nodes.detections` (the name `build` actually calls) to return fixed rows."""

    def _fake(page):
        return rows

    return _fake


def det(cls, bbox, score):
    return {"cls": cls, "bbox": [float(v) for v in bbox], "score": float(score)}


# -- arrowheads are not nodes ---------------------------------------------------------------


def test_an_arrowhead_detection_does_not_become_a_node(monkeypatch):
    rows = [det("arrowhead", [0, 0, 5, 5], 0.9), det("rectangle", [10, 10, 20, 20], 0.9)]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert [n.shape for n in assembled.nodes] == ["rectangle"]
    assert len(assembled.arrowheads) == 1
    assert assembled.arrowheads[0]["cls"] == "arrowhead"


def test_an_arrowhead_is_never_counted_among_the_dropped_detections(monkeypatch):
    rows = [det("arrowhead", [0, 0, 5, 5], 0.01)]  # score below MIN_SCORE too
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert assembled.dropped == []
    assert len(assembled.arrowheads) == 1


# -- the score threshold ---------------------------------------------------------------------


def test_a_detection_below_min_score_is_dropped_not_kept(monkeypatch):
    rows = [det("rectangle", [0, 0, 10, 10], nodes.MIN_SCORE - 0.01)]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert assembled.nodes == []
    assert len(assembled.dropped) == 1
    assert assembled.dropped[0]["reason"] == "low-score"


def test_a_detection_at_exactly_min_score_is_kept(monkeypatch):
    rows = [det("rectangle", [0, 0, 10, 10], nodes.MIN_SCORE)]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert len(assembled.nodes) == 1


# -- de-duplication ----------------------------------------------------------------------------


def test_two_overlapping_boxes_of_different_classes_are_deduplicated(monkeypatch):
    # A rectangle and a rounded-rect on (nearly) the same ink: 9.1's NMS is class-wise and
    # cannot see this, so `build`'s own cross-class IoU check must.
    rows = [
        det("rectangle", [0, 0, 100, 100], 0.9),
        det("rounded-rect", [0, 0, 100, 100], 0.8),
    ]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert len(assembled.nodes) == 1
    assert assembled.nodes[0].shape == "rectangle"  # the higher-scoring one survives
    assert len(assembled.dropped) == 1
    assert assembled.dropped[0]["reason"] == "duplicate"


def test_two_distant_boxes_of_the_same_class_are_not_deduplicated(monkeypatch):
    rows = [
        det("rectangle", [0, 0, 10, 10], 0.9),
        det("rectangle", [1000, 1000, 10, 10], 0.8),
    ]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert len(assembled.nodes) == 2
    assert assembled.dropped == []


def test_dedupe_false_disables_duplicate_suppression(monkeypatch):
    rows = [
        det("rectangle", [0, 0, 100, 100], 0.9),
        det("rectangle", [0, 0, 100, 100], 0.8),
    ]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage(), dedupe=False)
    assert len(assembled.nodes) == 2
    assert assembled.dropped == []


def test_nest_fraction_none_by_default_keeps_a_contained_box(monkeypatch):
    # A small box entirely inside a larger one is not a duplicate by IoU, and NEST_FRACTION is
    # None by default, so both should survive as separate nodes.
    rows = [
        det("rectangle", [0, 0, 100, 100], 0.9),
        det("rectangle", [10, 10, 5, 5], 0.8),
    ]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert len(assembled.nodes) == 2


def test_nest_fraction_set_drops_the_contained_box(monkeypatch):
    rows = [
        det("rectangle", [0, 0, 100, 100], 0.9),
        det("rectangle", [10, 10, 5, 5], 0.8),
    ]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage(), nest=0.85)
    assert len(assembled.nodes) == 1
    assert assembled.dropped[0]["reason"] == "nested-inside"


# -- id stability --------------------------------------------------------------------------


def test_node_ids_are_assigned_in_descending_score_order(monkeypatch):
    rows = [
        det("rectangle", [0, 0, 10, 10], 0.4),
        det("circle", [50, 50, 10, 10], 0.9),
    ]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    ids_by_shape = {n.shape: n.id for n in assembled.nodes}
    assert ids_by_shape["circle"] == "n0"  # higher score, sorted first
    assert ids_by_shape["rectangle"] == "n1"


def test_building_the_same_detections_twice_produces_identical_ids(monkeypatch):
    rows = [
        det("rectangle", [0, 0, 10, 10], 0.4),
        det("circle", [50, 50, 10, 10], 0.9),
        det("diamond", [80, 80, 10, 10], 0.9),  # tie broken by class, deterministically
    ]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    first = [n.id for n in nodes.build(FakePage()).nodes]
    second = [n.id for n in nodes.build(FakePage()).nodes]
    assert first == second


# -- bbox stays on the page -------------------------------------------------------------------


@pytest.mark.parametrize("source", ["fa_bresler", "flowchartseg", "hdbpmn"])
def test_every_built_node_bbox_is_within_the_pages_own_size(source):
    pages = corpus.pages(source=source)
    if not pages:
        pytest.skip(f"no held-out pages for {source}")
    page = pages[0]
    assembled = nodes.build(page)
    w, h = page.size
    for node in assembled.nodes:
        x, y, bw, bh = node.bbox
        assert x >= -1e-6
        assert y >= -1e-6
        assert x + bw <= w + 1e-6
        assert y + bh <= h + 1e-6


# -- Assembled / build() contract -------------------------------------------------------------


def test_assembled_nodes_property_mirrors_the_diagram_nodes(monkeypatch):
    rows = [det("rectangle", [0, 0, 10, 10], 0.9)]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    assert assembled.nodes is assembled.diagram.nodes


def test_a_node_carries_no_text_and_says_so_in_its_attrs(monkeypatch):
    rows = [det("rectangle", [0, 0, 10, 10], 0.9)]
    monkeypatch.setattr(nodes, "detections", fake_detections(rows))
    assembled = nodes.build(FakePage())
    node = assembled.nodes[0]
    assert node.text == ""
    assert node.attrs["text_source"] == "none"
    assert node.attrs["shape_overridden"] is False


# -- matching -------------------------------------------------------------------------------


def test_match_is_one_to_one_even_when_two_predictions_fit_one_truth_box():
    from src.ir.model import Node

    predicted = [
        Node(id="p0", shape="rectangle", bbox=[0, 0, 10, 10]),
        Node(id="p1", shape="rectangle", bbox=[1, 1, 10, 10]),
    ]
    actual = [Node(id="a0", shape="rectangle", bbox=[0, 0, 10, 10])]
    pairs = nodes.match(predicted, actual, threshold=0.1)
    assert len(pairs) == 1
    matched_pred_index = pairs[0][0]
    assert matched_pred_index == 0  # the exact-overlap prediction wins, not just the first one

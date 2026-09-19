"""`src.assemble.edgelabels` - the parts that decide *what gets decoded*, without a decoder.

The recogniser is the expensive half and the well-tested half; it is S3, and 10.1.2 measures it.
What is new here is the cheap half: which predicted edges get a crop at all, and whether the read
labels land on the right edges. Both are pure geometry and both are where a mistake would be
silent - a gate that lets every over-segmented fragment through would hand blank paper to an
autoregressive decoder, which answers anyway, and 9.3 measured what that produces.

So `read_page` is stubbed throughout. These tests never load a model.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.assemble import edgelabels


@pytest.fixture
def page(tmp_path):
    """A page with a horizontal line across it and a blob of ink above the middle."""
    import cv2

    image = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(image, (40, 100), (360, 100), 0, 2)
    cv2.rectangle(image, (190, 72), (210, 90), 0, -1)  # the "label", written beside the line
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), image)
    return SimpleNamespace(image=path, source="fa_bresler", id="p1", name="p1")


def _diagram(edges):
    return {
        "nodes": [
            {"id": "n0", "bbox": [10, 60, 60, 60]},
            {"id": "n1", "bbox": [330, 60, 60, 60]},
        ],
        "edges": edges,
    }


def test_an_edge_with_no_polyline_is_never_a_candidate(page):
    ids, _ = edgelabels.candidates(page, _diagram([{"id": "e0"}]))
    assert ids == []


def test_a_degenerate_polyline_is_never_a_candidate(page):
    """A zero-length polyline has no midpoint to crop beside."""
    edge = {"id": "e0", "polyline": [[100.0, 100.0], [100.0, 100.0]]}
    ids, _ = edgelabels.candidates(page, _diagram([edge]))
    assert ids == []


def test_the_edge_with_ink_beside_it_is_selected(page):
    edge = {"id": "e0", "polyline": [[40.0, 100.0], [360.0, 100.0]]}
    ids, patches = edgelabels.candidates(page, _diagram([edge]))
    assert ids == ["e0"]
    assert patches and patches[0].size > 0


def test_a_bare_connector_with_nothing_written_beside_it_is_dropped(page, tmp_path):
    """The gate's whole job: a line with no handwriting must not reach the decoder."""
    import cv2

    image = np.full((200, 400), 255, dtype=np.uint8)
    cv2.line(image, (40, 100), (360, 100), 0, 2)
    bare = tmp_path / "bare.png"
    cv2.imwrite(str(bare), image)
    blank_page = SimpleNamespace(image=bare, source="fa_bresler", id="p2", name="p2")

    edge = {"id": "e0", "polyline": [[40.0, 100.0], [360.0, 100.0]]}
    ids, _ = edgelabels.candidates(blank_page, _diagram([edge]))
    assert ids == []


def test_an_unreadable_page_yields_nothing_rather_than_raising(tmp_path):
    missing = SimpleNamespace(image=tmp_path / "nope.png", source="fa_bresler", id="x", name="x")
    assert edgelabels.candidates(missing, _diagram([])) == ([], [])


def test_apply_writes_labels_onto_dict_edges(page, monkeypatch):
    diagram = _diagram(
        [
            {"id": "e0", "polyline": [[40.0, 100.0], [360.0, 100.0]], "label": ""},
            {"id": "e1", "polyline": [[40.0, 150.0], [360.0, 150.0]], "label": ""},
        ]
    )
    monkeypatch.setattr(edgelabels, "cached", lambda _page: {"e0": "a", "e1": ""})
    written = edgelabels.apply(page, diagram)
    assert written == 1
    assert diagram["edges"][0]["label"] == "a"
    # An empty read must not overwrite with a blank, and must not count as written.
    assert diagram["edges"][1]["label"] == ""


def test_apply_writes_labels_onto_object_edges(page, monkeypatch):
    class Edge:
        def __init__(self, id):
            self.id = id
            self.label = ""

    class Diagram:
        def __init__(self):
            self.edges = [Edge("e0"), Edge("e1")]

        def to_dict(self):
            return {"nodes": [], "edges": [{"id": e.id} for e in self.edges]}

    diagram = Diagram()
    monkeypatch.setattr(edgelabels, "cached", lambda _page: {"e0": "b"})
    assert edgelabels.apply(page, diagram) == 1
    assert diagram.edges[0].label == "b"
    assert diagram.edges[1].label == ""


def test_a_page_with_no_labels_read_is_a_no_op(page, monkeypatch):
    diagram = _diagram([{"id": "e0", "polyline": [[40.0, 100.0], [360.0, 100.0]]}])
    monkeypatch.setattr(edgelabels, "cached", lambda _page: {})
    monkeypatch.setattr(edgelabels, "read_page", lambda *a, **k: {})
    assert edgelabels.apply(page, diagram) == 0


def test_the_unit_falls_back_when_a_diagram_has_no_boxed_nodes():
    """A predicted diagram can lose every box; the crop still needs a scale."""
    unit = edgelabels._unit({"nodes": []}, (600.0, 400.0))
    assert unit >= 6.0


def test_the_ink_floor_is_the_value_the_sweep_chose():
    """Pinned so the constant cannot drift back up without the sweep being redone.

    The sweep is in the module docstring: precision *falls* as this rises, so a well-meaning
    increase would cost both recall and precision.
    """
    assert edgelabels.MIN_INK == 0.001

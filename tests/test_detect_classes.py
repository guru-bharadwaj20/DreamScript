"""Phase 9.1.2 - the class set is frozen, and the boxes it admits are the ones that exist."""

from __future__ import annotations

import numpy as np
import pytest

from src.detect import classes, dataset
from src.ir.vocab import SHAPES


def page(source="hdbpmn", nodes=(), edges=(), size=(1000, 800)):
    return {
        "id": "p1",
        "diagram_type": "flowchart",
        "nodes": list(nodes),
        "edges": list(edges),
        "meta": {"source": source, "image_size": list(size), "image": ""},
    }


def node(node_id, shape, bbox):
    return {
        "id": node_id,
        "shape": shape,
        "bbox": list(bbox),
        "text": "",
        "semantic_role": "process",
    }


# -- the frozen list ------------------------------------------------------------------------


def test_class_ids_are_positional_and_unique():
    """A checkpoint stores integers, so a reorder silently relabels every prediction."""
    assert len(set(classes.CLASSES)) == len(classes.CLASSES)
    assert list(classes.CLASS_INDEX.values()) == list(range(len(classes.CLASSES)))


def test_every_class_is_a_shape_the_ir_vocabulary_knows_or_is_the_derived_arrowhead():
    for name in classes.CLASSES:
        assert name in SHAPES or name == "arrowhead"


def test_every_shape_is_either_included_or_has_a_written_reason():
    """The point of the module: no class disappears without saying why."""
    for shape in SHAPES:
        assert shape in classes.CLASS_INDEX or shape in classes.EXCLUDED, shape


def test_the_ui_widget_classes_are_excluded_and_the_reason_names_the_geometry():
    for name in ("ui-input", "ui-button", "ui-label", "ui-image"):
        assert "sketch2code" in classes.EXCLUDED[name]


def test_didi_and_sketch2code_are_rejected_sources():
    assert set(classes.REJECTED_SOURCES) == {"didi", "sketch2code"}
    assert not set(classes.SOURCES) & set(classes.REJECTED_SOURCES)


# -- node boxes -----------------------------------------------------------------------------


def test_a_node_of_an_included_shape_becomes_an_annotated_box():
    found = classes.boxes_for(page(nodes=[node("n1", "diamond", (10, 10, 40, 40))]))
    assert [(b["cls"], b["basis"]) for b in found] == [("diamond", "annotated")]


def test_a_node_of_an_excluded_shape_is_dropped_not_relabelled():
    found = classes.boxes_for(page(nodes=[node("n1", "text-block", (10, 10, 40, 40))]))
    assert found == []


def test_a_node_without_a_box_is_dropped():
    n = node("n1", "rectangle", (0, 0, 1, 1))
    n["bbox"] = None
    assert classes.boxes_for(page(nodes=[n])) == []


def test_a_page_without_an_image_size_yields_nothing():
    p = page(nodes=[node("n1", "rectangle", (10, 10, 40, 40))])
    p["meta"]["image_size"] = None
    assert classes.boxes_for(p) == []


def test_a_box_running_off_the_page_is_trimmed_to_it():
    found = classes.boxes_for(page(nodes=[node("n1", "rectangle", (-50, -50, 200, 200))]))
    x, y, w, h = found[0]["bbox"]
    assert (x, y) == (0.0, 0.0)
    assert w == 150 and h == 150


def test_a_box_entirely_off_the_page_is_dropped():
    assert classes.boxes_for(page(nodes=[node("n1", "rectangle", (-500, -500, 100, 100))])) == []


# -- the derived arrowhead ------------------------------------------------------------------


def test_the_arrowhead_sits_behind_the_tip_along_the_final_segment():
    box = classes.arrowhead_box([[0.0, 0.0], [100.0, 0.0]], side=20.0)
    x, y, w, h = box
    assert (w, h) == (20.0, 20.0)
    # tip at x=100, centre half a side back => 90; box spans 80..100
    assert x == pytest.approx(80.0)
    assert y == pytest.approx(-10.0)


def test_a_degenerate_polyline_yields_no_arrowhead():
    """fa_bresler's traces contain runs of identical points; they must not become boxes."""
    assert classes.arrowhead_box([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], 10.0) is None
    assert classes.arrowhead_box([[5.0, 5.0]], 10.0) is None
    assert classes.arrowhead_box([], 10.0) is None


def test_the_last_real_segment_is_used_when_the_tail_repeats():
    box = classes.arrowhead_box([[0.0, 0.0], [100.0, 0.0], [100.0, 0.0]], side=20.0)
    assert box[0] == pytest.approx(80.0)


def test_arrowheads_are_derived_for_hdbpmn_only():
    """fa_bresler polylines are raw ink whose direction is whichever way the pen moved."""
    edges = [{"id": "e1", "src": "a", "dst": "b", "directed": True, "polyline": [[0, 0], [100, 0]]}]
    nodes = [node("n1", "rectangle", (10, 10, 100, 100))]
    hd = classes.boxes_for(page("hdbpmn", nodes, edges))
    fa = classes.boxes_for(page("fa_bresler", nodes, edges))
    assert any(b["cls"] == "arrowhead" for b in hd)
    assert not any(b["cls"] == "arrowhead" for b in fa)


def test_an_undirected_edge_gets_no_head():
    edges = [
        {"id": "e1", "src": "a", "dst": "b", "directed": False, "polyline": [[0, 0], [100, 0]]}
    ]
    found = classes.boxes_for(page("hdbpmn", [node("n1", "rectangle", (10, 10, 100, 100))], edges))
    assert not any(b["cls"] == "arrowhead" for b in found)


def test_arrowheads_can_be_switched_off_entirely():
    edges = [{"id": "e1", "src": "a", "dst": "b", "directed": True, "polyline": [[0, 0], [100, 0]]}]
    nodes = [node("n1", "rectangle", (10, 10, 100, 100))]
    found = classes.boxes_for(page("hdbpmn", nodes, edges), arrowheads=False)
    assert not any(b["cls"] == "arrowhead" for b in found)


def test_the_head_size_is_clipped_to_a_band_of_the_page_diagonal():
    diagonal = float(np.hypot(1000, 800))
    tiny = page(nodes=[node("n1", "rectangle", (0, 0, 4, 4))])
    huge = page(nodes=[node("n1", "rectangle", (0, 0, 900, 700))])
    assert classes.arrowhead_side(tiny) == pytest.approx(classes.ARROWHEAD_MIN_DIAG * diagonal)
    assert classes.arrowhead_side(huge) == pytest.approx(classes.ARROWHEAD_MAX_DIAG * diagonal)


# -- the export -----------------------------------------------------------------------------


def test_yolo_lines_are_normalised_centres():
    record = {
        "width": 200,
        "height": 100,
        "boxes": [{"cls": "diamond", "bbox": [50.0, 25.0, 100.0, 50.0], "basis": "annotated"}],
    }
    (line,) = dataset.yolo_lines(record)
    cls, cx, cy, w, h = line.split()
    assert int(cls) == classes.CLASS_INDEX["diamond"]
    assert float(cx) == pytest.approx(0.5)
    assert float(cy) == pytest.approx(0.5)
    assert float(w) == pytest.approx(0.5)
    assert float(h) == pytest.approx(0.5)


def test_fa_splits_are_writer_disjoint():
    diagrams = [
        {"id": f"w{w:03d}_fa_{i:03d}", "meta": {"scribe_id": f"fa-writer{w:03d}"}}
        for w in range(25)
        for i in range(12)
    ]
    assigned = dataset.fa_splits(diagrams)
    by_writer: dict[str, set[str]] = {}
    for key, split in assigned.items():
        writer = key.split(":")[1].split("_")[0]
        by_writer.setdefault(writer, set()).add(split)
    assert all(len(splits) == 1 for splits in by_writer.values())
    assert set(assigned.values()) == {"train", "val", "test"}


def test_the_manifest_split_names_are_translated_to_yolos():
    assert dataset.SPLIT_ALIAS["validation"] == "val"
    assert set(dataset.SPLITS) == {"train", "val", "test"}

"""Phase 2.1.3 - the Python model and the JSON Schema must not drift apart."""

from __future__ import annotations

import json

import pytest

from src.ir.model import Diagram, Edge, IRError, Node


def tiny() -> Diagram:
    return Diagram(
        id="demo-001",
        diagram_type="flowchart",
        nodes=[
            Node("n1", "ellipse", [10, 10, 60, 40], "start", "start"),
            Node("n2", "rectangle", [10, 80, 90, 50], "do it", "process"),
        ],
        edges=[Edge("e1", "n1", "n2", True, "", [[40, 50], [40, 80]])],
        meta={"source": "unit-test", "geometry": "annotated", "image_size": [200, 200]},
    )


def test_tiny_diagram_is_valid():
    assert tiny().problems() == []


def test_roundtrip_through_dict_is_lossless():
    d = tiny()
    assert Diagram.from_dict(d.to_dict()).to_dict() == d.to_dict()


def test_roundtrip_through_disk_is_lossless(tmp_path):
    d = tiny()
    path = d.save(tmp_path / "demo.ir.json")
    assert Diagram.load(path).to_dict() == d.to_dict()


def test_unset_optionals_are_omitted_not_nulled():
    """`"attrs": null` is not schema-legal, so it must never be written."""
    raw = tiny().to_dict()
    assert "attrs" not in raw["nodes"][0]
    assert "source_id" not in raw["edges"][0]


def test_save_refuses_to_write_invalid_ir(tmp_path):
    d = tiny()
    d.diagram_type = "sudoku"
    with pytest.raises(IRError):
        d.save(tmp_path / "bad.ir.json")
    assert not (tmp_path / "bad.ir.json").exists()


def test_meta_requires_source_and_geometry():
    d = tiny()
    d.meta.pop("geometry")
    assert any("geometry" in p for p in d.problems())


def test_graph_helpers():
    d = tiny()
    assert d.node("n2").text == "do it"
    assert d.node("nope") is None
    assert [e.id for e in d.out_edges("n1")] == ["e1"]
    assert [e.id for e in d.in_edges("n1")] == []
    assert d.node_ids == {"n1", "n2"}
    assert d.nodes[0].centre == (40, 30)


def test_written_file_is_utf8_and_newline_terminated(tmp_path):
    d = tiny()
    d.nodes[0].text = "café ☕"
    path = d.save(tmp_path / "u.ir.json")
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8"))["nodes"][0]["text"] == "café ☕"


# -- Phase 2.1.6, ambiguity -----------------------------------------------------------


def test_ambiguity_arrays_are_required_and_default_to_empty():
    raw = tiny().to_dict()
    assert raw["unresolved_edges"] == []
    assert raw["crossed_out"] == []
    assert raw["low_conf_text"] == []
    del raw["crossed_out"]
    from src.ir import schema

    assert schema.problems("ir", raw)


def test_recording_an_unresolved_edge_derives_its_reason():
    d = tiny()
    d.edges[0].dst = None
    d.record_unresolved(d.edges[0], candidates=["n2"])
    assert d.unresolved_edges == [{"edge": "e1", "reason": "no-target", "candidates": ["n2"]}]
    assert d.problems() == []


def test_sync_unresolved_is_idempotent():
    d = tiny()
    d.edges[0].src = None
    assert d.sync_unresolved() == 1
    assert d.sync_unresolved() == 0
    assert d.unresolved_edges[0]["reason"] == "no-source"


def test_both_ends_open_is_its_own_reason():
    d = tiny()
    d.edges[0].src = d.edges[0].dst = None
    d.sync_unresolved()
    assert d.unresolved_edges[0]["reason"] == "both-ends-open"


def test_crossed_out_and_low_conf_text_roundtrip(tmp_path):
    d = tiny()
    d.crossed_out.append({"ref": "n2", "bbox": [10, 80, 90, 50], "note": "struck through twice"})
    d.low_conf_text.append({"ref": "n2", "confidence": 0.3, "alternatives": ["do it", "clo it"]})
    assert d.problems() == []
    assert Diagram.load(d.save(tmp_path / "a.ir.json")).crossed_out == d.crossed_out


def test_ambiguity_entries_are_schema_checked():
    d = tiny()
    d.unresolved_edges.append({"edge": "e1", "reason": "it looked odd"})
    assert d.problems()

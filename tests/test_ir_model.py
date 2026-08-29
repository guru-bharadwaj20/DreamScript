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

"""Phase 2.2.1 - the Label Studio bridge.

The live end-to-end proof lives in `scripts/labelstudio_verify.py`, which needs the separate
`.venv-labelstudio` interpreter and a running server. These tests cover what can be checked
without either: that the config and the vocabularies agree, and that IR survives a trip
through Label Studio's data model.
"""

from __future__ import annotations

import pytest

from src.ir import labelstudio
from src.ir.model import Diagram, Edge, Node
from src.ir.vocab import ROLES, SHAPES


def sample() -> Diagram:
    return Diagram(
        id="demo",
        diagram_type="flowchart",
        nodes=[
            Node("n1", "circle", [10, 10, 40, 40], "start", "start"),
            Node("n2", "diamond", [10, 100, 80, 60], "ok?", "decision"),
            Node("n3", "rectangle", [200, 100, 90, 50], "ship it", "process"),
        ],
        edges=[
            Edge("e1", "n1", "n2", True, "", None),
            Edge("e2", "n2", "n3", True, "yes", None),
        ],
        meta={
            "source": "test",
            "geometry": "annotated",
            "image": "a/b.jpg",
            "image_size": [400, 300],
        },
    )


def test_config_matches_the_frozen_vocabularies():
    assert labelstudio.check() == []


def test_config_lists_every_shape_and_role():
    vocab = labelstudio.config_vocabularies()
    assert vocab["shapes"] == list(SHAPES)
    assert vocab["role"] == list(ROLES)


def test_config_offers_a_way_to_express_doubt():
    """An annotator with no way to say 'I am not sure' will guess, and the guess is invisible."""
    flags = labelstudio.config_vocabularies()["flags"]
    assert {"crossed-out", "text-uncertain", "shape-uncertain"} <= set(flags)
    assert "uncertain" in labelstudio.config_vocabularies()["relations"]


def test_task_carries_the_image_and_a_prediction():
    t = labelstudio.task(sample())
    assert t["data"]["image"] == "/data/local-files/?d=a/b.jpg"
    assert len(t["predictions"]) == 1
    assert t["predictions"][0]["result"]


def test_prediction_geometry_is_percentages():
    result = labelstudio.prediction(sample())["result"]
    boxes = [r for r in result if r["type"] == "rectanglelabels"]
    assert len(boxes) == 3
    first = boxes[0]["value"]
    assert first["x"] == pytest.approx(2.5)  # 10 / 400
    assert first["height"] == pytest.approx(40 / 300 * 100)


def test_prediction_relations_carry_the_branch_label():
    result = labelstudio.prediction(sample())["result"]
    names = [r["labels"][0] for r in result if r["type"] == "relation"]
    assert names == ["flow", "flow-yes"]


def test_prediction_refuses_without_an_image_size():
    d = sample()
    d.meta.pop("image_size")
    with pytest.raises(ValueError, match="image_size"):
        labelstudio.prediction(d)


def test_ir_survives_the_round_trip_through_label_studio():
    """IR -> pre-annotation -> IR. Ids and geometry change representation; shapes, roles, text
    and connectivity must not."""
    original = sample()
    annotation = labelstudio.prediction(original)
    back = labelstudio.to_ir(
        annotation, diagram_id=original.id, diagram_type=original.diagram_type, image="a/b.jpg"
    )
    assert back.problems() == []
    assert [n.shape for n in back.nodes] == [n.shape for n in original.nodes]
    assert [n.semantic_role for n in back.nodes] == [n.semantic_role for n in original.nodes]
    assert [n.text for n in back.nodes] == [n.text for n in original.nodes]
    for got, want in zip(back.nodes, original.nodes, strict=True):
        assert got.bbox == pytest.approx(want.bbox)
    edges = {(e.src, e.dst, e.label) for e in back.edges}
    assert edges == {("r_n1", "r_n2", ""), ("r_n2", "r_n3", "yes")}


def test_annotator_flags_become_ambiguity_records():
    result = labelstudio.prediction(sample())["result"]
    result.append(
        {
            "id": "r_n2",
            "type": "choices",
            "from_name": "flags",
            "to_name": "image",
            "original_width": 400,
            "original_height": 300,
            "value": {"choices": ["crossed-out", "text-uncertain"]},
        }
    )
    back = labelstudio.to_ir({"result": result}, diagram_id="demo", diagram_type="flowchart")
    assert [c["ref"] for c in back.crossed_out] == ["r_n2"]
    assert [t["ref"] for t in back.low_conf_text] == ["r_n2"]
    assert back.problems() == []


def test_uncertain_relation_lands_in_unresolved_edges():
    result = [
        {"type": "relation", "from_id": "r_a", "to_id": "r_b", "labels": ["uncertain"]},
        {
            "id": "r_a",
            "type": "rectanglelabels",
            "from_name": "shape",
            "to_name": "image",
            "original_width": 100,
            "original_height": 100,
            "value": {"x": 0, "y": 0, "width": 10, "height": 10, "rectanglelabels": ["circle"]},
        },
        {
            "id": "r_b",
            "type": "rectanglelabels",
            "from_name": "shape",
            "to_name": "image",
            "original_width": 100,
            "original_height": 100,
            "value": {"x": 50, "y": 0, "width": 10, "height": 10, "rectanglelabels": ["circle"]},
        },
    ]
    back = labelstudio.to_ir({"result": result}, diagram_id="d", diagram_type="flowchart")
    assert len(back.unresolved_edges) == 1
    assert back.unresolved_edges[0]["reason"] == "ambiguous-endpoint"
    assert back.edges[0].confidence < 1.0

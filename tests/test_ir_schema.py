"""Phase 2.1 - the IR schemas are a contract; these tests are what makes it one."""

from __future__ import annotations

import pytest

from src.ir import schema


def node(**over):
    base = {
        "id": "n1",
        "shape": "rectangle",
        "bbox": [10, 20, 100, 40],
        "text": "do the thing",
        "semantic_role": "process",
        "confidence": 1.0,
    }
    base.update(over)
    return base


def test_schemas_are_themselves_valid():
    assert schema.main([]) == 0


def test_minimal_node_validates():
    assert schema.validate_node(node()) == []


@pytest.mark.parametrize(
    "field",
    ["id", "shape", "bbox", "text", "semantic_role", "confidence"],
)
def test_every_field_is_required(field):
    bad = node()
    del bad[field]
    assert schema.validate_node(bad), f"{field} should be required"


def test_bbox_may_be_null_but_not_short():
    assert schema.validate_node(node(bbox=None)) == []
    assert schema.validate_node(node(bbox=[1, 2, 3]))


def test_confidence_is_bounded():
    assert schema.validate_node(node(confidence=1.2))
    assert schema.validate_node(node(confidence=-0.1))


def test_text_may_be_empty_but_not_null():
    assert schema.validate_node(node(text="")) == []
    assert schema.validate_node(node(text=None))


def test_unknown_fields_are_rejected():
    assert schema.validate_node(node(colour="blue"))


def edge(**over):
    base = {
        "id": "e1",
        "src": "n1",
        "dst": "n2",
        "directed": True,
        "label": "yes",
        "polyline": [[10, 10], [50, 10], [50, 60]],
        "confidence": 0.9,
    }
    base.update(over)
    return base


def test_minimal_edge_validates():
    assert schema.problems("edge", edge()) == []


@pytest.mark.parametrize(
    "field", ["id", "src", "dst", "directed", "label", "polyline", "confidence"]
)
def test_every_edge_field_is_required(field):
    bad = edge()
    del bad[field]
    assert schema.problems("edge", bad), f"{field} should be required"


def test_dangling_endpoints_are_representable():
    """A drawn arrow whose owner is unknown is a fact about the diagram, not a schema error."""
    assert schema.problems("edge", edge(src=None)) == []
    assert schema.problems("edge", edge(dst=None)) == []


def test_polyline_needs_two_points_and_pairs():
    assert schema.problems("edge", edge(polyline=[[1, 2]]))
    assert schema.problems("edge", edge(polyline=[[1, 2, 3], [4, 5, 6]]))
    assert schema.problems("edge", edge(polyline=None)) == []


def test_directed_is_not_coerced_from_a_string():
    assert schema.problems("edge", edge(directed="true"))

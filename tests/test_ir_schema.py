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

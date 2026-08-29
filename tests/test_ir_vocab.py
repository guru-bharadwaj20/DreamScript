"""Phase 2.1.4 / 2.1.5 - a vocabulary is only frozen if something breaks when it thaws."""

from __future__ import annotations

import json

import pytest

from src.ir import schema, vocab


def test_shape_schema_matches_the_code():
    """The schema is generated; this test fails if someone edits one and not the other."""
    on_disk = json.loads(schema.schema_path("shape").read_text(encoding="utf-8"))
    assert on_disk["enum"] == list(vocab.SHAPES)


def test_shape_vocabulary_covers_the_plan():
    """plan.md 2.1.4 names these ten; they must all survive any later edit."""
    planned = {
        "rectangle",
        "rounded-rect",
        "diamond",
        "ellipse",
        "circle",
        "parallelogram",
        "arrow",
        "line",
        "text-block",
        "freeform",
    }
    assert planned <= set(vocab.SHAPES)


def test_no_duplicate_or_uppercase_shapes():
    assert len(set(vocab.SHAPES)) == len(vocab.SHAPES)
    assert all(s == s.lower() for s in vocab.SHAPES)


def test_every_shape_has_a_geometric_description():
    assert set(vocab.SHAPE_GEOMETRY) == set(vocab.SHAPES)


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("oval", "ellipse"),
        ("Oval", "ellipse"),
        ("box", "rectangle"),
        ("doublecircle", "double-circle"),
        ("double_circle", "double-circle"),
        ("rounded rect", "rounded-rect"),
        ("rectangle", "rectangle"),
        ("hexagram", "freeform"),
        ("", "freeform"),
    ],
)
def test_canonical_shape(raw, want):
    assert vocab.canonical_shape(raw) == want


def test_every_alias_target_is_in_the_vocabulary():
    assert set(vocab.SHAPE_ALIASES.values()) <= set(vocab.SHAPES)


def test_node_schema_rejects_a_shape_outside_the_vocabulary():
    node = {
        "id": "n1",
        "shape": "trapezium",
        "bbox": None,
        "text": "",
        "semantic_role": "process",
        "confidence": 1.0,
    }
    assert schema.validate_node(node)
    node["shape"] = "parallelogram"
    assert schema.validate_node(node) == []

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
    """contributing.md 2.1.4 names these ten; they must all survive any later edit."""
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


def test_role_schema_matches_the_code():
    on_disk = json.loads(schema.schema_path("role").read_text(encoding="utf-8"))
    assert on_disk["enum"] == list(vocab.ROLES)


def test_role_vocabulary_covers_the_plan():
    planned = {
        "start",
        "end",
        "process",
        "decision",
        "io",
        "state",
        "transition",
        "entity",
        "attribute",
        "relationship",
        "container",
        "ui-input",
        "ui-button",
        "ui-label",
        "ui-image",
        "component",
        "wire",
    }
    assert planned <= set(vocab.ROLES)


def test_every_role_is_reachable_from_some_diagram_type():
    reachable = set().union(*vocab.ROLES_BY_TYPE.values())
    assert set(vocab.ROLES) <= reachable


def test_role_type_map_covers_every_diagram_type():
    from src.ir import model  # noqa: F401  (import kept local to the assertion it supports)

    declared = set(
        json.loads(schema.schema_path("ir").read_text(encoding="utf-8"))["properties"][
            "diagram_type"
        ]["enum"]
    )
    assert declared == set(vocab.ROLES_BY_TYPE)


def test_every_role_has_a_shape_prior():
    assert set(vocab.SHAPE_ROLE_PRIOR) == set(vocab.ROLES)
    for role, shapes in vocab.SHAPE_ROLE_PRIOR.items():
        assert set(shapes) <= set(vocab.SHAPES), role


def test_every_role_alias_target_is_in_the_vocabulary():
    assert set(vocab.ROLE_ALIASES.values()) <= set(vocab.ROLES)


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("task", "process"),
        ("startEvent", "start"),
        ("exclusiveGateway", "decision"),
        ("parallelGateway", "fork"),
        ("final state", "final-state"),
        ("ui-button", "ui-button"),
        ("button", "ui-button"),
        ("wire", "wire"),
        ("banana", "unknown"),
        ("", "unknown"),
    ],
)
def test_canonical_role(raw, want):
    assert vocab.canonical_role(raw) == want


def test_edge_like_roles_are_real_roles():
    assert set(vocab.ROLES) >= vocab.EDGE_LIKE_ROLES


def test_node_schema_rejects_a_role_outside_the_vocabulary():
    node = {
        "id": "n1",
        "shape": "rectangle",
        "bbox": None,
        "text": "",
        "semantic_role": "subroutine",
        "confidence": 1.0,
    }
    assert schema.validate_node(node)

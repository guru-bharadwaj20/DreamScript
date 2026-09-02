"""Phase 7.3.1 - the frozen state space, the role mapping, and the derived states."""

from __future__ import annotations

import pytest

from src.parse import roles


@pytest.fixture
def diamond_branch():
    """A start, a decision, two branches and an end - the smallest diagram with every rule in it."""
    return {
        "id": "toy",
        "diagram_type": "flowchart",
        "nodes": [
            {
                "id": "a",
                "shape": "ellipse",
                "bbox": [0, 0, 10, 10],
                "text": "start",
                "semantic_role": "start",
            },
            {
                "id": "b",
                "shape": "diamond",
                "bbox": [0, 20, 10, 10],
                "text": "ok?",
                "semantic_role": "decision",
            },
            {
                "id": "c",
                "shape": "rectangle",
                "bbox": [0, 40, 10, 10],
                "text": "yes path",
                "semantic_role": "process",
            },
            {
                "id": "d",
                "shape": "rectangle",
                "bbox": [20, 40, 10, 10],
                "text": "no path",
                "semantic_role": "process",
            },
            {
                "id": "e",
                "shape": "ellipse",
                "bbox": [0, 60, 10, 10],
                "text": "end",
                "semantic_role": "end",
            },
        ],
        "edges": [
            {"id": "e0", "src": "a", "dst": "b"},
            {"id": "e1", "src": "b", "dst": "c", "label": "yes"},
            {"id": "e2", "src": "b", "dst": "d", "label": "no"},
            {"id": "e3", "src": "c", "dst": "e"},
        ],
    }


# -- the frozen space -----------------------------------------------------------------------------


def test_the_state_space_is_the_nine_the_plan_names():
    assert roles.STATES == (
        "start",
        "input",
        "process",
        "decision",
        "branch-true",
        "branch-false",
        "loop-back",
        "output",
        "terminal",
    )


def test_the_index_agrees_with_the_tuple():
    assert all(roles.STATES[i] == name for name, i in roles.STATE_INDEX.items())


def test_every_mapped_role_lands_in_the_space():
    assert set(roles.ROLE_TO_STATE.values()) <= set(roles.STATES)


def test_io_is_not_mapped_by_role_alone():
    """Direction is a property of the graph; a role table cannot decide it."""
    assert "io" not in roles.ROLE_TO_STATE


def test_the_derived_states_are_exactly_the_ones_no_annotation_carries():
    assert set(roles.DERIVED_STATES) == set(roles.STATES) - set(roles.ROLE_TO_STATE.values())


def test_the_judgement_calls_are_recorded_for_the_roles_they_are_about():
    assert set(roles.JUDGEMENT_CALLS) <= set(roles.ROLE_TO_STATE)


# -- branch labels --------------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["yes", "YES", " True ", "1", "ok"])
def test_true_labels_are_recognised_however_they_are_written(label):
    assert roles.branch_of_label(label) == "branch-true"


@pytest.mark.parametrize("label", ["no", "FALSE", "0", "reject"])
def test_false_labels_are_recognised(label):
    assert roles.branch_of_label(label) == "branch-false"


def test_a_label_that_names_nothing_returns_none():
    """hdbpmn labels its edges with message names; those must fall through to the ordering rule."""
    assert roles.branch_of_label("claim") is None
    assert roles.branch_of_label("") is None


# -- the derivation -------------------------------------------------------------------------------


def test_labels_decide_the_branches_when_they_are_present(diamond_branch):
    order = ["a", "b", "c", "e", "d"]
    states = roles.derive_states(diamond_branch, order, set())
    assert states["c"] == "branch-true"
    assert states["d"] == "branch-false"


def test_the_ordering_decides_when_the_labels_do_not(diamond_branch):
    for edge in diamond_branch["edges"]:
        edge["label"] = "message"
    order = ["a", "b", "c", "e", "d"]
    states = roles.derive_states(diamond_branch, order, set())
    assert states["c"] == "branch-true"
    assert states["d"] == "branch-false"


def test_a_back_edge_target_becomes_a_loop_back_whatever_else_it_is(diamond_branch):
    """Precedence rule 1: the flow returning here outranks the node's own annotation."""
    states = roles.derive_states(diamond_branch, ["a", "b", "c", "e", "d"], {("e", "b")})
    assert states["b"] == "loop-back"


def test_an_annotated_role_survives_when_no_rule_fires(diamond_branch):
    states = roles.derive_states(diamond_branch, ["a", "b", "c", "e", "d"], set())
    assert states["a"] == "start"
    assert states["e"] == "terminal"


def test_a_source_io_is_input_and_a_sink_io_is_output():
    diagram = {
        "nodes": [
            {"id": "i", "shape": "parallelogram", "bbox": [0, 0, 1, 1], "semantic_role": "io"},
            {"id": "p", "shape": "rectangle", "bbox": [0, 1, 1, 1], "semantic_role": "process"},
            {"id": "o", "shape": "parallelogram", "bbox": [0, 2, 1, 1], "semantic_role": "io"},
        ],
        "edges": [{"id": "e0", "src": "i", "dst": "p"}, {"id": "e1", "src": "p", "dst": "o"}],
    }
    states = roles.derive_states(diagram, ["i", "p", "o"], set())
    assert states["i"] == "input"
    assert states["o"] == "output"


def test_provenance_separates_annotation_from_rule(diamond_branch):
    source = roles.provenance(diamond_branch, ["a", "b", "c", "e", "d"], set())
    assert source["a"] == "annotation"
    assert source["c"] == "label"


def test_provenance_covers_every_node(diamond_branch):
    order = ["a", "b", "c", "e", "d"]
    assert set(roles.provenance(diamond_branch, order, set())) == set(
        roles.derive_states(diamond_branch, order, set())
    )


# -- what counts as a sequence --------------------------------------------------------------------


def test_a_wireframe_is_not_a_sequence():
    assert not roles.is_sequential(
        {"nodes": [{"id": "n", "semantic_role": "ui-button"}], "edges": [{"id": "e"}]}
    )


def test_a_diagram_with_no_edges_is_not_a_sequence(diamond_branch):
    diamond_branch["edges"] = []
    assert not roles.is_sequential(diamond_branch)


def test_a_flowchart_with_edges_is(diamond_branch):
    assert roles.is_sequential(diamond_branch)


def test_a_dangling_edge_still_counts_for_the_end_it_has(diamond_branch):
    """12% of the corpus's edges are dangling; calling those nodes sinks would be a lie."""
    diamond_branch["edges"].append({"id": "x", "src": "c", "dst": None})
    incoming, outgoing = roles.degrees(diamond_branch)
    assert outgoing["c"] == 2
    assert incoming["e"] == 1

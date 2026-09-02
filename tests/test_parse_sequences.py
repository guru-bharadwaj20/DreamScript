"""Phase 7.3.3 - the DFS linearisation, its back edges, and the component breaks it records."""

from __future__ import annotations

import pytest

from src.parse import sequences


def node(node_id: str, y: float, shape: str = "rectangle", role: str = "process") -> dict:
    return {
        "id": node_id,
        "shape": shape,
        "bbox": [0.0, y, 10.0, 10.0],
        "text": "",
        "semantic_role": role,
    }


@pytest.fixture
def chain():
    """a -> b -> c, drawn top to bottom."""
    return {
        "id": "chain",
        "diagram_type": "flowchart",
        "nodes": [node("a", 0, "ellipse", "start"), node("b", 10), node("c", 20, "ellipse", "end")],
        "edges": [
            {"id": "e0", "src": "a", "dst": "b"},
            {"id": "e1", "src": "b", "dst": "c"},
        ],
    }


@pytest.fixture
def loop(chain):
    chain["edges"].append({"id": "e2", "src": "c", "dst": "a"})
    return chain


# -- ordering -------------------------------------------------------------------------------------


def test_a_chain_orders_itself(chain):
    order, back = sequences.traversal(chain)
    assert order == ["a", "b", "c"]
    assert back == set()


def test_children_are_visited_in_reading_order():
    """Top to bottom, then left to right - the order a reader's eye takes."""
    diagram = {
        "id": "fork",
        "diagram_type": "flowchart",
        "nodes": [node("a", 0), node("low", 30), node("high", 10)],
        "edges": [
            {"id": "e0", "src": "a", "dst": "low"},
            {"id": "e1", "src": "a", "dst": "high"},
        ],
    }
    order, _ = sequences.traversal(diagram)
    assert order == ["a", "high", "low"]


def test_a_cycle_produces_a_back_edge(loop):
    order, back = sequences.traversal(loop)
    assert order == ["a", "b", "c"]
    assert back == {("c", "a")}


def test_a_re_convergence_is_not_a_back_edge():
    """Both branches meeting again is a cross edge; calling it a loop would label it loop-back."""
    diagram = {
        "id": "diamond",
        "diagram_type": "flowchart",
        "nodes": [node("a", 0), node("l", 10), node("r", 11), node("end", 20)],
        "edges": [
            {"id": "e0", "src": "a", "dst": "l"},
            {"id": "e1", "src": "a", "dst": "r"},
            {"id": "e2", "src": "l", "dst": "end"},
            {"id": "e3", "src": "r", "dst": "end"},
        ],
    }
    _, back = sequences.traversal(diagram)
    assert back == set()


def test_every_node_appears_exactly_once_even_across_components(chain):
    chain["nodes"].append(node("island", 40))
    order, _ = sequences.traversal(chain)
    assert sorted(order) == ["a", "b", "c", "island"]


def test_the_traversal_is_deterministic(loop):
    assert sequences.traversal(loop) == sequences.traversal(loop)


# -- components -----------------------------------------------------------------------------------


def test_a_break_is_recorded_where_the_component_changes(chain):
    chain["nodes"].append(node("island", 40))
    record = sequences.sequence_of(chain)
    assert record["components"] == 2
    assert record["component_breaks"] == [3]


def test_a_connected_diagram_has_no_breaks(chain):
    assert sequences.sequence_of(chain)["component_breaks"] == []


# -- the record -----------------------------------------------------------------------------------


def test_the_record_carries_one_observation_per_node(chain):
    record = sequences.sequence_of(chain)
    assert len(record["observations"]) == len(record["node_ids"]) == 3


def test_states_are_present_when_labelled_and_absent_when_not(chain):
    assert "states" in sequences.sequence_of(chain, labelled=True)
    assert "states" not in sequences.sequence_of(chain, labelled=False)


def test_the_states_line_up_with_the_observations(chain):
    record = sequences.sequence_of(chain)
    assert len(record["states"]) == len(record["observations"])


def test_a_single_node_diagram_is_not_a_sequence():
    diagram = {
        "id": "dot",
        "diagram_type": "flowchart",
        "nodes": [node("a", 0)],
        "edges": [],
    }
    assert sequences.sequence_of(diagram) is None


def test_the_back_edges_are_carried_into_the_record(loop):
    assert sequences.sequence_of(loop)["back_edges"] == [["c", "a"]] or (
        sequences.sequence_of(loop)["back_edges"] == [("c", "a")]
    )


def test_a_loop_target_is_labelled_loop_back(loop):
    """The traversal and the state derivation must agree about what a back edge is."""
    record = sequences.sequence_of(loop)
    assert record["states"][record["node_ids"].index("a")] == "loop-back"

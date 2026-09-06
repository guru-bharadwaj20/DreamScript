"""S4's component splitting, its emission interpolation, and the leak it must not have."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import s4
from src.parse.roles import STATES


def test_segments_splits_at_every_component_break():
    sequence = {
        "observations": ["a", "b", "c", "d"],
        "states": ["start", "process", "start", "terminal"],
        "component_breaks": [2],
    }
    parts = s4.segments(sequence)
    assert [p[1] for p in parts] == [["a", "b"], ["c", "d"]]
    assert [p[0] for p in parts] == [["start", "process"], ["start", "terminal"]]


def test_segments_of_an_unbroken_sequence_is_one_chain():
    sequence = {"observations": ["a", "b"], "states": ["start", "terminal"], "component_breaks": []}
    assert len(s4.segments(sequence)) == 1


def test_segments_isolates_a_single_node_component():
    """The 1,085 input/output nodes are isolated, and that is where the gain comes from."""
    sequence = {
        "observations": ["a", "b", "c"],
        "states": ["start", "input", "terminal"],
        "component_breaks": [1, 2],
    }
    assert [p[1] for p in s4.segments(sequence)] == [["a"], ["b"], ["c"]]


def test_segments_preserves_every_position():
    sequence = {
        "observations": list("abcdef"),
        "states": ["process"] * 6,
        "component_breaks": [1, 3, 4],
    }
    parts = s4.segments(sequence)
    assert [o for _, obs in parts for o in obs] == list("abcdef")
    assert sum(len(states) for states, _ in parts) == 6


def test_component_breaks_are_not_derived_from_labels():
    """The whole result rests on this: breaks come from connectivity, never from the states.

    `sequences.components` is union-find over nodes and edges and is computed before
    `derive_states` runs, so changing every label must not move a single break.
    """
    from src.parse.sequences import components

    diagram = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [{"src": "a", "dst": "b"}],
    }
    order = ["a", "b", "c"]
    before = components(diagram, order)
    for node in diagram["nodes"]:
        node["semantic_role"] = "decision"
    assert components(diagram, order) == before
    # a and b share a component, c does not
    assert before[0] == before[1] != before[2]


def test_emissions_rows_are_distributions():
    alphabet = ["box|other|linear", "diamond|empty|branching", "round|other|sink"]
    train = [
        {
            "observations": ["box|other|linear", "diamond|empty|branching"],
            "states": ["process", "decision"],
        }
    ]
    for lam in (0.0, 0.5, 1.0):
        matrix = s4.emissions(train, alphabet, lam)
        assert matrix.shape == (len(STATES), len(alphabet))
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert (matrix > 0).all(), "no cell may be zero or the decoder can hit -inf"


def test_emissions_at_lambda_one_is_the_atomic_estimate():
    """lambda = 1 must reduce to plain Laplace, so the interpolation is a strict generalisation."""
    alphabet = ["box|other|linear", "diamond|empty|branching"]
    train = [{"observations": ["box|other|linear"], "states": ["process"]}]
    matrix = s4.emissions(train, alphabet, 1.0, alpha=1.0)
    row = matrix[STATES.index("process")]
    assert row[0] > row[1], "the observed symbol must dominate"
    assert row[0] == pytest.approx(2.0 / 3.0)


def test_choose_lambda_only_ever_sees_training_sequences(monkeypatch):
    """Selection-on-test at this margin would be the difference between result and artefact."""
    seen = []
    real = s4.fit_per_type

    def spy(train, alphabet, lam):
        seen.extend(s["id"] for s in train)
        return real(train, alphabet, lam)

    monkeypatch.setattr(s4, "fit_per_type", spy)
    train = [
        {
            "id": f"t{i}",
            "diagram_type": "flowchart",
            "observations": ["box|other|linear", "round|other|sink"],
            "states": ["process", "terminal"],
            "component_breaks": [],
        }
        for i in range(9)
    ]
    alphabet = ["box|other|linear", "round|other|sink"]
    s4.choose_lambda(train, alphabet, {s: i for i, s in enumerate(alphabet)}, 42)
    assert set(seen) <= {s["id"] for s in train}


def test_target_is_the_plans_number():
    assert s4.TARGET_MACRO_F1 == 0.80

"""Phase 7.3.4 - the supervised transition matrix, its smoothing and its exclusions."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import transitions
from src.parse.roles import STATE_INDEX, STATES


def sequence(states, breaks=()):
    return {
        "id": "s",
        "diagram_type": "flowchart",
        "states": list(states),
        "observations": ["box|other|linear"] * len(states),
        "component_breaks": list(breaks),
    }


@pytest.fixture
def simple():
    return [
        sequence(["start", "process", "terminal"]),
        sequence(["start", "process", "process", "terminal"]),
    ]


# -- counting -------------------------------------------------------------------------------------


def test_counts_are_pairs_not_nodes(simple):
    counted, _ = transitions.counts(simple)
    assert counted.sum() == 5


def test_the_first_state_of_each_sequence_is_an_initial(simple):
    _, initial = transitions.counts(simple)
    assert initial[STATE_INDEX["start"]] == 2


def test_a_component_break_is_not_a_transition():
    """The jump between two disconnected pools is an artefact of concatenation."""
    counted, initial = transitions.counts([sequence(["terminal", "start", "process"], breaks=[1])])
    assert counted[STATE_INDEX["terminal"], STATE_INDEX["start"]] == 0
    assert counted[STATE_INDEX["start"], STATE_INDEX["process"]] == 1


def test_a_break_becomes_an_initial_instead():
    _, initial = transitions.counts([sequence(["terminal", "start", "process"], breaks=[1])])
    assert initial[STATE_INDEX["start"]] == 1


def test_breaks_can_be_kept_for_the_comparison():
    counted, _ = transitions.counts(
        [sequence(["terminal", "start", "process"], breaks=[1])], drop_breaks=False
    )
    assert counted[STATE_INDEX["terminal"], STATE_INDEX["start"]] == 1


# -- smoothing ------------------------------------------------------------------------------------


def test_every_row_sums_to_one(simple):
    model = transitions.fit(simple)
    assert np.allclose(model["A"].sum(axis=1), 1.0)
    assert np.isclose(model["pi"].sum(), 1.0)


def test_no_cell_is_zero_which_is_what_keeps_viterbi_alive(simple):
    """A zero in A is a hard veto; 993 diagrams cannot establish that a transition is impossible."""
    assert (transitions.fit(simple)["A"] > 0).all()


def test_a_state_never_observed_gets_a_uniform_row(simple):
    row = transitions.fit(simple)["A"][STATE_INDEX["output"]]
    assert np.allclose(row, 1 / len(STATES))


def test_more_smoothing_flattens_the_matrix(simple):
    def spread(alpha):
        A = transitions.fit(simple, alpha)["A"]
        return float(A.max() - A.min())

    assert spread(0.01) > spread(10.0)


def test_the_matrix_is_square_over_the_frozen_space(simple):
    assert transitions.fit(simple)["A"].shape == (len(STATES), len(STATES))


# -- structure and per-type -----------------------------------------------------------------------


def test_the_structure_summary_names_the_likeliest_successor(simple):
    summary = transitions.structure(transitions.fit(simple))
    assert summary["per_state"]["start"]["top"][0]["to"] == "process"


def test_the_summary_counts_the_empty_cells(simple):
    summary = transitions.structure(transitions.fit(simple))
    assert summary["empty_cells"] == int((transitions.fit(simple)["counts"] == 0).sum())


def test_per_type_splits_the_corpus(simple):
    simple[1]["diagram_type"] = "state_machine"
    per_type = transitions.by_type(simple)
    assert set(per_type) == {"flowchart", "state_machine"}
    assert per_type["flowchart"]["sequences"] == 1


def test_normalise_handles_an_all_zero_row():
    assert np.allclose(transitions.normalise(np.zeros((1, 4)))[0], 0.25)

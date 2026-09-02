"""Phase 7.3.5 - the emission matrix and the memoryless baseline it defines."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import emissions
from src.parse.roles import STATE_INDEX, STATES


def sequence(pairs):
    states, symbols = zip(*pairs, strict=True)
    return {
        "id": "s",
        "diagram_type": "flowchart",
        "states": list(states),
        "observations": list(symbols),
        "component_breaks": [],
    }


@pytest.fixture
def simple():
    return [
        sequence(
            [
                ("start", "round|other|source"),
                ("process", "box|other|linear"),
                ("terminal", "round|other|sink"),
            ]
        ),
        sequence(
            [
                ("start", "round|other|source"),
                ("decision", "diamond|empty|branching"),
                ("process", "box|other|linear"),
            ]
        ),
    ]


def test_the_alphabet_is_what_the_corpus_used(simple):
    model = emissions.fit(simple)
    assert model["alphabet"] == sorted({s for seq in simple for s in seq["observations"]})


def test_the_matrix_is_states_by_symbols(simple):
    model = emissions.fit(simple)
    assert model["B"].shape == (len(STATES), len(model["alphabet"]))


def test_every_row_is_a_distribution(simple):
    assert np.allclose(emissions.fit(simple)["B"].sum(axis=1), 1.0)


def test_no_cell_is_zero(simple):
    assert (emissions.fit(simple)["B"] > 0).all()


def test_counts_land_where_they_were_observed(simple):
    model = emissions.fit(simple)
    column = model["alphabet"].index("diamond|empty|branching")
    assert model["counts"][STATE_INDEX["decision"], column] == 1


def test_a_symbol_outside_the_alphabet_is_ignored_rather_than_crashing(simple):
    model = emissions.fit(simple, alphabet=["box|other|linear"])
    assert model["counts"].sum() == 2


def test_the_symbol_only_decoder_reports_its_own_accuracy(simple):
    """Two symbols map to one state each here, so the memoryless rule is perfect."""
    result = emissions.symbol_only_decoder(emissions.fit(simple))
    assert result["accuracy"] == 1.0


def test_the_symbol_only_decoder_counts_the_states_it_can_reach(simple):
    result = emissions.symbol_only_decoder(emissions.fit(simple))
    assert result["states_ever_chosen"] <= len(STATES)


def test_an_ambiguous_symbol_costs_the_memoryless_decoder():
    shared = [
        sequence([("branch-true", "box|other|linear"), ("branch-false", "box|other|linear")]),
    ]
    assert emissions.symbol_only_decoder(emissions.fit(shared))["accuracy"] == 0.5


def test_the_structure_summary_names_the_likeliest_symbol(simple):
    summary = emissions.structure(emissions.fit(simple))
    assert summary["decision"]["top"][0]["symbol"] == "diamond|empty|branching"


def test_a_state_with_no_rows_has_uniform_emissions(simple):
    model = emissions.fit(simple)
    row = model["B"][STATE_INDEX["output"]]
    assert np.allclose(row, row[0])


def test_more_smoothing_flattens_the_rows(simple):
    def spread(alpha):
        B = emissions.fit(simple, alpha=alpha)["B"]
        return float(B.max() - B.min())

    assert spread(0.01) > spread(10.0)

"""Phase 7.3.7 - the hand-written decoder, checked against hmmlearn and against known paths."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import viterbi
from src.parse.roles import STATES


@pytest.fixture
def two_state():
    """A model with an obvious answer: state 0 emits symbol 0, state 1 emits symbol 1."""
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.array([[0.95, 0.05], [0.05, 0.95]])
    return {"pi": pi, "A": A, "B": B}


@pytest.fixture
def sticky():
    """Transitions so sticky that one odd observation should not break the run."""
    pi = np.array([0.5, 0.5])
    A = np.array([[0.999, 0.001], [0.001, 0.999]])
    B = np.array([[0.7, 0.3], [0.3, 0.7]])
    return {"pi": pi, "A": A, "B": B}


# -- the recursion --------------------------------------------------------------------------------


def test_the_obvious_path_is_recovered(two_state):
    path = viterbi.decode([0, 0, 1, 1], *viterbi.logs(two_state))
    assert path == [0, 0, 1, 1]


def test_an_empty_sequence_decodes_to_an_empty_path(two_state):
    assert viterbi.decode([], *viterbi.logs(two_state)) == []


def test_a_single_observation_uses_pi_and_b_only(two_state):
    assert viterbi.decode([1], *viterbi.logs(two_state)) == [1]


def test_the_path_is_as_long_as_the_sequence(two_state):
    assert len(viterbi.decode([0, 1, 0, 1, 0], *viterbi.logs(two_state))) == 5


def test_a_sticky_transition_overrules_one_odd_observation(sticky):
    """This is the whole point of a path decoder: local evidence loses to global structure."""
    assert viterbi.decode([0, 0, 1, 0, 0], *viterbi.logs(sticky)) == [0, 0, 0, 0, 0]


def test_the_decoder_survives_a_long_sequence_without_underflowing(sticky):
    """In plain probabilities this underflows; in logs it does not."""
    path = viterbi.decode([0] * 400, *viterbi.logs(sticky))
    assert path == [0] * 400


# -- against the reference implementation ---------------------------------------------------------


def path_log_probability(path, symbols, model) -> float:
    log_pi, log_A, log_B = viterbi.logs(model)
    total = log_pi[path[0]] + log_B[path[0], symbols[0]]
    for t in range(1, len(symbols)):
        total += log_A[path[t - 1], path[t]] + log_B[path[t], symbols[t]]
    return float(total)


def test_it_scores_the_same_as_hmmlearn_on_a_symmetric_model(two_state):
    """A symmetric model has exact ties, so the paths may differ - the score may not."""
    hmm = pytest.importorskip("hmmlearn.hmm")

    machine = hmm.CategoricalHMM(n_components=2, init_params="")
    machine.startprob_ = two_state["pi"]
    machine.transmat_ = two_state["A"]
    machine.emissionprob_ = two_state["B"]
    symbols = [0, 1, 1, 0, 1, 0, 0]
    _, reference = machine.decode(np.array(symbols).reshape(-1, 1), algorithm="viterbi")
    ours = viterbi.decode(symbols, *viterbi.logs(two_state))
    assert path_log_probability(ours, symbols, two_state) == pytest.approx(
        path_log_probability(list(reference), symbols, two_state)
    )


def test_it_agrees_with_hmmlearn_when_there_are_no_ties():
    hmm = pytest.importorskip("hmmlearn.hmm")

    model = {
        "pi": np.array([0.6, 0.4]),
        "A": np.array([[0.85, 0.15], [0.3, 0.7]]),
        "B": np.array([[0.9, 0.1], [0.25, 0.75]]),
    }
    machine = hmm.CategoricalHMM(n_components=2, init_params="")
    machine.startprob_, machine.transmat_, machine.emissionprob_ = (
        model["pi"],
        model["A"],
        model["B"],
    )
    symbols = [0, 1, 1, 0, 1, 0, 0]
    _, reference = machine.decode(np.array(symbols).reshape(-1, 1), algorithm="viterbi")
    assert viterbi.decode(symbols, *viterbi.logs(model)) == list(reference)


def test_it_agrees_with_hmmlearn_on_a_random_nine_state_model():
    hmm = pytest.importorskip("hmmlearn.hmm")

    rng = np.random.default_rng(7)

    def rows(shape):
        values = rng.random(shape) + 0.05
        return values / values.sum(axis=-1, keepdims=True)

    model = {"pi": rows(9), "A": rows((9, 9)), "B": rows((9, 20))}
    machine = hmm.CategoricalHMM(n_components=9, init_params="")
    machine.startprob_, machine.transmat_, machine.emissionprob_ = (
        model["pi"],
        model["A"],
        model["B"],
    )
    symbols = rng.integers(0, 20, size=50).tolist()
    _, reference = machine.decode(np.array(symbols).reshape(-1, 1), algorithm="viterbi")
    assert viterbi.decode(symbols, *viterbi.logs(model)) == list(reference)


# -- the harness ----------------------------------------------------------------------------------


def test_the_folds_partition_the_sequences():
    data = [{"i": i} for i in range(23)]
    for train, test in viterbi.folds(data, 5):
        assert len(train) + len(test) == 23
        assert not [row for row in test if row in train]


def test_every_sequence_is_tested_exactly_once():
    data = [{"i": i} for i in range(23)]
    seen = [row["i"] for _, test in viterbi.folds(data, 5) for row in test]
    assert sorted(seen) == list(range(23))


def test_decode_sequence_returns_state_names():
    sequences = [
        {
            "id": "s",
            "diagram_type": "flowchart",
            "states": ["start", "process", "terminal"],
            "observations": ["round|other|source", "box|other|linear", "round|other|sink"],
            "component_breaks": [],
        }
    ]
    alphabet = sorted(set(sequences[0]["observations"]))
    model = viterbi.build_model(sequences, alphabet, 1.0)
    decoded = viterbi.decode_sequence(sequences[0], model, alphabet)
    assert len(decoded) == 3
    assert set(decoded) <= set(STATES)


def test_score_reports_accuracy_and_macro_f1():
    result = viterbi.score(["start", "process"], ["start", "terminal"])
    assert result["accuracy"] == 0.5
    assert 0.0 <= result["macro_f1"] <= 1.0

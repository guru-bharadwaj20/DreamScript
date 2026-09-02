"""Phase 7.3.8 - forward-backward marginals, and the confidence the IR carries."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import posteriors
from src.parse.viterbi import build_model, logs


@pytest.fixture
def model():
    return {
        "pi": np.array([0.6, 0.4]),
        "A": np.array([[0.8, 0.2], [0.3, 0.7]]),
        "B": np.array([[0.9, 0.1], [0.2, 0.8]]),
    }


# -- the recursion --------------------------------------------------------------------------------


def test_every_position_is_a_distribution(model):
    gamma = posteriors.forward_backward([0, 1, 1, 0], *logs(model))
    assert np.allclose(gamma.sum(axis=1), 1.0)


def test_the_shape_is_positions_by_states(model):
    assert posteriors.forward_backward([0, 1, 0], *logs(model)).shape == (3, 2)


def test_an_empty_sequence_gives_an_empty_matrix(model):
    assert posteriors.forward_backward([], *logs(model)).shape == (0, 2)


def test_a_decisive_observation_produces_a_confident_marginal(model):
    gamma = posteriors.forward_backward([0], *logs(model))
    assert gamma[0, 0] > 0.85


def test_the_marginal_agrees_with_hmmlearn(model):
    hmm = pytest.importorskip("hmmlearn.hmm")

    machine = hmm.CategoricalHMM(n_components=2, init_params="")
    machine.startprob_, machine.transmat_, machine.emissionprob_ = (
        model["pi"],
        model["A"],
        model["B"],
    )
    symbols = [0, 1, 1, 0, 0, 1]
    reference = machine.predict_proba(np.array(symbols).reshape(-1, 1))
    ours = posteriors.forward_backward(symbols, *logs(model))
    assert np.allclose(ours, reference, atol=1e-9)


def test_the_log_likelihood_agrees_with_hmmlearn(model):
    hmm = pytest.importorskip("hmmlearn.hmm")

    machine = hmm.CategoricalHMM(n_components=2, init_params="")
    machine.startprob_, machine.transmat_, machine.emissionprob_ = (
        model["pi"],
        model["A"],
        model["B"],
    )
    symbols = [0, 1, 1, 0]
    assert posteriors.log_likelihood(symbols, *logs(model)) == pytest.approx(
        float(machine.score(np.array(symbols).reshape(-1, 1)))
    )


def test_a_long_sequence_does_not_underflow(model):
    gamma = posteriors.forward_backward([0] * 500, *logs(model))
    assert np.isfinite(gamma).all()
    assert np.allclose(gamma.sum(axis=1), 1.0)


# -- the IR field ---------------------------------------------------------------------------------


def test_annotate_writes_a_role_and_a_confidence():
    diagram = {
        "id": "d",
        "diagram_type": "flowchart",
        "nodes": [
            {"id": "a", "shape": "ellipse", "bbox": [0, 0, 1, 1], "text": "start"},
            {"id": "b", "shape": "rectangle", "bbox": [0, 2, 1, 1], "text": "do it"},
        ],
        "edges": [{"id": "e", "src": "a", "dst": "b"}],
    }
    sequence = {
        "node_ids": ["a", "b"],
        "observations": ["round|terminal-word|source", "box|other|sink"],
        "states": ["start", "terminal"],
        "component_breaks": [],
        "diagram_type": "flowchart",
        "id": "d",
    }
    alphabet = sorted(set(sequence["observations"]))
    fitted = build_model([sequence], alphabet, 1.0)
    annotated = posteriors.annotate(diagram, sequence, fitted, alphabet)
    for node in annotated["nodes"]:
        assert node["semantic_role"]
        assert 0.0 <= node["confidence"] <= 1.0
        assert node["attrs"]["role_posterior"]


def test_the_recorded_confidence_belongs_to_the_recorded_role():
    """Reporting a different state's marginal is the subtle version of 7.2.4's mistake."""
    diagram = {
        "id": "d",
        "diagram_type": "flowchart",
        "nodes": [{"id": "a", "shape": "diamond", "bbox": [0, 0, 1, 1], "text": "ok?"}],
        "edges": [],
    }
    sequence = {
        "node_ids": ["a"],
        "observations": ["diamond|question|branching"],
        "states": ["decision"],
        "component_breaks": [],
        "diagram_type": "flowchart",
        "id": "d",
    }
    alphabet = sorted(set(sequence["observations"]))
    fitted = build_model([sequence], alphabet, 1.0)
    node = posteriors.annotate(diagram, sequence, fitted, alphabet)["nodes"][0]
    recorded = node["attrs"]["role_posterior"].get(node["semantic_role"])
    assert recorded == pytest.approx(node["confidence"], abs=1e-4)


# -- the reliability curve ------------------------------------------------------------------------


def test_reliability_buckets_by_confidence():
    confidence = np.array([0.2, 0.6, 0.95, 0.995])
    correct = np.array([0.0, 1.0, 1.0, 1.0])
    curve = posteriors.reliability(confidence, correct)
    assert set(curve) == {"0.0-0.5", "0.5-0.7", "0.9-0.99", "0.99-1.01"}
    assert curve["0.0-0.5"]["accuracy"] == 0.0


def test_the_gap_is_confidence_minus_accuracy():
    curve = posteriors.reliability(np.array([0.8, 0.8]), np.array([1.0, 0.0]))
    assert curve["0.7-0.9"]["gap"] == pytest.approx(0.3)

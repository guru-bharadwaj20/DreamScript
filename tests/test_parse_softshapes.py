"""Phase 7.4.6 - learned components as the HMM's shape factor, hard and soft."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import softshapes
from src.parse.roles import STATES


def sequence(observations, states, node_ids=None, responsibilities=None, k=3):
    n = len(observations)
    return {
        "id": "page",
        "diagram_type": "flowchart",
        "node_ids": node_ids or [f"n{i}" for i in range(n)],
        "observations": list(observations),
        "states": list(states),
        "component_breaks": [],
        "R": np.array(responsibilities) if responsibilities is not None else np.full((n, k), 1 / k),
    }


# -- the declared design -----------------------------------------------------------------------------


def test_the_three_modes_are_the_ones_being_compared():
    assert softshapes.MODES == ("annotated", "hard", "soft")


# -- the symbol rewrite ------------------------------------------------------------------------------


def test_the_annotated_mode_leaves_the_symbol_alone():
    seq = sequence(["box|other|linear"], ["process"])
    assert softshapes.symbols_for(seq, "annotated", 3) == ["box|other|linear"]


def test_the_hard_mode_replaces_only_the_shape_factor():
    """The keyword and degree factors must survive; only the first third is learned."""
    seq = sequence(["box|question|branching"], ["decision"], responsibilities=[[0.1, 0.8, 0.1]])
    assert softshapes.symbols_for(seq, "hard", 3) == ["g1|question|branching"]


def test_the_hard_mode_takes_the_argmax_component():
    seq = sequence(["box|other|linear"], ["process"], responsibilities=[[0.7, 0.2, 0.1]])
    assert softshapes.symbols_for(seq, "hard", 3) == ["g0|other|linear"]


def test_the_alphabet_is_every_component_crossed_with_every_tail():
    seqs = [sequence(["box|other|linear", "diamond|question|branching"], ["process", "decision"])]
    assert len(softshapes.alphabet_for(seqs, "hard", 3)) == 3 * 2


def test_the_annotated_alphabet_is_just_the_symbols_seen():
    seqs = [sequence(["box|other|linear", "box|other|linear"], ["process", "process"])]
    assert softshapes.alphabet_for(seqs, "annotated", 3) == ["box|other|linear"]


# -- the emission matrix ------------------------------------------------------------------------------


def test_every_emission_row_is_a_distribution():
    seqs = [sequence(["box|other|linear", "diamond|question|branching"], ["process", "decision"])]
    alphabet = softshapes.alphabet_for(seqs, "soft", 3)
    B = softshapes.train(seqs, alphabet, "soft", 3)["B"]
    assert np.allclose(B.sum(axis=1), 1.0)
    assert B.shape == (len(STATES), len(alphabet))


def test_soft_counting_spreads_one_node_across_the_components():
    """A node contributes r[c] of a count to each component, not a whole count to one."""
    seqs = [sequence(["box|other|linear"], ["process"], responsibilities=[[0.5, 0.5, 0.0]])]
    alphabet = softshapes.alphabet_for(seqs, "soft", 3)
    B = softshapes.train(seqs, alphabet, "soft", 3, alpha=0.0)["B"]
    row = B[STATES.index("process")]
    assert row[alphabet.index("g0|other|linear")] == pytest.approx(0.5)
    assert row[alphabet.index("g1|other|linear")] == pytest.approx(0.5)


def test_hard_counting_gives_the_whole_count_to_the_argmax():
    seqs = [sequence(["box|other|linear"], ["process"], responsibilities=[[0.5, 0.5, 0.0]])]
    alphabet = softshapes.alphabet_for(seqs, "hard", 3)
    B = softshapes.train(seqs, alphabet, "hard", 3, alpha=0.0)["B"]
    row = B[STATES.index("process")]
    assert row[alphabet.index("g0|other|linear")] == pytest.approx(1.0)
    assert row[alphabet.index("g1|other|linear")] == pytest.approx(0.0)


def test_training_and_decoding_are_both_soft_or_the_experiment_is_a_mismatch():
    """Decoding softly from a hard-counted matrix would just be smoothing wearing a costume."""
    seqs = [sequence(["box|other|linear"], ["process"], responsibilities=[[0.5, 0.5, 0.0]])]
    alphabet = softshapes.alphabet_for(seqs, "soft", 3)
    soft = softshapes.train(seqs, alphabet, "soft", 3, alpha=0.0)["B"]
    hard = softshapes.train(seqs, alphabet, "hard", 3, alpha=0.0)["B"]
    assert not np.allclose(soft, hard)


# -- decoding ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", softshapes.MODES)
def test_every_mode_decodes_one_state_per_node(mode):
    seqs = [
        sequence(
            ["round|other|source", "box|other|linear", "round|other|sink"],
            ["start", "process", "terminal"],
        )
    ]
    alphabet = softshapes.alphabet_for(seqs, mode, 3)
    model = softshapes.train(seqs, alphabet, mode, 3)
    path = softshapes.decode(seqs[0], model, alphabet, mode, 3)
    assert len(path) == 3
    assert set(path) <= set(STATES)


def test_a_confident_responsibility_makes_soft_agree_with_hard():
    """A one-hot responsibility vector is the hard case, so the two must coincide there."""
    one_hot = [[0.0, 1.0, 0.0]] * 3
    seqs = [
        sequence(
            ["round|other|source", "box|other|linear", "round|other|sink"],
            ["start", "process", "terminal"],
            responsibilities=one_hot,
        )
    ]
    alphabet = softshapes.alphabet_for(seqs, "soft", 3)
    model = softshapes.train(seqs, alphabet, "soft", 3)
    assert softshapes.decode(seqs[0], model, alphabet, "soft", 3) == softshapes.decode(
        seqs[0], model, alphabet, "hard", 3
    )


def test_the_soft_decoder_matches_the_shared_viterbi_on_a_one_hot_sequence():
    """The local Viterbi in `decode` exists only to index emissions by position; on a degenerate
    mixture it must reproduce the shared implementation exactly."""
    seqs = [
        sequence(
            ["round|other|source", "box|other|linear", "box|other|linear", "round|other|sink"],
            ["start", "process", "process", "terminal"],
            responsibilities=[[1.0, 0.0]] * 4,
            k=2,
        )
    ]
    alphabet = softshapes.alphabet_for(seqs, "soft", 2)
    model = softshapes.train(seqs, alphabet, "soft", 2)
    assert softshapes.decode(seqs[0], model, alphabet, "soft", 2) == softshapes.decode(
        seqs[0], model, alphabet, "hard", 2
    )


# -- the missing-descriptor case ------------------------------------------------------------------------


def test_a_node_with_no_descriptor_gets_a_uniform_vector_rather_than_being_deleted():
    """Deleting it would renumber the sequence and desync the states beside it."""
    seqs = [
        {
            "id": "page",
            "diagram_type": "flowchart",
            "node_ids": ["a", "b"],
            "observations": ["box|other|linear", "box|other|linear"],
            "states": ["process", "process"],
            "component_breaks": [],
        }
    ]
    attached, missing, total = softshapes.attach(seqs, {"page:a": np.array([1.0, 0.0, 0.0])}, 3)
    assert (missing, total) == (1, 2)
    assert len(attached[0]["node_ids"]) == 2
    assert attached[0]["R"][1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_the_missing_count_is_reported_not_swallowed():
    seqs = [sequence(["box|other|linear"], ["process"])]
    _, missing, total = softshapes.attach(seqs, {}, 3)
    assert missing == total == 1

"""Phase 7.3.6 - EM on unlabelled pages, and the alignment that makes it comparable."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import baumwelch
from src.parse.roles import STATES

pytest.importorskip("hmmlearn")

ALPHABET = ["a", "b", "c"]


def sequence(observations, states=None):
    return {
        "observations": list(observations),
        "states": list(states) if states else ["process"] * len(observations),
    }


# -- encoding --------------------------------------------------------------------------------------


def test_encode_returns_hmmlearns_concatenated_convention():
    flat, lengths = baumwelch.encode([sequence("abc"), sequence("ba")], ALPHABET)
    assert flat.shape == (5, 1)
    assert lengths == [3, 2]


def test_the_lengths_sum_to_the_flat_length():
    flat, lengths = baumwelch.encode([sequence("abc"), sequence("ba")], ALPHABET)
    assert sum(lengths) == len(flat)


def test_a_sequence_too_short_to_have_a_transition_is_dropped():
    """A one-node page carries no transition evidence and hmmlearn will not accept it."""
    _, lengths = baumwelch.encode([sequence("a"), sequence("abc")], ALPHABET)
    assert lengths == [3]


def test_symbols_outside_the_alphabet_are_skipped_not_crashed():
    flat, lengths = baumwelch.encode([sequence(["a", "zzz", "b"])], ALPHABET)
    assert lengths == [2]
    assert flat.ravel().tolist() == [0, 1]


def test_encoding_is_by_alphabet_position():
    flat, _ = baumwelch.encode([sequence("cab")], ALPHABET)
    assert flat.ravel().tolist() == [2, 0, 1]


# -- initialisation --------------------------------------------------------------------------------


@pytest.fixture
def supervised():
    rng = np.random.default_rng(0)
    A = rng.random((len(STATES), len(STATES)))
    B = rng.random((len(STATES), len(ALPHABET)))
    pi = rng.random(len(STATES))
    return {
        "A": A / A.sum(axis=1, keepdims=True),
        "B": B / B.sum(axis=1, keepdims=True),
        "pi": pi / pi.sum(),
    }


@pytest.mark.parametrize("init", ["random", "uniform"])
def test_every_initialisation_is_row_stochastic(init):
    machine = baumwelch.model(init, ALPHABET)
    assert np.allclose(machine.transmat_.sum(axis=1), 1.0)
    assert np.allclose(machine.emissionprob_.sum(axis=1), 1.0)
    assert machine.startprob_.sum() == pytest.approx(1.0)


def test_the_supervised_start_is_the_matrices_it_was_given(supervised):
    """`init_params=""` is what stops hmmlearn silently overwriting them."""
    machine = baumwelch.model("supervised", ALPHABET, supervised)
    assert np.allclose(machine.transmat_, supervised["A"])
    assert np.allclose(machine.emissionprob_, supervised["B"])


def test_the_supervised_matrices_are_copied_not_aliased(supervised):
    machine = baumwelch.model("supervised", ALPHABET, supervised)
    machine.transmat_[0, 0] = 0.5
    assert supervised["A"][0, 0] != 0.5


def test_supervised_without_matrices_refuses_rather_than_guessing():
    with pytest.raises(ValueError, match="supervised init needs"):
        baumwelch.model("supervised", ALPHABET, None)


def test_the_uniform_start_is_symmetric_under_permutation():
    """The reason it is a degenerate fixed point: EM cannot break a symmetry it starts in."""
    machine = baumwelch.model("uniform", ALPHABET)
    assert len(np.unique(machine.transmat_)) == 1
    assert len(np.unique(machine.emissionprob_)) == 1


def test_random_is_seeded_and_reproducible():
    a = baumwelch.model("random", ALPHABET, seed=3).transmat_
    b = baumwelch.model("random", ALPHABET, seed=3).transmat_
    assert np.allclose(a, b)


def test_a_different_seed_gives_a_different_draw():
    a = baumwelch.model("random", ALPHABET, seed=3).transmat_
    b = baumwelch.model("random", ALPHABET, seed=4).transmat_
    assert not np.allclose(a, b)


def test_nothing_is_left_out_of_re_estimation():
    """`params="ste"` - a missing letter would freeze a matrix EM is supposed to move."""
    assert set(baumwelch.model("random", ALPHABET).params) == set("ste")


# -- the alignment ---------------------------------------------------------------------------------


def test_alignment_recovers_a_permuted_but_perfect_decoder():
    """EM's state numbering is arbitrary; reading state #3 as `process` would be arithmetic."""

    class Permuted:
        """Emits each node's true state, relabelled by a fixed permutation."""

        def __init__(self, labelled):
            self.answers = [STATES.index(s) for seq in labelled for s in seq["states"]]

        def predict(self, X):
            taken, self.answers = self.answers[: len(X)], self.answers[len(X) :]
            return np.array([(i + 4) % len(STATES) for i in taken])

    labelled = [sequence("abc", ["start", "process", "terminal"])]
    mapping, agreement = baumwelch.align(Permuted(labelled), labelled, ALPHABET)
    assert agreement == 1.0
    assert mapping[(STATES.index("start") + 4) % len(STATES)] == "start"


def test_a_decoder_that_answers_one_state_for_everything_scores_its_share():
    class Constant:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    labelled = [sequence("abca", ["start", "process", "process", "terminal"])]
    _, agreement = baumwelch.align(Constant(), labelled, ALPHABET)
    assert agreement == pytest.approx(0.5)


def test_the_mapping_is_a_permutation_of_the_state_space():
    class Constant:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    labelled = [sequence("abc", ["start", "process", "terminal"])]
    mapping, _ = baumwelch.align(Constant(), labelled, ALPHABET)
    assert sorted(mapping.values()) == sorted(STATES)


def test_nothing_to_align_gives_zero_rather_than_a_division_error():
    class Never:
        def predict(self, X):  # pragma: no cover - never reached
            raise AssertionError

    assert baumwelch.align(Never(), [sequence("a", ["start"])], ALPHABET)[1] == 0.0


# -- the declared budget ---------------------------------------------------------------------------


def test_the_iteration_cap_and_tolerance_are_declared():
    assert baumwelch.MAX_ITER == 100
    assert baumwelch.TOLERANCE == 1e-4

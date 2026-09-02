"""Phase 7.2 - the Naive Bayes text prior, and the four ways of spending it."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import prior

# -- the declared design ---------------------------------------------------------------------------


def test_the_four_modes_are_the_ones_the_plan_names():
    assert prior.MODES == ("none", "features", "multiply", "interpolate")


def test_the_weight_sweep_spans_both_endpoints():
    """0 has to be present as the untouched control and 1 as Naive Bayes alone."""
    assert prior.WEIGHTS[0] == 0.0
    assert prior.WEIGHTS[-1] == 1.0


# -- the interpolation -----------------------------------------------------------------------------


@pytest.fixture
def two_rows():
    down = np.array([[0.7, 0.3], [0.2, 0.8]])
    upstream = np.array([[0.1, 0.9], [0.6, 0.4]])
    return down, upstream


def test_weight_zero_is_the_downstream_model_untouched(two_rows):
    down, upstream = two_rows
    assert np.allclose(prior.combine(down, upstream, 0.0), down)


def test_weight_one_is_the_prior_alone(two_rows):
    down, upstream = two_rows
    assert np.allclose(prior.combine(down, upstream, 1.0), upstream)


def test_every_mixture_is_still_a_distribution(two_rows):
    down, upstream = two_rows
    for weight in prior.WEIGHTS:
        assert np.allclose(prior.combine(down, upstream, weight).sum(axis=1), 1.0)


def test_a_half_weight_is_the_renormalised_product(two_rows):
    down, upstream = two_rows
    expected = np.sqrt(down * upstream)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(prior.combine(down, upstream, 0.5), expected)


def test_an_exact_zero_does_not_propagate_a_nan():
    """Both models can emit a hard zero; log(0) would poison the whole row."""
    down = np.array([[1.0, 0.0]])
    upstream = np.array([[0.0, 1.0]])
    mixed = prior.combine(down, upstream, 0.5)
    assert np.isfinite(mixed).all()
    assert mixed.sum() == pytest.approx(1.0)


def test_the_mixture_moves_monotonically_toward_the_prior(two_rows):
    down, upstream = two_rows
    trace = [prior.combine(down, upstream, w)[0, 1] for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert trace == sorted(trace)


# -- the alignment ---------------------------------------------------------------------------------


def test_alignment_reorders_rather_than_concatenating_by_position():
    """Two tables built in different orders would silently mispair without this."""
    table = {
        "ids": np.array(["c", "a", "b"], dtype=object),
        "proba": np.array([[3.0], [1.0], [2.0]]),
    }
    assert list(prior._align(table, ["a", "b", "c"]).ravel()) == [1.0, 2.0, 3.0]


def test_a_missing_id_raises_rather_than_guessing():
    table = {"ids": np.array(["a"], dtype=object), "proba": np.array([[1.0]])}
    with pytest.raises(KeyError, match="no Naive Bayes prior"):
        prior._align(table, ["a", "ghost"])

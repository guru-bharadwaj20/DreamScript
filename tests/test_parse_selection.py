"""Phase 7.4.3 - the K sweep, and the difference between an elbow and a minimum."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import selection

# -- the declared sweep ------------------------------------------------------------------------------


def test_the_sweep_covers_the_plans_range():
    assert selection.K_RANGE[0] == 3
    assert selection.K_RANGE[-1] == 15


def test_the_sweep_is_evenly_spaced():
    """The discrete second difference is only a curvature if the spacing is constant."""
    assert len(set(np.diff(selection.K_RANGE))) == 1


# -- the elbow ---------------------------------------------------------------------------------------


def test_an_elbow_is_where_the_curve_stops_falling_fast():
    ks = [3, 4, 5, 6, 7]
    # Falls steeply to K=5, then flattens: the corner is at 5.
    values = [100.0, 60.0, 30.0, 28.0, 27.0]
    assert selection.elbow(ks, values) == 5


def test_the_elbow_is_not_the_minimum():
    """A curve can still be falling at its own elbow, and BIC on a big table usually is."""
    ks = [3, 4, 5, 6, 7]
    values = [100.0, 60.0, 30.0, 28.0, 27.0]
    assert selection.elbow(ks, values) != ks[int(np.argmin(values))]


def test_a_straight_line_has_no_corner_to_find():
    ks = [3, 4, 5, 6, 7]
    assert selection.elbow(ks, [100.0, 90.0, 80.0, 70.0, 60.0]) in ks


def test_a_sweep_too_short_to_have_curvature_returns_its_first_k():
    assert selection.elbow([3, 4], [10.0, 5.0]) == 3


def test_the_elbow_is_one_of_the_ks_swept():
    rng = np.random.default_rng(0)
    ks = list(range(3, 16))
    values = list(np.sort(rng.random(len(ks)))[::-1])
    assert selection.elbow(ks, values) in ks


def test_a_sharper_corner_moves_the_elbow():
    ks = [3, 4, 5, 6, 7]
    early = [100.0, 30.0, 29.0, 28.0, 27.0]
    late = [100.0, 90.0, 80.0, 30.0, 29.0]
    assert selection.elbow(ks, early) < selection.elbow(ks, late)


# -- the sweep rows ----------------------------------------------------------------------------------


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    centres = rng.normal(0, 10, size=(3, 22))
    X = np.vstack([rng.normal(c, 0.5, size=(40, 22)) for c in centres])
    labels = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40, dtype=object)
    return X, labels


def test_every_k_produces_a_row(blobs):
    X, labels = blobs
    rows = selection.sweep(X, labels, "diag", ks=(3, 4, 5), folds=3)
    assert [row["k"] for row in rows] == [3, 4, 5]


def test_each_row_carries_all_three_criteria(blobs):
    X, labels = blobs
    row = selection.sweep(X, labels, "diag", ks=(3,), folds=3)[0]
    assert {"bic", "aic", "held_out_log_likelihood", "adjusted_rand"} <= set(row)


def test_free_parameters_grow_with_k(blobs):
    X, labels = blobs
    rows = selection.sweep(X, labels, "diag", ks=(3, 5, 7), folds=3)
    counts = [row["free_parameters"] for row in rows]
    assert counts == sorted(counts)
    assert len(set(counts)) == 3


def test_aic_is_never_above_bic_on_a_table_this_size(blobs):
    """AIC's penalty is 2k and BIC's is k*log(n); for n > e^2 the BIC penalty is larger."""
    X, labels = blobs
    for row in selection.sweep(X, labels, "diag", ks=(3, 5), folds=3):
        assert row["aic"] < row["bic"]


def test_the_smallest_component_is_reported_so_a_collapsed_fit_is_visible(blobs):
    """A K that only fits by leaving a component with three nodes in it has not found a shape."""
    X, labels = blobs
    row = selection.sweep(X, labels, "diag", ks=(12,), folds=3)[0]
    assert row["smallest_component"] >= 0


def test_a_k_matching_the_truth_agrees_better_than_a_k_of_one_more_than_nothing(blobs):
    X, labels = blobs
    rows = {r["k"]: r for r in selection.sweep(X, labels, "diag", ks=(3, 15), folds=3)}
    assert rows[3]["adjusted_rand"] > rows[15]["adjusted_rand"]

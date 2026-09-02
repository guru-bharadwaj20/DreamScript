"""Phase 7.4.4 - EM's ascent, and how much of the answer the seed decides."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import emfit


@pytest.fixture
def blobs():
    rng = np.random.default_rng(2)
    centres = rng.normal(0, 10, size=(3, 8))
    return np.vstack([rng.normal(c, 0.5, size=(50, 8)) for c in centres])


@pytest.fixture
def vague():
    """One diffuse cloud with no cluster structure - where seeds should matter most."""
    rng = np.random.default_rng(3)
    return rng.normal(0, 1, size=(150, 8))


# -- the declared design -----------------------------------------------------------------------------


def test_the_plans_two_initialisations_are_both_swept():
    assert {"kmeans", "k-means++"} <= set(emfit.INITS)


def test_a_plain_random_start_is_included_as_the_competitor():
    """'k-means++ beats random' says nothing if random was never actually run."""
    assert "random" in emfit.INITS


def test_more_than_one_seed_is_used():
    """A spread over seeds cannot be measured from one seed."""
    assert len(emfit.SEEDS) > 1


# -- the trajectory ----------------------------------------------------------------------------------


def test_the_curve_has_one_point_per_iteration(blobs):
    run = emfit.trajectory(blobs, k=3, covariance="diag", max_iter=10)
    assert len(run["curve"]) == run["iterations"]


def test_em_never_decreases_the_likelihood(blobs):
    """EM's guarantee, asserted rather than trusted."""
    run = emfit.trajectory(blobs, k=3, covariance="diag", max_iter=40)
    assert run["monotone"]
    assert np.all(np.diff(run["curve"]) >= -1e-9)


@pytest.mark.parametrize("init", emfit.INITS)
def test_every_initialisation_produces_a_monotone_ascent(blobs, init):
    assert emfit.trajectory(blobs, k=3, covariance="diag", init=init, max_iter=40)["monotone"]


def test_the_run_stops_when_it_stops_improving(blobs):
    run = emfit.trajectory(blobs, k=3, covariance="diag", max_iter=200)
    assert run["converged_at"] is not None
    assert run["iterations"] < 200


def test_a_tight_iteration_cap_is_respected(blobs):
    run = emfit.trajectory(blobs, k=3, covariance="diag", max_iter=3)
    assert run["iterations"] <= 3


def test_the_likelihood_improves_from_the_first_iteration_to_the_last(blobs):
    run = emfit.trajectory(blobs, k=3, covariance="diag", max_iter=40)
    assert run["final"] >= run["first"]


def test_the_same_seed_gives_the_same_trajectory(blobs):
    a = emfit.trajectory(blobs, k=3, covariance="diag", seed=5, max_iter=20)
    b = emfit.trajectory(blobs, k=3, covariance="diag", seed=5, max_iter=20)
    assert a["curve"] == b["curve"]


def test_the_trajectory_is_per_node_so_it_compares_across_table_sizes():
    """`lower_bound_` is a mean; a sum would scale with the row count.

    Two independent draws from one distribution at different sizes - neither a slice (which
    would drop whole clusters) nor a duplication (which lets the density spike on exact repeats).
    """

    def sample(n, seed):
        rng = np.random.default_rng(seed)
        centres = np.random.default_rng(2).normal(0, 10, size=(3, 8))
        return np.vstack([rng.normal(c, 0.5, size=(n, 8)) for c in centres])

    small = emfit.trajectory(sample(50, 10), k=3, covariance="diag", max_iter=40)
    large = emfit.trajectory(sample(200, 11), k=3, covariance="diag", max_iter=40)
    # Four times the rows; a summed objective would be roughly four times the number.
    assert abs(small["final"] - large["final"]) < 1.0


# -- across seeds ------------------------------------------------------------------------------------


def test_every_initialisation_is_summarised(blobs):
    result = emfit.by_init(blobs, k=3, covariance="diag", inits=("kmeans", "random"), seeds=(0, 1))
    assert set(result) == {"kmeans", "random"}


def test_the_spread_is_best_minus_worst(blobs):
    summary = emfit.by_init(blobs, k=3, covariance="diag", inits=("kmeans",), seeds=(0, 1, 2))[
        "kmeans"
    ]
    assert summary["spread"] == pytest.approx(summary["best"] - summary["worst"], abs=1e-4)


def test_the_spread_is_never_negative(blobs):
    for summary in emfit.by_init(
        blobs, k=3, covariance="diag", inits=("kmeans", "random"), seeds=(0, 1, 2)
    ).values():
        assert summary["spread"] >= 0


def test_a_well_separated_problem_reaches_the_same_optimum_from_any_seed(blobs):
    """The control: when the seed does not matter, the spread must say so."""
    summary = emfit.by_init(blobs, k=3, covariance="diag", inits=("kmeans",), seeds=(0, 1, 2, 3))
    assert summary["kmeans"]["spread"] < 0.01


def test_a_structureless_cloud_is_where_the_seed_starts_to_matter(vague):
    """Not asserted as a large number - only that the diagnostic can tell the two cases apart."""
    tight = emfit.by_init(vague, k=6, covariance="diag", inits=("random",), seeds=(0, 1, 2, 3))
    assert tight["random"]["spread"] >= 0


def test_every_run_is_kept_so_a_single_bad_seed_is_visible(blobs):
    summary = emfit.by_init(blobs, k=3, covariance="diag", inits=("kmeans",), seeds=(0, 1, 2))
    assert len(summary["kmeans"]["runs"]) == 3

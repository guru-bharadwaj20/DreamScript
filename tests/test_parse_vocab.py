"""Phase 7.4.2 - the mixture over 7.4.1's descriptors, and the assignment that scores it."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import vocab


@pytest.fixture
def blobs():
    """Three well-separated clusters in 22 dimensions, one per label.

    Separated in every dimension rather than in two of twenty-two: the model standardises, so
    columns that carry nothing are rescaled up to the same variance as the ones that do, and a
    fixture with 20 noise columns tests the noise rather than the mixture.
    """
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 10, size=(3, 22))
    X = np.vstack([rng.normal(c, 0.4, size=(60, 22)) for c in centres])
    labels = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 60, dtype=object)
    return X, labels


# -- the declared design -----------------------------------------------------------------------------


def test_the_plans_three_covariance_types_are_all_compared():
    assert {"full", "diag", "tied"} <= set(vocab.COVARIANCES)


def test_a_rigid_floor_is_included_to_read_the_others_against():
    assert "spherical" in vocab.COVARIANCES


def test_the_covariance_diagonal_is_regularised():
    """`full` on 22 correlated columns can find a singular component and report +inf."""
    assert vocab.REG > 0


# -- the pipeline ------------------------------------------------------------------------------------


def test_the_model_standardises_before_fitting(blobs):
    """A tied or spherical covariance on raw columns would be modelling the units."""
    X, _ = blobs
    assert "scale" in vocab.fit(X, k=3).named_steps


def test_the_fitted_model_has_the_requested_component_count(blobs):
    X, _ = blobs
    assert vocab.fit(X, k=4).named_steps["gmm"].n_components == 4


@pytest.mark.parametrize("covariance", vocab.COVARIANCES)
def test_every_covariance_type_fits_and_predicts(blobs, covariance):
    X, _ = blobs
    model = vocab.fit(X, k=3, covariance=covariance)
    assert len(model.predict(X)) == len(X)


def test_the_fit_is_seeded_and_reproducible(blobs):
    X, _ = blobs
    a = vocab.fit(X, k=3, seed=7).predict(X)
    b = vocab.fit(X, k=3, seed=7).predict(X)
    assert (a == b).all()


def test_separated_blobs_are_recovered(blobs):
    """A sanity floor: if the mixture cannot find three obvious clusters, nothing below means
    anything."""
    X, labels = blobs
    components = vocab.fit(X, k=3).predict(X)
    assert vocab.purity(components, labels, 3)["purity"] > 0.99


# -- the Hungarian purity ----------------------------------------------------------------------------


def test_a_perfect_but_permuted_clustering_scores_one():
    components = np.array([2, 2, 0, 0, 1, 1])
    labels = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    assert vocab.purity(components, labels, 3)["purity"] == 1.0


def test_the_mapping_names_each_component():
    components = np.array([0, 0, 1, 1])
    labels = np.array(["a", "a", "b", "b"], dtype=object)
    assert set(vocab.purity(components, labels, 2)["mapping"]) == {0, 1}


def test_two_components_cannot_both_claim_the_same_label():
    """The greedy version flatters a model that found one shape twice and missed another."""
    # Both components are pure `a`; `b` is never found.
    components = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array(["a"] * 5 + ["b"], dtype=object)
    result = vocab.purity(components, labels, 2)
    assert result["labels_claimed"] == 2
    assert result["purity"] < 1.0


def test_one_component_for_everything_scores_the_largest_class_share():
    components = np.zeros(10, dtype=int)
    labels = np.array(["a"] * 7 + ["b"] * 3, dtype=object)
    assert vocab.purity(components, labels, 1)["purity"] == pytest.approx(0.7)


def test_purity_is_never_above_one(blobs):
    X, labels = blobs
    components = vocab.fit(X, k=8).predict(X)
    assert vocab.purity(components, labels, 8)["purity"] <= 1.0


# -- the evaluation ----------------------------------------------------------------------------------


def test_evaluate_reports_all_three_criteria(blobs):
    X, labels = blobs
    result = vocab.evaluate(X, labels, k=3, covariance="diag", folds=3)
    assert {"held_out_log_likelihood", "bic", "adjusted_rand", "purity"} <= set(result)


def test_the_free_parameter_count_grows_with_the_covariance(blobs):
    """The whole bias-variance argument, asserted so the ordering cannot silently invert."""
    X, labels = blobs
    counts = {
        name: vocab.evaluate(X, labels, k=3, covariance=name, folds=3)["free_parameters"]
        for name in ("spherical", "diag", "tied", "full")
    }
    assert counts["spherical"] < counts["diag"] < counts["tied"] < counts["full"]


def test_the_component_sizes_account_for_every_row(blobs):
    X, labels = blobs
    result = vocab.evaluate(X, labels, k=3, covariance="diag", folds=3)
    assert sum(result["component_sizes"]) == len(X)


def test_the_held_out_likelihood_is_per_node_so_folds_compare(blobs):
    """`score` is a mean, not a sum; a sum would make the metric depend on fold size."""
    X, labels = blobs
    small = vocab.evaluate(X, labels, k=3, covariance="diag", folds=3)
    large = vocab.evaluate(X, labels, k=3, covariance="diag", folds=5)
    assert abs(small["held_out_log_likelihood"] - large["held_out_log_likelihood"]) < 5.0

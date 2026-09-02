"""Phase 8.5 - DBSCAN, and the difference between an outlier and a novel shape."""

from __future__ import annotations

import numpy as np
import pytest

from src.cluster import dbscan
from src.features.descriptors import NAMES

# -- the declared design ---------------------------------------------------------------------------


def test_min_samples_follows_the_two_times_dimensionality_rule():
    assert 2 * len(NAMES) == dbscan.MIN_SAMPLES


def test_the_eps_sweep_is_centred_on_the_chosen_value():
    """A sweep that never includes 1.0 would report neighbours of the elbow but not the elbow."""
    assert 1.0 in dbscan.EPS_FACTORS
    assert min(dbscan.EPS_FACTORS) < 1.0 < max(dbscan.EPS_FACTORS)


# -- choosing eps ----------------------------------------------------------------------------------


def test_the_k_distance_curve_is_sorted_and_one_per_point():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(200, 5))
    curve = dbscan.k_distances(X, k=5)
    assert len(curve) == 200
    assert np.all(np.diff(curve) >= 0)


def test_the_elbow_lands_on_the_knee_of_a_hockey_stick():
    """Flat then steep: the knee is where it turns, not at either end."""
    curve = np.concatenate([np.linspace(1.0, 1.2, 90), np.linspace(1.2, 20.0, 10)])
    chosen = dbscan.eps_elbow(curve)
    assert 1.1 < chosen < 6.0


def test_the_elbow_of_a_straight_line_is_not_an_endpoint():
    curve = np.linspace(0.0, 10.0, 50)
    assert dbscan.eps_elbow(curve) not in (curve[0], curve[-1])


# -- the clustering --------------------------------------------------------------------------------


@pytest.fixture
def two_blobs_and_strays():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.2, size=(80, 4))
    b = rng.normal(6.0, 0.2, size=(80, 4))
    strays = np.array([[40.0] * 4, [-40.0] * 4, [0.0, 40.0, 0.0, 0.0]])
    labels = np.array(["a"] * 80 + ["b"] * 80 + ["odd"] * 3, dtype=object)
    return np.vstack([a, b, strays]), labels


def test_two_blobs_and_three_strays_give_two_clusters_and_three_outliers(two_blobs_and_strays):
    """The floor: where the answer is unambiguous, DBSCAN must find it."""
    X, labels = two_blobs_and_strays
    result = dbscan.evaluate(X, labels, eps=1.0, min_samples=5)
    assert result["clusters"] == 2
    assert result["noise"] == 3


def test_a_larger_eps_never_produces_more_noise(two_blobs_and_strays):
    """Monotone by construction; if it is not, the sweep is not varying what it claims to."""
    X, labels = two_blobs_and_strays
    noise = [dbscan.evaluate(X, labels, eps=e, min_samples=5)["noise"] for e in (1.0, 3.0, 10.0)]
    assert noise == sorted(noise, reverse=True)


def test_the_noise_and_clustered_label_mixes_are_both_reported(two_blobs_and_strays):
    X, labels = two_blobs_and_strays
    result = dbscan.evaluate(X, labels, eps=1.0, min_samples=5)
    assert result["noise_label_mix"] == {"odd": 1.0}
    assert "a" in result["clustered_label_mix"]


# -- enrichment ------------------------------------------------------------------------------------


def test_a_label_over_represented_among_the_outliers_scores_above_one():
    assert dbscan.enrichment({"circle": 0.4}, {"circle": 0.2})["circle"] == 2.0


def test_a_label_absent_from_the_outliers_scores_zero():
    assert dbscan.enrichment({}, {"rectangle": 0.5})["rectangle"] == 0.0


def test_a_class_with_no_corpus_share_is_skipped_rather_than_dividing_by_zero():
    assert dbscan.enrichment({"a": 0.5}, {"a": 0.5, "b": 0.0}) == {"a": 1.0}


# -- the two kinds of outlier ----------------------------------------------------------------------


def test_slivers_and_compact_outliers_are_separated_rather_than_averaged():
    """The outlier mean aspect and the outlier median aspect point opposite ways; both are real."""
    X = np.zeros((6, len(NAMES)))
    aspect = NAMES.index("rect_aspect")
    X[:, aspect] = [1.0, 1.1, 1.2, 50.0, 60.0, 1.0]
    assignment = np.array([-1, -1, -1, -1, -1, 0])
    labels = np.array(["circle"] * 3 + ["rectangle"] * 2 + ["circle"], dtype=object)

    kinds = dbscan.outlier_kinds(X, assignment, labels)
    assert kinds["outliers"] == 5
    assert kinds["sliver"]["n"] == 2
    assert kinds["compact"]["n"] == 3
    assert kinds["sliver"]["labels"] == {"rectangle": 2}


def test_the_split_reports_both_the_mean_and_the_median_aspect():
    """Reporting only the mean would have described 118 of 142 outliers backwards."""
    X = np.zeros((4, len(NAMES)))
    X[:, NAMES.index("rect_aspect")] = [1.0, 1.0, 100.0, 1.0]
    kinds = dbscan.outlier_kinds(X, np.array([-1, -1, -1, 0]), np.array(["a"] * 4, dtype=object))
    assert kinds["aspect"]["outlier_mean"] > kinds["aspect"]["outlier_median"]


def test_the_shares_of_the_two_kinds_add_to_one():
    X = np.zeros((5, len(NAMES)))
    X[:, NAMES.index("rect_aspect")] = [1.0, 1.0, 50.0, 1.0, 1.0]
    kinds = dbscan.outlier_kinds(X, np.full(5, -1), np.array(["a"] * 5, dtype=object))
    assert kinds["sliver"]["share_of_outliers"] + kinds["compact"]["share_of_outliers"] == 1.0


def test_the_sliver_threshold_sits_outside_the_normal_range():
    """The clustered p90 aspect is 2.90 on the corpus; the threshold must not clip normal shapes."""
    assert dbscan.SLIVER_ASPECT > 2.9

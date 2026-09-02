"""Phase 8.8 - silhouette, Davies-Bouldin, Calinski-Harabasz, against the answer key."""

from __future__ import annotations

import numpy as np
import pytest

from src.cluster import validity

# -- the declared design ---------------------------------------------------------------------------


def test_all_three_indices_the_plan_names_are_computed():
    assert set(validity.INDICES) == {"silhouette", "davies_bouldin", "calinski_harabasz"}


def test_the_direction_of_each_index_is_declared_rather_than_assumed():
    """Davies-Bouldin is the odd one out; hard-coding its sign three times is how it gets flipped."""
    assert validity.BETTER["davies_bouldin"] == "low"
    assert validity.BETTER["silhouette"] == "high"
    assert validity.BETTER["calinski_harabasz"] == "high"


# -- the indices themselves ------------------------------------------------------------------------


@pytest.fixture
def separated():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.2, (60, 4)), rng.normal(10, 0.2, (60, 4))])
    return X, np.array([0] * 60 + [1] * 60)


def test_a_clean_partition_scores_well_on_all_three(separated):
    X, assignment = separated
    scores = validity.indices(X, assignment)
    assert scores["silhouette"] > 0.8
    assert scores["davies_bouldin"] < 0.5
    assert scores["calinski_harabasz"] > 100


def test_a_random_partition_scores_badly_on_all_three(separated):
    X, _ = separated
    rng = np.random.default_rng(1)
    scores = validity.indices(X, rng.integers(0, 2, size=len(X)))
    assert scores["silhouette"] < 0.2
    assert scores["davies_bouldin"] > 1.0


def test_a_single_cluster_returns_nulls_rather_than_raising(separated):
    """Every index is undefined on one cluster; the row still has to exist in the table."""
    X, _ = separated
    scores = validity.indices(X, np.zeros(len(X), dtype=int))
    assert all(scores[name] is None for name in validity.INDICES)


# -- the answer-key comparison ---------------------------------------------------------------------


def _row(name, sil, db, ch, ari):
    return {
        "partition": name,
        "k": 3,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "ari": ari,
        "purity": 0.4,
        "smallest_cluster": 10,
        "largest_share": 0.5,
    }


def test_an_index_that_tracks_the_answer_key_scores_positive():
    rows = [
        _row("a", 0.1, 3.0, 100.0, 0.1),
        _row("b", 0.5, 2.0, 300.0, 0.5),
        _row("c", 0.9, 1.0, 900.0, 0.9),
    ]
    result = validity.agreement(rows)
    assert result["silhouette"]["spearman_with_ari"] == pytest.approx(1.0)
    assert result["calinski_harabasz"]["spearman_with_ari"] == pytest.approx(1.0)


def test_the_davies_bouldin_sign_is_flipped_so_positive_always_means_agreement():
    """It is lower-is-better; without the flip a perfectly truthful index would read as -1."""
    rows = [
        _row("a", 0.1, 3.0, 100.0, 0.1),
        _row("b", 0.5, 2.0, 300.0, 0.5),
        _row("c", 0.9, 1.0, 900.0, 0.9),
    ]
    assert validity.agreement(rows)["davies_bouldin"]["spearman_with_ari"] == pytest.approx(1.0)


def test_an_index_that_ranks_backwards_scores_negative():
    rows = [
        _row("a", 0.9, 1.0, 900.0, 0.1),
        _row("b", 0.5, 2.0, 300.0, 0.5),
        _row("c", 0.1, 3.0, 100.0, 0.9),
    ]
    assert validity.agreement(rows)["silhouette"]["spearman_with_ari"] == pytest.approx(-1.0)


def test_the_best_partition_by_each_index_is_named():
    rows = [_row("a", 0.9, 1.0, 900.0, 0.1), _row("b", 0.1, 3.0, 100.0, 0.9)]
    result = validity.agreement(rows)
    assert result["silhouette"]["best_partition"] == "a"
    assert result["davies_bouldin"]["best_partition"] == "a"
    assert result["best_by_ari"] == "b"


def test_rows_with_an_undefined_index_are_excluded_rather_than_ranked_as_zero():
    rows = [
        _row("a", 0.1, 3.0, 100.0, 0.1),
        _row("b", 0.5, 2.0, 300.0, 0.5),
        _row("c", 0.9, 1.0, 900.0, 0.9),
        {**_row("degenerate", None, None, None, 0.0)},
    ]
    assert validity.agreement(rows)["silhouette"]["spearman_with_ari"] == pytest.approx(1.0)


# -- the within-family split -----------------------------------------------------------------------


def test_a_pooled_coefficient_can_cancel_two_opposite_effects():
    """The reason the split exists: CH ranks K backwards and linkage rules correctly."""
    rows = [
        _row("kmeans k=2", 0.4, 1.0, 900.0, 0.0),
        _row("kmeans k=3", 0.3, 1.1, 600.0, 0.3),
        _row("kmeans k=4", 0.2, 1.2, 300.0, 0.6),
        _row("single k=4", 0.9, 0.1, 50.0, 0.0),
        _row("ward k=4", 0.4, 0.9, 250.0, 0.4),
        _row("average k=4", 0.8, 0.4, 80.0, 0.1),
    ]
    split = validity.agreement_within_families(rows)
    assert split["kmeans_only"]["calinski_harabasz"]["spearman_with_ari"] < 0
    assert split["linkage_only"]["calinski_harabasz"]["spearman_with_ari"] > 0


def test_a_family_too_small_to_correlate_is_omitted_rather_than_reported_as_noise():
    rows = [_row("kmeans k=2", 0.4, 1.0, 900.0, 0.0), _row("ward k=4", 0.4, 0.9, 250.0, 0.4)]
    assert validity.agreement_within_families(rows) == {}

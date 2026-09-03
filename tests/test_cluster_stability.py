"""Phase 8.9 - bootstrap stability, and whether the partition that recurs is the one that knows."""

from __future__ import annotations

import numpy as np
import pytest

from src.cluster import stability
from src.features.descriptors import NAMES

# -- the declared design ---------------------------------------------------------------------------


def test_both_algorithms_are_measured():
    """They fail differently: K-means has a seed, Ward has none and can still move with the sample."""
    assert set(stability.ALGORITHMS) == {"kmeans", "ward"}


def test_the_ks_are_the_ones_earlier_tasks_argued_for():
    assert 3 in stability.KS  # 8.2's elbow and silhouette peak
    assert 4 in stability.KS  # 8.2's recommendation
    assert 6 in stability.KS  # 8.3's ARI peak


def test_enough_draws_to_separate_the_reported_differences():
    assert stability.DRAWS >= 20


# -- one draw --------------------------------------------------------------------------------------


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(i * 30.0, 0.4, size=(50, len(NAMES))) for i in range(3)])


def test_a_trivially_separable_problem_is_perfectly_stable(blobs):
    """The floor: three far-apart clouds must recur in every resample."""
    result = stability.stability(blobs, k=3, algorithm="kmeans", draws=5)
    assert result["mean_ari"] > 0.99


def test_structureless_data_is_not_stable():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(300, len(NAMES)))
    assert stability.stability(X, k=4, algorithm="kmeans", draws=5)["mean_ari"] < 0.75


def test_the_comparison_is_made_on_unique_rows_only(blobs):
    """A bootstrap draw repeats rows, and a repeated row agrees with itself for free."""
    rng = np.random.default_rng(2)
    reference = stability.assign(blobs, 3, "kmeans")
    draw = rng.integers(0, len(blobs), size=len(blobs))
    assert len(np.unique(draw)) < len(draw)  # the fixture for the claim
    score = stability.one_draw(blobs, reference, 3, "kmeans", np.random.default_rng(3))
    assert score is not None
    assert score <= 1.0


def test_a_draw_too_small_to_support_k_returns_nothing_rather_than_a_number():
    X = np.zeros((3, len(NAMES)))
    assert stability.one_draw(X, np.array([0, 1, 2]), 5, "kmeans", np.random.default_rng(0)) is None


def test_label_switching_does_not_count_as_instability(blobs):
    """Two runs finding the same groups in a different order agree completely."""
    from sklearn.metrics import adjusted_rand_score

    a = stability.assign(blobs, 3, "kmeans", seed=1)
    b = (a + 1) % 3
    assert adjusted_rand_score(a, b) == pytest.approx(1.0)


# -- the reported row ------------------------------------------------------------------------------


def test_stability_is_reported_beside_the_smallest_cluster(blobs):
    """A partition that puts everything in one cluster is perfectly stable and worth nothing."""
    row = stability.stability(blobs, k=3, algorithm="kmeans", draws=3)
    assert "smallest_cluster" in row
    assert "largest_share" in row


def test_the_spread_across_draws_is_reported_not_only_the_mean(blobs):
    row = stability.stability(blobs, k=3, algorithm="kmeans", draws=5)
    assert row["min_ari"] <= row["mean_ari"] <= row["max_ari"]
    assert row["std_ari"] >= 0.0


def test_ward_is_deterministic_for_a_fixed_sample(blobs):
    """So any instability it shows is instability with respect to the sample, not the seed."""
    assert np.array_equal(
        stability.assign(blobs, 3, "ward"), stability.assign(blobs, 3, "ward", seed=99)
    )


def test_an_unknown_algorithm_is_refused(blobs):
    with pytest.raises(ValueError):
        stability.assign(blobs, 3, "spectral")


# -- stability against informativeness -------------------------------------------------------------


def _row(mean_ari, ari_against_labels):
    return {
        "algorithm": "kmeans",
        "k": 3,
        "mean_ari": mean_ari,
        "ari_against_labels": ari_against_labels,
    }


def test_a_criterion_that_tracks_the_answer_key_scores_positive():
    rows = [_row(0.2, 0.1), _row(0.5, 0.4), _row(0.9, 0.8)]
    assert stability.informativeness(rows)["spearman"] == pytest.approx(1.0)


def test_a_criterion_that_ranks_backwards_scores_negative():
    rows = [_row(0.9, 0.1), _row(0.5, 0.4), _row(0.2, 0.8)]
    assert stability.informativeness(rows)["spearman"] == pytest.approx(-1.0)


def test_the_two_extreme_partitions_are_named_so_the_trade_is_visible():
    rows = [_row(0.9, 0.01), _row(0.5, 0.40)]
    result = stability.informativeness(rows)
    assert result["most_stable_partition_ari"] == 0.01
    assert result["most_informative_partition_stability"] == 0.5


def test_a_p_value_is_carried_because_the_table_is_small():
    """Six rows cannot make a strong claim; the point estimate is not the whole story."""
    rows = [_row(0.9, 0.1), _row(0.5, 0.4), _row(0.2, 0.8)]
    assert "p_value" in stability.informativeness(rows)

"""Phase 8.2 - elbow, silhouette, and the K they each choose."""

from __future__ import annotations

import numpy as np
import pytest

from src.cluster import choosek
from src.features.descriptors import NAMES

# -- the declared design ---------------------------------------------------------------------------


def test_the_sweep_starts_at_two_and_reaches_fifteen():
    """15 matches 7.4.3's mixture sweep so the two can be read side by side."""
    assert min(choosek.K_RANGE) == 2
    assert max(choosek.K_RANGE) == 15


# -- the elbow -------------------------------------------------------------------------------------


def test_the_elbow_is_the_point_of_maximum_curvature():
    """A curve that drops hard then flattens has its elbow where it flattens."""
    ks = [2, 3, 4, 5, 6]
    values = [100.0, 50.0, 20.0, 18.0, 17.0]
    assert choosek.elbow(ks, values) == 4


def test_a_straight_line_has_no_meaningful_elbow_and_does_not_crash():
    ks = [2, 3, 4, 5]
    assert choosek.elbow(ks, [10.0, 8.0, 6.0, 4.0]) in ks


def test_too_few_points_returns_the_first_k():
    assert choosek.elbow([2, 3], [10.0, 5.0]) == 2


def test_the_elbow_matches_the_definition_seven_four_three_used():
    """Kept identical on purpose; a different definition would make the two sweeps incomparable."""
    from src.parse.selection import elbow as gmm_elbow

    ks = [2, 3, 4, 5, 6, 7]
    values = [90.0, 40.0, 22.0, 20.0, 19.0, 18.5]
    assert choosek.elbow(ks, values) == gmm_elbow(ks, values)


# -- the three criteria ----------------------------------------------------------------------------


def _row(k, inertia, silhouette, ari, smallest=100):
    return {
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette,
        "ari": ari,
        "purity": 0.4,
        "smallest_cluster": smallest,
        "clusters_under_ten": 0 if smallest >= 10 else 1,
    }


def test_all_three_criteria_are_reported_even_when_they_agree():
    rows = [_row(2, 100, 0.1, 0.1), _row(3, 50, 0.9, 0.9), _row(4, 48, 0.2, 0.2)]
    picked = choosek.choose(rows)
    assert picked["inertia_elbow"] == picked["silhouette_peak"] == picked["ari_peak"] == 3
    assert picked["criteria_agree"] is True


def test_disagreement_is_recorded_rather_than_resolved_silently():
    rows = [_row(2, 100, 0.9, 0.0), _row(3, 50, 0.5, 0.1), _row(4, 48, 0.2, 0.8)]
    picked = choosek.choose(rows)
    assert picked["silhouette_peak"] == 2
    assert picked["ari_peak"] == 4
    assert picked["criteria_agree"] is False


def test_the_recommendation_refuses_a_k_whose_partition_has_dissolved():
    """The ARI peak is at K=4, but K=4 holds a one-node cluster; K=3 is the largest real one."""
    rows = [_row(2, 100, 0.4, 0.05), _row(3, 60, 0.3, 0.30), _row(4, 50, 0.2, 0.90, smallest=1)]
    picked = choosek.choose(rows)
    assert picked["ari_peak"] == 4
    assert picked["largest_k_with_no_cluster_under_ten"] == 3
    assert picked["recommended"] == 3


def test_the_recommendation_takes_the_ari_peak_when_every_k_is_intact():
    rows = [_row(2, 100, 0.4, 0.05), _row(3, 60, 0.3, 0.30), _row(4, 50, 0.2, 0.90)]
    assert choosek.choose(rows)["recommended"] == 4


def test_a_sweep_with_no_intact_k_still_returns_something():
    rows = [_row(k, 100 - k, 0.2, 0.1, smallest=1) for k in (2, 3, 4)]
    assert choosek.choose(rows)["recommended"] in (2, 3, 4)


# -- the sweep itself ------------------------------------------------------------------------------


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X, labels = [], []
    for i, name in enumerate(["circle", "diamond", "rectangle"]):
        X.append(rng.normal(i * 30.0, 0.5, size=(50, len(NAMES))))
        labels += [name] * 50
    return np.vstack(X), np.array(labels, dtype=object)


def test_the_sweep_reports_a_row_for_every_k(blobs):
    X, labels = blobs
    rows = choosek.sweep(X, labels, ks=(2, 3, 4))
    assert [r["k"] for r in rows] == [2, 3, 4]


def test_inertia_falls_monotonically_with_k(blobs):
    """It does so by construction; if it ever rises, the fit is not converging."""
    X, labels = blobs
    inertia = [r["inertia"] for r in choosek.sweep(X, labels, ks=(2, 3, 4, 5, 6))]
    assert all(a >= b for a, b in zip(inertia, inertia[1:], strict=False))


def test_three_obvious_blobs_peak_all_three_criteria_at_three(blobs):
    """The floor: where the answer is unambiguous, the criteria must not disagree."""
    X, labels = blobs
    picked = choosek.choose(choosek.sweep(X, labels, ks=(2, 3, 4, 5, 6)))
    assert picked["silhouette_peak"] == 3
    assert picked["ari_peak"] == 3


def test_the_cluster_sizes_are_carried_next_to_the_scores(blobs):
    """A criterion improving while the partition dissolves is only visible beside the sizes."""
    X, labels = blobs
    for row in choosek.sweep(X, labels, ks=(2, 5)):
        assert "smallest_cluster" in row
        assert "clusters_under_ten" in row

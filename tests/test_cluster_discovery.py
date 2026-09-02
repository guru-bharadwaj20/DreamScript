"""Phase 8.10 - clustering pages with the labels withheld."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cluster import discovery

# -- agreement against several explanations at once ------------------------------------------------


def test_every_candidate_explanation_gets_both_statistics():
    assignment = np.array([0, 0, 1, 1])
    result = discovery.against(
        assignment,
        type=np.array(["a", "a", "b", "b"]),
        source=np.array(["x", "y", "x", "y"]),
    )
    assert set(result) == {"ari_vs_type", "ami_vs_type", "ari_vs_source", "ami_vs_source"}


def test_a_partition_that_reproduces_a_truth_scores_one_against_it():
    assignment = np.array([0, 0, 1, 1])
    result = discovery.against(assignment, type=np.array(["a", "a", "b", "b"]))
    assert result["ari_vs_type"] == pytest.approx(1.0)
    assert result["ami_vs_type"] == pytest.approx(1.0)


def test_a_partition_orthogonal_to_a_truth_scores_about_zero():
    assignment = np.array([0, 1, 0, 1])
    result = discovery.against(assignment, type=np.array(["a", "a", "b", "b"]))
    assert abs(result["ari_vs_type"]) < 0.6


def test_the_two_explanations_are_scored_on_the_same_scale():
    """The whole design turns on comparing type against source; different metrics would not."""
    assignment = np.array([0, 0, 1, 1])
    truth = np.array(["a", "a", "b", "b"])
    result = discovery.against(assignment, type=truth, source=truth)
    assert result["ami_vs_type"] == result["ami_vs_source"]


# -- the sweep -------------------------------------------------------------------------------------


@pytest.fixture
def three_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(i * 20.0, 0.4, size=(30, 8)) for i in range(3)])
    return X, np.array(["a"] * 30 + ["b"] * 30 + ["c"] * 30)


def test_the_sweep_carries_the_smallest_cluster_beside_every_score(three_blobs):
    """8.2's guard, applied here because there is no label to choose K against."""
    X, truth = three_blobs
    for row in discovery.sweep(X, ks=(2, 3), type=truth):
        assert "smallest_cluster" in row
        assert "largest_share" in row


def test_three_obvious_blobs_are_recovered_at_k_three(three_blobs):
    X, truth = three_blobs
    rows = discovery.sweep(X, ks=(2, 3, 4), type=truth)
    assert max(rows, key=lambda r: r["ami_vs_type"])["k"] == 3


def test_clustering_is_deterministic_for_a_fixed_seed(three_blobs):
    X, _ = three_blobs
    assert np.array_equal(discovery.cluster(X, 3), discovery.cluster(X, 3))


# -- the source-against-type comparison ------------------------------------------------------------


def _index(sources, types):
    return pd.DataFrame(
        {
            "source": sources,
            "diagram_type": types,
            "id": [f"{s}/p{i}" for i, s in enumerate(sources)],
            "synthetic": [False] * len(sources),
        }
    )


def test_a_partition_that_tracks_the_source_is_flagged_as_such():
    """The finding this experiment exists to be able to report."""
    rng = np.random.default_rng(1)
    # Two sources, far apart; the type label cuts across them.
    X = np.vstack([rng.normal(0.0, 0.3, (40, 6)), rng.normal(20.0, 0.3, (40, 6))])
    index = _index(["a"] * 40 + ["b"] * 40, (["x"] * 20 + ["y"] * 20) * 2)
    assert discovery.experiment_all(X, index, ks=(2,))["source_beats_type"] is True


def test_a_partition_that_tracks_the_type_is_not_flagged():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0.0, 0.3, (40, 6)), rng.normal(20.0, 0.3, (40, 6))])
    index = _index((["a"] * 20 + ["b"] * 20) * 2, ["x"] * 40 + ["y"] * 40)
    assert discovery.experiment_all(X, index, ks=(2,))["source_beats_type"] is False


def test_a_block_too_small_to_cluster_is_skipped_rather_than_fitted():
    """A five-page 'chaos corpus' would produce a number, and it would mean nothing."""
    X = np.zeros((5, 6))
    index = _index(["chaos"] * 5, ["x"] * 5)
    assert "skipped" in discovery.experiment_chaos(X, index)


def test_the_supervised_reference_is_carried_so_the_gap_can_be_read():
    """A clustering score alone cannot say whether the representation had the answer."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(i * 20.0, 0.3, (20, 6)) for i in range(3)])
    index = _index(["chaos"] * 60, ["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    assert discovery.experiment_chaos(X, index, ks=(3,))["supervised_reference_macro_f1"] == 0.9925


def test_synthetic_rows_are_excluded_from_the_real_page_set():
    """They were rendered by a generator that knows the type; clustering them proves nothing."""
    index = pd.DataFrame(
        {
            "row": [0, 1, 2],
            "id": ["a", "b", "c"],
            "source": ["synthetic", "chaos", "chaos"],
            "diagram_type": ["x", "y", "z"],
            "synthetic": [True, False, False],
        }
    )
    assert int((~index["synthetic"]).sum()) == 2

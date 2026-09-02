"""Phase 8.1 - K-means over the shape descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from src.cluster import kmeans
from src.features.descriptors import NAMES

# -- the declared design ---------------------------------------------------------------------------


def test_both_ks_the_project_has_argued_for_are_swept():
    """5 is 4.1.3's declared vocabulary, 6 is 7.4.3's measured elbow."""
    assert set(kmeans.KS) == {5, 6}


def test_the_sweep_includes_a_random_initialisation():
    """Without it, 'k-means++ helps' would be an assumption rather than a measurement."""
    assert "random" in kmeans.INITS
    assert "k-means++" in kmeans.INITS


def test_restarts_are_not_one():
    """A single restart measures one draw, not the algorithm."""
    assert kmeans.RESTARTS > 1


# -- the fit ---------------------------------------------------------------------------------------


@pytest.fixture
def blobs():
    """Three well-separated clouds over the real descriptor width."""
    rng = np.random.default_rng(0)
    X, labels = [], []
    for i, name in enumerate(["circle", "diamond", "rectangle"]):
        centre = np.full(len(NAMES), i * 30.0)
        X.append(rng.normal(centre, 0.5, size=(40, len(NAMES))))
        labels += [name] * 40
    return np.vstack(X), np.array(labels, dtype=object)


def test_the_scaler_travels_with_the_model(blobs):
    """K-means minimises Euclidean distance, so an unscaled column would own the partition."""
    X, _ = blobs
    model = kmeans.fit(X, k=3)
    assert "scale" in model.named_steps
    assert "kmeans" in model.named_steps


def test_separated_blobs_are_recovered(blobs):
    """A floor: if the fit cannot find three obvious clouds, no corpus number means anything."""
    from sklearn.metrics import adjusted_rand_score

    X, labels = blobs
    model = kmeans.fit(X, k=3)
    assert adjusted_rand_score(labels, model.named_steps["kmeans"].labels_) > 0.95


def test_scaling_is_actually_applied_and_changes_the_answer():
    """One column a thousand times wider than the rest must not decide the partition alone."""
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(1)
    # Everything but two columns is held constant, so that standardisation cannot turn a wall of
    # narrow noise columns into a wall of unit-width ones and bury the structure either way.
    X = np.zeros((200, len(NAMES)))
    X[:, 0] = rng.normal(0, 0.1, size=200)
    X[:100, 0] += 2.0  # the real structure, in a normal-width column
    X[:, 1] = rng.normal(0, 1000, size=200)  # noise, ten thousand times wider

    scaled = kmeans.fit(X, k=2).named_steps["kmeans"].labels_
    raw = KMeans(n_clusters=2, n_init=10, random_state=kmeans.SEED).fit_predict(X)

    truth = np.array([0] * 100 + [1] * 100)
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(truth, scaled) > adjusted_rand_score(truth, raw)


# -- the reported quantities -----------------------------------------------------------------------


def test_evaluate_reports_every_yardstick_the_docstring_names(blobs):
    X, labels = blobs
    result = kmeans.evaluate(X, labels, k=3, init="k-means++")
    for key in ("ari", "purity", "silhouette", "inertia", "smallest_cluster", "labels_claimed"):
        assert key in result


def test_a_perfect_partition_scores_ari_one():
    labels = np.array(["a"] * 20 + ["b"] * 20, dtype=object)
    rng = np.random.default_rng(2)
    X = np.vstack(
        [rng.normal(0, 0.01, size=(20, len(NAMES))), rng.normal(50, 0.01, size=(20, len(NAMES)))]
    )
    assert kmeans.evaluate(X, labels, k=2, init="k-means++")["ari"] > 0.99


def test_cluster_means_are_in_the_original_units(blobs):
    """Scaled means are what the algorithm optimised; unscaled means are what a reader can check."""
    X, _ = blobs
    assignment = kmeans.fit(X, k=3).named_steps["kmeans"].labels_
    means = kmeans.cluster_means(X, assignment, 3)
    centres = sorted(round(m[NAMES[0]]) for m in means)
    assert centres == [0, 30, 60]


def test_cluster_means_survive_an_empty_cluster():
    """K-means can return fewer than k occupied clusters; the row must still exist."""
    assignment = np.array([0, 0, 2, 2])
    X = np.zeros((4, len(NAMES)))
    means = kmeans.cluster_means(X, assignment, 3)
    assert [m["cluster"] for m in means] == [0, 1, 2]
    assert means[1]["size"] == 0


def test_the_label_mix_names_the_dominant_label_and_its_share():
    assignment = np.array([0, 0, 0, 1])
    labels = np.array(["rectangle", "rectangle", "circle", "diamond"], dtype=object)
    mix = kmeans.label_mix(assignment, labels, 2)
    assert mix[0]["dominant"] == "rectangle"
    assert mix[0]["dominant_share"] == 0.6667
    assert mix[1]["mix"] == {"diamond": 1}


def test_the_majority_baseline_is_the_largest_class_share():
    labels = np.array(["rectangle"] * 7 + ["circle"] * 3, dtype=object)
    assert kmeans.majority_baseline(labels) == 0.7


# -- the seed control ------------------------------------------------------------------------------


def test_the_seed_spread_compares_runs_to_each_other_not_only_to_the_labels(blobs):
    """Two runs can each score the same ARI against the truth and still disagree with each other."""
    X, labels = blobs
    spread = kmeans.seed_spread(X, labels, k=3, seeds=range(3))
    assert "agreement_between_seeds" in spread
    assert spread["agreement_between_seeds"]["mean"] == pytest.approx(1.0, abs=1e-6)


def test_the_same_seed_gives_the_same_partition(blobs):
    X, _ = blobs
    a = kmeans.fit(X, k=3, seed=7).named_steps["kmeans"].labels_
    b = kmeans.fit(X, k=3, seed=7).named_steps["kmeans"].labels_
    assert np.array_equal(a, b)

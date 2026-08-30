"""Phase 5.1.2 - k-nearest neighbours, CPU and GPU."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src.classify import gpu, knn
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(1)
    blocks = [rng.normal(centre, 0.7, size=(50, 5)) for centre in (0.0, 3.5, 7.0)]
    X = np.vstack(blocks)
    y = np.repeat(["a", "b", "c"], 50).astype(object)
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=[f"f{i}" for i in range(5)],
        corpus="test",
    )


def test_the_sweep_covers_the_full_grid():
    assert len(knn.K_VALUES) * len(knn.METRICS) * len(knn.WEIGHTINGS) == 36


def test_the_selected_neighbour_rule_is_recorded():
    model = knn.best_estimator().named_steps["model"]
    assert model.n_neighbors == knn.BEST_PARAMS["n_neighbors"]
    assert model.metric == knn.BEST_PARAMS["metric"]


def test_it_fits_and_predicts_through_the_scaler(toy):
    fitted = knn.best_estimator().fit(toy.X, toy.y)
    assert fitted.score(toy.X, toy.y) > 0.9
    assert [name for name, _ in fitted.steps] == ["prepare", "model"]


def test_the_cpu_sweep_ranks_every_configuration(toy):
    result = knn.sweep_cpu(toy, folds=3, n_jobs=2)
    assert result["configurations"] == 36
    assert result["best"]["macro_f1"] == max(row["macro_f1"] for row in result["all"])


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device")
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_gpu_neighbours_agree_with_sklearn(toy, metric, weights):
    """Including the two rules that are easy to get subtly wrong: exact matches and ties."""
    classes = np.unique(toy.y)
    encoded = np.searchsorted(classes, toy.y)
    reference = KNeighborsClassifier(n_neighbors=5, metric=metric, weights=weights).fit(
        toy.X, encoded
    )
    predicted, _ = gpu.knn_predict(toy.X, toy.y, toy.X, k=5, metric=metric, weights=weights)
    assert (predicted == classes[reference.predict(toy.X)]).mean() == 1.0


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device")
def test_probabilities_are_a_distribution(toy):
    _, probability = gpu.knn_predict(toy.X, toy.y, toy.X, k=7)
    assert probability.shape == (len(toy.y), 3)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device")
def test_one_distance_matrix_serves_every_k(toy):
    """The reason the GPU path exists: k changes, the neighbours do not have to be recomputed."""
    distances = gpu.pairwise(toy.X, toy.X, "euclidean")
    for k in (1, 3, 11):
        shared, _ = gpu.knn_predict(toy.X, toy.y, toy.X, k=k, distances=distances)
        fresh, _ = gpu.knn_predict(toy.X, toy.y, toy.X, k=k, metric="euclidean")
        assert (shared == fresh).all()


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device")
def test_both_sweeps_reach_the_same_ranking(toy):
    cpu = knn.sweep_cpu(toy, folds=3, n_jobs=2)
    device = knn.sweep_gpu(toy, folds=3)
    report = knn.agreement(cpu, device)
    assert report["configurations_compared"] == 36
    assert report["max_macro_f1_difference"] < 1e-9
    assert report["identical_best"]


def test_an_unknown_metric_is_rejected(toy):
    if not gpu.available():
        pytest.skip("no CUDA device")
    with pytest.raises(ValueError):
        gpu.pairwise(toy.X, toy.X, "mahalanobis")


def test_the_gpu_sweep_reports_unavailability_rather_than_failing(toy, monkeypatch):
    monkeypatch.setattr(gpu, "available", lambda: False)
    assert knn.sweep_gpu(toy, folds=2) == {"available": False}
    assert knn.agreement({"all": [], "best": {}, "seconds": 1}, {"available": False}) == {
        "available": False
    }

"""Phase 5.1.1 - logistic regression and its sweep."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.classify import gpu, linear
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """Three well-separated classes with a hole in one column, so the imputer is exercised."""
    rng = np.random.default_rng(0)
    blocks = [rng.normal(centre, 0.6, size=(40, 6)) for centre in (0.0, 4.0, 8.0)]
    X = np.vstack(blocks)
    X[::17, 1] = np.nan
    y = np.repeat(["a", "b", "c"], 40).astype(object)
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=[f"f{i}" for i in range(6)],
        corpus="test",
    )


def test_the_pipeline_scales_before_it_fits(toy):
    """4.2.4's leakage argument only holds if the scaler is inside the estimator."""
    steps = [name for name, _ in linear.pipeline().steps]
    assert steps == ["prepare", "model"]
    assert isinstance(linear.pipeline(), Pipeline)


def test_it_fits_a_table_that_still_has_holes(toy):
    fitted = linear.best_estimator().fit(toy.X, toy.y)
    assert set(fitted.classes_) == {"a", "b", "c"}
    assert fitted.score(toy.X, toy.y) > 0.9


def test_the_selected_configuration_is_recorded_not_reinvented():
    """5.2 and 5.3 must fit the identical model, so the choice lives in one constant."""
    assert set(linear.BEST_PARAMS) == {"penalty", "solver", "C", "class_weight"}
    model = linear.best_estimator().named_steps["model"]
    assert model.penalty == linear.BEST_PARAMS["penalty"]
    assert linear.BEST_PARAMS["C"] == model.C


def test_keyword_arguments_override_the_selected_configuration():
    model = linear.best_estimator(C=0.5).named_steps["model"]
    assert model.C == 0.5


def test_the_grid_covers_all_three_penalties():
    penalties = {p for block in linear.GRID for p in block["model__penalty"]}
    assert penalties == {"l1", "l2", "elasticnet"}
    assert all("model__C" in block for block in linear.GRID)


def test_the_sweep_scores_by_macro_f1_and_returns_a_ranking(toy):
    """Accuracy would let a model that never predicts the minority class win."""
    small = [{"model__penalty": ["l2"], "model__solver": ["lbfgs"], "model__C": [0.1, 1.0]}]
    original, linear.GRID = linear.GRID, small
    try:
        result = linear.sweep(toy, n_jobs=2, folds=3)
    finally:
        linear.GRID = original
    assert result["candidates"] == 2
    assert result["best"]["macro_f1"] == max(row["macro_f1"] for row in result["top5"])
    assert 0.0 <= result["best"]["macro_f1"] <= 1.0


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device")
def test_the_gpu_solver_agrees_with_sklearn(toy):
    """A GPU path that gives a different answer is not an acceleration."""
    from src.features.scaling import feature_scaler

    prepared = feature_scaler().fit_transform(toy.X)
    reference = LogisticRegression(C=1.0, max_iter=5000).fit(prepared, toy.y)
    coef, intercept = gpu.logreg_fit(prepared, toy.y, C=1.0)
    predicted = np.unique(toy.y)[gpu.logreg_decision(prepared, coef, intercept).argmax(axis=1)]
    assert (predicted == reference.predict(prepared)).mean() == 1.0


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device")
def test_the_gpu_report_is_honest_about_speed(toy):
    report = linear.gpu_agreement(toy)
    assert report["available"] is True
    assert report["prediction_agreement"] == 1.0
    assert report["cpu_seconds"] > 0 and report["gpu_seconds"] > 0


def test_the_gpu_report_degrades_cleanly_without_a_device(toy, monkeypatch):
    monkeypatch.setattr(gpu, "available", lambda: False)
    assert linear.gpu_agreement(toy) == {"available": False}

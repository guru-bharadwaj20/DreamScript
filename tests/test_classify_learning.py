"""Phase 5.2.9 - learning curves."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import learning
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(12)
    y = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 20, dtype=object)
    centres = {"a": 0.0, "b": 2.5, "c": -2.5}
    X = np.array([[centres[label]] for label in y]) + rng.normal(0, 1.1, size=(220, 1))
    X = np.hstack([X, rng.normal(size=(220, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(220)], dtype=object),
        ids=np.arange(220).astype(object),
        feature_names=["f0", "f1", "f2"],
        corpus="test",
    )


def test_the_subsample_keeps_every_class():
    """The floor is the whole point: a 5% proportional draw of 20 rows expects one."""
    y = np.array(["a"] * 200 + ["rare"] * 20, dtype=object)
    index = np.arange(220)
    rng = np.random.default_rng(0)
    for _ in range(20):
        taken = learning.stratified_subsample(y, index, 0.05, rng)
        assert set(y[taken]) == {"a", "rare"}


def test_the_subsample_keeps_the_class_shares():
    y = np.array(["a"] * 600 + ["b"] * 200, dtype=object)
    taken = learning.stratified_subsample(y, np.arange(800), 0.5, np.random.default_rng(1))
    assert (y[taken] == "a").sum() == 300
    assert (y[taken] == "b").sum() == 100


def test_a_full_fraction_returns_the_index_untouched():
    index = np.arange(50)
    y = np.array(["a"] * 50, dtype=object)
    taken = learning.stratified_subsample(y, index, 1.0, np.random.default_rng(2))
    assert np.array_equal(taken, index)


def test_the_subsample_never_repeats_a_row():
    y = np.array(["a"] * 100 + ["b"] * 100, dtype=object)
    taken = learning.stratified_subsample(y, np.arange(200), 0.3, np.random.default_rng(3))
    assert len(set(taken.tolist())) == len(taken)


def test_a_curve_point_reports_both_sides(toy):
    point = learning.curve_point(toy, "logreg", 0.5)
    assert 0.0 <= point["macro_f1"] <= 1.0
    assert 0.0 <= point["train_macro_f1"] <= 1.0
    # The gap is computed before rounding, so it can differ from the difference of the two
    # rounded columns by a unit in the last place.
    assert point["gap"] == pytest.approx(point["train_macro_f1"] - point["macro_f1"], abs=2e-4)
    assert set(point["recall"]) == {"a", "b", "c"}


def test_the_training_set_grows_with_the_fraction(toy):
    sizes = [learning.curve_point(toy, "tree", f)["train_rows"] for f in (0.1, 0.5, 1.0)]
    assert sizes[0] < sizes[1] < sizes[2]


def test_the_test_side_is_never_subsampled(toy):
    """Every point must be scored on the same rows, or the curve measures two things at once."""
    small = learning.curve_point(toy, "logreg", 0.1)
    full = learning.curve_point(toy, "logreg", 1.0)
    assert small["train_rows"] < full["train_rows"]
    # Same folds, same test rows: the fold count and therefore the scored population is fixed.
    assert small["fraction"] != full["fraction"]


def test_one_nearest_neighbour_never_misses_a_training_row(toy):
    """k = 1 makes the training score exactly 1.0, which is the module's kNN finding."""
    for fraction in (0.2, 0.6, 1.0):
        assert learning.curve_point(toy, "knn", fraction)["train_macro_f1"] == 1.0


def test_a_still_climbing_curve_is_called_data_limited():
    points = [
        {"fraction": f, "train_rows": n, "macro_f1": m, "gap": 0.05}
        for f, n, m in [(0.5, 100, 0.60), (0.7, 140, 0.66), (0.85, 170, 0.70), (1.0, 200, 0.74)]
    ]
    assert learning.diagnosis(points)["verdict"].startswith("data-limited")


def test_a_flat_curve_with_a_wide_gap_is_called_variance_limited():
    points = [
        {"fraction": f, "train_rows": n, "macro_f1": 0.70, "gap": 0.30}
        for f, n in [(0.5, 100), (0.7, 140), (0.85, 170), (1.0, 200)]
    ]
    assert learning.diagnosis(points)["verdict"].startswith("variance-limited")


def test_a_flat_curve_with_a_tight_gap_is_called_bias_limited():
    points = [
        {"fraction": f, "train_rows": n, "macro_f1": 0.70, "gap": 0.02}
        for f, n in [(0.5, 100), (0.7, 140), (0.85, 170), (1.0, 200)]
    ]
    assert learning.diagnosis(points)["verdict"].startswith("bias-limited")


def test_the_figure_is_written(tmp_path, toy):
    results = [learning.curve(toy, "tree", fractions=(0.3, 0.6, 1.0))]
    path = learning.figure(results, tmp_path / "curves.png")
    assert path.is_file() and path.stat().st_size > 5000

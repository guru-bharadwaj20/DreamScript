"""Phase 7.1.4 - four gradient boosters on one conceptual grid, and the label adapter they need."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import boosting
from src.classify.data import Dataset
from src.classify.labels import LabelSafe


@pytest.fixture
def toy():
    rng = np.random.default_rng(66)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (1.5, 2.5)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.9, size=(140, 2))
    X = np.hstack([X, rng.normal(size=(140, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(140)], dtype=object),
        ids=np.arange(140).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


SMALL = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 20,
    "subsample": 1.0,
    "colsample": 1.0,
}


# -- one conceptual grid, four parameter vocabularies --------------------------------------------


def test_both_sklearn_implementations_are_compared():
    """They are different algorithms, not one library - exact splits against histogram bins."""
    assert "sklearn_gb" in boosting.LIBRARIES
    assert "sklearn_hist" in boosting.LIBRARIES


def test_the_shared_grid_is_the_full_product():
    assert len(boosting.grid()) == (
        len(boosting.RATES)
        * len(boosting.DEPTHS)
        * len(boosting.ROUNDS)
        * len(boosting.SUBSAMPLE)
        * len(boosting.COLSAMPLE)
    )


def test_the_learning_rate_grid_brackets_every_librarys_default():
    """xgboost defaults to 0.3 and the others to 0.1; a shared default would favour one."""
    assert min(boosting.RATES) <= 0.1 <= max(boosting.RATES)
    assert 0.3 in boosting.RATES


@pytest.mark.parametrize("name", ["sklearn_gb", "sklearn_hist", "xgboost", "lightgbm"])
def test_the_shared_grid_reaches_each_librarys_own_parameter_names(name):
    if not boosting.available(name):
        pytest.skip(f"{name} not installed")
    model = boosting.build(
        name, learning_rate=0.03, max_depth=6, n_estimators=42, subsample=0.7, colsample=0.7
    )
    params = model.get_params()
    assert params["learning_rate"] == 0.03
    assert params["max_depth"] == 6
    # `n_estimators` is `max_iter` in HistGradientBoosting - the translation this test checks.
    assert params.get("n_estimators", params.get("max_iter")) == 42


def test_an_unknown_library_is_refused_by_name():
    with pytest.raises(ValueError, match="library must be one of"):
        boosting.build("catboost", 0.1, 3, 10, 1.0, 1.0)


def test_lightgbms_min_child_samples_is_lowered_for_the_40_row_class():
    """The default of 20 would forbid almost every split that isolates the smallest class."""
    if not boosting.available("lightgbm"):
        pytest.skip("lightgbm not installed")
    model = boosting.build("lightgbm", 0.1, 3, 10, 1.0, 1.0)
    assert model.get_params()["min_child_samples"] < 20


# -- the pipeline ------------------------------------------------------------------------------------


def test_the_imputer_is_kept_even_though_trees_ignore_scale(toy):
    """25 of the 161 hybrid columns have missing values and `GradientBoosting` rejects NaN."""
    estimator = boosting.pipeline("sklearn_hist", **SMALL)
    assert list(estimator.named_steps) == ["prepare", "model"]


def test_xgboost_is_wrapped_so_it_accepts_string_labels(toy):
    if not boosting.available("xgboost"):
        pytest.skip("xgboost not installed")
    inner = boosting.pipeline("xgboost", **SMALL).named_steps["model"]
    assert isinstance(inner, LabelSafe)


@pytest.mark.parametrize("name", ["sklearn_hist", "xgboost", "lightgbm"])
def test_every_library_fits_and_returns_string_labels(toy, name):
    if not boosting.available(name):
        pytest.skip(f"{name} not installed")
    fitted = boosting.pipeline(name, **SMALL).fit(toy.X, toy.y)
    predicted = fitted.predict(toy.X)
    assert set(predicted) <= set(toy.y)


# -- the label adapter -------------------------------------------------------------------------------


def test_label_safe_round_trips_string_labels(toy):
    from sklearn.tree import DecisionTreeClassifier

    wrapped = LabelSafe(DecisionTreeClassifier(random_state=0)).fit(toy.X, toy.y)
    assert wrapped.classes_.tolist() == ["a", "b", "c"]
    assert set(wrapped.predict(toy.X)) <= {"a", "b", "c"}


def test_label_safe_keeps_probability_columns_in_classes_order(toy):
    """5.2.1's harness reads probability columns through `classes_`; a mismatch is invisible."""
    from sklearn.tree import DecisionTreeClassifier

    wrapped = LabelSafe(DecisionTreeClassifier(random_state=0)).fit(toy.X, toy.y)
    proba = wrapped.predict_proba(toy.X)
    assert proba.shape == (len(toy.y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (wrapped.classes_[proba.argmax(axis=1)] == wrapped.predict(toy.X)).all()


def test_label_safe_does_not_mutate_the_estimator_it_was_given(toy):
    """It is cloned in `fit`, so one wrapper is reusable across cross-validation folds."""
    from sklearn.tree import DecisionTreeClassifier

    inner = DecisionTreeClassifier(random_state=0)
    LabelSafe(inner).fit(toy.X, toy.y)
    assert not hasattr(inner, "tree_")


def test_label_safe_is_cloneable(toy):
    from sklearn.base import clone
    from sklearn.tree import DecisionTreeClassifier

    original = LabelSafe(DecisionTreeClassifier(max_depth=4))
    copy = clone(original)
    assert copy.get_params()["estimator"].max_depth == 4


# -- the sweep ----------------------------------------------------------------------------------------


def test_a_cell_records_the_library_and_its_cost(toy):
    row = boosting.evaluate(toy, "lightgbm", SMALL, folds=3, n_jobs=2)
    assert row["library"] == "lightgbm"
    assert row["seconds"] >= 0
    assert 0.0 <= row["macro_f1"] <= 1.0


def test_the_marginal_takes_the_best_at_each_level():
    rows = [
        {"learning_rate": 0.1, "macro_f1": 0.90},
        {"learning_rate": 0.1, "macro_f1": 0.10},
        {"learning_rate": 0.3, "macro_f1": 0.85},
    ]
    result = boosting.marginal(rows, "learning_rate")
    assert result["0.1"] == 0.90
    assert result["0.3"] == 0.85


def test_a_missing_library_is_skipped_rather_than_crashing_the_sweep(toy, monkeypatch):
    monkeypatch.setattr(boosting, "available", lambda name: name == "lightgbm")
    monkeypatch.setattr(boosting, "grid", lambda *a, **k: [SMALL])
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.embed.hybrid",
        type("m", (), {"dataset": staticmethod(lambda *a, **k: toy)})(),
    )
    result = boosting.run("hybrid", n_jobs=2)
    assert result["libraries"] == ["lightgbm"]
    assert set(result["skipped"]) == set(boosting.LIBRARIES) - {"lightgbm"}

"""Phase 5.2.1 - the cross-validation harness."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import cv
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """Two writers per class, so grouped and stratified splitting differ."""
    rng = np.random.default_rng(4)
    blocks = [rng.normal(centre, 0.7, size=(40, 5)) for centre in (0.0, 4.0, 8.0)]
    y = np.repeat(["a", "b", "c"], 40).astype(object)
    groups = np.array([f"scribe:{i % 6}" for i in range(120)], dtype=object)
    return Dataset(
        X=np.vstack(blocks),
        y=y,
        groups=groups,
        ids=np.arange(120).astype(object),
        feature_names=[f"f{i}" for i in range(5)],
        corpus="test",
    )


def test_the_registry_holds_the_three_models_and_three_baselines():
    assert set(cv.MODELS) == {"logreg", "knn", "tree", "majority", "stratified", "uniform"}
    assert set(cv.REAL_MODELS) <= set(cv.MODELS)


def test_an_unknown_model_is_rejected(toy):
    with pytest.raises(KeyError):
        cv.out_of_fold("random_forest", toy)


def test_every_row_gets_exactly_one_out_of_fold_prediction(toy):
    result = cv.out_of_fold("tree", toy, n_splits=4)
    assert len(result.y_pred) == len(toy.y)
    assert not any(value is None for value in result.y_pred)
    assert set(result.fold.tolist()) == {0, 1, 2, 3}


def test_probabilities_line_up_with_the_class_order(toy):
    result = cv.out_of_fold("logreg", toy, n_splits=4)
    assert list(result.classes) == ["a", "b", "c"]
    assert result.y_proba.shape == (len(toy.y), 3)
    assert np.allclose(result.y_proba.sum(axis=1), 1.0, atol=1e-6)
    # the argmax of the probabilities must be the prediction
    assert (result.classes[result.y_proba.argmax(axis=1)] == result.y_pred).mean() > 0.99


def test_a_prediction_never_comes_from_a_model_that_saw_the_row(toy):
    """Fitting on everything would score far higher; this is the check that it did not."""
    honest = cv.out_of_fold("knn", toy, n_splits=4).macro_f1()
    fitted_on_all = cv.MODELS["knn"]().fit(toy.X, toy.y)
    from sklearn.metrics import f1_score

    cheating = f1_score(toy.y, fitted_on_all.predict(toy.X), average="macro")
    assert cheating >= honest


def test_grouped_folds_never_split_a_writer(toy):
    for train, test in cv.splitter(toy, "grouped", seed=42, n_splits=3):
        assert not set(toy.groups[train]) & set(toy.groups[test])


def test_stratified_folds_keep_the_class_balance(toy):
    for _, test in cv.splitter(toy, "stratified", seed=42, n_splits=4):
        counts = np.unique(toy.y[test], return_counts=True)[1]
        assert counts.max() - counts.min() <= 1


def test_an_unknown_strategy_is_an_error(toy):
    with pytest.raises(ValueError):
        list(cv.splitter(toy, "leave_one_out", seed=42))


def test_repeats_use_different_partitions(toy):
    first = cv.out_of_fold("tree", toy, seed=42, n_splits=4)
    second = cv.out_of_fold("tree", toy, seed=43, n_splits=4)
    assert not np.array_equal(first.fold, second.fold)


def test_the_same_seed_gives_the_identical_result(toy):
    first = cv.out_of_fold("tree", toy, seed=42, n_splits=4)
    second = cv.out_of_fold("tree", toy, seed=42, n_splits=4)
    assert np.array_equal(first.fold, second.fold)
    assert (first.y_pred == second.y_pred).all()


def test_evaluate_reports_one_row_per_fit(toy):
    result = cv.evaluate("tree", toy, seeds=(42, 43), n_splits=3)
    assert result["fits"] == 6
    assert len(result["per_fold"]) == 6
    assert len(result["macro_f1_by_repeat"]) == 2
    assert 0.0 <= result["macro_f1"] <= 1.0

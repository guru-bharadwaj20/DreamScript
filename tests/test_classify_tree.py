"""Phase 5.1.3 - the decision tree and its pruning path."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import tree
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """Three classes separated on one column, with noise columns and a hole."""
    rng = np.random.default_rng(2)
    signal = np.repeat([0.0, 5.0, 10.0], 60)[:, None]
    noise = rng.normal(size=(180, 4))
    X = np.hstack([signal + rng.normal(0, 0.6, size=(180, 1)), noise])
    X[::23, 2] = np.nan
    y = np.repeat(["a", "b", "c"], 60).astype(object)
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=[f"f{i}" for i in range(5)],
        corpus="test",
    )


def test_the_pipeline_imputes_so_the_tree_can_split(toy):
    """A tree cannot split on nan, and 4.2.3 measured 63.5% of real rows carrying one."""
    fitted = tree.best_estimator().fit(toy.X, toy.y)
    assert fitted.score(toy.X, toy.y) > 0.9


def test_standardising_cannot_change_a_split(toy):
    """It is monotone per column, so the tree is identical with or without it."""
    from sklearn.tree import DecisionTreeClassifier

    from src.features.impute import MedianImputer

    filled = MedianImputer().fit_transform(toy.X)
    params = dict(tree.BEST_PARAMS)
    plain = DecisionTreeClassifier(random_state=tree.SEED, **params).fit(filled, toy.y)
    scaled = tree.best_estimator().fit(toy.X, toy.y).named_steps["model"]
    assert plain.get_n_leaves() == scaled.get_n_leaves()


def test_the_selected_configuration_is_recorded():
    model = tree.best_estimator().named_steps["model"]
    assert model.criterion == tree.BEST_PARAMS["criterion"]
    assert model.ccp_alpha == tree.BEST_PARAMS["ccp_alpha"]
    assert tree.BEST_PARAMS["ccp_alpha"] > 0, "the selected tree is a pruned one"


def test_pruning_shrinks_the_tree_monotonically(toy):
    result = tree.pruning_path(toy, {"criterion": "entropy", "max_depth": None}, n_jobs=2, folds=3)
    nodes = [row["nodes"] for row in result["alphas"]]
    assert nodes == sorted(nodes, reverse=True)
    assert result["alphas"][0]["alpha"] == 0.0


def test_the_one_standard_error_tree_is_never_larger_than_the_best(toy):
    result = tree.pruning_path(toy, {"criterion": "entropy", "max_depth": None}, n_jobs=2, folds=3)
    assert result["one_standard_error"]["nodes"] <= result["best"]["nodes"]


def test_the_shape_sweep_ranks_by_macro_f1(toy):
    original, tree.GRID = tree.GRID, {"model__criterion": ["gini", "entropy"]}
    try:
        result = tree.sweep_shape(toy, n_jobs=2, folds=3)
    finally:
        tree.GRID = original
    assert result["candidates"] == 2
    assert result["best_macro_f1"] == max(row["macro_f1"] for row in result["top5"])


def test_the_grid_offers_both_criteria_and_a_class_weight():
    assert set(tree.GRID["model__criterion"]) == {"gini", "entropy"}
    assert None in tree.GRID["model__class_weight"]
    assert "balanced" in tree.GRID["model__class_weight"]


def test_the_tree_is_deterministic(toy):
    first = tree.best_estimator().fit(toy.X, toy.y).named_steps["model"]
    second = tree.best_estimator().fit(toy.X, toy.y).named_steps["model"]
    assert first.get_n_leaves() == second.get_n_leaves()
    assert (first.feature_importances_ == second.feature_importances_).all()

"""Phase 6.3.3 - the RBF grid, and the two degeneracies its diagnostics have to name."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import rbf
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(58)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (1.5, 3.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.7, size=(140, 2))
    X = np.hstack([X, rng.normal(size=(140, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(140)], dtype=object),
        ids=np.arange(140).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


# -- the grid ------------------------------------------------------------------------------------


def test_gamma_spans_six_orders_of_magnitude():
    """The extremes are in the grid to be diagnosed, not because they are candidates."""
    numeric = [value for value in rbf.GAMMAS if value != "scale"]
    assert min(numeric) == 1e-5
    assert max(numeric) == 1.0


def test_the_adaptive_default_is_in_the_grid():
    assert "scale" in rbf.GAMMAS


def test_the_grid_is_the_full_product():
    assert len(rbf.grid()) == len(rbf.GAMMAS) * len(rbf.CS)


def test_the_kernel_parameters_reach_the_estimator():
    model = rbf.pipeline(gamma=0.01, C=100.0).named_steps["model"]
    assert model.kernel == "rbf"
    assert (model.gamma, model.C) == (0.01, 100.0)


def test_the_linear_baseline_is_recorded_for_every_table():
    from src.classify.activations import TABLES

    assert set(rbf.LINEAR) == set(TABLES)


# -- the diagnostics -------------------------------------------------------------------------------


def test_a_memorising_kernel_is_named_as_one():
    row = {"distinct_predictions": 5, "support_vector_share": 1.0, "train_macro_f1": 1.0}
    assert rbf.diagnose(row, 5) == "memorised: every row a support vector"


def test_a_collapsed_kernel_is_named_as_one():
    row = {"distinct_predictions": 1, "support_vector_share": 0.4, "train_macro_f1": 0.2}
    assert rbf.diagnose(row, 5) == "collapsed to one class"


def test_a_kernel_missing_some_classes_is_named():
    row = {"distinct_predictions": 3, "support_vector_share": 0.4, "train_macro_f1": 0.8}
    assert rbf.diagnose(row, 5) == "predicts only 3 of 5 classes"


def test_a_healthy_cell_is_not_flagged():
    row = {"distinct_predictions": 5, "support_vector_share": 0.27, "train_macro_f1": 0.99}
    assert rbf.diagnose(row, 5) == "ok"


def test_every_row_is_a_support_vector_without_memorising_is_still_flagged():
    """The handcrafted table's worst cell: 99.4% support vectors and a training score of 0.20."""
    row = {"distinct_predictions": 4, "support_vector_share": 0.994, "train_macro_f1": 0.1985}
    assert rbf.diagnose(row, 5) == "every row a support vector"


# -- the two degeneracies, on real fits ---------------------------------------------------------------


def test_a_large_gamma_makes_every_row_a_support_vector(toy):
    """The predicted signature, and the one the worst cell of every table actually shows."""
    row = rbf.score_cell(toy, {"gamma": 100.0, "C": 1.0}, n_jobs=2)
    assert row["support_vector_share"] > 0.95
    assert row["train_macro_f1"] > 0.99
    assert row["macro_f1"] < row["train_macro_f1"] - 0.2


def test_a_tiny_gamma_agrees_with_the_linear_model_more_than_a_large_one_does(toy):
    """ "Degenerates toward a linear model" is a claim about the decision function, so it is
    checked on the predictions rather than on the score - two models can share a macro F1
    without agreeing on a single row."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from src.classify.svm import pipeline as linear_pipeline

    splitter = StratifiedKFold(5, shuffle=True, random_state=rbf.SEED)
    linear = cross_val_predict(linear_pipeline(C=1.0), toy.X, toy.y, cv=splitter, n_jobs=2)

    def agreement(gamma, C):
        predicted = cross_val_predict(
            rbf.pipeline(gamma=gamma, C=C), toy.X, toy.y, cv=splitter, n_jobs=2
        )
        return float((predicted == linear).mean())

    assert agreement(1e-6, 1e6) > agreement(10.0, 1.0)


def test_a_cell_carries_both_diagnostics_beside_its_score(toy):
    row = rbf.score_cell(toy, {"gamma": "scale", "C": 1.0}, n_jobs=2)
    assert {"macro_f1", "train_macro_f1", "support_vector_share", "distinct_predictions"} <= set(
        row
    )
    assert 0.0 <= row["support_vector_share"] <= 1.0
    assert row["train_macro_f1"] >= row["macro_f1"] - 0.3

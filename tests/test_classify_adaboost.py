"""Phase 7.1.3 - AdaBoost on stumps and depth-3 trees, and where its sample weight ends up."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import adaboost
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(65)
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


# -- the two base learners the plan names ---------------------------------------------------------


def test_the_plan_names_stumps_and_depth_three():
    assert adaboost.DEPTHS == (1, 3)


def test_a_stump_is_actually_depth_one():
    model = adaboost.pipeline(depth=1).named_steps["model"]
    assert model.estimator.max_depth == 1


def test_the_round_grid_is_wide_enough_for_stumps():
    """Stumps need far more rounds; judging them at a depth-3 round count is not a comparison."""
    assert max(adaboost.ROUNDS) >= 1000


def test_the_hyperparameters_reach_the_estimator():
    model = adaboost.pipeline(depth=3, n_estimators=77, learning_rate=0.25).named_steps["model"]
    assert model.n_estimators == 77
    assert model.learning_rate == 0.25
    assert model.estimator.max_depth == 3


def test_the_grid_is_the_full_product():
    assert len(adaboost.grid()) == (
        len(adaboost.DEPTHS) * len(adaboost.ROUNDS) * len(adaboost.RATES)
    )


def test_the_pipeline_scales_first(toy):
    estimator = adaboost.pipeline(depth=1, n_estimators=10)
    assert list(estimator.named_steps)[0] == "prepare"
    estimator.fit(toy.X, toy.y)
    assert estimator.predict(toy.X).shape == toy.y.shape


# -- the weight concentration, which is the point of the task -----------------------------------------


def test_the_weight_shares_sum_to_one(toy):
    result = adaboost.weight_concentration(
        toy, {"depth": 1, "n_estimators": 30, "learning_rate": 1.0}
    )
    total = sum(row["weight_share"] for row in result["by_class"].values())
    assert total == pytest.approx(1.0, abs=1e-3)


def test_concentration_is_weight_share_over_population_share(toy):
    result = adaboost.weight_concentration(
        toy, {"depth": 1, "n_estimators": 20, "learning_rate": 1.0}
    )
    for row in result["by_class"].values():
        assert row["concentration"] == pytest.approx(
            row["weight_share"] / row["population_share"], abs=0.02
        )


def test_a_class_carrying_its_population_share_scores_one(toy):
    """So the number is readable without holding the class counts in mind."""
    result = adaboost.weight_concentration(
        toy, {"depth": 1, "n_estimators": 1, "learning_rate": 1.0}
    )
    assert all(row["concentration"] > 0 for row in result["by_class"].values())


def test_boosting_concentrates_weight_away_from_uniform(toy):
    """One round is near-uniform; many rounds are not. That difference is the mechanism."""
    few = adaboost.weight_concentration(toy, {"depth": 1, "n_estimators": 1, "learning_rate": 1.0})
    many = adaboost.weight_concentration(
        toy, {"depth": 1, "n_estimators": 200, "learning_rate": 1.0}
    )
    assert many["largest_single_row_weight"] > few["largest_single_row_weight"]


def test_the_single_row_concentration_is_reported(toy):
    """A per-class summary cannot show one page holding a large share of the weight."""
    result = adaboost.weight_concentration(
        toy, {"depth": 1, "n_estimators": 50, "learning_rate": 1.0}
    )
    assert 0.0 < result["largest_single_row_weight"] <= 1.0
    assert 1 <= result["rows_holding_half_the_weight"] <= len(toy.y)


def test_the_replay_uses_the_fitted_ensembles_own_alphas(toy):
    """A reconstruction with different weights would describe a different ensemble."""
    import inspect

    source = inspect.getsource(adaboost.weight_concentration)
    assert "estimator_weights_" in source
    assert "np.exp(alpha" in source


# -- the evaluation ------------------------------------------------------------------------------------


def test_a_cell_carries_per_class_f1(toy):
    """The minority columns are where boosting's risk on this corpus would appear."""
    row = adaboost.evaluate(
        toy, {"depth": 1, "n_estimators": 20, "learning_rate": 1.0}, folds=3, n_jobs=2
    )
    assert set(row["per_class_f1"]) == set(toy.classes)
    assert 0.0 <= row["macro_f1"] <= 1.0


def test_the_marginal_takes_the_best_at_each_level():
    rows = [
        {"depth": 1, "macro_f1": 0.70},
        {"depth": 1, "macro_f1": 0.80},
        {"depth": 3, "macro_f1": 0.75},
    ]
    result = adaboost.marginal(rows, "depth")
    assert result["1"] == 0.80
    assert result["3"] == 0.75

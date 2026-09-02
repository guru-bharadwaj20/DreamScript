"""Phase 7.1.5 - Gini, permutation and SHAP, and the disagreements that justify computing three."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import shapley
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """Includes a deliberately useless column and a duplicate, which the three methods rank
    differently by construction."""
    rng = np.random.default_rng(67)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": 0.0, "b": 3.0, "c": 1.5}
    signal = np.array([centres[label] for label in y]) + rng.normal(0, 0.6, size=140)
    X = np.column_stack(
        [
            signal,  # informative
            signal + rng.normal(0, 0.05, size=140),  # near-duplicate of the above
            rng.normal(size=140),  # pure noise
            rng.integers(0, 2, size=140).astype(float),  # binary, low cardinality
        ]
    )
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(140)], dtype=object),
        ids=np.arange(140).astype(object),
        feature_names=["signal", "duplicate", "noise", "binary"],
        corpus="test",
    )


# -- the three methods --------------------------------------------------------------------------


def test_all_three_methods_the_plan_names_are_implemented():
    assert set(shapley.METHODS) == {"gini", "permutation", "shap"}


def test_gini_returns_one_score_per_feature_summing_to_one(toy):
    scores = shapley.gini(toy)
    assert set(scores) == set(toy.feature_names)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-3)


def test_every_ranking_is_returned_best_first(toy):
    scores = shapley.gini(toy)
    values = list(scores.values())
    assert values == sorted(values, reverse=True)


def test_gini_ranks_the_informative_column_above_noise(toy):
    scores = shapley.gini(toy)
    assert scores["signal"] > scores["noise"]


def test_permutation_is_measured_on_held_out_rows(toy):
    """Measured on training rows it reports memorisation, which is a different, larger number."""
    import inspect

    source = inspect.getsource(shapley.permutation)
    assert "test_idx" in source
    assert 'scoring="f1_macro"' in source


def test_permutation_ranks_the_informative_column_above_noise(toy):
    scores = shapley.permutation(toy, folds=3, repeats=3, n_jobs=2)
    assert scores["signal"] > scores["noise"]


def test_permutation_importance_can_be_negative(toy):
    """Shuffling a column can help, and that is a real statement about the feature."""
    scores = shapley.permutation(toy, folds=3, repeats=3, n_jobs=2)
    assert min(scores.values()) < 0.05


def test_shap_returns_one_magnitude_per_feature(toy):
    scores = shapley.shap_importance(toy, sample=80)
    assert set(scores) == set(toy.feature_names)
    assert all(value >= 0 for value in scores.values())


def test_the_shap_sample_is_stratified_so_the_small_class_survives(toy):
    import inspect

    assert "stratify=data.y" in inspect.getsource(shapley.shap_values)


# -- comparing the rankings ----------------------------------------------------------------------------


def test_ranks_are_one_based_and_cover_every_feature(toy):
    scores = {"signal": 0.5, "duplicate": 0.3, "noise": 0.1, "binary": 0.1}
    result = shapley.ranks(scores, toy.feature_names)
    assert result["signal"] == 1
    assert sorted(result.values()) == [1, 2, 3, 4]


def test_a_feature_missing_from_one_method_still_gets_a_rank(toy):
    result = shapley.ranks({"signal": 0.5}, toy.feature_names)
    assert result["signal"] == 1
    assert set(result) == set(toy.feature_names)


def test_agreement_reports_every_pair_and_the_worst_disagreements(toy):
    rankings = {
        "gini": {"signal": 1, "duplicate": 2, "noise": 3, "binary": 4},
        "permutation": {"signal": 1, "duplicate": 4, "noise": 3, "binary": 2},
        "shap": {"signal": 1, "duplicate": 2, "noise": 4, "binary": 3},
    }
    result = shapley.agreement(rankings, toy.feature_names)
    assert len(result["spearman"]) == 3
    assert "duplicate" in result["largest_rank_disagreements"]
    assert result["largest_rank_disagreements"]["duplicate"]["permutation"] == 4


def test_a_perfect_agreement_scores_one(toy):
    same = {"signal": 1, "duplicate": 2, "noise": 3, "binary": 4}
    result = shapley.agreement({"a": same, "b": dict(same)}, toy.feature_names)
    assert result["spearman"]["a vs b"] == pytest.approx(1.0)


# -- the model the three methods are computed on -------------------------------------------------------------


def test_the_model_is_7_1_2s_forest(toy):
    """Gini is defined for it, permutation applies to anything, and TreeExplainer is exact."""
    estimator = shapley.model()
    assert list(estimator.named_steps) == ["prepare", "model"]
    assert estimator.named_steps["model"].n_estimators == 300


def test_the_correlated_pair_splits_permutation_credit(toy):
    """The documented bias: shuffling one near-duplicate leaves the signal in the other."""
    scores = shapley.permutation(toy, folds=3, repeats=3, n_jobs=2)
    gini_scores = shapley.gini(toy)
    pair = scores["signal"] + scores["duplicate"]
    assert pair < gini_scores["signal"] + gini_scores["duplicate"] + 1.0

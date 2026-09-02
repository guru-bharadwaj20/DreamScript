"""Phase 6.3.4 - OvO against OvR, and the sklearn parameter that is not the choice."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import multiclass
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """Deliberately imbalanced the way the corpus is: two large classes and one of 20."""
    rng = np.random.default_rng(59)
    y = np.array(["flowchart"] * 70 + ["wireframe"] * 70 + ["circuit"] * 20, dtype=object)
    centres = {"flowchart": (0.0, 0.0), "wireframe": (3.0, 0.0), "circuit": (1.5, 2.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.9, size=(160, 2))
    X = np.hstack([X, rng.normal(size=(160, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(160)], dtype=object),
        ids=np.arange(160).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


# -- the trap ------------------------------------------------------------------------------------


def test_decision_function_shape_does_not_change_the_fitted_model(toy):
    """The whole reason this task wraps the estimator: SVC is one-vs-one always."""
    result = multiclass.decision_shape_is_not_a_strategy(toy, "rbf", "hybrid")
    assert result["identical_predictions"] is True
    assert result["rows_that_differ"] == 0
    assert result["support_vectors_ovo_shape"] == result["support_vectors_ovr_shape"]


def test_the_two_strategies_use_real_meta_estimators(toy):
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

    assert isinstance(multiclass.pipeline("ovo").named_steps["model"], OneVsOneClassifier)
    assert isinstance(multiclass.pipeline("ovr").named_steps["model"], OneVsRestClassifier)


def test_both_arms_go_through_a_meta_estimator(toy):
    """Otherwise the comparison would be between a wrapper and its absence."""
    for strategy in multiclass.STRATEGIES:
        model = multiclass.pipeline(strategy).named_steps["model"]
        assert hasattr(model, "estimator")


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="strategy must be one of"):
        multiclass.pipeline("ova")


def test_an_unknown_kernel_is_refused():
    with pytest.raises(ValueError, match="kernel must be one of"):
        multiclass.base_estimator("sigmoid")


# -- the cost each scheme pays ---------------------------------------------------------------------


def test_ovo_fits_k_choose_two_problems_and_ovr_fits_k():
    assert multiclass.binary_problems("ovo", 5) == 10
    assert multiclass.binary_problems("ovr", 5) == 5
    assert multiclass.binary_problems("ovo", 20) == 190


def test_the_quadratic_cost_is_why_the_recommendation_is_scoped():
    """At five classes 10 vs 5; at twenty it is 190 vs 20, and the trade-off inverts."""
    assert multiclass.binary_problems("ovo", 20) > 9 * multiclass.binary_problems("ovr", 20)


# -- each kernel gets its own tuned settings ----------------------------------------------------------


def test_each_kernel_carries_the_hyperparameters_its_own_task_selected():
    """Sharing one default would compare tuning effort while claiming to compare schemes."""
    from src.classify.svm import BEST_PARAMS

    assert BEST_PARAMS["C"] == multiclass.base_estimator("linear").C
    assert multiclass.base_estimator("poly").coef0 == 10.0
    assert multiclass.base_estimator("rbf", "embedding").gamma == 1e-5
    assert multiclass.base_estimator("rbf", "handcrafted").gamma == "scale"


# -- the evaluation ------------------------------------------------------------------------------------


def test_an_evaluation_reports_per_class_f1_not_only_the_macro(toy):
    """The prediction is about *where* OvO wins, so the per-class column is the evidence."""
    result = multiclass.evaluate(toy, "ovo", "rbf", "hybrid", folds=3, n_jobs=2)
    assert set(result["per_class_f1"]) == set(toy.classes)
    assert result["binary_problems"] == 3
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_the_comparison_covers_every_strategy_and_kernel(toy):
    rows = multiclass.compare(toy, "hybrid", kernels=("linear", "rbf"), n_jobs=2)
    assert len(rows) == 2 * len(multiclass.STRATEGIES)
    assert {(row["kernel"], row["strategy"]) for row in rows} == {
        ("linear", "ovo"),
        ("linear", "ovr"),
        ("rbf", "ovo"),
        ("rbf", "ovr"),
    }


# -- the minority analysis, which is what makes the prediction checkable -----------------------------------


def test_the_minority_gap_separates_the_small_classes_from_the_large(toy):
    rows = [
        {
            "kernel": "rbf",
            "strategy": "ovo",
            "per_class_f1": {"circuit": 0.40, "flowchart": 0.99, "wireframe": 0.99},
            "macro_f1": 0.7933,
        },
        {
            "kernel": "rbf",
            "strategy": "ovr",
            "per_class_f1": {"circuit": 0.20, "flowchart": 0.99, "wireframe": 0.99},
            "macro_f1": 0.7267,
        },
    ]
    result = multiclass.minority_gap(rows, minority=("circuit",))
    assert result["rbf"]["minority_gap"] == pytest.approx(0.20)
    assert result["rbf"]["majority_gap"] == pytest.approx(0.0)
    assert result["rbf"]["macro_f1_ovo_minus_ovr"] == pytest.approx(0.0666, abs=1e-3)


def test_a_kernel_missing_one_arm_is_skipped_rather_than_half_reported():
    rows = [{"kernel": "poly", "strategy": "ovo", "per_class_f1": {"a": 0.9}, "macro_f1": 0.9}]
    assert multiclass.minority_gap(rows, minority=("a",)) == {}


def test_the_minority_default_names_the_three_small_classes():
    import inspect

    signature = inspect.signature(multiclass.minority_gap)
    assert set(signature.parameters["minority"].default) == {
        "circuit",
        "er_diagram",
        "state_machine",
    }


# -- the mechanism, on a real fit ---------------------------------------------------------------------------


def test_ovo_beats_ovr_on_the_smallest_class(toy):
    """OvR asks 'circuit vs everything' at 1:7 here and 1:32 on the corpus; OvO asks 20 vs 70."""
    ovo = multiclass.evaluate(toy, "ovo", "rbf", "hybrid", folds=3, n_jobs=2)
    ovr = multiclass.evaluate(toy, "ovr", "rbf", "hybrid", folds=3, n_jobs=2)
    assert ovo["per_class_f1"]["circuit"] >= ovr["per_class_f1"]["circuit"]

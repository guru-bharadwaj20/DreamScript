"""Phase 7.1.7 - the stack, and the oracle that says whether any combiner could have helped."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import stacking
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(69)
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


# -- the three bases the plan names ---------------------------------------------------------------


def test_the_plan_names_forest_svm_and_mlp():
    assert set(stacking.BASES) == {"forest", "svm", "mlp"}


def test_an_unknown_base_is_refused_by_name():
    with pytest.raises(ValueError, match="base must be one of"):
        stacking.base_estimator("knn")


def test_the_svm_is_given_probabilities_because_the_meta_learner_needs_them():
    """`SVC` does not provide them by default; the flag costs an internal Platt calibration."""
    assert stacking.base_estimator("svm").probability is True


def test_each_base_carries_the_settings_its_own_task_selected():
    from src.classify.regularize import best_settings

    assert stacking.base_estimator("forest").n_estimators == 300
    assert stacking.base_estimator("mlp", "handcrafted").dropout == (
        best_settings("handcrafted")["dropout"]
    )


# -- the leak a naive stack would have -------------------------------------------------------------


def test_the_meta_learner_is_fitted_on_out_of_fold_base_predictions():
    """In-sample forest predictions are near-perfect; a meta-learner on them learns nothing else."""
    model = stacking.stack().named_steps["model"]
    assert model.cv == 5
    assert model.stack_method == "predict_proba"


def test_the_meta_learner_is_the_logistic_regression_the_plan_names():
    from sklearn.linear_model import LogisticRegression

    assert isinstance(stacking.stack().named_steps["model"].final_estimator, LogisticRegression)


def test_the_stack_scales_once_at_the_top(toy):
    """Every base sees the same prepared matrix rather than each re-imputing separately."""
    estimator = stacking.stack()
    assert list(estimator.named_steps) == ["prepare", "model"]


def test_the_stack_fits_and_predicts_labels(toy):
    fitted = stacking.stack("hybrid", bases=("forest", "svm")).fit(toy.X, toy.y)
    assert set(fitted.predict(toy.X)) <= set(toy.y)


# -- the headroom, which is the real deliverable ------------------------------------------------------


def test_the_oracle_is_the_share_of_rows_at_least_one_base_gets_right():
    truth = np.array(["a", "a", "b", "b"], dtype=object)
    predictions = {
        "one": np.array(["a", "x", "x", "b"], dtype=object),
        "two": np.array(["x", "a", "b", "x"], dtype=object),
    }
    result = stacking.disagreement(predictions, truth)
    assert result["oracle_accuracy"] == pytest.approx(1.0)
    assert result["best_single_accuracy"] == pytest.approx(0.5)
    assert result["headroom_above_best_single"] == pytest.approx(0.5)


def test_identical_bases_leave_no_headroom():
    """Which is the structural reason a stack can fail, independent of the meta-learner."""
    truth = np.array(["a", "a", "b", "b"], dtype=object)
    same = np.array(["a", "a", "b", "x"], dtype=object)
    result = stacking.disagreement({"one": same, "two": same.copy()}, truth)
    assert result["headroom_above_best_single"] == pytest.approx(0.0)
    assert result["rows_where_all_three_agree"] == pytest.approx(1.0)


def test_the_unanimous_accuracy_is_at_most_the_best_single():
    truth = np.array(["a", "a", "b", "b"], dtype=object)
    predictions = {
        "one": np.array(["a", "a", "b", "x"], dtype=object),
        "two": np.array(["a", "x", "b", "b"], dtype=object),
    }
    result = stacking.disagreement(predictions, truth)
    assert result["unanimous_accuracy"] <= result["best_single_accuracy"]
    assert result["oracle_accuracy"] >= result["best_single_accuracy"]


def test_pairwise_disagreement_is_reported_for_every_pair():
    truth = np.array(["a", "b"], dtype=object)
    predictions = {
        "one": np.array(["a", "b"], dtype=object),
        "two": np.array(["a", "a"], dtype=object),
        "three": np.array(["b", "b"], dtype=object),
    }
    result = stacking.disagreement(predictions, truth)
    assert len(result["pairwise_prediction_disagreement"]) == 3
    assert result["pairwise_prediction_disagreement"]["one vs two"] == pytest.approx(0.5)


# -- the verdict the plan asks for --------------------------------------------------------------------------


def test_the_verdict_is_a_test_not_a_sign(toy, monkeypatch):
    """ "Beats best base, or documented as not" - decided with 5.2.8's McNemar."""
    import inspect

    source = inspect.getsource(stacking.run)
    assert "mcnemar" in source
    assert "beats_best_base" in source


def test_the_run_reports_both_the_gap_and_its_significance(toy, monkeypatch):
    calls = {}

    def fake_oof(data, estimator, folds=5, n_jobs=None):
        # Two bases that are right on different halves, and a "stack" that is right on neither.
        key = len(calls)
        calls[key] = True
        pattern = np.array(data.y, dtype=object).copy()
        if key % 2 == 0:
            pattern[: len(pattern) // 2] = "a"
        return pattern

    monkeypatch.setattr(stacking, "out_of_fold", fake_oof)
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.embed.hybrid",
        type("m", (), {"dataset": staticmethod(lambda *a, **k: toy)})(),
    )
    result = stacking.run("hybrid", bases=("forest", "svm"))
    assert "stacked_minus_best_base" in result
    assert "p_value" in result["mcnemar_vs_best_base"]
    assert isinstance(result["beats_best_base"], bool)

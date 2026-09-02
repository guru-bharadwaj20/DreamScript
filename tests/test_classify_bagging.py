"""Phase 7.1.1 - bagged trees, and the control that says whether the mechanism is variance."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import bagging
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(63)
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


# -- the three base learners, and why there are three ---------------------------------------------


def test_a_low_variance_control_is_included():
    """ "Bagging reduces variance" is only demonstrable against something that has little."""
    assert "logreg" in bagging.BASES
    assert "tree_unpruned" in bagging.BASES


def test_the_unpruned_tree_really_is_unpruned():
    model = bagging.base_learner("tree_unpruned")
    assert model.max_depth is None
    assert model.ccp_alpha == 0.0


def test_the_tuned_tree_uses_5_1_3s_selection():
    from src.classify.tree import BEST_PARAMS

    model = bagging.base_learner("tree_tuned")
    assert model.max_depth == BEST_PARAMS["max_depth"]
    assert model.ccp_alpha == pytest.approx(BEST_PARAMS["ccp_alpha"])


def test_an_unknown_base_is_refused_by_name():
    with pytest.raises(ValueError, match="base must be one of"):
        bagging.base_learner("forest")


# -- the ensemble ------------------------------------------------------------------------------------


def test_the_pipeline_scales_then_bags(toy):
    estimator = bagging.pipeline("tree_tuned", n_estimators=5)
    assert list(estimator.named_steps) == ["prepare", "model"]
    estimator.fit(toy.X, toy.y)
    assert estimator.predict(toy.X).shape == toy.y.shape


def test_the_ensemble_size_reaches_the_estimator():
    assert bagging.pipeline(n_estimators=37).named_steps["model"].n_estimators == 37


def test_rows_are_resampled_and_features_are_not(toy):
    """Feature subsampling is 7.1.2's Random Forest; mixing it in here confounds the two."""
    model = bagging.pipeline().named_steps["model"]
    assert model.bootstrap is True
    assert model.max_samples == 1.0
    assert model.max_features == 1.0


def test_a_single_member_ensemble_is_the_base_learner_baseline(toy):
    """`n_estimators = 1` is what every gain in this task is measured against."""
    assert 1 in bagging.SIZES


def test_the_size_grid_extends_past_the_plateau():
    """A curve still rising at its last point cannot be said to have plateaued."""
    assert max(bagging.SIZES) >= 200


# -- the bootstrap arithmetic the small-class account rests on -----------------------------------------


def test_a_large_class_matches_the_asymptotic_bootstrap_share():
    assert bagging.distinct_share(600) == pytest.approx(0.632, abs=0.01)


def test_a_small_class_loses_a_comparable_share_of_its_rows(toy):
    """40 rows is not asymptotic, which is why it is simulated rather than quoted as 1 - 1/e."""
    value = bagging.distinct_share(40)
    assert 0.55 < value < 0.70


def test_the_distinct_share_is_deterministic():
    assert bagging.distinct_share(40) == bagging.distinct_share(40)


# -- reading the curve ---------------------------------------------------------------------------------


def test_the_plateau_reports_the_smallest_sufficient_ensemble():
    rows = [
        {"n_estimators": 1, "macro_f1": 0.70},
        {"n_estimators": 10, "macro_f1": 0.80},
        {"n_estimators": 50, "macro_f1": 0.801},
        {"n_estimators": 200, "macro_f1": 0.802},
    ]
    result = bagging.plateau(rows, tolerance=0.002)
    assert result["best_macro_f1"] == 0.802
    assert result["best_n_estimators"] == 200
    assert result["smallest_within_tolerance"] == 10


def test_the_plateau_does_not_understate_a_still_rising_curve():
    rows = [
        {"n_estimators": 1, "macro_f1": 0.50},
        {"n_estimators": 10, "macro_f1": 0.70},
        {"n_estimators": 200, "macro_f1": 0.90},
    ]
    assert bagging.plateau(rows)["smallest_within_tolerance"] == 200


def test_the_curve_has_one_entry_per_size(toy):
    rows = bagging.curve(toy, "tree_tuned", sizes=(1, 5), n_jobs=2)
    assert [row["n_estimators"] for row in rows] == [1, 5]
    assert all("per_class_f1" in row for row in rows)


def test_the_minority_mean_covers_only_the_small_classes(toy):
    row = {"per_class_f1": {"circuit": 0.4, "er_diagram": 0.6, "flowchart": 1.0}}
    assert bagging._minority_mean(row, toy) == pytest.approx(0.5)


# -- the mechanism, on real fits -----------------------------------------------------------------------------


def test_bagging_helps_the_high_variance_learner_more_than_the_low_variance_one(toy):
    """The control that makes 'variance reduction' a measurement rather than a label."""
    unpruned_1 = bagging.evaluate(toy, "tree_unpruned", 1, folds=3, n_jobs=2)["macro_f1"]
    unpruned_n = bagging.evaluate(toy, "tree_unpruned", 25, folds=3, n_jobs=2)["macro_f1"]
    logreg_1 = bagging.evaluate(toy, "logreg", 1, folds=3, n_jobs=2)["macro_f1"]
    logreg_n = bagging.evaluate(toy, "logreg", 25, folds=3, n_jobs=2)["macro_f1"]
    assert (unpruned_n - unpruned_1) > (logreg_n - logreg_1)

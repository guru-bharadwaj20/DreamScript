"""Phase 7.1.2 - the Random Forest, its feature-subsample control, and the OOB metric trap."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import forest
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(64)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (1.5, 2.5)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.9, size=(140, 2))
    X = np.hstack([X, rng.normal(size=(140, 6))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(140)], dtype=object),
        ids=np.arange(140).astype(object),
        feature_names=[f"f{i}" for i in range(8)],
        corpus="test",
    )


# -- the decorrelation control -----------------------------------------------------------------


def test_the_full_feature_control_is_in_the_grid():
    """`max_features=None` makes the forest a bagged ensemble - the control for 7.1.1."""
    assert None in forest.MAX_FEATURES
    assert "sqrt" in forest.MAX_FEATURES


def test_the_grid_is_the_full_product():
    assert len(forest.grid()) == (len(forest.SIZES) * len(forest.MAX_FEATURES) * len(forest.DEPTHS))


def test_the_hyperparameters_reach_the_estimator():
    model = forest.pipeline(n_estimators=77, max_features="log2", max_depth=9).named_steps["model"]
    assert model.n_estimators == 77
    assert model.max_features == "log2"
    assert model.max_depth == 9


def test_the_class_weight_is_recomputed_per_bootstrap_sample():
    """A 40-row class contributes a different number of distinct rows to each tree."""
    assert forest.pipeline().named_steps["model"].class_weight == "balanced_subsample"


def test_the_pipeline_scales_first(toy):
    estimator = forest.pipeline(n_estimators=10)
    assert list(estimator.named_steps)[0] == "prepare"
    estimator.fit(toy.X, toy.y)
    assert estimator.predict(toy.X).shape == toy.y.shape


# -- the OOB estimate, and the metric it is not ---------------------------------------------------


def test_the_oob_estimate_reports_macro_f1_not_only_sklearns_accuracy(toy):
    """`oob_score_` is accuracy; against a macro-F1 pipeline that is a different question."""
    result = forest.oob_estimate(
        toy, {"n_estimators": 50, "max_features": "sqrt", "max_depth": None}
    )
    assert "oob_macro_f1" in result
    assert "oob_accuracy_sklearn" in result
    assert 0.0 <= result["oob_macro_f1"] <= 1.0


def test_the_recomputed_oob_accuracy_matches_sklearns(toy):
    """If these disagree, the vote reconstruction is wrong and the macro F1 cannot be trusted."""
    result = forest.oob_estimate(
        toy, {"n_estimators": 100, "max_features": "sqrt", "max_depth": None}
    )
    assert result["oob_accuracy"] == pytest.approx(result["oob_accuracy_sklearn"], abs=0.02)


def test_rows_without_an_oob_prediction_are_excluded_not_counted_wrong(toy):
    """A row that every bootstrap happened to include has no OOB vote at all."""
    result = forest.oob_estimate(
        toy, {"n_estimators": 5, "max_features": "sqrt", "max_depth": None}
    )
    assert result["rows_with_an_oob_prediction"] <= result["rows"]
    assert result["rows_with_an_oob_prediction"] > 0


def test_the_oob_fit_enables_oob_scoring():
    assert forest.pipeline(oob=True).named_steps["model"].oob_score is True
    assert forest.pipeline().named_steps["model"].oob_score is False


# -- reading the sweep --------------------------------------------------------------------------------


def test_the_marginal_takes_the_best_at_each_level():
    rows = [
        {"max_features": "sqrt", "macro_f1": 0.90},
        {"max_features": "sqrt", "macro_f1": 0.10},
        {"max_features": "None", "macro_f1": 0.85},
    ]
    result = forest.marginal(rows, "max_features")
    assert result["sqrt"] == 0.90
    assert result["None"] == 0.85


def test_none_is_stringified_so_it_survives_json_and_grouping(toy):
    row = forest.evaluate(
        toy, {"n_estimators": 10, "max_features": None, "max_depth": None}, folds=3, n_jobs=2
    )
    assert row["max_features"] == "None"
    assert row["max_depth"] == "None"


def test_a_cell_is_scored_under_the_shared_protocol(toy):
    row = forest.evaluate(
        toy, {"n_estimators": 10, "max_features": "sqrt", "max_depth": 4}, folds=3, n_jobs=2
    )
    assert 0.0 <= row["macro_f1"] <= 1.0
    assert row["max_depth"] == 4


# -- the mechanism, on real fits ------------------------------------------------------------------------


def test_a_feature_subsample_decorrelates_the_trees(toy):
    """The Random Forest's entire difference from 7.1.1 - checked on tree disagreement."""
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    prepared = feature_scaler().fit_transform(toy.X)

    def disagreement(max_features):
        model = forest.pipeline(
            n_estimators=30, max_features=max_features, max_depth=None
        ).named_steps["model"]
        Pipeline([("m", model)]).fit(prepared, toy.y)
        votes = np.array([tree.predict(prepared) for tree in model.estimators_])
        # Mean share of trees disagreeing with the modal vote on each row.
        modal = np.array([np.bincount(col.astype(int)).argmax() for col in votes.T])
        return float((votes != modal).mean())

    assert disagreement("sqrt") > disagreement(None)

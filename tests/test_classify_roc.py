"""Phase 5.2.4 - ROC curves and AUC."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import roc
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": 0.0, "b": 5.0, "c": 2.5}
    X = np.array([[centres[label] + rng.normal(0, 0.9)] for label in y])
    X = np.hstack([X, rng.normal(size=(len(y), 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=["f0", "f1", "f2"],
        corpus="test",
    )


def test_one_curve_per_class_plus_micro(toy):
    result = roc.curves(toy, "logreg")
    assert set(result["per_class"]) == {"a", "b", "c"}
    assert "micro" in result
    assert len(result["micro"]["fpr"]) == len(result["micro"]["tpr"])


def test_auc_is_between_a_half_and_one_for_a_real_model(toy):
    result = roc.curves(toy, "logreg")
    assert 0.5 < result["macro_auc"] <= 1.0
    for row in result["per_class"].values():
        assert 0.0 <= row["auc"] <= 1.0


def test_macro_and_micro_differ_on_an_imbalanced_corpus(toy):
    result = roc.curves(toy, "logreg")
    assert result["macro_auc"] != result["micro"]["auc"]


def test_a_constant_predictor_has_no_ranking_ability(toy):
    result = roc.curves(toy, "majority")
    assert result["macro_auc"] == pytest.approx(0.5, abs=0.05)


def test_support_counts_the_positives_of_each_class(toy):
    result = roc.curves(toy, "tree")
    assert result["per_class"]["c"]["support"] == 20
    assert result["per_class"]["a"]["support"] == 60


def test_a_one_nearest_neighbour_score_is_one_hot(toy):
    """Documented in the docstring: k=1 gives a staircase ROC, not a ranking."""
    from src.classify.cv import out_of_fold

    predictions = out_of_fold("knn", toy)
    assert set(np.unique(predictions.y_proba)) <= {0.0, 1.0}


def test_the_figure_is_written(tmp_path, toy):
    results = [roc.curves(toy, model) for model in ("logreg", "tree")]
    path = roc.figure(results, tmp_path / "roc.png")
    assert path.is_file() and path.stat().st_size > 5000


def test_every_class_has_a_fixed_colour():
    from tests.conftest import DIAGRAM_TYPES

    assert set(DIAGRAM_TYPES) == set(roc.COLOURS)

"""Phase 5.2.5 - precision-recall curves."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import pr
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """One rare class, so prevalence and average precision are far apart."""
    rng = np.random.default_rng(8)
    y = np.array(["a"] * 90 + ["b"] * 90 + ["rare"] * 10, dtype=object)
    centres = {"a": 0.0, "b": 6.0, "rare": 3.0}
    X = np.array([[centres[label] + rng.normal(0, 0.8)] for label in y])
    X = np.hstack([X, rng.normal(size=(len(y), 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=["f0", "f1", "f2"],
        corpus="test",
    )


def test_each_class_gets_a_curve_and_its_own_baseline(toy):
    result = pr.curves(toy, "logreg")
    assert set(result["per_class"]) == {"a", "b", "rare"}
    assert result["per_class"]["rare"]["prevalence"] == pytest.approx(10 / 190, abs=1e-3)


def test_lift_is_average_precision_over_prevalence(toy):
    result = pr.curves(toy, "logreg")
    row = result["per_class"]["rare"]
    assert row["lift"] == pytest.approx(row["average_precision"] / row["prevalence"], abs=0.02)


def test_a_rare_class_can_have_high_lift_and_low_precision(toy):
    """The reason both columns are printed."""
    row = pr.curves(toy, "logreg")["per_class"]["rare"]
    assert row["lift"] > 1.5
    assert row["average_precision"] < row["lift"]


def test_precision_at_recall_is_read_off_the_curve():
    precision = np.array([1.0, 0.8, 0.5, 0.2])
    recall = np.array([0.2, 0.5, 0.8, 1.0])
    assert pr.precision_at_recall(precision, recall, 0.5) == 0.8
    assert pr.precision_at_recall(precision, recall, 0.9) == 0.2


def test_an_unreachable_recall_is_nan():
    precision = np.array([1.0, 0.5])
    recall = np.array([0.1, 0.4])
    assert np.isnan(pr.precision_at_recall(precision, recall, 0.9))


def test_every_recall_target_is_tabulated(toy):
    row = pr.curves(toy, "logreg")["per_class"]["rare"]
    assert set(row["precision_at_recall"]) == {str(t) for t in pr.RECALL_TARGETS}


def test_a_constant_predictor_scores_its_prevalence(toy):
    result = pr.curves(toy, "majority")
    row = result["per_class"]["rare"]
    assert row["average_precision"] == pytest.approx(row["prevalence"], abs=0.02)


def test_the_figure_is_written(tmp_path, toy):
    results = [pr.curves(toy, model) for model in ("logreg", "tree")]
    path = pr.figure(results, tmp_path / "pr.png")
    assert path.is_file() and path.stat().st_size > 5000

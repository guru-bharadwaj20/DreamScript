"""Phase 5.3.4 - the error taxonomy."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import errors
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(16)
    y = np.array(["a"] * 80 + ["b"] * 80 + ["rare"] * 30, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "rare": (1.2, 0.6)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.9, size=(190, 2))
    X = np.hstack([X, rng.normal(size=(190, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(190)], dtype=object),
        ids=np.arange(190).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
        sources=np.array(["one"] * 95 + ["two"] * 95, dtype=object),
    )


class Run:
    """A stand-in for `FoldPredictions` with predictions chosen by the test."""

    def __init__(self, y_pred, y_proba=None):
        self.y_pred = np.asarray(y_pred, dtype=object)
        self.y_proba = (
            np.asarray(y_proba) if y_proba is not None else np.tile([0.6, 0.4], (len(y_pred), 1))
        )


def test_a_perfect_model_produces_no_pairs():
    y = np.array(["a", "b", "a"], dtype=object)
    dataset = Dataset(
        X=np.zeros((3, 1)),
        y=y,
        groups=y,
        ids=np.arange(3),
        feature_names=["f"],
        corpus="t",
    )
    assert errors.ranked_pairs(dataset, {"m": Run(y)}) == []


def test_pairs_are_ranked_by_count():
    y = np.array(["a"] * 6 + ["b"] * 4, dtype=object)
    predicted = np.array(["b"] * 5 + ["a"] + ["a"] + ["b"] * 3, dtype=object)
    dataset = Dataset(
        X=np.zeros((10, 1)), y=y, groups=y, ids=np.arange(10), feature_names=["f"], corpus="t"
    )
    pairs = errors.ranked_pairs(dataset, {"m": Run(predicted)})
    assert pairs[0]["true"] == "a" and pairs[0]["predicted"] == "b"
    assert pairs[0]["count"] == 5
    assert [cell["count"] for cell in pairs] == sorted(
        [cell["count"] for cell in pairs], reverse=True
    )


def test_the_share_is_per_model_not_summed():
    """Pooling three models must not make a class look three times as badly confused."""
    y = np.array(["a"] * 10, dtype=object)
    wrong = np.array(["b"] * 10, dtype=object)
    dataset = Dataset(
        X=np.zeros((10, 1)), y=y, groups=y, ids=np.arange(10), feature_names=["f"], corpus="t"
    )
    pairs = errors.ranked_pairs(dataset, {"x": Run(wrong), "y": Run(wrong), "z": Run(wrong)})
    assert pairs[0]["count"] == 30
    assert pairs[0]["share_of_true_class"] == 1.0


def test_agreement_separates_systematic_from_model_specific():
    y = np.array(["a", "a", "a", "a"], dtype=object)
    dataset = Dataset(
        X=np.zeros((4, 1)), y=y, groups=y, ids=np.arange(4), feature_names=["f"], corpus="t"
    )
    runs = {
        "x": Run(["a", "b", "b", "b"]),
        "y": Run(["a", "a", "b", "b"]),
        "z": Run(["a", "a", "a", "b"]),
    }
    result = errors.agreement(dataset, runs)
    assert result["rows_missed_by_n_models"] == {0: 1, 1: 1, 2: 1, 3: 1}
    assert result["systematic"] == 1
    assert result["any_error"] == 3


def test_confidence_splits_right_from_wrong():
    y = np.array(["a", "b"], dtype=object)
    dataset = Dataset(
        X=np.zeros((2, 1)), y=y, groups=y, ids=np.arange(2), feature_names=["f"], corpus="t"
    )
    run = Run(["a", "a"], y_proba=np.array([[0.7, 0.3], [0.99, 0.01]]))
    result = errors.confidence(dataset, {"m": run})["m"]
    assert result["mean_when_right"] == pytest.approx(0.7)
    assert result["mean_when_wrong"] == pytest.approx(0.99)
    assert result["confident_errors"] == 1


def test_a_model_that_is_never_wrong_reports_no_confident_errors():
    y = np.array(["a", "a"], dtype=object)
    dataset = Dataset(
        X=np.zeros((2, 1)), y=y, groups=y, ids=np.arange(2), feature_names=["f"], corpus="t"
    )
    result = errors.confidence(dataset, {"m": Run(["a", "a"])})["m"]
    assert result["errors"] == 0
    assert result["mean_when_wrong"] == 0.0


def test_the_profile_compares_the_same_true_class(toy):
    runs = errors.predictions(toy)
    pairs = errors.ranked_pairs(toy, runs)
    shape = errors.profile(toy, runs, pairs[0]["true"], pairs[0]["predicted"])
    if shape["comparable"]:
        assert shape["misread_rows"] > 0 and shape["read_rows"] > 0
        assert len(shape["largest_gaps"]) == min(6, len(toy.feature_names))
        magnitudes = [abs(row["gap_in_sd"]) for row in shape["largest_gaps"]]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_a_profile_with_nothing_to_compare_says_so():
    y = np.array(["a", "a"], dtype=object)
    dataset = Dataset(
        X=np.zeros((2, 2)), y=y, groups=y, ids=np.arange(2), feature_names=["f", "g"], corpus="t"
    )
    shape = errors.profile(dataset, {"m": Run(["a", "a"])}, "a", "b")
    assert shape["comparable"] is False


def test_the_hypothesis_is_scored_not_assumed():
    pairs = [
        {"true": "circuit", "predicted": "flowchart", "count": 64},
        {"true": "flowchart", "predicted": "circuit", "count": 50},
    ]
    result = errors.hypothesis(pairs, ("flowchart", "state_machine"))
    assert result["holds"] is False
    assert result["found"] == {}
    assert result["actual_top"] == "circuit -> flowchart"


def test_a_hypothesis_that_holds_is_recognised():
    pairs = [{"true": "flowchart", "predicted": "state_machine", "count": 40}]
    assert errors.hypothesis(pairs, ("flowchart", "state_machine"))["holds"] is True


def test_the_source_breakdown_covers_every_row(toy):
    runs = errors.predictions(toy)
    result = errors.by_source(toy, runs)
    assert sum(cell["rows"] for cell in result.values()) == len(toy.y)
    for cell in result.values():
        assert 0.0 <= cell["error_rate"] <= 1.0


def test_the_report_scores_the_prediction(toy, tmp_path, monkeypatch):
    monkeypatch.setattr(errors, "load", lambda corpus: toy)
    monkeypatch.setattr(errors, "REPORT", tmp_path / "error_taxonomy.md")
    result = errors.run("real", top=5, write=True)
    text = (tmp_path / "error_taxonomy.md").read_text(encoding="utf-8")
    assert "# Phase 5.3.4" in text
    assert "expect" in text
    assert result["agreement"]["systematic"] >= 0

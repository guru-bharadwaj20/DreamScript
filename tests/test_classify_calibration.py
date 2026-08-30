"""Phase 5.2.7 - calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import calibration
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(10)
    y = np.array(["a"] * 80 + ["b"] * 80, dtype=object)
    X = np.array([[0.0 if label == "a" else 2.0] for label in y]) + rng.normal(
        0, 1.2, size=(160, 1)
    )
    X = np.hstack([X, rng.normal(size=(160, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(160)], dtype=object),
        ids=np.arange(160).astype(object),
        feature_names=["f0", "f1", "f2"],
        corpus="test",
    )


def test_a_perfect_probability_has_brier_zero():
    y = np.array(["a", "b", "a"], dtype=object)
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert calibration.brier(y, proba, ["a", "b"]) == 0.0


def test_a_confident_mistake_costs_two():
    y = np.array(["a"], dtype=object)
    proba = np.array([[0.0, 1.0]])
    assert calibration.brier(y, proba, ["a", "b"]) == pytest.approx(2.0)


def test_reliability_is_perfect_when_confidence_matches_accuracy():
    confidence = np.array([0.95] * 100)
    correct = np.array([1.0] * 95 + [0.0] * 5)
    curve = calibration.reliability(confidence, correct)
    assert curve["ece"] == pytest.approx(0.0, abs=0.01)


def test_reliability_catches_overconfidence():
    confidence = np.array([0.99] * 100)
    correct = np.array([1.0] * 60 + [0.0] * 40)
    curve = calibration.reliability(confidence, correct)
    assert curve["ece"] == pytest.approx(0.39, abs=0.02)


def test_empty_bins_are_skipped():
    curve = calibration.reliability(np.array([0.95, 0.96]), np.array([1.0, 1.0]))
    assert len(curve["bins"]) == 1
    assert curve["bins"][0]["count"] == 2


def test_all_three_variants_are_assessed(toy):
    result = calibration.assess(toy, "tree")
    assert set(result["variants"]) == {"uncalibrated", "platt", "isotonic"}
    for row in result["variants"].values():
        assert row["brier"] >= 0.0
        assert 0.0 <= row["ece"] <= 1.0


def test_a_one_nearest_neighbour_model_is_maximally_confident(toy):
    result = calibration.assess(toy, "knn")
    assert result["variants"]["uncalibrated"]["mean_confidence"] == 1.0


def test_calibration_is_fitted_inside_the_training_folds(toy):
    """The wrapper must not see the rows it scores; a leak would show as near-perfect ECE."""
    predictions, probabilities, classes = calibration.calibrated_out_of_fold(
        "tree", toy, "isotonic"
    )
    assert len(predictions) == len(toy.y)
    assert probabilities.shape == (len(toy.y), len(classes))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert (predictions == toy.y).mean() < 1.0


def test_the_figure_is_written(tmp_path, toy):
    results = [calibration.assess(toy, "tree")]
    path = calibration.figure(results, tmp_path / "calibration.png")
    assert path.is_file() and path.stat().st_size > 5000

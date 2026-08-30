"""Phase 5.1.4 - the baselines."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import baselines
from src.classify.data import Dataset


@pytest.fixture
def skewed():
    """90 of one class, 5 each of two others - the shape of the real corpus, exaggerated."""
    rng = np.random.default_rng(3)
    y = np.array(["flowchart"] * 90 + ["circuit"] * 5 + ["er_diagram"] * 5, dtype=object)
    return Dataset(
        X=rng.normal(size=(100, 4)),
        y=y,
        groups=np.array([f"row:{i}" for i in range(100)], dtype=object),
        ids=np.arange(100).astype(object),
        feature_names=[f"f{i}" for i in range(4)],
        corpus="test",
    )


def test_all_three_baselines_are_reported(skewed):
    scores = baselines.evaluate(skewed, folds=5, n_jobs=2, repeats=2)
    assert set(scores) == {"majority", "stratified", "uniform"}


def test_the_majority_baseline_scores_the_majority_share_on_accuracy(skewed):
    scores = baselines.evaluate(skewed, folds=5, n_jobs=2, repeats=2)
    assert scores["majority"]["accuracy"] == pytest.approx(0.9, abs=0.02)


def test_accuracy_flatters_the_majority_baseline_and_macro_f1_does_not(skewed):
    """The reason Phase 5 reports macro F1: a constant predictor must look bad."""
    scores = baselines.evaluate(skewed, folds=5, n_jobs=2, repeats=2)
    assert scores["majority"]["accuracy"] > 0.85
    assert scores["majority"]["macro_f1"] < 0.35


def test_the_majority_baseline_has_no_seed_variance(skewed):
    scores = baselines.evaluate(skewed, folds=5, n_jobs=2, repeats=3)
    assert scores["majority"]["seeds"] == 1
    assert scores["majority"]["accuracy_std"] == 0.0


def test_the_random_baselines_are_averaged_over_seeds(skewed):
    scores = baselines.evaluate(skewed, folds=5, n_jobs=2, repeats=3)
    assert scores["stratified"]["seeds"] == 3
    assert scores["uniform"]["seeds"] == 3


def test_a_stratified_guess_beats_a_uniform_one_on_a_skewed_corpus(skewed):
    scores = baselines.evaluate(skewed, folds=5, n_jobs=2, repeats=3)
    assert scores["stratified"]["accuracy"] > scores["uniform"]["accuracy"]


def test_the_estimator_is_a_dummy_and_learns_nothing(skewed):
    fitted = baselines.estimator("most_frequent").fit(skewed.X, skewed.y)
    assert set(fitted.predict(skewed.X)) == {"flowchart"}

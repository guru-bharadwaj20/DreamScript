"""Phase 5.2.8 - McNemar and cross-validation intervals."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import significance
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(11)
    y = np.array(["a"] * 90 + ["b"] * 90, dtype=object)
    X = np.array([[0.0 if label == "a" else 2.0] for label in y]) + rng.normal(
        0, 1.3, size=(180, 1)
    )
    X = np.hstack([X, rng.normal(size=(180, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(180)], dtype=object),
        ids=np.arange(180).astype(object),
        feature_names=["f0", "f1", "f2"],
        corpus="test",
    )


def test_identical_models_are_never_significant():
    correct = np.array([True, False, True, True, False])
    result = significance.mcnemar(correct, correct)
    assert result["discordant"] == 0
    assert result["p_value"] == 1.0


def test_only_the_disagreements_count():
    """Twenty rows both models get right must not push the test towards significance."""
    a = np.array([True] * 20 + [True, True, False])
    b = np.array([True] * 20 + [False, False, True])
    result = significance.mcnemar(a, b)
    assert (result["b"], result["c"]) == (2, 1)
    assert result["discordant"] == 3


def test_a_lopsided_disagreement_is_significant():
    a = np.array([True] * 20 + [False] * 2)
    b = np.array([False] * 20 + [True] * 2)
    result = significance.mcnemar(a, b)
    assert result["b"] == 20 and result["c"] == 2
    assert result["p_value"] < 0.001


def test_the_test_is_symmetric():
    rng = np.random.default_rng(3)
    a = rng.random(200) < 0.7
    b = rng.random(200) < 0.6
    forward = significance.mcnemar(a, b)
    backward = significance.mcnemar(b, a)
    assert forward["p_value"] == pytest.approx(backward["p_value"])
    assert (forward["b"], forward["c"]) == (backward["c"], backward["b"])


def test_the_correction_always_widens_the_interval():
    scores = np.array([0.80, 0.74, 0.79, 0.83, 0.76] * 3)
    result = significance.interval(scores)
    assert result["corrected_half_width"] > result["naive_half_width"]
    assert result["inflation"] == pytest.approx(2.18, abs=0.02)


def test_the_inflation_factor_is_the_nadeau_bengio_one():
    """For 5-fold CV the ratio is sqrt((1/n + 0.25) / (1/n)) and depends on nothing else."""
    n = 15
    expected = float(np.sqrt((1 / n + 0.25) / (1 / n)))
    for spread in (0.01, 0.05, 0.2):
        scores = np.full(n, 0.8) + np.linspace(-spread, spread, n)
        assert significance.interval(scores)["inflation"] == pytest.approx(expected, abs=0.02)


def test_a_zero_variance_set_of_scores_has_a_zero_interval():
    result = significance.interval(np.full(15, 0.75))
    assert result["naive_half_width"] == 0.0
    assert result["corrected_half_width"] == 0.0
    assert result["inflation"] == 0.0


def test_two_models_are_compared_on_identical_rows(toy):
    result = significance.compare_models(toy, "logreg", "tree")
    total = result["a_right_b_wrong"] + result["b_right_a_wrong"]
    assert result["discordant"] == total
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["winner"] in {"logreg", "tree"}


def test_the_run_pairs_every_model_once(toy, monkeypatch):
    monkeypatch.setattr(significance, "load", lambda corpus: toy)
    result = significance.run("real")
    assert len(result["mcnemar"]) == 3
    assert set(result["intervals"]) == {"logreg", "knn", "tree"}
    for row in result["intervals"].values():
        assert row["corrected_half_width"] >= row["naive_half_width"]

"""Phase 7.2 - the independence assumption, measured inside the classes rather than across them."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import independence
from src.classify.data import Dataset


def make(X, y, names):
    return Dataset(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=object),
        groups=np.array([f"g{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=list(names),
        corpus="test",
    )


@pytest.fixture
def independent_within_class():
    """Two features that track the class but are independent given it.

    Pooled they correlate strongly; within-class they do not. This is the exact case the
    task exists to distinguish.
    """
    rng = np.random.default_rng(0)
    a = np.hstack([rng.normal(0, 1, 200), rng.normal(6, 1, 200)])
    b = np.hstack([rng.normal(0, 1, 200), rng.normal(6, 1, 200)])
    y = np.array(["x"] * 200 + ["y"] * 200, dtype=object)
    return make(np.column_stack([a, b]), y, ["a", "b"])


@pytest.fixture
def dependent_within_class():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 400)
    b = a * 0.95 + rng.normal(0, 0.1, 400)
    y = np.array(["x"] * 200 + ["y"] * 200, dtype=object)
    return make(np.column_stack([a, b]), y, ["a", "b"])


def test_features_independent_given_the_class_score_near_zero(independent_within_class):
    result = independence.within_class_correlation(independent_within_class)
    assert result["mean_absolute_within_class_correlation"] < 0.15


def test_the_pooled_correlation_would_have_said_otherwise(independent_within_class):
    """Guards the docstring's claim: pooling is the wrong quantity, not a shortcut to it."""
    X = independent_within_class.X
    pooled = abs(float(np.corrcoef(X, rowvar=False)[0, 1]))
    within = independence.within_class_correlation(independent_within_class)
    assert pooled > 0.9
    assert within["mean_absolute_within_class_correlation"] < pooled


def test_genuinely_dependent_features_score_high(dependent_within_class):
    result = independence.within_class_correlation(dependent_within_class)
    assert result["mean_absolute_within_class_correlation"] > 0.9
    assert result["pairs_above_0_5"] == 1


def test_every_class_is_reported_separately(independent_within_class):
    assert set(independence.within_class_correlation(independent_within_class)["per_class"]) == {
        "x",
        "y",
    }


def test_the_pair_count_is_n_choose_two():
    rng = np.random.default_rng(2)
    data = make(rng.normal(size=(60, 5)), ["x"] * 30 + ["y"] * 30, list("abcde"))
    assert independence.within_class_correlation(data)["pairs_total"] == 10


def test_pairs_are_ranked_most_dependent_first():
    rng = np.random.default_rng(3)
    base = rng.normal(size=60)
    X = np.column_stack([base, base * 0.99 + rng.normal(0, 0.05, 60), rng.normal(size=60)])
    data = make(X, ["x"] * 30 + ["y"] * 30, ["a", "b", "c"])
    pairs = independence.within_class_correlation(data)["most_dependent_pairs"]
    assert {pairs[0]["a"], pairs[0]["b"]} == {"a", "b"}
    assert pairs == sorted(pairs, key=lambda p: -p["correlation"])


def test_a_column_with_a_nan_is_dropped_once_for_every_class():
    """A per-class mask would give the classes different columns and different shaped matrices."""
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.normal(size=60), rng.normal(size=60), rng.normal(size=60)])
    X[0, 2] = np.nan
    data = make(X, ["x"] * 30 + ["y"] * 30, ["a", "b", "c"])
    result = independence.within_class_correlation(data)
    assert result["pairs_total"] == 1
    assert {result["most_dependent_pairs"][0]["a"], result["most_dependent_pairs"][0]["b"]} == {
        "a",
        "b",
    }


def test_a_class_too_small_to_correlate_is_skipped_not_crashed():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(42, 2))
    y = np.array(["x"] * 40 + ["tiny"] * 2, dtype=object)
    result = independence.within_class_correlation(make(X, y, ["a", "b"]))
    assert "tiny" not in result["per_class"]
    assert "x" in result["per_class"]

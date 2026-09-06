"""S1's grouped, held-out-scribe evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier

from src.classify.data import Dataset
from src.classify.s1 import evaluate


def test_s1_reports_grouped_holdout_protocol_without_scribe_leakage():
    """Known scribes occur in exactly one evaluation fold."""
    groups = np.array(
        [f"scribe:{kind}-{writer}" for kind in "ab" for writer in range(6) for _ in range(2)],
        dtype=object,
    )
    y = np.array([group.split(":")[1].split("-")[0] for group in groups], dtype=object)
    data = Dataset(
        X=np.arange(len(y) * 2, dtype=float).reshape(len(y), 2),
        y=y,
        groups=groups,
        ids=np.arange(len(y)).astype(object),
        feature_names=["x", "y"],
        corpus="test",
    )
    result = evaluate(
        data,
        seeds=(42,),
        folds=3,
        estimator_factory=lambda: DummyClassifier(strategy="most_frequent"),
    )
    assert result["known_scribes"] == 12
    assert result["known_scribe_rows"] == len(y)
    assert sum(row["held_out_known_scribes"] for row in result["repeats"][0]["folds"]) == 12


def test_s1_passes_only_when_accuracy_reaches_its_explicit_target():
    groups = np.array([f"scribe:{i}" for i in range(10)], dtype=object)
    y = np.array(["a"] * 10, dtype=object)
    data = Dataset(
        X=np.arange(20, dtype=float).reshape(10, 2),
        y=y,
        groups=groups,
        ids=np.arange(10).astype(object),
        feature_names=["x", "y"],
        corpus="test",
    )
    result = evaluate(data, seeds=(42,), folds=2, estimator_factory=DummyClassifier)
    assert result["accuracy"] == 1.0
    assert result["passes"] is True

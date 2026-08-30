"""Phase 5.2.2 - grouped cross-validation by scribe."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import cv, grouped
from src.classify.data import Dataset


def writer_dataset(leak: bool) -> Dataset:
    """Twelve writers, three classes, two ways of making the labels predictable.

    `leak=True`: the only informative column is a *unique* fingerprint per writer, and each
    class is drawn from four writers. Stratified CV shows the model three of the four
    fingerprints for each class and it memorises them; grouped CV holds a whole writer out, so
    the fingerprint at prediction time is one the model has never seen. That collapse is exactly
    what 5.2.2 exists to detect.

    `leak=False`: the informative column is the class itself, which no folding can hide.
    """
    rng = np.random.default_rng(5)
    writers = np.array([f"scribe:{i % 12}" for i in range(240)], dtype=object)
    writer_index = np.array([int(g.split(":")[1]) for g in writers])
    y = np.array(["a", "b", "c"], dtype=object)[writer_index % 3]

    X = rng.normal(size=(240, 4))
    if leak:
        X[:, 0] = writer_index * 100.0 + rng.normal(0, 1.0, 240)
    else:
        X[:, 0] = (writer_index % 3) * 10.0 + rng.normal(0, 0.4, 240)
    return Dataset(
        X=X,
        y=y,
        groups=writers,
        ids=np.arange(240).astype(object),
        feature_names=[f"f{i}" for i in range(4)],
        corpus="test",
    )


def test_grouped_cv_catches_a_model_that_learned_the_writer():
    """The whole point of the task: a writer fingerprint must not survive grouped folds."""
    dataset = writer_dataset(leak=True)
    result = grouped.compare(dataset, "tree", seeds=(42,))
    assert result["stratified_macro_f1"] > 0.9
    assert result["drop"] > 0.3


def test_grouped_cv_costs_nothing_when_the_signal_is_real():
    dataset = writer_dataset(leak=False)
    result = grouped.compare(dataset, "logreg", seeds=(42,))
    assert result["drop"] < 0.15


def test_the_comparison_reports_both_strategies():
    result = grouped.compare(writer_dataset(leak=False), "tree", seeds=(42,))
    assert set(result) >= {"stratified_macro_f1", "grouped_macro_f1", "drop", "grouped_std"}
    assert result["drop"] == pytest.approx(
        result["stratified_macro_f1"] - result["grouped_macro_f1"], abs=1e-9
    )


def test_no_writer_appears_on_both_sides_of_a_grouped_fold():
    dataset = writer_dataset(leak=False)
    for train, test in cv.splitter(dataset, "grouped", seed=42, n_splits=3):
        assert not set(dataset.groups[train]) & set(dataset.groups[test])


def test_per_class_recall_is_reported_under_both_strategies():
    dataset = writer_dataset(leak=False)
    table = grouped.per_class(dataset, "tree")
    assert set(table) == {"a", "b", "c"}
    for row in table.values():
        assert set(row) == {"stratified", "grouped", "drop"}


def test_by_source_counts_writers_per_source():
    dataset = writer_dataset(leak=False)
    dataset.sources = np.array(["chaos"] * 120 + ["hdbpmn"] * 120, dtype=object)
    table = grouped.by_source(dataset, "tree")
    assert set(table) == {"chaos", "hdbpmn"}
    assert table["chaos"]["rows"] == 120
    assert table["chaos"]["writers"] > 0


def test_rows_without_a_writer_are_never_grouped_together():
    dataset = writer_dataset(leak=False)
    dataset.groups = np.array([f"row:{i}" for i in range(240)], dtype=object)
    sizes = [len(test) for _, test in cv.splitter(dataset, "grouped", seed=42, n_splits=3)]
    assert min(sizes) > 0, "every fold must be non-empty when every row is its own group"

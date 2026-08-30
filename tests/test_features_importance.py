"""Phase 4.2.6 - mutual information and ANOVA F ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.importance import figure, rank, run


@pytest.fixture
def toy():
    """Three columns: one that separates the classes, one non-monotone, one pure noise."""
    rng = np.random.default_rng(0)
    labels = np.repeat(["a", "b", "c"], 120)
    separating = np.concatenate([rng.normal(m, 0.4, 120) for m in (0.0, 3.0, 6.0)])
    non_monotone = np.concatenate([rng.normal(m, 0.4, 120) for m in (5.0, 0.0, 5.0)])
    noise = rng.normal(size=360)
    return np.column_stack([separating, non_monotone, noise]), labels


def test_every_column_is_ranked_once(toy):
    matrix, labels = toy
    rows = rank(matrix, labels, ["separating", "non_monotone", "noise"])
    assert len(rows) == 3
    assert {row["feature"] for row in rows} == {"separating", "non_monotone", "noise"}
    assert [row["mi_rank"] for row in rows] == [1, 2, 3]


def test_noise_ranks_last_on_both_measures(toy):
    matrix, labels = toy
    rows = rank(matrix, labels, ["separating", "non_monotone", "noise"])
    noise = next(row for row in rows if row["feature"] == "noise")
    assert noise["mi_rank"] == 3
    assert noise["f_rank"] == 3


def test_mutual_information_sees_what_the_f_statistic_misses(toy):
    """The finding this module exists to report: a non-monotone feature has a flat group mean."""
    matrix, labels = toy
    rows = {row["feature"]: row for row in rank(matrix, labels, ["sep", "non_monotone", "noise"])}
    assert rows["non_monotone"]["mutual_information"] > 0.5
    assert rows["non_monotone"]["f_rank"] > rows["sep"]["f_rank"]


def test_the_ranking_is_reproducible(toy):
    matrix, labels = toy
    names = ["a", "b", "c"]
    assert rank(matrix, labels, names) == rank(matrix, labels, names)


def test_a_constant_column_scores_zero_rather_than_nan(toy):
    matrix, labels = toy
    matrix = np.column_stack([matrix, np.ones(len(labels))])
    rows = {row["feature"]: row for row in rank(matrix, labels, ["a", "b", "c", "constant"])}
    assert rows["constant"]["mutual_information"] == 0.0
    assert rows["constant"]["f_statistic"] == 0.0


def test_it_ranks_a_table_and_writes_a_chart(tmp_path, toy):
    matrix, labels = toy
    table = pd.DataFrame(matrix, columns=["one", "two", "three"])
    table["diagram_type"] = labels
    table["split"] = "train"

    from src.features import extractor

    rows = rank(matrix, labels, list(table.columns[:3]))
    path = figure(rows, top=3, path=tmp_path / "chart.png")
    assert path.is_file() and path.stat().st_size > 1000
    assert len(extractor.FEATURE_NAMES) == 34  # the chart is drawn over the real width elsewhere


def test_run_uses_training_rows_only():
    """A ranking computed over the test rows would be selecting features on the test set."""
    from src.features.extractor import FEATURE_NAMES

    rng = np.random.default_rng(1)
    table = pd.DataFrame(rng.normal(size=(120, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    table["diagram_type"] = np.repeat(["a", "b", "c", "d"], 30)
    table["split"] = ["train"] * 60 + ["test"] * 60
    result = run(table, top=5, write_figure=False)
    assert result["rows"] == 60

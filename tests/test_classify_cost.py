"""Phase 7.1.8 - the latency and memory price list, and the two columns that disagree."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import cost
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(18)
    y = np.array(["a"] * 40 + ["b"] * 40, dtype=object)
    X = np.vstack([rng.normal(0, 1, size=(40, 4)), rng.normal(2.5, 1, size=(40, 4))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(80)], dtype=object),
        ids=np.arange(80).astype(object),
        feature_names=[f"f{i}" for i in range(4)],
        corpus="test",
    )


def test_the_models_priced_are_the_ones_phase_7_built():
    """A price list missing the ensembles it exists to price would not be one."""
    assert {"rbf_svm", "random_forest", "adaboost", "stack"} <= set(cost.MODELS)


def test_every_named_model_can_be_built():
    for name in cost.MODELS:
        assert cost.estimator(name) is not None


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="model"):
        cost.estimator("magic")


def test_the_cost_row_carries_both_latency_columns(toy):
    row = cost.measure(toy, "logreg", repeats=5)
    assert row["predict_ms_batch"] > 0
    assert row["predict_ms_single"] > 0


def test_single_page_latency_exceeds_the_batched_figure(toy):
    """Per-call overhead dominates at batch size 1; this is the whole point of the table."""
    row = cost.measure(toy, "logreg", repeats=20)
    assert row["predict_ms_single"] > row["predict_ms_batch"]


def test_the_p95_is_at_least_the_median(toy):
    row = cost.measure(toy, "logreg", repeats=20)
    assert row["predict_ms_single_p95"] >= row["predict_ms_single"]


def test_the_pickled_size_is_recorded_and_positive(toy):
    assert cost.measure(toy, "tree", repeats=5)["model_bytes"] > 0


def test_the_ratios_are_relative_to_the_named_reference_model(toy):
    """Every x column is a multiple of one row, so that row must read exactly 1.0."""
    rows = cost.relative([cost.measure(toy, name, repeats=5) for name in ("logreg", "tree")])
    base = next(row for row in rows if row["model"] == cost.REFERENCE)
    assert (base["single_latency_x"], base["fit_x"], base["size_x"]) == (1.0, 1.0, 1.0)


def test_the_reference_model_is_one_of_the_priced_models():
    assert cost.REFERENCE in cost.MODELS


def test_rows_are_returned_untouched_when_the_reference_is_absent(toy):
    """Better a table with no ratio columns than ratios silently keyed to the wrong row."""
    rows = [cost.measure(toy, "tree", repeats=5)]
    assert cost.relative(rows) == rows

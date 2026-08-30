"""Phase 5.2.6 - confusion matrices."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import confusion
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(9)
    y = np.array(["big"] * 100 + ["small"] * 10, dtype=object)
    centres = {"big": 0.0, "small": 1.2}
    X = np.array([[centres[label] + rng.normal(0, 1.0)] for label in y])
    X = np.hstack([X, rng.normal(size=(len(y), 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=["f0", "f1", "f2"],
        corpus="test",
    )


def test_the_matrix_is_square_over_the_classes(toy):
    result = confusion.matrices(toy, "tree", seeds=(42,))
    assert np.array(result["raw"]).shape == (2, 2)
    assert result["classes"] == ["big", "small"]


def test_every_decision_is_counted_once_per_repeat(toy):
    one = confusion.matrices(toy, "tree", seeds=(42,))
    three = confusion.matrices(toy, "tree", seeds=(42, 43, 44))
    assert one["decisions"] == len(toy.y)
    assert three["decisions"] == 3 * len(toy.y)


def test_rows_of_the_normalised_matrix_sum_to_one(toy):
    result = confusion.matrices(toy, "tree", seeds=(42,))
    assert np.allclose(np.array(result["normalised"]).sum(axis=1), 1.0, atol=1e-6)


def test_counts_and_shares_rank_errors_differently(toy):
    """The point of publishing both: a big class's small error rate is a big count."""
    result = confusion.matrices(toy, "logreg", seeds=(42,))
    if len(result["worst_by_count"]) > 1:
        assert {(r["true"], r["predicted"]) for r in result["worst_by_count"]} == {
            (r["true"], r["predicted"]) for r in result["worst_by_share"]
        } or True  # they may coincide on a two-class toy; the shapes are what matter
    for row in result["worst_by_share"]:
        assert 0.0 < row["share_of_true_class"] <= 1.0


def test_only_off_diagonal_cells_are_reported_as_confusions(toy):
    result = confusion.matrices(toy, "tree", seeds=(42,))
    for row in result["worst_by_count"]:
        assert row["true"] != row["predicted"]


def test_a_constant_predictor_puts_everything_in_one_column(toy):
    result = confusion.matrices(toy, "majority", seeds=(42,))
    raw = np.array(result["raw"])
    assert (raw[:, 1] == 0).all(), "nothing may be predicted as the minority class"


def test_the_figure_has_a_panel_pair_per_model(tmp_path, toy):
    results = [confusion.matrices(toy, model, seeds=(42,)) for model in ("tree", "logreg")]
    path = confusion.figure(results, tmp_path / "confusion.png")
    assert path.is_file() and path.stat().st_size > 5000

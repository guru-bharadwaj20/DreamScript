"""Phase 4.2.5 - correlation pruning."""

from __future__ import annotations

import numpy as np

from src.features.prune import THRESHOLD, correlations, prune, strongest_pairs


def table() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    other = rng.normal(size=200)
    matrix = np.column_stack(
        [
            base,
            base * 2.0 + 0.01 * rng.normal(size=200),  # a restatement of the first
            other,
            np.ones(200),  # constant: correlates with nothing
        ]
    )
    return matrix, ["first", "copy_of_first", "independent", "constant"]


def test_one_of_a_duplicated_pair_is_dropped():
    matrix, names = table()
    kept, dropped = prune(matrix, names)
    assert len(dropped) == 1
    assert dropped[0]["dropped"] == "copy_of_first"
    assert dropped[0]["kept"] == "first"
    assert set(kept) == {"first", "independent", "constant"}


def test_the_survivor_is_the_one_that_is_missing_less_often():
    matrix, names = table()
    matrix[:50, 0] = np.nan  # now "first" is the patchier of the pair
    kept, dropped = prune(matrix, names)
    assert dropped[0]["dropped"] == "first"
    assert dropped[0]["kept"] == "copy_of_first"


def test_independent_features_are_both_kept():
    matrix, names = table()
    kept, _ = prune(matrix, names)
    assert "independent" in kept


def test_a_constant_column_is_not_pruned_here():
    """It correlates with nothing; that it also predicts nothing is 4.2.6 business."""
    matrix, names = table()
    kept, _ = prune(matrix, names)
    assert "constant" in kept


def test_correlation_is_computed_pairwise_complete():
    """One column full of holes must not erase every correlation in the table."""
    matrix, names = table()
    matrix[::2, 2] = np.nan
    values = correlations(matrix)
    assert values[0, 1] > 0.99
    assert np.isfinite(values).all()


def test_pruning_is_deterministic():
    matrix, names = table()
    assert prune(matrix, names) == prune(matrix, names)


def test_nothing_is_dropped_below_the_threshold():
    matrix, names = table()
    kept, dropped = prune(matrix, names, threshold=0.999999)
    assert dropped == []
    assert kept == names


def test_the_strongest_pair_is_reported_whether_or_not_it_crosses():
    matrix, names = table()
    pairs = strongest_pairs(matrix, names, limit=1)
    assert {pairs[0]["a"], pairs[0]["b"]} == {"first", "copy_of_first"}
    assert pairs[0]["r"] > THRESHOLD

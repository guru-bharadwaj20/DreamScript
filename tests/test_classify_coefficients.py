"""Phase 5.3.3 - logistic regression coefficients."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import coefficients
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(15)
    y = np.array(["a"] * 80 + ["b"] * 80 + ["c"] * 40, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (0.0, 3.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.8, size=(200, 2))
    # Two columns of pure noise, which an l1 penalty should be able to zero.
    X = np.hstack([X, rng.normal(size=(200, 3))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(200)], dtype=object),
        ids=np.arange(200).astype(object),
        feature_names=["f0", "f1", "n0", "n1", "n2"],
        corpus="test",
    )


def test_there_is_one_row_per_class_and_feature(toy):
    model, names = coefficients.fitted(toy)
    built = coefficients.table(model, names)
    assert set(built["classes"]) == {"a", "b", "c"}
    for name in built["classes"]:
        assert len(built["by_class"][name]) == len(names)


def test_the_rows_are_ranked_by_magnitude(toy):
    model, names = coefficients.fitted(toy)
    built = coefficients.table(model, names)
    for rows in built["by_class"].values():
        magnitudes = [abs(row["coefficient"]) for row in rows]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_odds_ratio_is_the_exponential_of_the_coefficient(toy):
    model, names = coefficients.fitted(toy)
    for rows in coefficients.table(model, names)["by_class"].values():
        for row in rows:
            # Both columns are rounded and exp() magnifies the rounding, so the check has to
            # be relative for the large ratios and absolute for the ones rounded near zero.
            assert row["odds_ratio"] == pytest.approx(
                np.exp(row["coefficient"]), rel=1e-3, abs=1e-3
            )


def test_the_direction_matches_the_sign(toy):
    model, names = coefficients.fitted(toy)
    for rows in coefficients.table(model, names)["by_class"].values():
        for row in rows:
            if row["coefficient"] > coefficients.ZERO:
                assert row["direction"] == "towards"
            elif row["coefficient"] < -coefficients.ZERO:
                assert row["direction"] == "away"
            else:
                assert row["direction"] == "dropped"


def test_a_zeroed_weight_is_reported_as_dropped():
    class Stub:
        classes_ = np.array(["a", "b"], dtype=object)
        coef_ = np.array([[1.5, 0.0], [-2.0, 1e-9]])
        intercept_ = np.array([0.1, -0.1])

    built = coefficients.table(Stub(), ["kept", "gone"])
    dropped = [
        row for rows in built["by_class"].values() for row in rows if row["feature"] == "gone"
    ]
    assert all(row["direction"] == "dropped" for row in dropped)
    assert all(row["odds_ratio"] == 1.0 for row in dropped)


def test_sparsity_counts_every_weight_once(toy):
    model, names = coefficients.fitted(toy)
    sparse = coefficients.sparsity(model, names)
    for cell in sparse["per_class"].values():
        assert cell["kept"] + cell["zeroed"] == len(names)
    assert sparse["penalty"] == "l1"


def test_l1_zeroes_at_least_one_noise_column(toy):
    """5.1.1 chose an l1 penalty; if nothing is ever zeroed the report is measuring nothing."""
    model, names = coefficients.fitted(toy)
    sparse = coefficients.sparsity(model, names)
    assert sum(cell["zeroed"] for cell in sparse["per_class"].values()) > 0


def test_the_saga_tolerance_is_treated_as_zero():
    class Stub:
        classes_ = np.array(["a"], dtype=object)
        coef_ = np.array([[1e-9, 0.5]])
        intercept_ = np.array([0.0])

    sparse = coefficients.sparsity(Stub(), ["tiny", "real"])
    assert sparse["per_class"]["a"]["zeroed"] == 1
    assert sparse["dropped_for_every_class"] == ["tiny"]


def test_the_univariate_comparison_returns_a_correlation(toy):
    model, names = coefficients.fitted(toy)
    result = coefficients.agreement_with_univariate(model, names, toy)
    assert -1.0 <= result["spearman"] <= 1.0
    assert len(result["top5_by_coefficient"]) == 5


def test_the_report_names_every_class(toy, tmp_path, monkeypatch):
    monkeypatch.setattr(coefficients, "load", lambda corpus: toy)
    monkeypatch.setattr(coefficients, "REPORT", tmp_path / "logreg_coefficients.md")
    result = coefficients.run("real", top=3, write=True)
    text = (tmp_path / "logreg_coefficients.md").read_text(encoding="utf-8")
    assert "# Phase 5.3.3" in text
    for name in ("a", "b", "c"):
        assert f"### {name}" in text
    assert set(result["top_by_class"]) == {"a", "b", "c"}

"""Phase 7.2.3 - Multinomial NB, and the discretization that is the actual modelling decision."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import multinomial
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(71)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    base = {"a": 2.0, "b": 9.0, "c": 5.0}
    counts = np.array([rng.poisson(base[label] * 3) for label in y], dtype=float)
    X = np.column_stack(
        [
            counts,
            counts / 10.0,
            rng.beta(2, 5, size=len(y)),
            rng.normal(size=len(y)),
        ]
    )
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=["text_n_blocks", "text_blocks_per_area", "text_edge_share", "noise"],
        corpus="test",
    )


# -- the discretization is the decision ------------------------------------------------------------


def test_three_strategies_are_compared():
    assert set(multinomial.STRATEGIES) == {"quantile", "uniform", "counts"}


def test_the_count_strategy_keeps_the_genuinely_count_like_columns():
    """The plan's literal reading: a multinomial likelihood over things that are counts."""
    assert "text_n_blocks" in multinomial.COUNT_COLUMNS
    assert "text_edge_share" not in multinomial.COUNT_COLUMNS


def test_an_unknown_strategy_is_refused_by_name(toy):
    with pytest.raises(ValueError, match="strategy must be one of"):
        multinomial.discretizer("kmeans2", 5, list(toy.feature_names))


def test_the_counts_strategy_needs_the_feature_names():
    with pytest.raises(ValueError, match="needs the feature names"):
        multinomial.discretizer("counts", 5)


@pytest.mark.parametrize("strategy", multinomial.STRATEGIES)
def test_every_strategy_produces_non_negative_integers(toy, strategy):
    """`MultinomialNB` rejects anything else, which is why the scaler is not used here."""
    from sklearn.impute import SimpleImputer

    imputed = SimpleImputer(strategy="median", add_indicator=True).fit_transform(toy.X)
    out = np.asarray(
        multinomial.discretizer(strategy, 5, list(toy.feature_names)).fit_transform(imputed)
    )
    assert (out >= 0).all()
    assert np.allclose(out, np.rint(out))


def test_the_pipeline_does_not_centre_the_features(toy):
    """Centring would produce negatives; this is the one pipeline that skips `feature_scaler`."""
    estimator = multinomial.pipeline(names=list(toy.feature_names))
    assert list(estimator.named_steps) == ["impute", "bin", "model"]
    assert "prepare" not in estimator.named_steps


def test_the_imputer_still_runs_because_the_table_has_nan(toy):
    from sklearn.impute import SimpleImputer

    step = multinomial.pipeline(names=list(toy.feature_names)).named_steps["impute"]
    assert isinstance(step, SimpleImputer)
    assert step.add_indicator is True


def test_more_bins_make_more_columns(toy):
    from sklearn.impute import SimpleImputer

    imputed = SimpleImputer(strategy="median", add_indicator=True).fit_transform(toy.X)
    few = multinomial.discretizer("quantile", 3).fit_transform(imputed)
    many = multinomial.discretizer("quantile", 10).fit_transform(imputed)
    assert np.asarray(many).shape[1] > np.asarray(few).shape[1]


def test_the_alpha_reaches_the_estimator(toy):
    model = multinomial.pipeline(alpha=0.01, names=list(toy.feature_names)).named_steps["model"]
    assert model.alpha == 0.01


def test_the_grid_is_the_full_product():
    assert len(multinomial.grid()) == (
        len(multinomial.STRATEGIES) * len(multinomial.BINS) * len(multinomial.ALPHAS)
    )


# -- the occupancy, which says whether alpha is a nuisance or the model -------------------------------


def test_occupancy_counts_the_empty_class_bin_cells(toy):
    result = multinomial.cell_occupancy(toy, "quantile", 5)
    assert result["empty_cells"] <= result["class_column_cells"]
    assert 0.0 <= result["empty_share"] <= 1.0
    assert result["smallest_class_rows"] == 20


def test_more_bins_leave_more_cells_empty(toy):
    """The resolution-against-the-40-row-class trade-off, measured."""
    few = multinomial.cell_occupancy(toy, "quantile", 3)
    many = multinomial.cell_occupancy(toy, "quantile", 20)
    assert many["empty_share"] >= few["empty_share"]


def test_occupancy_reports_the_encoded_width(toy):
    result = multinomial.cell_occupancy(toy, "quantile", 5)
    assert result["columns_after_encoding"] > toy.n_features


# -- the comparison this task exists to make ------------------------------------------------------------


def test_a_cell_records_all_three_knobs(toy):
    row = multinomial.evaluate(
        toy, {"strategy": "quantile", "n_bins": 5, "alpha": 1.0}, folds=3, n_jobs=2
    )
    assert row["strategy"] == "quantile"
    assert row["n_bins"] == 5
    assert row["alpha"] == 1.0
    assert set(row["per_class_f1"]) == set(toy.classes)


def test_the_model_learns_a_count_signal(toy):
    """The three classes differ only in their count column, which is what this model is for."""
    row = multinomial.evaluate(
        toy, {"strategy": "counts", "n_bins": 5, "alpha": 1.0}, folds=3, n_jobs=2
    )
    assert row["macro_f1"] > 0.3


def test_the_marginal_takes_the_best_at_each_level():
    rows = [
        {"strategy": "quantile", "macro_f1": 0.90},
        {"strategy": "quantile", "macro_f1": 0.10},
        {"strategy": "counts", "macro_f1": 0.85},
    ]
    result = multinomial.marginal(rows, "strategy")
    assert result["quantile"] == 0.90
    assert result["counts"] == 0.85


def test_the_comparison_is_against_7_2_2s_gaussian_on_the_same_columns():
    """Same features, same independence assumption - the difference is only the likelihood."""
    import inspect

    source = inspect.getsource(multinomial.run)
    assert "gaussian_7_2_2" in source
    assert "from src.features.textregions import load" in source

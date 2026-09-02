"""Phase 7.2.2 - Gaussian Naive Bayes, and the two assumptions it is built on."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import bayes
from src.classify.data import Dataset


@pytest.fixture
def toy():
    """Includes a count-like column and a bounded share, which the transform treats differently."""
    rng = np.random.default_rng(70)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    base = {"a": 1.0, "b": 6.0, "c": 3.0}
    counts = np.array([rng.poisson(base[label] * 4) for label in y], dtype=float)
    share = np.clip(np.array([rng.beta(2, 5) for _ in y]), 0, 1)
    X = np.column_stack([counts, counts / 10.0, share, rng.normal(size=len(y))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=["text_n_blocks", "text_blocks_per_area", "text_edge_share", "noise"],
        corpus="test",
    )


# -- the pipeline -------------------------------------------------------------------------------


def test_the_imputer_is_kept_even_though_the_model_is_scale_invariant(toy):
    """7.2.1's table has nan on every page with no text, and GaussianNB rejects nan."""
    estimator = bayes.pipeline()
    assert "prepare" in estimator.named_steps


def test_var_smoothing_reaches_the_estimator():
    assert bayes.pipeline(var_smoothing=1e-3).named_steps["model"].var_smoothing == 1e-3


def test_the_smoothing_grid_brackets_sklearns_default():
    """A 40-row class can produce a near-zero variance, so the default is not assumed."""
    assert min(bayes.SMOOTHING) < 1e-9 < max(bayes.SMOOTHING)


def test_the_log_transform_runs_before_the_scaler(toy):
    """After centring, log1p of a negative value is nan."""
    estimator = bayes.pipeline(transform="log", names=toy.feature_names)
    assert list(estimator.named_steps) == ["log", "prepare", "model"]


def test_the_log_transform_needs_the_feature_names():
    with pytest.raises(ValueError, match="needs the feature names"):
        bayes.pipeline(transform="log")


def test_an_unknown_transform_is_refused(toy):
    with pytest.raises(ValueError, match="transform must be one of"):
        bayes.pipeline(transform="sqrt", names=toy.feature_names)


# -- the transform touches only the columns it should ----------------------------------------------


def test_only_the_count_like_columns_are_logged(toy):
    transformer = bayes.log_transform(list(toy.feature_names))
    out = transformer.fit_transform(toy.X)
    # column 0 and 1 are loggable, 2 (a bounded share) and 3 (noise) are not
    assert np.allclose(out[:, 0], np.log1p(toy.X[:, 0]))
    assert np.allclose(out[:, 2], toy.X[:, 2])
    assert np.allclose(out[:, 3], toy.X[:, 3])


def test_the_bounded_shares_are_left_alone():
    """log1p of a share that was never skewed compresses a range and buys nothing."""
    assert "text_edge_share" not in bayes.LOGGABLE
    assert "text_row_alignment" not in bayes.LOGGABLE
    assert "text_n_blocks" in bayes.LOGGABLE


def test_the_transform_does_not_mutate_its_input(toy):
    before = toy.X.copy()
    bayes.log_transform(list(toy.feature_names)).fit_transform(toy.X)
    assert np.array_equal(toy.X, before)


def test_the_transform_survives_a_negative_value(toy):
    """Clipped rather than allowed to produce nan, so one odd row cannot kill a fold."""
    X = toy.X.copy()
    X[0, 0] = -5.0
    out = bayes.log_transform(list(toy.feature_names)).fit_transform(X)
    assert np.isfinite(out).all()


# -- the normality diagnostic ------------------------------------------------------------------------


def test_normality_is_measured_per_feature_and_class(toy):
    result = bayes.normality(toy)
    assert set(result["per_feature"]) == set(toy.feature_names)
    assert 0.0 <= result["share_rejecting"] <= 1.0


def test_a_gaussian_column_is_not_rejected(toy):
    """The control: if this fired, the diagnostic would be measuring something else."""
    rng = np.random.default_rng(3)
    y = np.array(["a"] * 200 + ["b"] * 200, dtype=object)
    X = np.column_stack([rng.normal(size=400), rng.normal(size=400)])
    data = Dataset(
        X=X,
        y=y,
        groups=np.arange(400).astype(object),
        ids=np.arange(400).astype(object),
        feature_names=["g1", "g2"],
        corpus="test",
    )
    assert bayes.normality(data)["share_rejecting"] < 0.5


def test_a_constant_column_is_skipped_not_counted(toy):
    """`normaltest` on a constant raises; skipping it is the honest thing."""
    X = np.column_stack([np.ones(len(toy.y)), toy.X[:, 0]])
    data = Dataset(
        X=X,
        y=toy.y,
        groups=toy.groups,
        ids=toy.ids,
        feature_names=["constant", "counts"],
        corpus="test",
    )
    result = bayes.normality(data)
    assert result["per_feature"]["constant"]["classes_tested"] == 0


# -- the evaluation -----------------------------------------------------------------------------------


def test_a_cell_records_both_knobs_and_the_per_class_scores(toy):
    row = bayes.evaluate(toy, var_smoothing=1e-9, transform="none", folds=3, n_jobs=2)
    assert row["var_smoothing"] == 1e-9
    assert row["transform"] == "none"
    assert set(row["per_class_f1"]) == set(toy.classes)


def test_the_model_can_learn_a_separable_signal(toy):
    """The table is meant to be cheap, not useless."""
    row = bayes.evaluate(toy, folds=3, n_jobs=2)
    assert row["macro_f1"] > 0.3


def test_both_transforms_are_evaluated():
    assert set(bayes.TRANSFORMS) == {"none", "log"}

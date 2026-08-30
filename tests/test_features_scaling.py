"""Phase 4.2.4 - scaling, and the leakage test."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.scaling import feature_scaler, leak_check


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 1.0, size=(200, 6))
    matrix[::7, 2] = np.nan
    return matrix


@pytest.fixture
def held_out():
    """Deliberately a different distribution: shifted mean, inflated variance."""
    rng = np.random.default_rng(1)
    matrix = rng.normal(5.0, 3.0, size=(60, 6))
    matrix[::5, 2] = np.nan
    return matrix


def test_training_rows_come_out_standardised(train):
    scaled = feature_scaler().fit_transform(train)
    assert np.abs(scaled.mean(axis=0)).max() < 1e-9
    assert not np.isnan(scaled).any()


def test_fitting_on_train_only_leaves_the_test_set_uncentred(train, held_out):
    """If the test rows were standardised by their own statistics, something leaked."""
    scaler = feature_scaler().fit(train)
    scaled = scaler.transform(held_out)
    assert np.abs(scaled.mean(axis=0)).max() > 1.0


def test_no_test_row_influences_a_fitted_statistic(train, held_out):
    """The leakage test named in the docstring: T1 == T2 only if nothing leaked."""
    honest = feature_scaler().fit(train).transform(held_out)
    refit = feature_scaler().fit(train).transform(held_out)
    assert np.allclose(honest, refit, atol=1e-12)


def test_a_leaky_fit_is_detectably_different(train, held_out):
    """Proves the test above can fail: fitting on everything moves the numbers a long way."""
    result = leak_check(train, held_out)
    assert result["leak_would_be_visible"]
    assert result["mean_absolute_difference"] > 0.1


def test_holes_do_not_survive_scaling(train, held_out):
    scaled = feature_scaler().fit(train).transform(held_out)
    assert not np.isnan(scaled).any()


def test_the_imputer_runs_before_the_scaler(train):
    """Scaling first would compute statistics over columns that still hold nan."""
    steps = [name for name, _ in feature_scaler().steps]
    assert steps.index("impute") < steps.index("scale")


def test_indicator_columns_are_carried_through(train):
    with_indicator = feature_scaler(add_indicator=True).fit_transform(train)
    without = feature_scaler(add_indicator=False).fit_transform(train)
    assert with_indicator.shape[1] == without.shape[1] + 1

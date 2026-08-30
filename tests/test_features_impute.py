"""Phase 4.2.3 - median imputation and the missingness indicator."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.impute import SUFFIX, MedianImputer


@pytest.fixture
def matrix():
    return np.array(
        [
            [1.0, 10.0, np.nan],
            [3.0, 20.0, np.nan],
            [5.0, np.nan, np.nan],
            [7.0, 40.0, np.nan],
        ]
    )


def test_holes_are_filled_with_the_column_median(matrix):
    filled = MedianImputer(add_indicator=False).fit_transform(matrix)
    assert not np.isnan(filled).any()
    assert filled[2, 1] == pytest.approx(np.nanmedian(matrix[:, 1]))


def test_a_column_that_was_never_observed_is_filled_with_zero(matrix):
    filled = MedianImputer(add_indicator=False).fit_transform(matrix)
    assert (filled[:, 2] == 0.0).all()


def test_an_indicator_is_added_only_for_columns_that_went_missing(matrix):
    imputer = MedianImputer().fit(matrix)
    out = imputer.transform(matrix)
    assert out.shape == (4, 3 + 2), "one indicator each for the two columns with holes"
    assert list(imputer.missing_columns_) == [1, 2]


def test_the_indicator_marks_exactly_the_missing_rows(matrix):
    out = MedianImputer().fit_transform(matrix)
    assert out[:, 3].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert out[:, 4].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_names_follow_the_columns(matrix):
    imputer = MedianImputer().fit(matrix)
    names = list(imputer.get_feature_names_out(["a", "b", "c"]))
    assert names == ["a", "b", "c", f"b{SUFFIX}", f"c{SUFFIX}"]


def test_the_median_comes_from_fit_not_from_transform(matrix):
    """The whole point: a test row must not move the value used to fill it."""
    imputer = MedianImputer(add_indicator=False).fit(matrix)
    other = np.array([[100.0, np.nan, np.nan]])
    assert imputer.transform(other)[0, 1] == pytest.approx(np.nanmedian(matrix[:, 1]))


def test_a_width_mismatch_is_an_error_not_a_silent_reshape(matrix):
    imputer = MedianImputer().fit(matrix)
    with pytest.raises(ValueError):
        imputer.transform(np.zeros((2, 5)))


def test_indicators_can_be_turned_off_for_the_phase_14_ablation(matrix):
    assert MedianImputer(add_indicator=False).fit_transform(matrix).shape == (4, 3)


def test_it_is_a_normal_sklearn_transformer(matrix):
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([("impute", MedianImputer()), ("scale", StandardScaler())])
    out = pipeline.fit_transform(matrix)
    assert out.shape[0] == len(matrix)
    assert clone(MedianImputer()).get_params() == MedianImputer().get_params()

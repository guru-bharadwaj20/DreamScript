"""Phase 4.2.1 - the sklearn-compatible FeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.features import context as ctx
from src.features.extractor import FAMILIES, FEATURE_NAMES, FeatureExtractor, from_context, to_frame


@pytest.fixture
def paths(fixtures_dir):
    return [fixtures_dir / f"{name}.png" for name in ("flowchart", "wireframe", "state_machine")]


def test_every_family_contributes_its_own_names():
    expected = [name for family in FAMILIES for name in family.NAMES]
    assert list(FEATURE_NAMES) == expected
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES), "a feature name is used twice"


def test_the_matrix_is_one_row_per_image_and_one_column_per_name(paths):
    matrix = FeatureExtractor(n_jobs=1).fit_transform(paths)
    assert matrix.shape == (len(paths), len(FEATURE_NAMES))
    assert matrix.dtype == float


def test_column_order_matches_get_feature_names_out(paths):
    extractor = FeatureExtractor(n_jobs=1).fit(paths)
    assert list(extractor.get_feature_names_out()) == list(FEATURE_NAMES)
    row = FeatureExtractor(n_jobs=1).fit_transform(paths[:1])[0]
    values = from_context(ctx.from_image(paths[0]))
    for index, name in enumerate(FEATURE_NAMES):
        assert row[index] == pytest.approx(values[name], nan_ok=True), name


def test_parallel_and_serial_give_the_identical_array(paths):
    """Row order must never depend on which thread finished first."""
    serial = FeatureExtractor(n_jobs=1).fit_transform(paths)
    parallel = FeatureExtractor(n_jobs=4).fit_transform(paths)
    assert np.array_equal(serial, parallel, equal_nan=True)


def test_a_page_that_cannot_be_read_is_a_row_of_nan(tmp_path, paths):
    broken = tmp_path / "not_an_image.png"
    broken.write_bytes(b"this is not a png")
    matrix = FeatureExtractor(n_jobs=1).fit_transform([paths[0], broken])
    assert not np.isnan(matrix[0]).all()
    assert np.isnan(matrix[1]).all()


def test_transform_is_stateless_so_nothing_can_leak(paths):
    """Fitting on one set and transforming another must give the same rows as fitting on it."""
    fitted_elsewhere = FeatureExtractor(n_jobs=1).fit(paths[:1]).transform(paths)
    fitted_here = FeatureExtractor(n_jobs=1).fit(paths).transform(paths)
    assert np.array_equal(fitted_elsewhere, fitted_here, equal_nan=True)


def test_an_empty_batch_is_an_empty_matrix_of_the_right_width():
    assert FeatureExtractor().fit_transform([]).shape == (0, len(FEATURE_NAMES))


def test_it_survives_sklearn_clone_and_get_params():
    extractor = FeatureExtractor(n_jobs=3)
    assert extractor.get_params()["n_jobs"] == 3
    assert clone(extractor).get_params() == extractor.get_params()


def test_it_works_as_the_first_step_of_a_pipeline(paths):
    pipeline = Pipeline(
        [
            ("features", FeatureExtractor(n_jobs=1)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    pipeline.fit(paths, ["flowchart", "wireframe", "state_machine"])
    assert len(pipeline.predict(paths)) == len(paths)


def test_the_frame_carries_the_names(paths):
    frame = to_frame(paths, n_jobs=1)
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == len(paths)

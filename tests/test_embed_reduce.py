"""Phase 6.1.3 - PCA over the embedding cache."""

from __future__ import annotations

import numpy as np
import pytest

from src.embed import reduce as reducer


@pytest.fixture
def embedded():
    """160 rows in 40 dimensions where only the first four carry the label."""
    rng = np.random.default_rng(30)
    y = np.array(["a"] * 80 + ["b"] * 80, dtype=object)
    signal = np.array([[0.0] * 4 if label == "a" else [3.0] * 4 for label in y])
    noise = rng.normal(0, 1.0, size=(160, 36))
    return np.hstack([signal + rng.normal(0, 0.6, size=(160, 4)), noise]).astype(np.float32), y


def test_the_variance_curve_is_monotone(embedded):
    curve = reducer.variance_curve(embedded[0], widths=(2, 4, 8, 16))
    retained = [curve["retained"][key] for key in ("2", "4", "8", "16")]
    assert retained == sorted(retained)
    assert 0.0 < retained[0] <= retained[-1] <= 1.0


def test_more_components_are_needed_for_more_variance(embedded):
    curve = reducer.variance_curve(embedded[0], widths=(4, 8))
    assert (
        curve["components_for_90_percent"]
        <= curve["components_for_95_percent"]
        <= curve["components_for_99_percent"]
    )


def test_a_width_wider_than_the_matrix_does_not_index_past_the_end(embedded):
    curve = reducer.variance_curve(embedded[0], widths=(1000,))
    assert curve["retained"]["1000"] == pytest.approx(1.0, abs=1e-6)


def test_the_variance_curve_ignores_unreadable_rows(embedded):
    matrix, _ = embedded
    matrix = matrix.copy()
    matrix[0] = np.nan
    assert reducer.variance_curve(matrix, widths=(4,))["rows"] == len(matrix) - 1


def test_the_probe_reports_the_unreduced_width_when_asked_for_none(embedded):
    matrix, labels = embedded
    result = reducer.probe_at(matrix, labels, None)
    assert result["width"] == matrix.shape[1]
    assert result["reduced"] is False


def test_reducing_to_the_signal_dimensions_keeps_the_score(embedded):
    """The point of the module: variance is not the same question as separability."""
    matrix, labels = embedded
    full = reducer.probe_at(matrix, labels, None)["macro_f1"]
    reduced = reducer.probe_at(matrix, labels, 8)["macro_f1"]
    assert reduced > 0.8
    assert reduced >= full - 0.15


def test_the_projection_keeps_every_row_and_marks_the_dead_ones(embedded):
    matrix, _ = embedded
    matrix = matrix.copy()
    matrix[3] = np.nan
    reduced, _ = reducer.project(matrix, width=6)
    assert reduced.shape == (len(matrix), 6)
    assert np.isnan(reduced[3]).all()
    assert not np.isnan(reduced[0]).any()


def test_the_projection_is_deterministic(embedded):
    first, _ = reducer.project(embedded[0], width=6)
    second, _ = reducer.project(embedded[0], width=6)
    assert np.allclose(first, second)


def test_the_written_matrix_reads_back_identical(tmp_path, embedded):
    reduced, _ = reducer.project(embedded[0], width=5)
    path = reducer.write(reduced, tmp_path / "pca.npy")
    assert path.is_file()
    assert np.allclose(np.load(path), reduced, equal_nan=True)


def test_the_write_leaves_no_temporary_behind(tmp_path, embedded):
    reduced, _ = reducer.project(embedded[0], width=5)
    reducer.write(reduced, tmp_path / "pca.npy")
    assert not list(tmp_path.glob("*.tmp"))

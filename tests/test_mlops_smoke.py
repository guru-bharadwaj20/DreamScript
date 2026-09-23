"""Phase 15.9 - the smoke train has to be able to fail, or it is a green light wired to nothing.

The risk with a smoke test is that it passes for reasons unrelated to the thing it checks. A
DecisionTree will happily fit a matrix of zeros and predict it back perfectly, so "fit succeeded"
proves nothing on its own. These tests pin the checks that make the smoke train mean something.
"""

from __future__ import annotations

import numpy as np

from src.mlops import smoke


def test_the_fixtures_are_committed_so_ci_needs_no_dataset():
    """0.2.7 committed these precisely so something can be trained with no corpus present."""
    images = smoke.fixture_images()
    assert len(images) == len(smoke.KINDS)
    for path in images:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_the_whole_path_runs_on_the_fixtures():
    result = smoke.run()
    assert result["ok"] is True
    for stage in ("fixtures", "extract", "fit", "predict"):
        assert result["stages"][stage]["ok"] is True, stage


def test_extraction_produces_features_that_actually_vary():
    """An all-NaN or all-constant matrix still fits and still predicts.

    That is the silent failure this smoke train exists to catch: a broken extractor would let
    every later stage report success. So the check is not "did fit work" but "is there anything
    in the matrix to fit".
    """
    result = smoke.run()
    extract = result["stages"]["extract"]
    assert extract["finite_values"] > 0
    assert extract["distinct_columns"] > 5, "features are degenerate; the extractor is not working"


def test_the_matrix_is_one_row_per_fixture():
    result = smoke.run()
    rows, columns = result["stages"]["extract"]["shape"]
    assert rows == len(smoke.KINDS)
    assert columns == result["stages"]["extract"]["features"]


def test_an_all_nan_matrix_is_reported_as_a_failure(monkeypatch):
    """The check has to be able to fail, so make it fail."""

    class DeadExtractor:
        def __init__(self, *a, **k):
            pass

        def fit_transform(self, paths):
            return np.full((len(paths), 4), np.nan)

        def get_feature_names_out(self):
            return np.array(["a", "b", "c", "d"])

    monkeypatch.setattr("src.features.extractor.FeatureExtractor", DeadExtractor)
    result = smoke.run()
    assert result["ok"] is False
    assert "NaN" in result.get("error", "")


def test_missing_fixtures_are_reported_rather_than_passing(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "FIXTURES", tmp_path)
    result = smoke.run()
    assert result["ok"] is False
    assert "no fixtures" in result.get("error", "")


def test_the_accuracy_is_not_presented_as_a_result():
    """Five images, one per class - a tree memorises it. The report must say so."""
    result = smoke.run()
    predict = result["stages"]["predict"]
    assert "not a measurement" in predict["note"]
    # And there is no top-level "accuracy" key that a reader could mistake for one.
    assert "accuracy" not in result


def test_check_exits_non_zero_when_a_stage_is_broken(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "FIXTURES", tmp_path)
    assert smoke.main(["--check"]) == 1


def test_check_exits_zero_when_the_path_is_sound():
    assert smoke.main(["--check"]) == 0


def test_it_is_fast_enough_to_belong_in_ci():
    """A smoke train nobody waits for is a smoke train nobody runs."""
    result = smoke.run()
    assert result["seconds"] < 60

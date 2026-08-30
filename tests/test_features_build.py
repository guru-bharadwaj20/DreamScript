"""Phase 4.2.2 - the feature table build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build
from src.features.extractor import FEATURE_NAMES


@pytest.fixture
def rows(fixtures_dir):
    return [
        {
            "id": f"fixture/{name}",
            "source": "fixtures",
            "diagram_type": name,
            "split": "train",
            "scribe_id": None,
            "adverse": False,
            "synthetic": True,
            "path": fixtures_dir / f"{name}.png",
        }
        for name in ("flowchart", "wireframe", "circuit")
    ]


def test_the_table_carries_identity_and_features_in_that_order(rows):
    table = build.build(rows, n_jobs=1)
    assert list(table.columns) == [*build.IDENTITY, *FEATURE_NAMES]
    assert len(table) == len(rows)


def test_identity_columns_are_not_features():
    """A model trained on `id` has memorised the corpus; on `source`, it has learned the leak."""
    assert not set(build.IDENTITY) & set(FEATURE_NAMES)


def test_rows_keep_the_order_they_were_collected_in(rows):
    table = build.build(rows, n_jobs=1)
    assert list(table["id"]) == [row["id"] for row in rows]
    assert list(table["diagram_type"]) == [row["diagram_type"] for row in rows]


def test_an_empty_build_is_an_empty_table_with_the_right_columns():
    table = build.build([], n_jobs=1)
    assert list(table.columns) == [*build.IDENTITY, *FEATURE_NAMES]
    assert table.empty


def test_writing_is_atomic_and_round_trips(tmp_path, rows):
    target = tmp_path / "handcrafted.parquet"
    written = build.write(build.build(rows, n_jobs=1), target)
    assert written == target
    assert not list(tmp_path.glob("*.tmp")), "the temporary file was left behind"
    reloaded = pd.read_parquet(target)
    assert list(reloaded.columns) == [*build.IDENTITY, *FEATURE_NAMES]
    assert len(reloaded) == len(rows)


def test_a_second_write_replaces_rather_than_appends(tmp_path, rows):
    target = tmp_path / "handcrafted.parquet"
    build.write(build.build(rows, n_jobs=1), target)
    build.write(build.build(rows[:1], n_jobs=1), target)
    assert len(pd.read_parquet(target)) == 1


def test_the_build_is_deterministic(rows):
    """A row is a pure function of the image bytes, as in 3.2.9."""
    first = build.build(rows, n_jobs=1)[list(FEATURE_NAMES)].to_numpy()
    second = build.build(rows, n_jobs=2)[list(FEATURE_NAMES)].to_numpy()
    assert np.array_equal(first, second, equal_nan=True)


def test_the_summary_reports_missingness_rather_than_hiding_it(rows):
    summary = build.summarise(build.build(rows, n_jobs=1))
    assert summary["rows"] == len(rows)
    assert summary["features"] == len(FEATURE_NAMES)
    assert 0.0 <= summary["missing_share"] <= 1.0
    assert set(summary["by_type"]) == {row["diagram_type"] for row in rows}


def test_collect_skips_images_that_are_not_on_disk(monkeypatch, tmp_path):
    missing = [
        {
            "id": "gone",
            "source": "s",
            "diagram_type": "flowchart",
            "split": "train",
            "scribe_id": None,
            "adverse": False,
            "synthetic": True,
            "path": tmp_path / "nothing.png",
        }
    ]
    monkeypatch.setattr(build, "_synthetic_rows", lambda limit: missing)
    monkeypatch.setattr(build, "_real_rows", lambda limit: [])
    assert build.collect(10) == []

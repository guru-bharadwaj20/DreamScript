"""Phase 5 - the shared dataset contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.classify import data
from src.features.extractor import FEATURE_NAMES


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    rows = 40
    table = pd.DataFrame(rng.normal(size=(rows, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    table["id"] = [f"row{i}" for i in range(rows)]
    table["source"] = ["hdbpmn"] * 20 + ["sketch2code"] * 20
    table["diagram_type"] = ["flowchart"] * 20 + ["wireframe"] * 20
    table["split"] = "train"
    table["scribe_id"] = [f"w{i % 5}" for i in range(20)] + [None] * 20
    table["adverse"] = False
    table["synthetic"] = [False] * 30 + [True] * 10
    return table


def test_the_real_corpus_is_the_default(frame):
    dataset = data.load(frame=frame)
    assert dataset.corpus == "real"
    assert len(dataset.y) == 30


def test_the_two_corpora_are_never_mixed_unless_asked(frame):
    """4.2.7 measured them separable at AUC 0.953; pooling them silently would hide that."""
    assert len(data.load("synthetic", frame=frame).y) == 10
    assert len(data.load("all", frame=frame).y) == len(frame)


def test_an_unknown_corpus_is_an_error(frame):
    with pytest.raises(ValueError):
        data.load("everything", frame=frame)


def test_the_pruned_feature_is_gone_and_the_leaky_one_is_not(frame):
    dataset = data.load(frame=frame)
    assert "layout_node_density" not in dataset.feature_names
    assert "global_aspect" in dataset.feature_names


def test_dropping_the_leak_removes_exactly_one_column(frame):
    kept = data.load(frame=frame)
    dropped = data.load(frame=frame, drop_leaky=True)
    assert dropped.n_features == kept.n_features - 1
    assert "global_aspect" not in dropped.feature_names


def test_columns_stay_in_the_frozen_order(frame):
    dataset = data.load(frame=frame)
    expected = [n for n in FEATURE_NAMES if n != "layout_node_density"]
    assert dataset.feature_names == expected


def test_a_row_without_a_writer_gets_its_own_group(frame):
    """One shared "unknown" group would put every wireframe in the same fold."""
    dataset = data.load(frame=frame)
    unknown = [g for g in dataset.groups if g.startswith("row:")]
    assert len(unknown) == 10
    assert len(set(unknown)) == 10


def test_writers_are_shared_groups(frame):
    dataset = data.load(frame=frame)
    known = [g for g in dataset.groups if g.startswith("scribe:")]
    assert len(known) == 20
    assert len(set(known)) == 5


def test_the_matrix_and_the_labels_line_up(frame):
    dataset = data.load(frame=frame)
    assert dataset.X.shape == (len(dataset.y), dataset.n_features)
    assert len(dataset.ids) == len(dataset.y) == len(dataset.groups)


def test_the_summary_reports_the_imbalance(frame):
    summary = data.load(frame=frame).summary()
    assert summary["classes"] == {"flowchart": 20, "wireframe": 10}
    assert summary["rows_with_a_known_scribe"] == 20
    assert summary["minority_share"] == pytest.approx(10 / 30, abs=1e-4)


def test_a_missing_table_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load(table=tmp_path / "nothing.parquet")

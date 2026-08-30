"""Phase 4.2.7 - feature-space projections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import viz
from src.features.extractor import FEATURE_NAMES


@pytest.fixture
def table():
    """Three well-separated clusters, half of each marked as a different corpus."""
    rng = np.random.default_rng(0)
    blocks = [rng.normal(centre, 0.5, size=(60, len(FEATURE_NAMES))) for centre in (0.0, 6.0, 12.0)]
    frame = pd.DataFrame(np.vstack(blocks), columns=list(FEATURE_NAMES))
    frame["diagram_type"] = np.repeat(["flowchart", "wireframe", "circuit"], 60)
    frame["split"] = "train"
    frame["synthetic"] = np.tile([True, False], 90)
    frame.iloc[::11, 0] = np.nan
    return frame


def test_every_projection_is_two_dimensional(table):
    scaled = viz.projections(
        np.asarray(table[list(FEATURE_NAMES)].fillna(0.0).to_numpy(float)), seed=0
    )
    for key in ("pca", "tsne"):
        assert scaled[key].shape == (len(table), 2)


def test_the_figure_is_written(tmp_path, table):
    matrix = table[list(FEATURE_NAMES)].fillna(0.0).to_numpy(float)
    embeddings = viz.projections(matrix, seed=0)
    path = viz.figure(
        embeddings,
        table["diagram_type"].to_numpy(),
        table["synthetic"].to_numpy(),
        tmp_path / "space.png",
    )
    assert path.is_file() and path.stat().st_size > 5000


def test_it_runs_end_to_end_over_a_table_with_holes(table, monkeypatch, tmp_path):
    monkeypatch.setattr(viz, "FIGURE", tmp_path / "space.png")
    result = viz.run(table, sample=0)
    assert result["rows"] == len(table)
    assert result["columns_after_scaling"] >= len(FEATURE_NAMES)
    assert len(result["pca_explained_variance"]) == 2


def test_corpus_separability_detects_two_distinguishable_corpora(table):
    """The check the fourth panel exists for, as a number rather than an impression."""
    indistinguishable = viz.corpus_separability(table, seed=0)
    assert 0.3 < indistinguishable < 0.7, "randomly assigned corpora must not be separable"

    separable = table.copy()
    separable["synthetic"] = separable["diagram_type"] == "circuit"
    assert viz.corpus_separability(separable, seed=0) > 0.9


def test_separability_is_nan_when_there_is_only_one_corpus(table):
    single = table.copy()
    single["synthetic"] = True
    assert np.isnan(viz.corpus_separability(single, seed=0))


def test_every_diagram_type_has_a_colour():
    from tests.conftest import DIAGRAM_TYPES

    assert set(DIAGRAM_TYPES) == set(viz.COLOURS)

"""Phase 5.3.1 - kNN decision boundaries in two dimensions."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import boundaries
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(13)
    y = np.array(["a"] * 80 + ["b"] * 80 + ["rare"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "rare": (1.5, 3.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.6, size=(180, 2))
    X = np.hstack([X, rng.normal(size=(180, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(180)], dtype=object),
        ids=np.arange(180).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


def test_pca_gives_two_columns_and_named_axes(toy):
    transform, names = boundaries.project(toy, "pca")
    assert transform(toy.X).shape == (len(toy.y), 2)
    assert len(names) == 2 and names[0].startswith("PC1")


def test_top2_names_real_features(toy):
    transform, names = boundaries.project(toy, "top2")
    assert transform(toy.X).shape == (len(toy.y), 2)
    assert all(name in toy.feature_names for name in names)


def test_an_unknown_projection_is_refused(toy):
    with pytest.raises(ValueError, match="pca or top2"):
        boundaries.project(toy, "umap")


def test_the_projection_only_learns_from_the_rows_it_is_given(toy):
    """Fitting on a subset must produce a different map, or `fit_rows` is doing nothing."""
    everything, _ = boundaries.project(toy, "pca")
    subset, _ = boundaries.project(toy, "pca", fit_rows=np.arange(60))
    assert not np.allclose(everything(toy.X), subset(toy.X))


def test_the_mesh_covers_every_point(toy):
    embedded = boundaries.project(toy, "pca")[0](toy.X)
    result = boundaries.boundary(embedded, toy.y, k=5, steps=40)
    left, right, bottom, top = result["extent"]
    assert left <= embedded[:, 0].min() and right >= embedded[:, 0].max()
    assert bottom <= embedded[:, 1].min() and top >= embedded[:, 1].max()


def test_the_region_map_has_one_cell_per_mesh_point(toy):
    embedded = boundaries.project(toy, "pca")[0](toy.X)
    result = boundaries.boundary(embedded, toy.y, k=1, steps=32)
    assert result["regions"].shape == (32, 32)
    assert result["regions"].max() < len(result["classes"])


def test_the_area_shares_sum_to_the_whole_plane(toy):
    embedded = boundaries.project(toy, "pca")[0](toy.X)
    result = boundaries.boundary(embedded, toy.y, k=5, steps=48)
    assert sum(result["area_share"].values()) == pytest.approx(1.0, abs=1e-3)


def test_one_neighbour_never_misses_a_training_point(toy):
    embedded = boundaries.project(toy, "pca")[0](toy.X)
    assert boundaries.boundary(embedded, toy.y, k=1, steps=16)["resubstitution_accuracy"] == 1.0


def test_smoothing_shrinks_the_minority_region(toy):
    """The module's kNN finding: a large k erases the small class from the plane."""
    embedded = boundaries.project(toy, "pca")[0](toy.X)
    small = boundaries.boundary(embedded, toy.y, k=1, steps=64)["area_share"]["rare"]
    smoothed = boundaries.boundary(embedded, toy.y, k=25, steps=64)["area_share"]["rare"]
    assert smoothed < small


def test_the_projected_score_is_a_real_cross_validated_number(toy):
    score = boundaries.projected_score(toy, "pca", k=5)
    assert 0.0 <= score <= 1.0
    # A leaked projection would push this to the resubstitution accuracy; it must not.
    assert score < 1.0


def test_the_figure_is_written(tmp_path, toy):
    sheets = [boundaries.panel(toy, "pca", k_values=(1, 5))]
    path = boundaries.figure(sheets, toy.y, {1: 0.8}, tmp_path / "boundaries.png")
    assert path.is_file() and path.stat().st_size > 5000

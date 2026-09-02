"""Phase 6.3.6 - kernel decision surfaces, and the honesty number every panel has to carry."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import surface
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(61)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (1.5, 2.5)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.7, size=(140, 2))
    X = np.hstack([X, rng.normal(size=(140, 6))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(140)], dtype=object),
        ids=np.arange(140).astype(object),
        feature_names=[f"f{i}" for i in range(8)],
        corpus="test",
    )


# -- the leak the projection could introduce ---------------------------------------------------


def test_the_projection_is_part_of_the_estimator_so_cv_refits_it_per_fold(toy):
    """A PCA fitted on all rows then cross-validated has chosen its axes using the test fold."""
    estimator = surface.projected_estimator("linear")
    assert list(estimator.named_steps) == ["project", "model"]
    inner = estimator.named_steps["project"]
    assert list(inner.named_steps) == ["prepare", "pca"]


def test_the_projection_scales_before_it_rotates(toy):
    """PCA on unscaled columns finds the largest-variance column, not the largest structure."""
    inner = surface.projector()
    assert list(inner.named_steps)[0] == "prepare"


def test_the_projection_keeps_two_components(toy):
    fitted = surface.projector().fit(toy.X)
    assert fitted.named_steps["pca"].n_components == 2
    assert fitted.transform(toy.X).shape == (len(toy.y), 2)


def test_the_retained_variance_is_reported(toy):
    value = surface.retained_variance(toy)
    assert 0.0 < value <= 1.0


# -- the honesty number --------------------------------------------------------------------------


def test_the_projected_model_is_scored_not_just_drawn(toy):
    """5.3.1's rule: a boundary plot is a picture of a different model, so the model is scored."""
    score = surface.score_projection(toy, "linear", folds=3, n_jobs=2)
    assert 0.0 <= score <= 1.0


def test_the_full_dimension_reference_exists_for_every_kernel_drawn():
    assert set(surface.FULL_DIMENSION) >= set(surface.KERNELS)


def test_the_summary_reports_what_the_picture_cost(toy, monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.embed.hybrid",
        type("m", (), {"dataset": staticmethod(lambda *a, **k: toy)})(),
    )
    result = surface.run("hybrid", kernels=("linear",), n_jobs=2, write=False)
    assert result["cost_of_two_dimensions"]["linear"] == pytest.approx(
        surface.FULL_DIMENSION["linear"] - result["projected_macro_f1"]["linear"], abs=1e-4
    )


# -- the surface itself --------------------------------------------------------------------------------


def test_the_surface_grid_covers_the_projected_points(toy):
    panel = surface.surface(toy, "linear", resolution=60)
    assert panel["xx"].shape == panel["zz"].shape
    assert panel["points"].shape == (len(toy.y), 2)
    assert panel["xx"].min() <= panel["points"][:, 0].min()
    assert panel["xx"].max() >= panel["points"][:, 0].max()


def test_the_grid_is_classified_in_projected_space(toy):
    """Pushing the mesh back through the projection would draw a different transform."""
    import inspect

    source = inspect.getsource(surface.surface)
    assert "model.predict(grid)" in source


def test_the_area_shares_sum_to_one(toy):
    panel = surface.surface(toy, "rbf", resolution=60)
    assert sum(panel["area_share"].values()) == pytest.approx(1.0, abs=1e-3)


def test_a_class_with_no_region_is_reported(toy):
    """er_diagram gets exactly 0.0% of the plane on the real corpus, under all three kernels."""
    panel = surface.surface(toy, "linear", resolution=60)
    zeros = [name for name, share in panel["area_share"].items() if share == 0.0]
    assert panel["classes_with_no_region"] == zeros


@pytest.mark.parametrize("kernel", surface.KERNELS)
def test_every_kernel_produces_a_surface(toy, kernel):
    panel = surface.surface(toy, kernel, resolution=40)
    assert panel["kernel"] == kernel
    assert panel["zz"].shape == (40, 40)


# -- the figure ----------------------------------------------------------------------------------------


def test_the_figure_is_written(toy, tmp_path):
    panels = [surface.surface(toy, kernel, resolution=40) for kernel in ("linear", "rbf")]
    scores = {"linear": 0.5, "rbf": 0.51}
    path = surface.figure(panels, toy.y, scores, {"linear": 0.96, "rbf": 0.97}, tmp_path / "s.png")
    assert path.is_file()
    assert path.stat().st_size > 5000


def test_each_panel_title_carries_both_numbers():
    """So the figure cannot be quoted without its own refutation attached."""
    import inspect

    source = inspect.getsource(surface.figure)
    assert "2-D" in source and "161-D" in source

"""Phase 6.2.1 - MLP topology search."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import mlp
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(50)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 60, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (1.5, 3.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.7, size=(180, 2))
    X = np.hstack([X, rng.normal(size=(180, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(180)], dtype=object),
        ids=np.arange(180).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


def test_the_grid_covers_one_two_and_three_layers():
    shapes = mlp.topologies((64, 128))
    depths = {len(shape) for shape in shapes}
    assert depths == {1, 2, 3}
    assert (64,) in shapes and (128, 128, 128) in shapes


def test_the_tapering_shapes_are_included():
    """A fixed-width grid cannot express these, and they are the usual answer on a small corpus."""
    shapes = mlp.topologies()
    assert (256, 128) in shapes
    assert (512, 256, 128) in shapes


def test_every_topology_is_distinct():
    shapes = mlp.topologies()
    assert len(shapes) == len(set(shapes))


def test_the_parameter_count_matches_the_hand_calculation():
    # 4 -> 8 -> 3:  (4*8 + 8) + (8*3 + 3) = 40 + 27 = 67
    assert mlp.parameter_count(4, (8,), 3) == 67


def test_the_parameter_count_chains_through_every_layer():
    # 10 -> 5 -> 2 -> 3:  (10*5+5) + (5*2+2) + (2*3+3) = 55 + 12 + 9 = 76
    assert mlp.parameter_count(10, (5, 2), 3) == 76


def test_a_wider_first_layer_costs_more_parameters():
    narrow = mlp.parameter_count(161, (64,), 5)
    wide = mlp.parameter_count(161, (512,), 5)
    assert wide > narrow * 7


def test_the_pipeline_scales_before_the_network(toy):
    """An unscaled column saturates the first layer and kills the gradient through it."""
    estimator = mlp.pipeline(hidden_layer_sizes=(8,), max_iter=50)
    assert list(estimator.named_steps) == ["prepare", "model"]
    estimator.fit(toy.X, toy.y)
    assert estimator.predict(toy.X).shape == toy.y.shape


def test_early_stopping_is_on_by_default():
    assert mlp.pipeline().named_steps["model"].early_stopping is True


def test_the_best_estimator_uses_the_recorded_parameters():
    model = mlp.best_estimator().named_steps["model"]
    assert model.hidden_layer_sizes == mlp.BEST_PARAMS["hidden_layer_sizes"]


def test_the_best_estimator_can_be_overridden():
    model = mlp.best_estimator(hidden_layer_sizes=(16,)).named_steps["model"]
    assert model.hidden_layer_sizes == (16,)


def test_the_search_ranks_and_reports_every_candidate(toy):
    result = mlp.search(toy, widths=(8, 16), n_jobs=2)
    assert result["candidates"] == len(mlp.topologies((8, 16)))
    assert len(result["all"]) == result["candidates"]
    scores = [row["macro_f1"] for row in result["all"]]
    assert scores == sorted(scores, reverse=True)
    assert result["best"]["macro_f1"] >= result["worst"]["macro_f1"]


def test_the_search_reports_the_best_at_each_depth(toy):
    result = mlp.search(toy, widths=(8,), n_jobs=2)
    assert set(result["best_by_depth"]) >= {"1", "2", "3"}
    assert all(0.0 <= value <= 1.0 for value in result["best_by_depth"].values())


def test_the_search_counts_parameters_for_the_real_feature_width(toy):
    result = mlp.search(toy, widths=(8,), n_jobs=2)
    for row in result["all"]:
        assert row["parameters"] == mlp.parameter_count(
            toy.n_features, tuple(row["hidden_layer_sizes"]), 3
        )

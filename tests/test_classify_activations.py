"""Phase 6.2.2 - the activation comparison, and the dead-unit count behind it."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import activations
from src.classify.data import Dataset

pytest.importorskip("torch")


@pytest.fixture
def toy():
    rng = np.random.default_rng(51)
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


SMALL = {"max_epochs": 25, "device": "cpu"}


@pytest.fixture(autouse=True)
def tiny_topology(monkeypatch):
    """The module pins 6.2.1's (512, 256); the tests are about the comparison, not the size."""
    monkeypatch.setattr(activations, "TOPOLOGY", (8, 8))


# -- the comparison ------------------------------------------------------------------------


def test_it_compares_exactly_the_four_activations_the_plan_names():
    from src.classify.torchnet import ACTIVATIONS

    assert set(ACTIVATIONS) == {"relu", "leaky_relu", "gelu", "tanh"}


def test_the_ranking_is_ordered_and_covers_every_activation(toy):
    rows = activations.compare(toy, folds=3, max_epochs=25, device="cpu")
    assert {row["activation"] for row in rows} == set(activations.ACTIVATIONS)
    scores = [row["macro_f1"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_every_row_carries_both_sides_of_the_overfit_gap(toy):
    """6.2.3's opening position is read off this table, so it has to be on it."""
    rows = activations.compare(toy, ("relu", "gelu"), folds=3, max_epochs=25, device="cpu")
    for row in rows:
        assert row["overfit_gap"] == pytest.approx(
            row["train_macro_f1"] - row["macro_f1"], abs=1e-4
        )


def test_the_topology_is_held_fixed_so_the_table_is_about_the_activation():
    """A comparison that also varied the width would not be an activation comparison."""
    import inspect

    source = inspect.getsource(activations.compare)
    assert "hidden_layer_sizes=TOPOLOGY" in source


# -- the dead-unit measurement --------------------------------------------------------------


def test_relu_is_the_only_activation_that_can_produce_a_dead_unit(toy):
    """tanh cannot emit an exact zero and GELU only does so in the limit."""
    for name in ("leaky_relu", "gelu", "tanh"):
        result = activations.dead_unit_fraction(toy, name, **SMALL)
        assert result["dead_units"] == 0, name


def test_the_dead_unit_count_is_out_of_every_hidden_unit(toy):
    result = activations.dead_unit_fraction(toy, "relu", **SMALL)
    assert result["hidden_units"] == 16
    assert len(result["dead_by_layer"]) == 2
    assert sum(result["dead_by_layer"]) == result["dead_units"]


def test_the_dead_fraction_is_the_count_over_the_units(toy):
    result = activations.dead_unit_fraction(toy, "relu", **SMALL)
    assert result["dead_fraction"] == pytest.approx(
        result["dead_units"] / result["hidden_units"], abs=1e-4
    )
    assert 0.0 <= result["dead_fraction"] <= 1.0


def test_a_unit_is_dead_by_what_it_emits_not_by_its_pre_activation(toy):
    """A live ReLU unit is negative for most rows; only silence on *every* row is death."""
    import inspect

    source = inspect.getsource(activations.dead_unit_fraction)
    assert "register_forward_hook" in source
    assert "max(axis=0)" in source


# -- the summary the plan row is graded on ---------------------------------------------------


def test_the_spread_is_reported_against_the_networks_own_fold_noise(toy, monkeypatch):
    """0.013 of activation effect is only readable beside a 0.017 fold-to-fold spread."""
    monkeypatch.setattr(
        activations,
        "compare",
        lambda *a, **k: [
            {
                "activation": "gelu",
                "macro_f1": 0.94,
                "std": 0.02,
                "train_macro_f1": 0.99,
                "overfit_gap": 0.05,
                "epochs": 20.0,
                "seconds": 1.0,
            },
            {
                "activation": "relu",
                "macro_f1": 0.93,
                "std": 0.01,
                "train_macro_f1": 0.99,
                "overfit_gap": 0.06,
                "epochs": 20.0,
                "seconds": 1.0,
            },
        ],
    )
    monkeypatch.setattr(activations, "dead_unit_fraction", lambda *a, **k: {})
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.embed.hybrid",
        type("m", (), {"dataset": staticmethod(lambda *a, **k: toy)})(),
    )
    summary = activations.run("hybrid", dead_units=False)
    assert summary["best"] == "gelu"
    assert summary["spread"] == pytest.approx(0.01, abs=1e-6)
    assert summary["spread_in_fold_std"] == pytest.approx(0.5, abs=1e-2)


def test_every_table_is_scored_against_6_1_4s_logistic_regression():
    """The activation ranking is the small question; beating the linear model is the large one."""
    assert set(activations.LOGREG) == set(activations.TABLES)
    assert activations.LOGREG["embedding"] == 0.9554

"""Phase 6.2.4 - five optimizers, each at its own learning rate, and the figure that separates
convergence from generalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import optimizers
from src.classify.data import Dataset

pytest.importorskip("torch")


@pytest.fixture
def toy():
    rng = np.random.default_rng(55)
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


# -- the grid, and the variable it controls for --------------------------------------------


def test_every_optimizer_the_plan_names_is_swept():
    from src.classify.torchnet import OPTIMIZERS

    assert set(optimizers.ORDER) == set(OPTIMIZERS)
    assert optimizers.ORDER[0] == "sgd"


def test_each_optimizer_has_its_own_learning_rate_grid():
    """Comparing them at one rate measures the rate; that is the whole design of this task."""
    assert set(optimizers.RATES) == set(optimizers.ORDER)
    assert max(optimizers.RATES["sgd"]) > max(optimizers.RATES["adam"]) * 10


def test_the_sgd_grid_reaches_the_rates_sgd_actually_wants():
    """It selected 0.3; a grid topping out at 0.01 would have reported SGD as broken."""
    assert max(optimizers.RATES["sgd"]) >= 0.3
    assert max(optimizers.RATES["momentum"]) >= 0.1


def test_the_adaptive_grids_bracket_their_default():
    for name in ("adam", "adamw", "rmsprop"):
        assert min(optimizers.RATES[name]) < 1e-3 < max(optimizers.RATES[name])


def test_the_grid_is_every_optimizer_at_every_one_of_its_rates():
    configs = optimizers.grid(base={"activation": "gelu"})
    assert len(configs) == sum(len(rates) for rates in optimizers.RATES.values())
    assert {config["activation"] for config in configs} == {"gelu"}
    for config in configs:
        assert config["learning_rate"] in optimizers.RATES[config["optimizer"]]


# -- reading the sweep ------------------------------------------------------------------------


def test_the_winner_per_optimizer_carries_the_rate_that_won():
    rows = [
        {"optimizer": "sgd", "learning_rate": 0.003, "macro_f1": 0.60},
        {"optimizer": "sgd", "learning_rate": 0.3, "macro_f1": 0.94},
        {"optimizer": "adam", "learning_rate": 0.001, "macro_f1": 0.95},
    ]
    best = optimizers.best_per_optimizer(rows)
    by_name = {row["optimizer"]: row for row in best}
    assert by_name["sgd"]["learning_rate"] == 0.3
    assert by_name["sgd"]["macro_f1"] == 0.94


def test_the_winners_come_back_in_the_plans_order():
    rows = [
        {"optimizer": name, "learning_rate": 0.01, "macro_f1": 0.9}
        for name in ("adamw", "sgd", "adam")
    ]
    assert [row["optimizer"] for row in optimizers.best_per_optimizer(rows)] == [
        "sgd",
        "adam",
        "adamw",
    ]


def test_convergence_speed_uses_a_shared_target():
    """Each optimizer's own final loss would let the worst converger 'arrive' first."""
    assert optimizers.epochs_to_reach([1.0, 0.5, 0.2, 0.1], 0.25) == 3
    assert optimizers.epochs_to_reach([1.0, 0.9], 0.25) is None


def test_the_target_is_reached_on_the_first_epoch_at_or_below_it():
    assert optimizers.epochs_to_reach([0.2, 0.1], 0.2) == 1


# -- the figure ---------------------------------------------------------------------------------


def test_the_figure_is_written_and_has_both_panels(tmp_path):
    traces = {
        name: {"learning_rate": 0.01, "loss": [1.0, 0.5, 0.2], "val": [0.5, 0.7, 0.8], "epochs": 3}
        for name in optimizers.ORDER
    }
    path = optimizers.figure(traces, tmp_path / "p6_optimizers.png")
    assert path.is_file()
    assert path.stat().st_size > 5000


def test_the_figure_survives_a_missing_optimizer(tmp_path):
    """A partial sweep should still draw, rather than raising in the reporting step."""
    traces = {"adam": {"learning_rate": 0.001, "loss": [1.0, 0.4], "val": [0.4, 0.9], "epochs": 2}}
    assert optimizers.figure(traces, tmp_path / "one.png").is_file()


def test_the_curves_come_from_one_fold_not_an_average(toy):
    """Folds stop at different epochs, so averaging pads or truncates a run that never happened."""
    import inspect

    source = inspect.getsource(optimizers.curves)
    assert "next(" in source and "split(" in source


def test_the_curves_are_recorded_per_optimizer(toy):
    rows = [
        {
            "optimizer": "adam",
            "learning_rate": 0.01,
            "hidden_layer_sizes": (8,),
            "activation": "gelu",
            "max_epochs": 8,
        },
        {
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "hidden_layer_sizes": (8,),
            "activation": "gelu",
            "max_epochs": 8,
        },
    ]
    traces = optimizers.curves(toy, rows, folds=3)
    assert set(traces) == {"adam", "sgd"}
    for trace in traces.values():
        assert trace["loss"] and trace["val"]
        assert len(trace["loss"]) == trace["epochs"]


def test_only_fit_arguments_reach_the_estimator(toy):
    """Sweep rows carry `macro_f1` and `seconds`; passing them to the constructor would raise."""
    rows = [
        {
            "optimizer": "adam",
            "learning_rate": 0.01,
            "hidden_layer_sizes": (8,),
            "activation": "gelu",
            "max_epochs": 5,
            "macro_f1": 0.9,
            "seconds": 1.0,
            "std": 0.01,
            "overfit_gap": 0.02,
            "epochs": 5.0,
        }
    ]
    traces = optimizers.curves(toy, rows, folds=3)
    assert traces["adam"]["loss"]


# -- the claim the docstring rests on --------------------------------------------------------------


def test_a_shared_rate_would_misrepresent_sgd(toy):
    """The measured version of this on the real corpus is 0.5970 against 0.9360."""
    from src.classify.torchnet import cross_validate

    settings = {
        "hidden_layer_sizes": (32, 16),
        "activation": "gelu",
        "max_epochs": 40,
        "device": "cpu",
        "folds": 3,
    }
    at_adams_rate = cross_validate(toy, optimizer="sgd", learning_rate=1e-4, **settings)
    at_its_own = cross_validate(toy, optimizer="sgd", learning_rate=0.3, **settings)
    assert at_its_own["macro_f1"] > at_adams_rate["macro_f1"]

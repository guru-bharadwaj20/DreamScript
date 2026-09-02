"""Phase 6.2.5 - four schedules, and the horizon that makes three of them meaningless."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import schedules
from src.classify.data import Dataset
from src.classify.torchnet import SCHEDULES

pytest.importorskip("torch")


@pytest.fixture
def toy():
    rng = np.random.default_rng(53)
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


# -- the grid, and the argument behind it -----------------------------------------------------


def test_all_four_plan_schedules_are_swept():
    assert set(SCHEDULES) == {"none", "step", "cosine", "one_cycle"}


def test_the_horizon_is_swept_as_well_as_the_schedule():
    """A schedule is a function of the total length; one horizon asks the question once."""
    configs = schedules.grid()
    assert {config["max_epochs"] for config in configs} == set(schedules.HORIZONS)
    assert len(configs) == len(schedules.HORIZONS) * len(SCHEDULES)


def test_the_horizons_bracket_the_epochs_early_stopping_actually_uses():
    """6.2.2 and 6.2.3 measured 20-40 epochs; 600 is 6.2.1's untouched ceiling."""
    assert max(schedules.HORIZONS) == 600
    assert min(schedules.HORIZONS) <= 30


def test_the_constant_rate_is_the_control():
    assert schedules.CONTROL == "none"
    assert schedules.CONTROL in SCHEDULES


def test_early_stopping_stays_on_across_the_grid():
    """The plan wants the schedule chosen for the pipeline Phase 6 ships, and it stops early."""
    assert all("early_stopping" not in config for config in schedules.grid())


# -- the summaries -----------------------------------------------------------------------------


def test_rows_regroup_by_horizon_then_schedule():
    rows = [
        {"max_epochs": 600, "schedule": "none", "macro_f1": 0.90},
        {"max_epochs": 600, "schedule": "cosine", "macro_f1": 0.91},
        {"max_epochs": 30, "schedule": "none", "macro_f1": 0.92},
        {"max_epochs": 30, "schedule": "cosine", "macro_f1": 0.95},
    ]
    grouped = schedules.by_horizon(rows)
    assert grouped["600"]["cosine"] == 0.91
    assert grouped["30"]["cosine"] == 0.95


def test_the_horizon_effect_is_measured_against_the_constant_rate():
    rows = [
        {"max_epochs": 600, "schedule": "none", "macro_f1": 0.90},
        {"max_epochs": 600, "schedule": "cosine", "macro_f1": 0.901},
        {"max_epochs": 30, "schedule": "none", "macro_f1": 0.90},
        {"max_epochs": 30, "schedule": "cosine", "macro_f1": 0.94},
    ]
    effect = schedules.horizon_effect(rows)
    assert effect["600"]["gain_over_constant"] == pytest.approx(0.001, abs=1e-9)
    assert effect["30"]["gain_over_constant"] == pytest.approx(0.04, abs=1e-9)
    assert effect["30"]["best_schedule"] == "cosine"


def test_the_horizon_effect_skips_a_horizon_with_no_control():
    effect = schedules.horizon_effect([{"max_epochs": 30, "schedule": "cosine", "macro_f1": 0.9}])
    assert effect == {}


# -- the mechanism the whole task turns on -------------------------------------------------------


def scheduler_for(horizon: int, name: str = "cosine"):
    """One scheduler over `horizon` epochs, detached from any network."""
    import torch

    from src.classify.torchnet import _schedule

    optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
    return _schedule(name, optimizer, horizon, 17, 1e-3)[0]


def rates_over(horizon: int, epochs: int) -> list[float]:
    scheduler = scheduler_for(horizon)
    seen = []
    for _ in range(epochs):
        seen.append(scheduler.optimizer.param_groups[0]["lr"])
        scheduler.step()
    return seen


def test_a_cosine_over_600_epochs_has_barely_moved_by_epoch_25():
    """The docstring's central claim: at 6.2.1's ceiling the schedule is a constant rate."""
    rates = rates_over(600, 25)
    assert rates[-1] / rates[0] > 0.99


def test_a_cosine_over_30_epochs_is_a_real_cosine_by_epoch_25():
    """And at a horizon early stopping actually reaches, it has annealed by 90%."""
    rates = rates_over(30, 25)
    assert rates[-1] / rates[0] < 0.1


def test_the_rate_traces_are_read_off_the_run_not_recomputed(toy):
    """One-cycle steps per batch, so a per-epoch reconstruction is a curve nothing followed."""
    traces = schedules.rate_traces(
        toy, {"hidden_layer_sizes": (8,), "activation": "gelu"}, horizon=12
    )
    assert set(traces) == set(SCHEDULES)
    assert all(len(trace) == 12 for trace in traces.values())
    assert len(set(traces["none"])) == 1
    assert traces["cosine"][-1] < traces["cosine"][0]


def test_one_cycle_rises_before_it_falls(toy):
    """Its defining shape - and the thing a per-epoch step would never reach."""
    traces = schedules.rate_traces(
        toy, {"hidden_layer_sizes": (8,), "activation": "gelu"}, horizon=20
    )
    curve = traces["one_cycle"]
    assert max(curve) > curve[0]
    assert curve[-1] < max(curve)

"""Phase 6.2.6 - the batch-size sweep, and the prediction 6.2.2 left it to test."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import batchsize
from src.classify.data import Dataset

pytest.importorskip("torch")


@pytest.fixture
def toy():
    rng = np.random.default_rng(54)
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


# -- the grid ---------------------------------------------------------------------------------


def test_the_four_sizes_the_plan_names_are_swept():
    assert {16, 32, 64, 128} <= set(batchsize.SIZES)


def test_sklearns_default_is_swept_too():
    """200 is not arbitrary - it is the value 6.2.1's whole table was produced at."""
    assert 200 in batchsize.SIZES


def test_the_grid_varies_only_the_batch_size():
    configs = batchsize.grid(base={"activation": "gelu", "dropout": 0.5})
    assert [config["batch_size"] for config in configs] == list(batchsize.SIZES)
    assert {config["activation"] for config in configs} == {"gelu"}
    assert {config["dropout"] for config in configs} == {0.5}


def test_the_prediction_under_test_is_written_down():
    """A hypothesis formed while looking at something else has to be recorded to be falsifiable."""
    assert batchsize.PREDICTION["batch_64"] > batchsize.PREDICTION["batch_200"]
    assert batchsize.PREDICTION["table"] == "handcrafted"


# -- the confound the score alone cannot separate ------------------------------------------------


def test_a_smaller_batch_is_more_optimizer_steps_per_epoch():
    """The reason a small-batch win might be more training rather than better regularization."""
    assert batchsize.steps_per_epoch(1072, 16) == 67
    assert batchsize.steps_per_epoch(1072, 128) == 9
    assert batchsize.steps_per_epoch(1072, 16) > 7 * batchsize.steps_per_epoch(1072, 128)


def test_a_batch_larger_than_the_training_set_is_one_step():
    assert batchsize.steps_per_epoch(180, 100000) == 1


def test_the_last_partial_batch_still_counts_as_a_step():
    assert batchsize.steps_per_epoch(100, 32) == 4


def test_every_row_carries_the_update_count_not_just_the_score():
    rows = [
        {"batch_size": 16, "macro_f1": 0.85, "epochs": 20.0},
        {"batch_size": 128, "macro_f1": 0.80, "epochs": 20.0},
    ]
    annotated = batchsize.annotate(rows, 1072)
    by_size = {row["batch_size"]: row for row in annotated}
    assert by_size[16]["optimizer_steps"] == 67 * 20
    assert by_size[128]["optimizer_steps"] == 9 * 20
    assert by_size[16]["optimizer_steps"] > by_size[128]["optimizer_steps"]


def test_the_annotated_rows_come_back_in_batch_order():
    """The table is read as a curve in batch size, so an F1-sorted order would hide the shape."""
    rows = [{"batch_size": size, "macro_f1": 0.9, "epochs": 10.0} for size in (128, 16, 64, 32)]
    assert [row["batch_size"] for row in batchsize.annotate(rows, 1000)] == [16, 32, 64, 128]


# -- the verdict --------------------------------------------------------------------------------


def test_the_prediction_is_scored_directionally_and_against_the_fold_noise():
    rows = batchsize.annotate(
        [
            {"batch_size": 16, "macro_f1": 0.86, "std": 0.01, "epochs": 20.0},
            {"batch_size": 32, "macro_f1": 0.85, "std": 0.01, "epochs": 20.0},
            {"batch_size": 64, "macro_f1": 0.84, "std": 0.01, "epochs": 20.0},
            {"batch_size": 128, "macro_f1": 0.82, "std": 0.01, "epochs": 20.0},
            {"batch_size": 200, "macro_f1": 0.79, "std": 0.01, "epochs": 20.0},
        ],
        1072,
    )
    result = batchsize.verdict(rows, "handcrafted")
    assert result["best_batch"] == 16
    assert result["monotone_in_batch_size"] is True
    assert result["smallest_minus_largest"] == pytest.approx(0.07)
    assert result["larger_than_one_fold_std"] is True


def test_a_flat_curve_is_reported_as_not_worth_a_standard_deviation():
    rows = batchsize.annotate(
        [
            {"batch_size": size, "macro_f1": 0.95, "std": 0.02, "epochs": 20.0}
            for size in batchsize.SIZES
        ],
        1072,
    )
    result = batchsize.verdict(rows, "hybrid")
    assert result["spread"] == 0.0
    assert result["larger_than_one_fold_std"] is False


def test_a_non_monotone_curve_is_not_called_monotone():
    rows = batchsize.annotate(
        [
            {"batch_size": 16, "macro_f1": 0.80, "std": 0.01, "epochs": 20.0},
            {"batch_size": 32, "macro_f1": 0.90, "std": 0.01, "epochs": 20.0},
            {"batch_size": 64, "macro_f1": 0.85, "std": 0.01, "epochs": 20.0},
        ],
        1072,
    )
    assert batchsize.verdict(rows, "hybrid")["monotone_in_batch_size"] is False


# -- the mechanism, on a real fit -----------------------------------------------------------------


def test_the_batch_size_reaches_the_training_loop(toy):
    """A sweep whose knob never arrives at the optimizer is a sweep of nothing."""
    from src.classify.torchnet import TorchMLP

    settings = {
        "hidden_layer_sizes": (8,),
        "max_epochs": 5,
        "early_stopping": False,
        "device": "cpu",
        "activation": "gelu",
    }
    small = TorchMLP(batch_size=8, **settings).fit(toy.X, toy.y)
    large = TorchMLP(batch_size=180, **settings).fit(toy.X, toy.y)
    # Same epochs, different number of updates, so the two fits are genuinely different runs.
    assert small.loss_curve_ != large.loss_curve_


def test_the_study_is_built_on_6_2_3s_regularized_network_not_6_2_2s(monkeypatch, toy):
    """Sweeping the batch on an unregularized network would answer 6.2.2's question again."""
    captured = []

    def fake_sweep(data, configs, n_jobs=None, folds=5):
        captured.extend(configs)
        return [
            {**c, "macro_f1": 0.9, "std": 0.01, "overfit_gap": 0.02, "epochs": 20.0, "seconds": 1.0}
            for c in configs
        ]

    monkeypatch.setattr("src.classify.torchnet.sweep", fake_sweep)
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.embed.hybrid",
        type("m", (), {"dataset": staticmethod(lambda *a, **k: toy)})(),
    )
    from src.classify.regularize import best_settings

    batchsize.run("hybrid")
    expected = best_settings("hybrid")
    assert {config["dropout"] for config in captured} == {expected["dropout"]}
    assert {config["weight_decay"] for config in captured} == {expected["weight_decay"]}

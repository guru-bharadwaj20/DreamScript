"""Phase 6.2.3 - dropout, weight decay and early stopping against 6.2.2's overfit gap."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import regularize
from src.classify.data import Dataset

pytest.importorskip("torch")


@pytest.fixture
def toy():
    rng = np.random.default_rng(52)
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


# -- the grid ------------------------------------------------------------------------------


def test_the_dropout_rates_are_the_three_the_plan_names():
    assert regularize.DROPOUTS == (0.0, 0.2, 0.5)


def test_the_unpenalised_cell_is_in_the_grid():
    """A sweep starting at the smallest nonzero penalty cannot say what the penalty was worth."""
    configs = regularize.grid()
    assert any(c["dropout"] == 0.0 and c["weight_decay"] == 0.0 for c in configs)


def test_both_optimizers_are_swept_because_weight_decay_means_two_things():
    assert set(regularize.OPTIMIZERS) == {"adam", "adamw"}
    decayed = [c for c in regularize.grid() if c["weight_decay"] > 0]
    assert {c["optimizer"] for c in decayed} == {"adam", "adamw"}


def test_zero_decay_is_not_duplicated_across_the_two_optimizers():
    """With no decay they are the same algorithm, and a duplicate row skews the marginals."""
    zero = [c for c in regularize.grid() if c["weight_decay"] == 0.0]
    assert len(zero) == len(regularize.DROPOUTS)
    assert {c["optimizer"] for c in zero} == {"adam"}


def test_the_grid_size_is_the_product_minus_the_duplicates():
    expected = len(regularize.DROPOUTS) * (1 + (len(regularize.DECAYS) - 1) * 2)
    assert len(regularize.grid()) == expected


def test_every_cell_holds_the_topology_and_activation_fixed():
    """The grid is about the penalty; 6.2.1 chose the shape and 6.2.2 chose the nonlinearity."""
    from src.classify.activations import TOPOLOGY

    for config in regularize.grid():
        assert config["hidden_layer_sizes"] == TOPOLOGY
        assert config["activation"] == regularize.ACTIVATION


def test_a_narrowed_grid_is_still_well_formed():
    configs = regularize.grid(dropouts=(0.0, 0.5), decays=(0.0, 1e-3))
    assert len(configs) == 2 * (1 + 1 * 2)


# -- the marginals, which are the defensible claim -------------------------------------------


def test_the_marginals_average_each_knob_over_the_others():
    rows = [
        {"dropout": 0.0, "weight_decay": 0.0, "optimizer": "adam", "macro_f1": 0.90},
        {"dropout": 0.0, "weight_decay": 1e-3, "optimizer": "adamw", "macro_f1": 0.92},
        {"dropout": 0.5, "weight_decay": 0.0, "optimizer": "adam", "macro_f1": 0.94},
        {"dropout": 0.5, "weight_decay": 1e-3, "optimizer": "adamw", "macro_f1": 0.96},
    ]
    result = regularize.marginals(rows)
    assert result["by_dropout"]["0.0"] == pytest.approx(0.91)
    assert result["by_dropout"]["0.5"] == pytest.approx(0.95)
    assert result["dropout_range"] == pytest.approx(0.04)
    assert result["weight_decay_range"] == pytest.approx(0.02)


def test_the_marginals_cover_every_level_that_appears():
    rows = [
        {"dropout": d, "weight_decay": w, "optimizer": "adam", "macro_f1": 0.9}
        for d in regularize.DROPOUTS
        for w in regularize.DECAYS
    ]
    result = regularize.marginals(rows)
    assert len(result["by_dropout"]) == len(regularize.DROPOUTS)
    assert len(result["by_weight_decay"]) == len(regularize.DECAYS)


# -- the early-stopping study ----------------------------------------------------------------


def test_early_stopping_is_compared_at_the_best_cell_not_a_fresh_one():
    """The question is whether the penalty made it redundant, so the penalty has to be on."""
    best = {
        "dropout": 0.5,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "activation": "gelu",
        "hidden_layer_sizes": (8, 8),
        "macro_f1": 0.95,
    }
    pair = regularize.stopping_pair(best)
    assert {config["dropout"] for config in pair} == {0.5}
    assert {config["weight_decay"] for config in pair} == {1e-4}
    assert {config["early_stopping"] for config in pair} == {True, False}


def test_both_arms_hold_out_the_same_rows():
    """The confound this task had to remove: an un-stopped arm training on 12% more data."""
    pair = regularize.stopping_pair(
        {
            "dropout": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "activation": "gelu",
            "hidden_layer_sizes": (8,),
        }
    )
    off = next(config for config in pair if not config["early_stopping"])
    assert off["validation_fraction"] == 0.12
    assert {config["max_epochs"] for config in pair} == {regularize.UNSTOPPED_EPOCHS}


def test_the_stopping_pair_drops_result_keys_from_the_best_row():
    """`best` is a sweep result carrying `macro_f1` and `seconds`; those are not fit arguments."""
    pair = regularize.stopping_pair(
        {
            "dropout": 0.2,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "activation": "gelu",
            "hidden_layer_sizes": (8,),
            "macro_f1": 0.9,
            "seconds": 3.2,
            "std": 0.01,
        }
    )
    for config in pair:
        assert "macro_f1" not in config
        assert "seconds" not in config


def test_the_early_stopping_cost_reads_both_arms():
    rows = [
        {"early_stopping": True, "macro_f1": 0.90, "overfit_gap": 0.02, "epochs": 30.0},
        {"early_stopping": False, "macro_f1": 0.80, "overfit_gap": 0.05, "epochs": 200.0},
    ]
    result = regularize.early_stopping_cost(rows)
    assert result["cost_of_removing_it"] == pytest.approx(0.1)
    assert result["gap_without"] > result["gap_with"]
    assert result["epochs_without"] == 200.0


def test_a_negative_cost_means_early_stopping_hurt():
    """It does, on the hybrid table - so the sign has to survive the arithmetic."""
    rows = [
        {"early_stopping": True, "macro_f1": 0.9503, "overfit_gap": 0.046, "epochs": 33.0},
        {"early_stopping": False, "macro_f1": 0.9568, "overfit_gap": 0.036, "epochs": 200.0},
    ]
    assert regularize.early_stopping_cost(rows)["cost_of_removing_it"] < 0


def test_the_recorded_selection_covers_every_table_and_is_usable_as_fit_arguments():
    from src.classify.activations import TABLES

    assert set(regularize.BEST) == set(TABLES)
    for table in TABLES:
        settings = regularize.best_settings(table)
        assert set(settings) == {"dropout", "weight_decay", "optimizer"}


def test_the_handcrafted_selection_is_the_unpenalised_cell():
    """Every penalty tried made that table worse - which is why the grid contains a zero cell."""
    settings = regularize.best_settings("handcrafted")
    assert settings["dropout"] == 0.0
    assert settings["weight_decay"] == 0.0


def test_an_unknown_table_is_refused_rather_than_defaulted():
    """A silent unregularized fallback would make a later task answer 6.2.2's question again."""
    with pytest.raises(KeyError, match="no 6.2.3 selection"):
        regularize.best_settings("nonsense")


# -- the summary the plan row is graded on ---------------------------------------------------


def test_the_summary_reports_the_gap_on_both_sides(toy, monkeypatch):
    """ "Overfit gap reduced" needs the number it was reduced from, not just the winner."""
    rows = [
        {
            "dropout": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "macro_f1": 0.90,
            "overfit_gap": 0.09,
            "std": 0.01,
            "epochs": 20.0,
            "seconds": 1.0,
        },
        {
            "dropout": 0.5,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "macro_f1": 0.95,
            "overfit_gap": 0.02,
            "std": 0.01,
            "epochs": 40.0,
            "seconds": 2.0,
        },
    ]
    monkeypatch.setattr("src.classify.torchnet.sweep", lambda *a, **k: [dict(r) for r in rows])
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.embed.hybrid",
        type("m", (), {"dataset": staticmethod(lambda *a, **k: toy)})(),
    )
    summary = regularize.run("hybrid", stopping=False)

    assert summary["gap_before"] == 0.09
    assert summary["gap_after"] == 0.02
    assert summary["gap_reduced_by"] == pytest.approx(0.07)
    assert summary["macro_f1_gained"] == pytest.approx(0.05)
    assert summary["best_minus_logistic_regression"] == pytest.approx(0.95 - 0.9503, abs=1e-4)


def test_6_2_2s_unregularized_numbers_are_carried_for_every_table():
    from src.classify.activations import TABLES

    assert set(regularize.BASELINE) == set(TABLES)


# -- the penalties actually do something -----------------------------------------------------


def test_dropout_narrows_the_gap_between_training_and_held_out_score(toy):
    """The mechanism, on real fits: a penalised network memorises its training rows less."""
    from src.classify.torchnet import cross_validate

    settings = {
        "hidden_layer_sizes": (64, 64),
        "activation": "gelu",
        "max_epochs": 120,
        "early_stopping": False,
        "device": "cpu",
        "folds": 3,
    }
    plain = cross_validate(toy, dropout=0.0, **settings)
    dropped = cross_validate(toy, dropout=0.5, **settings)
    assert dropped["train_macro_f1"] < plain["train_macro_f1"]


def test_a_large_weight_decay_shrinks_the_weights(toy):
    """AdamW subtracts the decay from the weight, so the effect is visible in the norm."""
    import torch

    from src.classify.torchnet import TorchMLP

    settings = {
        "hidden_layer_sizes": (32,),
        "activation": "gelu",
        "max_epochs": 60,
        "optimizer": "adamw",
        "early_stopping": False,
        "device": "cpu",
    }
    norms = []
    for decay in (0.0, 0.1):
        model = TorchMLP(weight_decay=decay, **settings).fit(toy.X, toy.y)
        norms.append(float(sum(p.norm() for p in model.net_.parameters() if p.dim() == 2)))
    assert norms[1] < norms[0]
    assert torch.isfinite(torch.tensor(norms)).all()

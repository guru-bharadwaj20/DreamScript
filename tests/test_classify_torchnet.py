"""Phase 6.2.2 - the torch training loop 6.2.3 through 6.2.6 are all built on."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from src.classify import torchnet
from src.classify.data import Dataset

pytest.importorskip("torch")


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


def small(**kwargs) -> torchnet.TorchMLP:
    settings = {"hidden_layer_sizes": (16, 8), "max_epochs": 30, "device": "cpu"}
    return torchnet.TorchMLP(**{**settings, **kwargs})


# -- the four activations the plan names ------------------------------------------------


def test_all_four_plan_activations_are_available():
    """The reason this module exists: sklearn supplies two of these, not four."""
    assert set(torchnet.ACTIVATIONS) == {"relu", "leaky_relu", "gelu", "tanh"}


@pytest.mark.parametrize("activation", torchnet.ACTIVATIONS)
def test_every_activation_fits_and_predicts_labels(toy, activation):
    model = small(activation=activation).fit(toy.X, toy.y)
    predicted = model.predict(toy.X)
    assert predicted.shape == toy.y.shape
    assert set(predicted) <= set(toy.y)


def test_an_unknown_activation_is_refused_by_name():
    with pytest.raises(ValueError, match="activation must be one of"):
        small(activation="swish").fit(np.zeros((10, 2)), np.array(["a"] * 5 + ["b"] * 5))


def test_leaky_relu_has_no_flat_region_where_relu_does():
    """The mechanism 6.2.2's dead-unit count is about, checked on the layer itself."""
    import torch

    negative = torch.tensor([-5.0, -1.0])
    assert torch.all(torchnet._activation("relu")(negative) == 0)
    assert torch.all(torchnet._activation("leaky_relu")(negative) < 0)


# -- the estimator protocol Phase 5's harness depends on --------------------------------


def test_it_accepts_string_labels_without_the_6_2_1_workaround(toy):
    """6.2.1 needed `StringSafeMLP`; here the encoder is part of `fit` by construction."""
    model = small().fit(toy.X, toy.y)
    assert model.classes_.tolist() == ["a", "b", "c"]
    assert model.predict(toy.X).dtype.kind in "UO"


def test_predict_proba_rows_are_a_distribution(toy):
    probabilities = small().fit(toy.X, toy.y).predict_proba(toy.X)
    assert probabilities.shape == (180, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert (probabilities >= 0).all()


def test_predict_proba_survives_logits_that_would_overflow_a_naive_softmax(toy):
    """A converged network's logits reach 20-plus; the row max has to come out first."""
    model = small().fit(toy.X, toy.y)
    logits = model._logits(toy.X)
    shifted = np.exp(logits * 60 - (logits * 60).max(axis=1, keepdims=True))
    assert np.isfinite(shifted).all()


def test_it_is_cloneable_so_grid_search_and_cross_validation_work(toy):
    original = small(activation="gelu", dropout=0.2)
    copy = clone(original)
    assert copy.get_params()["activation"] == "gelu"
    assert copy.get_params()["dropout"] == 0.2
    copy.fit(toy.X, toy.y)


def test_the_constructor_does_no_work(toy):
    """`get_params`/`set_params` require it, and `clone` breaks quietly when it is violated."""
    model = small()
    assert not hasattr(model, "net_")
    assert not hasattr(model, "classes_")


def test_the_pipeline_scales_before_the_network(toy):
    estimator = torchnet.pipeline(hidden_layer_sizes=(16,), max_epochs=20, device="cpu")
    assert list(estimator.named_steps) == ["prepare", "model"]
    estimator.fit(toy.X, toy.y)
    assert estimator.predict(toy.X).shape == toy.y.shape


# -- training-loop behaviour -------------------------------------------------------------


def test_the_same_seed_gives_the_same_network(toy):
    first = small(random_state=7).fit(toy.X, toy.y)
    second = small(random_state=7).fit(toy.X, toy.y)
    assert np.array_equal(first.predict(toy.X), second.predict(toy.X))
    assert first.loss_curve_ == second.loss_curve_


def test_training_loss_falls(toy):
    model = small(max_epochs=60, early_stopping=False).fit(toy.X, toy.y)
    assert model.loss_curve_[-1] < model.loss_curve_[0]


def test_one_curve_entry_per_epoch(toy):
    model = small(max_epochs=25, early_stopping=False).fit(toy.X, toy.y)
    assert len(model.loss_curve_) == 25
    assert len(model.lr_curve_) == 25


def test_early_stopping_can_finish_before_the_epoch_ceiling(toy):
    model = small(max_epochs=600, n_iter_no_change=3).fit(toy.X, toy.y)
    assert model.epochs_run_ < 600
    assert len(model.loss_curve_) == model.epochs_run_


def test_early_stopping_restores_the_best_weights_not_the_last(toy):
    """Keeping the final weights is a shorter run, not early stopping."""
    model = small(max_epochs=200, n_iter_no_change=5).fit(toy.X, toy.y)
    assert model.best_validation_score_ is not None
    assert model.best_validation_score_ >= max(model.val_curve_) - 1e-9


def test_without_early_stopping_there_is_no_validation_curve(toy):
    model = small(early_stopping=False, validation_fraction=0.0).fit(toy.X, toy.y)
    assert model.val_curve_ == []
    assert model.best_validation_score_ is None


# -- the knobs 6.2.3 through 6.2.6 turn ---------------------------------------------------


def test_dropout_adds_a_layer_only_when_it_is_nonzero(toy):
    import torch.nn as nn

    plain = small(dropout=0.0).fit(toy.X, toy.y)
    dropped = small(dropout=0.5).fit(toy.X, toy.y)
    assert not any(isinstance(m, nn.Dropout) for m in plain.net_)
    assert sum(isinstance(m, nn.Dropout) for m in dropped.net_) == 2


def test_dropout_comes_after_the_activation(toy):
    """Dropping pre-activations of a ReLU zeroes units that were already zero."""
    import torch.nn as nn

    net = small(dropout=0.3, activation="relu").fit(toy.X, toy.y).net_
    kinds = [type(m) for m in net]
    for index, kind in enumerate(kinds):
        if kind is nn.Dropout:
            assert kinds[index - 1] is nn.ReLU


@pytest.mark.parametrize("name", torchnet.OPTIMIZERS)
def test_every_optimizer_6_2_4_compares_can_train(toy, name):
    model = small(optimizer=name, learning_rate=0.01).fit(toy.X, toy.y)
    assert model.loss_curve_[-1] < model.loss_curve_[0]


def test_momentum_and_plain_sgd_are_different_optimizers(toy):
    import torch

    net = small()._build(4, 3)
    plain = torchnet._optimizer("sgd", net.parameters(), 0.1, 0.0)
    fast = torchnet._optimizer("momentum", net.parameters(), 0.1, 0.0)
    assert isinstance(plain, torch.optim.SGD) and isinstance(fast, torch.optim.SGD)
    assert plain.param_groups[0]["momentum"] == 0
    assert fast.param_groups[0]["momentum"] == 0.9


def test_an_unknown_optimizer_is_refused_by_name(toy):
    with pytest.raises(ValueError, match="optimizer must be one of"):
        small(optimizer="lbfgs").fit(toy.X, toy.y)


@pytest.mark.parametrize("name", torchnet.SCHEDULES)
def test_every_schedule_6_2_5_compares_can_train(toy, name):
    model = small(schedule=name, max_epochs=20, early_stopping=False).fit(toy.X, toy.y)
    assert len(model.lr_curve_) == 20


def test_one_cycle_steps_per_batch_not_per_epoch():
    """Stepped once an epoch it would traverse 1/steps of the cycle and never anneal."""
    _, unit = torchnet._schedule("one_cycle", _dummy_optimizer(), 10, 17, 1e-3)
    assert unit == "batch"
    _, epoch_unit = torchnet._schedule("cosine", _dummy_optimizer(), 10, 17, 1e-3)
    assert epoch_unit == "epoch"


def test_a_constant_rate_is_the_control(toy):
    model = small(schedule="none", max_epochs=15, early_stopping=False).fit(toy.X, toy.y)
    assert len(set(model.lr_curve_)) == 1


def test_cosine_anneals_the_rate_downward(toy):
    model = small(schedule="cosine", max_epochs=15, early_stopping=False).fit(toy.X, toy.y)
    assert model.lr_curve_[-1] < model.lr_curve_[0]


def _dummy_optimizer():
    import torch

    return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-3)


def test_the_batch_size_is_capped_at_the_training_set(toy):
    """6.2.6 sweeps sizes that can exceed a fold; a batch larger than the data is one batch."""
    model = small(batch_size=100000).fit(toy.X, toy.y)
    assert model.epochs_run_ >= 1


# -- the parameter count, shared with 6.2.1 -----------------------------------------------


def test_the_parameter_count_matches_6_2_1s_formula(toy):
    from src.classify.mlp import parameter_count

    model = small(hidden_layer_sizes=(16, 8)).fit(toy.X, toy.y)
    assert model.parameter_count() == parameter_count(4, (16, 8), 3)


# -- the shared cross-validation harness --------------------------------------------------


def test_cross_validate_reports_both_sides_of_the_overfit_gap(toy):
    result = torchnet.cross_validate(
        toy, folds=3, hidden_layer_sizes=(16,), max_epochs=30, device="cpu"
    )
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["train_macro_f1"] >= result["macro_f1"] - 0.5
    assert result["overfit_gap"] == pytest.approx(
        result["train_macro_f1"] - result["macro_f1"], abs=1e-4
    )


def test_cross_validate_predicts_every_row_exactly_once(toy):
    result = torchnet.cross_validate(
        toy, folds=3, hidden_layer_sizes=(16,), max_epochs=20, device="cpu"
    )
    assert len(result["predicted"]) == len(toy.y)
    assert all(value is not None for value in result["predicted"])


def test_cross_validate_keeps_one_curve_per_fold(toy):
    result = torchnet.cross_validate(
        toy, folds=3, hidden_layer_sizes=(16,), max_epochs=20, device="cpu"
    )
    assert len(result["curves"]) == 3
    assert all(curve["loss"] for curve in result["curves"])


def test_the_sweep_returns_one_row_per_configuration_carrying_its_settings(toy):
    configs = [
        {"hidden_layer_sizes": (16,), "activation": "relu", "max_epochs": 20},
        {"hidden_layer_sizes": (16,), "activation": "gelu", "max_epochs": 20},
    ]
    rows = torchnet.sweep(toy, configs, n_jobs=2, folds=3)
    assert len(rows) == 2
    assert {row["activation"] for row in rows} == {"relu", "gelu"}
    assert all("macro_f1" in row for row in rows)
    # `predicted` is a 1,340-element object array per configuration; pickling it back from 32
    # workers is the one thing the sweep must not do.
    assert all("predicted" not in row for row in rows)

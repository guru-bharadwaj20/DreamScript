"""Phase 6.3.1 - the linear SVM, its C sweep, and the support vectors that explain the sweep."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import svm
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(56)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (1.5, 3.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.7, size=(140, 2))
    X = np.hstack([X, rng.normal(size=(140, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(140)], dtype=object),
        ids=np.arange(140).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


# -- the sweep --------------------------------------------------------------------------------


def test_c_spans_seven_orders_of_magnitude():
    """A narrow grid cannot tell a plateau from a peak, and both occur in this phase."""
    assert min(svm.CS) == 0.001
    assert max(svm.CS) == 1000.0
    assert len(svm.CS) == 7


def test_both_losses_are_swept_because_they_are_different_estimators():
    """The plan says 'hinge loss'; sklearn's default is the squared variant."""
    assert set(svm.LOSSES) == {"hinge", "squared_hinge"}


def test_the_pipeline_scales_before_the_margin(toy):
    """The margin is measured in the feature space's own units, so an unscaled column defines it."""
    estimator = svm.pipeline()
    assert list(estimator.named_steps) == ["prepare", "model"]
    estimator.fit(toy.X, toy.y)
    assert estimator.predict(toy.X).shape == toy.y.shape


def test_c_reaches_the_estimator(toy):
    assert svm.pipeline(C=0.01).named_steps["model"].C == 0.01
    assert svm.pipeline(loss="hinge").named_steps["model"].loss == "hinge"


def test_the_sweep_ranks_every_cell(toy):
    rows = svm.sweep(toy, cs=(0.01, 1.0), n_jobs=2)
    assert len(rows) == 2 * len(svm.LOSSES)
    scores = [row["macro_f1"] for row in rows]
    assert scores == sorted(scores, reverse=True)
    assert all({"C", "loss", "macro_f1", "accuracy"} <= set(row) for row in rows)


def test_the_recorded_best_is_a_cell_the_grid_contains():
    assert svm.BEST_PARAMS["C"] in svm.CS
    assert svm.BEST_PARAMS["loss"] in svm.LOSSES


def test_the_best_estimator_uses_the_recorded_parameters():
    model = svm.best_estimator().named_steps["model"]
    assert svm.BEST_PARAMS["C"] == model.C
    assert model.loss == svm.BEST_PARAMS["loss"]


def test_the_best_estimator_can_be_overridden():
    assert svm.best_estimator(C=42.0).named_steps["model"].C == 42.0


# -- the plateau, which is the honest reading of a flat sweep ------------------------------------


def test_the_plateau_reports_the_range_of_c_that_ties():
    rows = [
        {"C": 0.1, "loss": "squared_hinge", "macro_f1": 0.90},
        {"C": 1.0, "loss": "squared_hinge", "macro_f1": 0.899},
        {"C": 10.0, "loss": "squared_hinge", "macro_f1": 0.897},
        {"C": 1000.0, "loss": "squared_hinge", "macro_f1": 0.50},
    ]
    result = svm.plateau(rows, tolerance=0.005)
    assert result["cells_within_tolerance"] == 3
    assert result["c_min"] == 0.1
    assert result["c_max"] == 10.0
    assert result["orders_of_magnitude"] == pytest.approx(2.0)


def test_a_sharp_peak_is_reported_as_one_cell():
    rows = [
        {"C": 1.0, "loss": "hinge", "macro_f1": 0.90},
        {"C": 10.0, "loss": "hinge", "macro_f1": 0.70},
    ]
    result = svm.plateau(rows)
    assert result["cells_within_tolerance"] == 1
    assert result["orders_of_magnitude"] == 0.0


# -- the support vectors --------------------------------------------------------------------------


def test_a_smaller_c_makes_more_rows_support_vectors(toy):
    """A wide margin swallows more points - the mechanism behind the sweep's worst cells."""
    wide = svm.support_vector_count(toy, C=0.001)
    tight = svm.support_vector_count(toy, C=100.0)
    assert wide["support_vectors"] > tight["support_vectors"]


def test_the_support_vector_count_is_broken_down_by_class(toy):
    result = svm.support_vector_count(toy, C=1.0)
    assert set(result["by_class"]) == {"a", "b", "c"}
    assert sum(result["by_class"].values()) == result["support_vectors"]
    assert result["share_of_training_rows"] == pytest.approx(
        result["support_vectors"] / len(toy.y), abs=1e-4
    )


def test_the_minority_class_is_disproportionately_support_vectors(toy):
    """On the real corpus 39 of 40 circuit pages are SVs; the toy reproduces the direction."""
    result = svm.support_vector_count(toy, C=1.0)
    counts = toy.class_counts()
    shares = {name: result["by_class"][name] / counts[name] for name in counts}
    assert shares["c"] > shares["a"]


# -- the hinge loss the plan names ------------------------------------------------------------------


def test_the_hinge_loss_reads_the_margin_not_the_argmax(toy):
    """It is computed from decision values, which is what makes it different from macro F1."""
    fitted = svm.pipeline(C=1.0).fit(toy.X, toy.y)
    value = svm.hinge_loss_value(fitted, toy.X, toy.y)
    assert value >= 0.0
    assert np.isfinite(value)


def test_a_harder_problem_has_a_larger_hinge_loss(toy):
    """Shuffled labels destroy the margin, and the loss has to see that."""
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(toy.y)
    real = svm.hinge_loss_value(svm.pipeline().fit(toy.X, toy.y), toy.X, toy.y)
    noise = svm.hinge_loss_value(svm.pipeline().fit(toy.X, shuffled), toy.X, shuffled)
    assert noise > real

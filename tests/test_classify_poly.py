"""Phase 6.3.2 - the polynomial kernel on the plan's pair and on the one that is actually hard."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import poly
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(57)
    y = np.array(["circuit"] * 40 + ["flowchart"] * 80 + ["state_machine"] * 40, dtype=object)
    centres = {"circuit": (0.0, 0.0), "flowchart": (2.0, 0.0), "state_machine": (1.0, 4.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.8, size=(160, 2))
    X = np.hstack([X, rng.normal(size=(160, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(160)], dtype=object),
        ids=np.arange(160).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


# -- the grid the plan asks for -----------------------------------------------------------------


def test_the_degrees_are_the_three_the_plan_names():
    assert poly.DEGREES == (2, 3, 4)


def test_coef0_zero_is_in_the_grid_because_it_changes_what_the_kernel_can_represent():
    """At coef0 = 0 the kernel is homogeneous - and it is sklearn's default."""
    assert 0.0 in poly.COEF0S
    assert len(poly.COEF0S) >= 3


def test_the_adaptive_gamma_default_is_in_the_grid_not_replaced_by_it():
    """A fixed gamma on a 33-column table and a 161-column one are different kernels."""
    assert "scale" in poly.GAMMAS


def test_the_grid_is_the_full_product():
    configs = poly.grid()
    assert len(configs) == (len(poly.DEGREES) * len(poly.GAMMAS) * len(poly.COEF0S) * len(poly.CS))
    assert all({"degree", "gamma", "coef0", "C"} == set(config) for config in configs)


def test_the_kernel_parameters_reach_the_estimator():
    model = poly.pipeline(degree=4, coef0=10.0, gamma=0.01, C=3.0).named_steps["model"]
    assert model.kernel == "poly"
    assert (model.degree, model.coef0, model.gamma, model.C) == (4, 10.0, 0.01, 3.0)


# -- both pairs are studied -----------------------------------------------------------------------


def test_the_plans_pair_and_5_3_4s_pair_are_both_named():
    assert poly.PLAN_PAIR == ("flowchart", "state_machine")
    assert poly.HARD_PAIR == ("circuit", "flowchart")


def test_the_binary_subset_keeps_only_the_two_named_classes(toy):
    subset = poly.binary_subset(toy, ("circuit", "flowchart"))
    assert set(subset.y) == {"circuit", "flowchart"}
    assert len(subset.y) == 120
    assert len(subset.ids) == len(subset.y)
    assert len(subset.groups) == len(subset.y)


def test_the_binary_subset_keeps_the_feature_columns(toy):
    subset = poly.binary_subset(toy, poly.PLAN_PAIR)
    assert subset.X.shape[1] == toy.X.shape[1]
    assert subset.feature_names == toy.feature_names


def test_a_pair_that_is_not_in_the_corpus_is_refused(toy):
    with pytest.raises(KeyError):
        poly.binary_subset(toy, ("nonsense", "also_nonsense"))


# -- the marginals ----------------------------------------------------------------------------------


def test_the_marginal_takes_the_best_at_each_level_not_the_mean():
    """A polynomial grid contains genuinely broken cells; a mean would count them."""
    rows = [
        {"degree": 2, "macro_f1": 0.90},
        {"degree": 2, "macro_f1": 0.10},
        {"degree": 3, "macro_f1": 0.80},
    ]
    result = poly.marginal(rows, "degree")
    assert result["2"] == 0.90
    assert result["3"] == 0.80


def test_the_marginal_covers_every_level_present():
    rows = [{"coef0": value, "macro_f1": 0.5} for value in poly.COEF0S]
    assert len(poly.marginal(rows, "coef0")) == len(poly.COEF0S)


# -- the study --------------------------------------------------------------------------------------


def test_a_study_reports_the_kernels_gain_over_the_linear_baseline(toy):
    result = poly.study(
        toy,
        ("circuit", "flowchart"),
        configs=[{"degree": 2, "gamma": "scale", "coef0": 1.0, "C": 1.0}],
        n_jobs=2,
    )
    assert result["pair"] == ["circuit", "flowchart"]
    assert result["rows"] == 120
    assert result["kernel_minus_linear"] == pytest.approx(
        result["best"]["macro_f1"] - result["linear_svm_6_3_1"], abs=1e-4
    )


def test_a_study_ranks_its_cells(toy):
    configs = [
        {"degree": 2, "gamma": "scale", "coef0": 0.0, "C": 1.0},
        {"degree": 2, "gamma": "scale", "coef0": 10.0, "C": 1.0},
    ]
    result = poly.study(toy, poly.HARD_PAIR, configs=configs, n_jobs=2)
    scores = [row["macro_f1"] for row in result["all"]]
    assert scores == sorted(scores, reverse=True)
    assert result["best"]["macro_f1"] >= result["worst"]["macro_f1"]


def test_a_study_carries_the_class_counts_so_the_imbalance_is_visible(toy):
    result = poly.study(
        toy,
        poly.HARD_PAIR,
        configs=[{"degree": 2, "gamma": "scale", "coef0": 1.0, "C": 1.0}],
        n_jobs=2,
    )
    assert result["class_counts"] == {"circuit": 40, "flowchart": 80}


# -- the mechanism the coef0 finding rests on ----------------------------------------------------------


def test_a_homogeneous_kernel_cannot_represent_a_lower_degree_boundary(toy):
    """coef0 = 0 keeps only the degree-d monomials - the measured effect is 0.186 macro F1."""
    subset = poly.binary_subset(toy, poly.HARD_PAIR)
    homogeneous = poly.score_grid(
        subset, [{"degree": 3, "gamma": "scale", "coef0": 0.0, "C": 1.0}], n_jobs=2
    )[0]
    inhomogeneous = poly.score_grid(
        subset, [{"degree": 3, "gamma": "scale", "coef0": 10.0, "C": 1.0}], n_jobs=2
    )[0]
    assert inhomogeneous["macro_f1"] > homogeneous["macro_f1"]


def test_sklearns_default_coef0_is_the_degenerate_one():
    """Which is why a degree-only sweep at the default would have condemned the kernel."""
    from sklearn.svm import SVC

    assert SVC(kernel="poly").coef0 == 0.0
    assert poly.pipeline().named_steps["model"].coef0 == 0.0

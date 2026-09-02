"""Phase 6.3.7 - the feature-set x model cross-table, and the interaction it exists to test."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import crosstable
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(62)
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (3.0, 0.0), "c": (1.5, 2.5)}
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


GRID = {
    "logreg": {"handcrafted": 0.80, "embedding": 0.96, "hybrid": 0.95},
    "rbf_svm": {"handcrafted": 0.83, "embedding": 0.97, "hybrid": 0.97},
    "linear_svm": {"handcrafted": 0.77, "embedding": 0.97, "hybrid": 0.96},
}
TABLES = ("handcrafted", "embedding", "hybrid")


# -- the models span the phase, not just 6.3 ---------------------------------------------------


def test_the_table_covers_linear_baselines_the_network_and_the_kernels():
    """A grid of only SVMs could not tell 'kernels like this table' from 'capacity likes it'."""
    assert "logreg" in crosstable.MODELS
    assert "mlp" in crosstable.MODELS
    assert {"linear_svm", "poly_svm", "rbf_svm"} <= set(crosstable.MODELS)


def test_every_model_builds_a_pipeline_that_scales_first():
    for model in crosstable.MODELS:
        estimator = crosstable.estimator(model, "hybrid")
        assert list(estimator.named_steps)[0] == "prepare"


def test_an_unknown_model_is_refused_by_name():
    with pytest.raises(ValueError, match="model must be one of"):
        crosstable.estimator("nonsense")


def test_each_model_carries_the_settings_its_own_task_selected():
    from src.classify.svm import BEST_PARAMS

    assert BEST_PARAMS["C"] == crosstable.estimator("linear_svm").named_steps["model"].C
    assert crosstable.estimator("poly_svm").named_steps["model"].coef0 == 10.0


def test_the_network_uses_6_2_3s_per_table_regularization():
    from src.classify.regularize import best_settings

    model = crosstable.estimator("mlp", "handcrafted").named_steps["model"]
    assert model.dropout == best_settings("handcrafted")["dropout"]
    assert model.batch_size == 16


def test_the_network_is_not_nested_inside_a_parallel_pool(toy):
    """6.2.2 measured a torch fit inside many workers wedging the pool."""
    import inspect

    assert 'if model == "mlp" else' in inspect.getsource(crosstable.cell)


# -- reading the grid ----------------------------------------------------------------------------


def test_the_winners_are_reported_both_ways():
    result = crosstable.winners(GRID, TABLES)
    assert result["best_model_per_table"]["handcrafted"] == "rbf_svm"
    assert result["best_table_per_model"]["logreg"] == "embedding"
    assert result["best_overall"][0] == "rbf_svm"


def test_the_spreads_measure_each_axis_separately():
    result = crosstable.spreads(GRID, TABLES)
    # handcrafted: 0.83 - 0.77 = 0.06 ; embedding: 0.97 - 0.96 = 0.01
    assert result["model_choice_worth_on_table"]["handcrafted"] == pytest.approx(0.06)
    assert result["model_choice_worth_on_table"]["embedding"] == pytest.approx(0.01)
    assert result["table_choice_worth_for_model"]["linear_svm"] == pytest.approx(0.20)


def test_the_interaction_is_the_difference_between_the_model_spreads():
    """A pure main effect would make the model choice worth the same on every table."""
    result = crosstable.spreads(GRID, TABLES)
    values = result["model_choice_worth_on_table"].values()
    assert result["interaction"] == pytest.approx(max(values) - min(values))


def test_a_grid_with_no_interaction_reports_none():
    flat = {
        "a": {"handcrafted": 0.70, "embedding": 0.90},
        "b": {"handcrafted": 0.75, "embedding": 0.95},
    }
    result = crosstable.spreads(flat, ("handcrafted", "embedding"))
    assert result["interaction"] == pytest.approx(0.0, abs=1e-9)


def test_the_ranking_stability_compares_every_pair_of_tables():
    result = crosstable.ranking_stability(GRID, TABLES)
    assert len(result["spearman_between_tables"]) == 3
    assert set(result["ranking_per_table"]) == set(TABLES)
    assert result["ranking_per_table"]["handcrafted"][0] == "rbf_svm"


def test_the_ranking_is_ordered_best_first():
    result = crosstable.ranking_stability(GRID, TABLES)
    for table, order in result["ranking_per_table"].items():
        scores = [GRID[model][table] for model in order]
        assert scores == sorted(scores, reverse=True)


# -- the rendered table --------------------------------------------------------------------------------


def test_the_markdown_has_one_row_per_model_and_marks_the_winner():
    text = crosstable.to_markdown(GRID, TABLES)
    lines = text.splitlines()
    assert len(lines) == 2 + len(GRID)
    assert "`logreg`" in text
    assert "**0.9600**" in text


def test_the_markdown_names_each_models_best_table():
    text = crosstable.to_markdown(GRID, TABLES)
    row = next(line for line in text.splitlines() if "linear_svm" in line)
    assert row.strip().endswith("| embedding |")


# -- one real cell -----------------------------------------------------------------------------------------


def test_a_cell_is_a_macro_f1_under_the_shared_protocol(toy):
    value = crosstable.cell(toy, "logreg", "hybrid", folds=3, n_jobs=2)
    assert 0.0 <= value <= 1.0

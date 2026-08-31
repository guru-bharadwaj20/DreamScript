"""Phase 5.3.2 - the tree exported as rules."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import rules
from src.classify.data import Dataset


@pytest.fixture
def toy():
    rng = np.random.default_rng(14)
    y = np.array(["a"] * 90 + ["b"] * 90 + ["rare"] * 20, dtype=object)
    centres = {"a": (0.0, 0.0), "b": (4.0, 0.0), "rare": (2.0, 4.0)}
    X = np.array([centres[label] for label in y]) + rng.normal(0, 0.7, size=(200, 2))
    X = np.hstack([X, rng.normal(size=(200, 2))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(200)], dtype=object),
        ids=np.arange(200).astype(object),
        feature_names=["f0", "f1", "f2", "f3"],
        corpus="test",
    )


def extracted(dataset):
    estimator, model, prepare, names = rules.fitted(dataset)
    return rules.extract(
        model, names, model.classes_, prepare, rules.leaf_membership(estimator, dataset)
    )


def test_every_row_lands_in_exactly_one_leaf(toy):
    leaves = extracted(toy)
    assert sum(leaf["coverage"] for leaf in leaves) == len(toy.y)


def test_the_leaves_are_ranked_by_coverage(toy):
    coverage = [leaf["coverage"] for leaf in extracted(toy)]
    assert coverage == sorted(coverage, reverse=True)


def test_purity_counts_rows_not_class_weights(toy):
    """5.1.3 selected class_weight=balanced, so tree_.value is weighted and unusable here.

    A purity that is a share of rows must multiply back to a whole number of rows. A weighted
    one - which is what `tree_.value` would give - does not.
    """
    for leaf in extracted(toy):
        assert 0.0 < leaf["purity"] <= 1.0
        assert leaf["purity"] * leaf["coverage"] == pytest.approx(
            round(leaf["purity"] * leaf["coverage"]), abs=1e-6
        )


def test_leaf_membership_partitions_the_corpus(toy):
    estimator, _, _, _ = rules.fitted(toy)
    membership = rules.leaf_membership(estimator, toy)
    assert sum(len(members) for members in membership.values()) == len(toy.y)


def test_a_threshold_comes_back_in_the_features_own_units(toy):
    """The tree splits on standardised columns; a report in those units is unreadable."""
    _, model, prepare, _ = rules.fitted(toy)
    column = int(model.tree_.feature[0])
    standardised = float(model.tree_.threshold[0])
    raw = rules.unstandardise(prepare, column, standardised)
    assert raw == pytest.approx(
        prepare.named_steps["scale"].mean_[column]
        + standardised * prepare.named_steps["scale"].scale_[column]
    )
    # The toy's first two columns are centred far from zero, so the two forms must differ.
    assert raw != pytest.approx(standardised)


def test_a_repeated_feature_collapses_into_a_range():
    leaf = {
        "conditions": [
            {"feature": "x", "operator": ">", "threshold": 3.0},
            {"feature": "y", "operator": "<=", "threshold": 1.0},
            {"feature": "x", "operator": "<=", "threshold": 9.0},
        ]
    }
    assert rules.sentence(leaf) == "3 < x <= 9 and y <= 1"


def test_the_tightest_bound_wins_when_a_feature_is_tested_twice():
    leaf = {
        "conditions": [
            {"feature": "x", "operator": "<=", "threshold": 9.0},
            {"feature": "x", "operator": "<=", "threshold": 4.0},
        ]
    }
    assert rules.sentence(leaf) == "x <= 4"


def test_a_root_leaf_has_an_empty_rule():
    assert rules.sentence({"conditions": []}) == ""


def test_the_summary_accounts_for_every_row(toy):
    leaves = extracted(toy)
    stats = rules.summarise(leaves, top=3)
    assert stats["rows"] == len(toy.y)
    assert stats["leaves"] == len(leaves)
    assert 0.0 < stats["coverage_of_printed"] <= 1.0
    assert sum(stats["leaves_per_class"].values()) == len(leaves)


def test_the_report_names_the_rules_it_prints(toy):
    leaves = extracted(toy)
    text = rules.to_markdown(leaves, rules.summarise(leaves, top=5), toy, top=5)
    assert "# Phase 5.3.2" in text
    assert text.count("| **") == min(5, len(leaves))
    assert "<=" in text


def test_the_rendered_tree_is_written(tmp_path, toy):
    path = rules.figure(toy, depth=2, path=tmp_path / "tree.png")
    assert path.is_file() and path.stat().st_size > 5000


def test_the_run_writes_both_artefacts(tmp_path, toy, monkeypatch):
    monkeypatch.setattr(rules, "load", lambda corpus: toy)
    monkeypatch.setattr(rules, "REPORT", tmp_path / "tree_rules.md")
    monkeypatch.setattr(rules, "FIGURE", tmp_path / "p5_tree.png")
    result = rules.run("real", top=5, write=True)
    assert (tmp_path / "tree_rules.md").is_file()
    assert len(result["top_rules"]) == min(5, result["leaves"])
    assert result["report"].endswith("tree_rules.md")

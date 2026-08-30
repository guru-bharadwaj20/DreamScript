"""Phase 5.2.3 - the classification report."""

from __future__ import annotations

import numpy as np
import pytest

from src.classify import metrics
from src.classify.data import Dataset


@pytest.fixture
def skewed():
    """The shape of the real corpus: two big classes and one tiny one."""
    rng = np.random.default_rng(6)
    y = np.array(["flowchart"] * 60 + ["wireframe"] * 60 + ["circuit"] * 10, dtype=object)
    centres = {"flowchart": 0.0, "wireframe": 5.0, "circuit": 2.5}
    X = np.array([[centres[label] + rng.normal(0, 0.8)] for label in y])
    X = np.hstack([X, rng.normal(size=(len(y), 3))])
    return Dataset(
        X=X,
        y=y,
        groups=np.array([f"row:{i}" for i in range(len(y))], dtype=object),
        ids=np.arange(len(y)).astype(object),
        feature_names=[f"f{i}" for i in range(4)],
        corpus="test",
    )


def test_every_class_gets_a_row(skewed):
    result = metrics.report(skewed, "tree", seeds=(42,))
    assert set(result["per_class"]) == {"flowchart", "wireframe", "circuit"}
    assert result["per_class"]["circuit"]["support"] == 10


def test_macro_and_weighted_are_both_reported_and_differ(skewed):
    result = metrics.report(skewed, "tree", seeds=(42,))
    assert result["macro_minus_weighted"] == pytest.approx(
        result["macro_f1"] - result["weighted_f1"], abs=1e-3
    )
    assert result["macro_minus_weighted"] < 0.0, "the tiny class must drag macro F1 down"


def test_a_constant_predictor_scores_zero_on_every_class_it_ignores(skewed):
    result = metrics.report(skewed, "majority", seeds=(42,))
    assert result["per_class"]["circuit"]["f1"] == 0.0
    assert result["per_class"]["flowchart"]["recall"] == 1.0
    assert result["macro_f1"] < result["weighted_f1"]


def test_balanced_accuracy_is_the_mean_of_the_recalls(skewed):
    result = metrics.report(skewed, "tree", seeds=(42,))
    recalls = [cell["recall"] for cell in result["per_class"].values()]
    assert result["balanced_accuracy"] == pytest.approx(np.mean(recalls), abs=1e-3)


def test_the_report_averages_over_repeats(skewed):
    one = metrics.report(skewed, "tree", seeds=(42,))
    three = metrics.report(skewed, "tree", seeds=(42, 43, 44))
    assert one["repeats"] == 1
    assert three["repeats"] == 3
    assert three["per_class"]["circuit"]["f1_std"] >= 0.0


def test_the_markdown_has_a_row_per_class_and_a_headline(skewed):
    reports = [metrics.report(skewed, model, seeds=(42,)) for model in ("tree", "majority")]
    text = metrics.to_markdown(reports, skewed)
    assert "macro F1" in text
    for name in ("flowchart", "wireframe", "circuit"):
        assert text.count(f"| {name} |") == 2
    assert "`tree`" in text and "`majority`" in text


def test_writing_the_report_is_optional(skewed, monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "REPORT", tmp_path / "report.md")
    monkeypatch.setattr(metrics, "load", lambda corpus: skewed)
    result = metrics.run(models=("tree",), write=True)
    assert (tmp_path / "report.md").is_file()
    assert result["reports"][0]["model"] == "tree"

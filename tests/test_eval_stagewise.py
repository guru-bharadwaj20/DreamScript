"""Phase 14.2 - the stage-wise table's contract is the `input` line, so that is what is tested.

A stage-wise report is only meaningful if each number says what the stage was handed. These
tests pin that every stage declares its input, that a missing artefact degrades to *missing*
rather than to a plausible-looking number, and that the classification re-measurement really
runs the writer-grouped protocol rather than a shuffle that would leak a writer across folds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval import stagewise


def test_every_stage_declares_the_input_it_was_scored_on():
    result = stagewise.collect(fit=False)
    assert set(result) == {"classify", "detect", "ocr", "parse", "assemble", "codegen"}
    for stage, entry in result.items():
        assert entry["input"], stage
        assert "models" in entry, stage


def test_codegen_is_declared_as_a_gold_ir_measurement():
    """12.3.3's pass@1 is from the annotated IR; calling it end-to-end would be the error."""
    entry = stagewise.codegen_stage()
    assert "ground-truth IR" in entry["input"]
    assert entry["models"]["lora_r64"]["functional"] is not None


def test_assemble_is_declared_as_a_predicted_input_measurement():
    entry = stagewise.assemble_stage()
    assert "predicted" in entry["input"]


def test_a_missing_artefact_reads_missing_and_not_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(stagewise, "ROOT", tmp_path)
    entry = stagewise.ocr_stage()
    assert entry["models"]["trocr_large_selected"]["cer"] is None
    text = stagewise.render({"ocr": entry})
    assert "*missing*" in text
    assert "0.0" not in text


def test_read_walks_a_key_path(monkeypatch, tmp_path):
    monkeypatch.setattr(stagewise, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "x.json").write_text(json.dumps({"a": {"b": 2}}), encoding="utf-8")
    assert stagewise._read("reports/x.json", "a", "b") == 2
    assert stagewise._read("reports/x.json", "a", "c") is None


class _Data:
    """A two-writer dataset where the label is the writer, so any leak scores 1.0."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.X = np.vstack([rng.normal(0, 1, (40, 3)), rng.normal(0, 1, (40, 3))])
        self.y = np.array(["a"] * 40 + ["b"] * 40, dtype=object)
        self.groups = np.array([f"scribe:{label}" for label in self.y], dtype=object)


def test_cross_validation_holds_the_writer_out():
    from sklearn.dummy import DummyClassifier

    result = stagewise._cross_validate(_Data(), lambda: DummyClassifier(strategy="most_frequent"), (42,), 2)
    # With the writer held out the training side never contains the held-out class, so a model
    # that only knows the majority class must score zero - a leak would show up as > 0.
    assert result["accuracy"] == 0.0
    assert result["seeds"] == [42]


def test_cross_validation_reports_spread_over_seeds():
    from sklearn.linear_model import LogisticRegression

    data = _Data()
    data.groups = np.array([f"row:{i}" for i in range(len(data.y))], dtype=object)
    result = stagewise._cross_validate(data, lambda: LogisticRegression(max_iter=200), (42, 43), 2)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["accuracy_std"] >= 0.0
    assert result["worst_repeat_accuracy"] <= result["accuracy"]


def test_the_classification_section_states_the_corpus_it_measured():
    """The published S1 row and the surviving corpus disagree; the report must say so."""
    result = json.loads((stagewise.ROOT / "reports" / "stagewise.json").read_text("utf-8"))
    corpus = result["classify"]["corpus"]
    assert corpus["rows"] == sum(corpus["classes"].values())
    text = (stagewise.ROOT / "reports" / "stagewise.md").read_text("utf-8")
    assert str(corpus["rows"]) in text
    assert str(result["classify"]["published"]["s1_rows"]) in text


def test_render_lists_a_model_once_per_stage():
    text = stagewise.render(stagewise.collect(fit=False))
    assert text.count("## ocr") == 1
    assert text.count("| trocr_large_selected |") == 1


@pytest.mark.parametrize("stage", ["detect", "ocr", "parse", "assemble", "codegen"])
def test_read_only_stages_need_no_fitting(stage):
    entry = getattr(stagewise, f"{stage}_stage")()
    for model in entry["models"].values():
        assert "provenance" in model

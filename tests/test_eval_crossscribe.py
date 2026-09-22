"""Phase 14.6 - leave-one-scribe-out has one invariant: the held-out writer is really absent.

The second thing worth pinning is the refusal: per-writer OCR numbers must not be published from
a predictions file that does not reproduce the published aggregate, because that file belongs to
whichever run wrote it last and is not necessarily the incumbent checkpoint.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval import crossscribe


def test_writer_is_read_out_of_either_page_naming():
    assert crossscribe._writer_of("fa_bresler__writer015_fa_001") == "writer015"
    assert crossscribe._writer_of("hdbpmn__ex00_writer0065") == "writer0065"
    assert crossscribe._writer_of("odd_name") == "odd name".replace(" ", "_")


def test_spread_reports_the_tails_not_just_the_mean():
    spread = crossscribe._spread([0.1, 0.5, 0.5, 0.5, 0.9])
    assert spread["n"] == 5
    assert spread["median"] == 0.5
    assert spread["min"] == 0.1
    assert spread["max"] == 0.9
    assert spread["spread_p90_p10"] > 0


def test_spread_of_nothing_is_empty_rather_than_zero():
    assert crossscribe._spread([]) == {}


class _Data:
    """Three writers, each drawing both classes, with the class readable from the feature.

    A fold that kept the held-out writer's rows in the fit would score 1.0 on a model that
    memorised them; an honest fold scores 1.0 only if the feature generalises, which here it
    does - so the leak is caught by the *skipped* case below rather than by this one.
    """

    def __init__(self):
        rows, labels, groups = [], [], []
        for writer in ("one", "two", "three"):
            for index in range(6):
                label = "a" if index % 2 else "b"
                rows.append([0.0 if label == "a" else 1.0, float(index)])
                labels.append(label)
                groups.append(f"scribe:{writer}")
        self.X = np.array(rows)
        self.y = np.array(labels, dtype=object)
        self.groups = np.array(groups, dtype=object)

    def summary(self):
        return {}


def test_every_known_writer_is_held_out_in_turn(monkeypatch):
    monkeypatch.setattr("src.classify.data.load", lambda *a, **k: _Data())
    result = crossscribe.classification_loso("logreg")
    assert result["spread"]["n"] == 3
    assert result["rows_with_a_known_writer"] == 18
    assert [row["pages"] for row in result["per_writer"]] == [6, 6, 6]


def test_a_writer_whose_removal_leaves_one_class_is_skipped_and_counted(monkeypatch):
    """Such a fold measures a degenerate fit, not generalisation; it must not be averaged in."""
    data = _Data()
    data.y = np.array(["a"] * 6 + ["b"] * 12, dtype=object)
    monkeypatch.setattr("src.classify.data.load", lambda *a, **k: data)
    result = crossscribe.classification_loso("logreg")
    assert result["writers_skipped"] == ["one"]
    assert result["writers_scored"] == 2


def test_rows_without_a_writer_are_counted_not_hidden(monkeypatch):
    data = _Data()
    data.groups = np.array(["scribe:one"] * 6 + [f"row:{i}" for i in range(12)], dtype=object)
    monkeypatch.setattr("src.classify.data.load", lambda *a, **k: data)
    result = crossscribe.classification_loso("logreg")
    assert result["rows_without_one"] == 12
    assert result["spread"]["n"] == 1


def test_ocr_withholds_the_breakdown_when_the_rescore_disagrees(monkeypatch, tmp_path):
    report = tmp_path / "s3.json"
    report.write_text(json.dumps({"cer": 0.2006}), encoding="utf-8")
    monkeypatch.setattr(crossscribe, "S3_REPORT", report)
    monkeypatch.setattr(
        "src.ocr.s3.split", lambda name: (["a.png"], ["hello"], _frame(["writer01"], ["a.png"]))
    )
    monkeypatch.setattr(crossscribe, "ROOT", tmp_path)
    (tmp_path / "experiments" / "ocr").mkdir(parents=True)
    (tmp_path / "experiments" / "ocr" / "s3_val.json").write_text(
        json.dumps({"files": ["a.png"], "predictions": ["hellX"]}), encoding="utf-8"
    )

    entry = crossscribe.ocr_per_writer(rescore=False)

    assert entry["reproduces_published"] is False
    assert "per_writer" not in entry
    assert "withheld" in entry["note"]


def _frame(writers, files):
    import pandas as pd

    return pd.DataFrame({"scribe": writers, "file": files})


def test_ocr_publishes_the_breakdown_when_the_rescore_agrees(monkeypatch, tmp_path):
    report = tmp_path / "s3.json"
    report.write_text(json.dumps({"cer": 0.0}), encoding="utf-8")
    monkeypatch.setattr(crossscribe, "S3_REPORT", report)
    monkeypatch.setattr(
        "src.ocr.s3.split",
        lambda name: (["a.png", "b.png"], ["hi", "yo"], _frame(["w1", "w2"], ["a.png", "b.png"])),
    )
    monkeypatch.setattr(crossscribe, "ROOT", tmp_path)
    (tmp_path / "experiments" / "ocr").mkdir(parents=True)
    (tmp_path / "experiments" / "ocr" / "s3_val.json").write_text(
        json.dumps({"files": ["a.png", "b.png"], "predictions": ["hi", "yo"]}), encoding="utf-8"
    )

    entry = crossscribe.ocr_per_writer(rescore=False)

    assert entry["reproduces_published"] is True
    assert {row["writer"] for row in entry["per_writer"]} == {"w1", "w2"}


def test_page_level_reports_group_by_writer():
    rows = [
        {"page": "hdbpmn__ex00_writer0001", "source": "hdbpmn", "ged": 10.0},
        {"page": "hdbpmn__ex01_writer0001", "source": "hdbpmn", "ged": 20.0},
        {"page": "hdbpmn__ex00_writer0002", "source": "hdbpmn", "ged": 4.0},
    ]
    entry = crossscribe._per_writer_from_pages(rows, lambda row: row["ged"], "median_ged")
    assert entry["writers"] == 2
    by_writer = {row["writer"]: row["median_ged"] for row in entry["per_writer"]}
    assert by_writer["writer0001"] == 15.0
    assert by_writer["writer0002"] == 4.0


def test_the_cross_stage_question_is_left_open_when_the_splits_do_not_overlap():
    ocr = {"per_writer": [{"writer": "w1", "cer": 0.2}]}
    assembly = {"per_writer": [{"writer": "w9", "median_ged": 10.0}]}
    answer = crossscribe.overlap(ocr, assembly)
    assert answer["shared_writers"] == 0
    assert "left open" in answer["note"]


def test_render_survives_a_stage_with_no_measurement():
    text = crossscribe.render(
        {
            "classify": {"note": "not run"},
            "ocr": {"note": "not run"},
            "assemble": {"note": "not run"},
            "end_to_end": {"note": "not run"},
            "cross_stage": {"note": "not run"},
        }
    )
    assert "not run" in text
    assert "# Phase 14.6" in text

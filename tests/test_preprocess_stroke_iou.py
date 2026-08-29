"""Phase 3.3.1 - stroke IoU.

The corpus number is measured by `python -m src.preprocess.stroke_iou`; these tests pin the
mechanism and the reporting, on a handful of items so the suite stays quick.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import stroke_iou as si
from src.preprocess.evalset import build, f1


@pytest.fixture(scope="module")
def items():
    built = build(4)
    if not built:
        pytest.skip("FA InkML not available")
    return built


def test_pipeline_recovers_a_clean_render(items):
    """With no damage at all, the pipeline should recover the strokes almost exactly."""
    item = items[0]
    clean = np.where(item.mask, 35, 245).astype(np.uint8)
    scores = f1(si.pipeline(clean), item.mask)
    assert scores["iou"] >= 0.80


def test_pipeline_returns_a_boolean_mask_of_the_same_shape(items):
    predicted = si.pipeline(items[0].photo)
    assert predicted.dtype == bool
    assert predicted.shape == items[0].photo.shape


def test_measured_rows_carry_their_damage_and_scores(items):
    rows = si.measure(4)
    assert len(rows) == 4
    for row in rows:
        assert row["damage"], "every evaluation item is damaged"
        assert 0.0 <= row["iou"] <= 1.0
        assert row["iou"] <= row["f1"], "IoU is never above F1"


def test_summary_counts_items_below_the_bar():
    rows = [
        {"name": "a", "damage": ["blur"], "iou": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
        {"name": "b", "damage": ["jpeg"], "iou": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
    ]
    summary = si.summarise(rows)
    assert summary["items_below_target"] == 1
    assert summary["worst_iou"] == 0.5
    assert summary["mean_iou"] == 0.7


def test_damage_breakdown_is_worst_first():
    rows = [
        {"name": "a", "damage": ["blur"], "iou": 0.6, "precision": 1, "recall": 1, "f1": 1},
        {"name": "b", "damage": ["shadow"], "iou": 0.9, "precision": 1, "recall": 1, "f1": 1},
        {
            "name": "c",
            "damage": ["blur", "shadow"],
            "iou": 0.8,
            "precision": 1,
            "recall": 1,
            "f1": 1,
        },
    ]
    breakdown = si.by_damage(rows)
    assert list(breakdown) == ["blur", "shadow"]
    assert breakdown["blur"]["items"] == 2


def test_report_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "REPORT", tmp_path / "stroke_iou.md")
    rows = [
        {"name": "a", "damage": ["blur"], "iou": 0.85, "precision": 0.9, "recall": 0.95, "f1": 0.9}
    ]
    path = si.report(rows)
    text = path.read_text(encoding="utf-8")
    assert "stroke IoU" in text and "0.850" in text


def test_a_thickened_prediction_costs_iou_but_not_recall(items):
    """The failure mode the corpus number is made of, on one item."""
    item = items[0]
    fat = cv2.dilate(item.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    scores = f1(fat, item.mask)
    assert scores["recall"] > 0.99
    assert scores["iou"] < scores["recall"]

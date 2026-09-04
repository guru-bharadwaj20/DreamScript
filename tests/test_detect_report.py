"""Phase 9.1.4 - four readings of one target, and the arithmetic that separates them."""

from __future__ import annotations

import pytest

from src.detect import report
from src.detect.classes import CLASSES


def truth(image, cls, box):
    return {"image": image, "cls": cls, "xyxy": list(box)}


def prediction(image, cls, box, score=0.9):
    return {"image": image, "cls": cls, "xyxy": list(box), "score": score}


SOURCES = {"hdbpmn__a": "hdbpmn", "flowchartseg__b": "flowchartseg"}


def test_the_target_is_the_one_the_plan_states():
    assert report.TARGET == 0.80


def test_the_derived_class_is_the_one_9_1_2_derived():
    assert report.DERIVED == "arrowhead"


def test_hand_drawn_only_scores_the_photographs_alone():
    """The rendered page is perfect and the photograph is not; the two views must disagree."""
    truths = [
        truth("hdbpmn__a", "circle", (0, 0, 10, 10)),
        truth("flowchartseg__b", "circle", (0, 0, 10, 10)),
    ]
    preds = [prediction("flowchartseg__b", "circle", (0, 0, 10, 10))]
    views = report.views(preds, truths, SOURCES)
    assert views["hand_drawn_only"]["map50"] == 0.0
    assert views["pooled"]["map50"] > 0.0
    assert views["hand_drawn_only"]["pages"] == 1


def test_dropping_the_derived_class_changes_the_mean():
    truths = [
        truth("hdbpmn__a", "circle", (0, 0, 10, 10)),
        truth("hdbpmn__a", "arrowhead", (50, 50, 60, 60)),
    ]
    preds = [prediction("hdbpmn__a", "circle", (0, 0, 10, 10))]
    views = report.views(preds, truths, SOURCES)
    assert views["pooled"]["map50"] == pytest.approx(0.5)
    assert views["annotated_classes_only"]["map50"] == 1.0


def test_the_loose_view_rescues_a_box_that_found_the_object_but_missed_the_convention():
    truths = [truth("hdbpmn__a", "arrowhead", (0, 0, 10, 10))]
    preds = [prediction("hdbpmn__a", "arrowhead", (6, 0, 16, 10))]
    views = report.views(preds, truths, SOURCES)
    assert views["pooled"]["map50"] == 0.0
    assert views["loose_iou_0_25"]["map25"] == 1.0


def test_every_view_carries_its_own_verdict():
    truths = [truth("hdbpmn__a", "circle", (0, 0, 10, 10))]
    preds = [prediction("hdbpmn__a", "circle", (0, 0, 10, 10))]
    views = report.views(preds, truths, SOURCES)
    for name in ("pooled", "hand_drawn_only", "annotated_classes_only", "loose_iou_0_25"):
        assert views[name]["meets"] is True


def test_a_failing_view_says_so():
    truths = [truth("hdbpmn__a", "circle", (0, 0, 10, 10))]
    preds = []
    views = report.views(preds, truths, SOURCES)
    assert views["pooled"]["meets"] is False


def test_the_pr_curve_is_monotone_in_recall():
    truths = [truth("hdbpmn__a", "circle", (i * 20, 0, i * 20 + 10, 10)) for i in range(3)]
    preds = [
        prediction("hdbpmn__a", "circle", t["xyxy"], 0.9 - 0.1 * i) for i, t in enumerate(truths)
    ]
    recall, precision = report.pr_curve(preds, truths, "circle")
    assert list(recall) == sorted(recall)
    assert precision[-1] == pytest.approx(1.0)


def test_a_class_with_no_instances_has_an_empty_curve():
    recall, _ = report.pr_curve([], [], "parallelogram")
    assert len(recall) == 0


def test_the_written_report_lists_every_class_and_every_view(tmp_path):
    result = {
        "views": {
            "pooled": {"map50": 0.5, "meets": False},
            "hand_drawn_only": {"map50": 0.4, "meets": False},
            "annotated_classes_only": {"map50": 0.6, "meets": False},
            "loose_iou_0_25": {"map25": 0.9, "meets": True},
        },
        "per_class": {
            n: {"instances": 1, "ap50": 0.5, "ap50_95": 0.3, "ap25": 0.6} for n in CLASSES
        },
        "by_source": {"hdbpmn": {"map50": 0.4, "map50_95": 0.2}},
        "pages_per_source": {"hdbpmn": 128},
    }
    text = report.write_report(result, tmp_path / "detection.md").read_text(encoding="utf-8")
    for name in CLASSES:
        assert name in text
    assert "hand drawn only" in text
    assert "NO" in text

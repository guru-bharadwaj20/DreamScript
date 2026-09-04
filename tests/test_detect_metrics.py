"""Phase 9.1 - the shared evaluator, pinned on cases whose answer is known by hand."""

from __future__ import annotations

import numpy as np
import pytest

from src.detect.metrics import (
    IOU_THRESHOLDS,
    LOOSE_IOU,
    average_precision,
    by_source,
    evaluate,
    iou_matrix,
)


def box(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2), float(y2)]


def truth(image, cls, b):
    return {"image": image, "cls": cls, "xyxy": b}


def prediction(image, cls, b, score):
    return {"image": image, "cls": cls, "xyxy": b, "score": score}


# -- IoU ------------------------------------------------------------------------------------


def test_identical_boxes_have_iou_one():
    assert iou_matrix(np.array([box(0, 0, 10, 10)]), np.array([box(0, 0, 10, 10)]))[0, 0] == 1.0


def test_disjoint_boxes_have_iou_zero():
    assert iou_matrix(np.array([box(0, 0, 10, 10)]), np.array([box(20, 20, 30, 30)]))[0, 0] == 0.0


def test_a_half_overlap_is_one_third():
    """Intersection 50, union 150."""
    value = iou_matrix(np.array([box(0, 0, 10, 10)]), np.array([box(5, 0, 15, 10)]))[0, 0]
    assert value == pytest.approx(1 / 3)


def test_empty_sides_give_an_empty_matrix():
    assert iou_matrix(np.zeros((0, 4)), np.array([box(0, 0, 1, 1)])).shape == (0, 1)


# -- average precision ----------------------------------------------------------------------


def test_a_class_with_no_ground_truth_is_nan_not_zero():
    """Scoring it 0 would punish a model for a class the split does not contain."""
    assert np.isnan(average_precision(np.array([True]), np.array([0.9]), positives=0))


def test_perfect_detection_scores_one():
    assert average_precision(np.array([True, True]), np.array([0.9, 0.8]), 2) == 1.0


def test_missing_half_the_instances_caps_recall_at_a_half():
    """101-point sampling: 51 of the points sit at or below recall 0.5."""
    assert average_precision(np.array([True]), np.array([0.9]), 2) == pytest.approx(51 / 101)


def test_a_false_positive_ranked_first_costs_precision():
    high = average_precision(np.array([True, False]), np.array([0.9, 0.1]), 1)
    low = average_precision(np.array([False, True]), np.array([0.9, 0.1]), 1)
    assert high > low


# -- the evaluation -------------------------------------------------------------------------


def test_perfect_predictions_score_one_everywhere():
    truths = [truth("a", "diamond", box(0, 0, 10, 10)), truth("a", "diamond", box(20, 20, 30, 30))]
    preds = [prediction(t["image"], t["cls"], t["xyxy"], 0.9) for t in truths]
    result = evaluate(preds, truths)
    assert result["map50"] == 1.0
    assert result["map50_95"] == 1.0
    assert result["classes_scored"] == 1


def test_a_box_offset_to_iou_a_quarter_fails_at_half_and_passes_loose():
    """This is the gap 9.1.2's derived arrowheads are expected to live in."""
    truths = [truth("a", "arrowhead", box(0, 0, 10, 10))]
    preds = [prediction("a", "arrowhead", box(6, 0, 16, 10), 0.9)]
    entry = evaluate(preds, truths)["per_class"]["arrowhead"]
    assert entry["ap50"] == 0.0
    assert entry["ap25"] == 1.0


def test_a_duplicate_box_is_matched_only_once():
    """Greedy matching consumes a ground truth once, so the second copy is a false positive."""
    truths = [truth("a", "circle", box(0, 0, 10, 10))]
    preds = [
        prediction("a", "circle", box(0, 0, 10, 10), 0.9),
        prediction("a", "circle", box(0, 0, 10, 10), 0.8),
    ]
    from src.detect.metrics import _match

    matched, _, positives = _match(preds, truths, 0.5)
    assert list(matched) == [True, False]
    assert positives == 1


def test_a_duplicate_ranked_below_full_recall_costs_nothing():
    """Not a bug and worth pinning: COCO's monotone envelope keeps the earlier precision, so
    a redundant box that arrives after every instance is already found is free. Duplicate
    suppression therefore has to be judged by 9.1.6's own counts, not by mAP."""
    truths = [truth("a", "circle", box(0, 0, 10, 10))]
    preds = [
        prediction("a", "circle", box(0, 0, 10, 10), 0.9),
        prediction("a", "circle", box(0, 0, 10, 10), 0.8),
    ]
    assert evaluate(preds, truths)["per_class"]["circle"]["ap50"] == 1.0


def test_a_false_positive_ranked_above_the_hit_does_cost():
    truths = [truth("a", "circle", box(0, 0, 10, 10))]
    preds = [
        prediction("a", "circle", box(50, 50, 60, 60), 0.95),
        prediction("a", "circle", box(0, 0, 10, 10), 0.9),
    ]
    assert evaluate(preds, truths)["per_class"]["circle"]["ap50"] < 1.0


def test_a_prediction_on_the_wrong_image_never_matches():
    truths = [truth("a", "circle", box(0, 0, 10, 10))]
    preds = [prediction("b", "circle", box(0, 0, 10, 10), 0.9)]
    assert evaluate(preds, truths)["per_class"]["circle"]["ap50"] == 0.0


def test_a_prediction_of_the_wrong_class_never_matches():
    truths = [truth("a", "circle", box(0, 0, 10, 10))]
    preds = [prediction("a", "diamond", box(0, 0, 10, 10), 0.9)]
    assert evaluate(preds, truths)["per_class"]["circle"]["ap50"] == 0.0


def test_classes_absent_from_the_split_do_not_move_the_mean():
    truths = [truth("a", "circle", box(0, 0, 10, 10))]
    preds = [prediction("a", "circle", box(0, 0, 10, 10), 0.9)]
    result = evaluate(preds, truths)
    assert result["map50"] == 1.0
    assert result["per_class"]["parallelogram"]["ap50"] is None


def test_the_threshold_sweep_is_cocos_ten():
    assert IOU_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    assert LOOSE_IOU not in IOU_THRESHOLDS


def test_by_source_scores_each_source_on_its_own_pages():
    truths = [
        truth("hdbpmn__a", "circle", box(0, 0, 10, 10)),
        truth("fa__b", "circle", box(0, 0, 10, 10)),
    ]
    preds = [prediction("hdbpmn__a", "circle", box(0, 0, 10, 10), 0.9)]
    sources = {"hdbpmn__a": "hdbpmn", "fa__b": "fa_bresler"}
    result = by_source(preds, truths, sources)
    assert result["hdbpmn"]["map50"] == 1.0
    assert result["fa_bresler"]["map50"] == 0.0

"""Phase 9.1.6 - suppression applied to a fixed candidate set, and the nesting it must not eat."""

from __future__ import annotations

from src.detect import nms as tuning


def prediction(image, cls, box, score):
    return {"image": image, "cls": cls, "xyxy": list(box), "score": score}


def test_the_grid_spans_hard_and_loose_suppression():
    assert min(tuning.IOU_GRID) <= 0.3
    assert max(tuning.IOU_GRID) >= 0.9


def test_per_class_suppression_keeps_a_nested_pair():
    """A pool and the task inside it overlap by construction, not by duplication."""
    rows = [
        prediction("p", "rectangle", (0, 0, 100, 100), 0.9),
        prediction("p", "rounded-rect", (5, 5, 95, 95), 0.8),
    ]
    kept = tuning.suppress(rows, 0.5, agnostic=False)
    assert len(kept) == 2


def test_class_agnostic_suppression_eats_it():
    rows = [
        prediction("p", "rectangle", (0, 0, 100, 100), 0.9),
        prediction("p", "rounded-rect", (5, 5, 95, 95), 0.8),
    ]
    kept = tuning.suppress(rows, 0.5, agnostic=True)
    assert len(kept) == 1
    assert kept[0]["cls"] == "rectangle"


def test_suppression_never_crosses_pages():
    rows = [
        prediction("a", "circle", (0, 0, 10, 10), 0.9),
        prediction("b", "circle", (0, 0, 10, 10), 0.8),
    ]
    assert len(tuning.suppress(rows, 0.5, agnostic=True)) == 2


def test_a_higher_threshold_keeps_at_least_as_many_boxes():
    rows = [
        prediction("p", "circle", (0, 0, 10, 10), 0.9),
        prediction("p", "circle", (2, 2, 12, 12), 0.8),
        prediction("p", "circle", (4, 4, 14, 14), 0.7),
    ]
    counts = [len(tuning.suppress(rows, t, False)) for t in (0.3, 0.6, 0.9)]
    assert counts == sorted(counts)


def test_nesting_counts_a_contained_box_of_a_different_class():
    rows = [
        prediction("p", "rectangle", (0, 0, 100, 100), 0.9),
        prediction("p", "rounded-rect", (10, 10, 40, 40), 0.8),
    ]
    assert tuning.nesting(rows)["nested_pairs"] == 1


def test_nesting_ignores_a_contained_box_of_the_same_class():
    """Same-class containment is what suppression is *for*; it is not the structure to protect."""
    rows = [
        prediction("p", "rectangle", (0, 0, 100, 100), 0.9),
        prediction("p", "rectangle", (10, 10, 40, 40), 0.8),
    ]
    assert tuning.nesting(rows)["nested_pairs"] == 0


def test_nesting_ignores_low_confidence_boxes():
    rows = [
        prediction("p", "rectangle", (0, 0, 100, 100), 0.9),
        prediction("p", "rounded-rect", (10, 10, 40, 40), 0.01),
    ]
    assert tuning.nesting(rows)["nested_pairs"] == 0

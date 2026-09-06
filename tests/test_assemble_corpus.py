"""Phase 10 - the held-out page set, and the one coordinate frame every task measures in."""

from __future__ import annotations

import pytest

from src.assemble import corpus


def test_only_pages_the_detector_never_trained_on_are_offered():
    """Assembly measured on the detector's training pages would be measuring memorisation."""
    assert {page.split for page in corpus.pages()} <= {"val", "test"}


def test_every_offered_page_has_both_an_image_and_a_ground_truth_graph():
    for page in corpus.pages():
        assert page.image.is_file()
        assert page.ir_path.is_file()


def test_the_page_order_is_fixed_so_two_runs_enumerate_identically():
    assert [p.name for p in corpus.pages()] == sorted(p.name for p in corpus.pages())


def test_the_truth_is_returned_in_detector_pixels_and_says_so():
    page = next(p for p in corpus.pages() if p.source == "hdbpmn")
    diagram = corpus.truth(page)
    assert diagram.meta["assemble_frame"] == "detect-pixels"
    assert diagram.meta["assemble_scale"] == pytest.approx(page.scale)


def test_scaling_the_truth_puts_the_boxes_inside_the_detector_image():
    """The whole point of the frame: a scaled box and a predicted box are comparable."""
    page = next(p for p in corpus.pages() if p.source == "hdbpmn")
    width, height = page.size
    boxed = [n for n in corpus.truth(page).nodes if n.bbox]
    assert boxed, "an hdbpmn page has boxes"
    for node in boxed:
        x, y, w, h = node.bbox
        assert -1.0 <= x <= width + 1.0
        assert -1.0 <= y <= height + 1.0


def test_a_box_is_not_scaled_twice():
    page = corpus.pages()[0]
    once = corpus.truth(page)
    twice = corpus.truth(page)
    assert [n.bbox for n in once.nodes] == [n.bbox for n in twice.nodes]


def test_scale_box_is_a_pure_multiplication_of_all_four_numbers():
    assert corpus.scale_box([10, 20, 30, 40], 0.5) == [5.0, 10.0, 15.0, 20.0]


def test_iou_of_a_box_with_itself_is_one_and_with_a_disjoint_box_is_zero():
    box = [0.0, 0.0, 10.0, 10.0]
    assert corpus.iou(box, box) == pytest.approx(1.0)
    assert corpus.iou(box, [100.0, 100.0, 10.0, 10.0]) == 0.0


def test_iou_of_a_zero_area_box_does_not_divide_by_zero():
    assert corpus.iou([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 10.0, 10.0]) == 0.0


def test_containment_is_not_overlap():
    """A small box wholly inside a large one is fully contained and barely overlapping."""
    small = [4.0, 4.0, 2.0, 2.0]
    large = [0.0, 0.0, 100.0, 100.0]
    assert corpus.contains_fraction(small, large) == pytest.approx(1.0)
    assert corpus.iou(small, large) < 0.01


def test_half_a_box_hanging_outside_is_half_contained():
    assert corpus.contains_fraction([-5.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) == (
        pytest.approx(0.5)
    )


def test_the_detection_cache_covers_every_held_out_page():
    boxes = corpus.detections()
    assert set(boxes) == {p.name for p in corpus.pages()}


def test_detections_come_back_sorted_by_score_so_a_threshold_is_a_prefix():
    page = corpus.pages()[0]
    scores = [row["score"] for row in corpus.detections(page)]
    assert scores == sorted(scores, reverse=True)


def test_a_detection_is_width_height_not_corners():
    """`[x, y, w, h]` throughout Phase 10; an xyxy box smuggled in here would be silently wrong."""
    for row in corpus.detections(corpus.pages()[0]):
        assert row["bbox"][2] > 0 and row["bbox"][3] > 0

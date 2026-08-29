"""Phase 3.2.4 - line segments."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess.primitives import segments as seg


def box_page() -> tuple[np.ndarray, np.ndarray]:
    gray = np.full((400, 600), 240, np.uint8)
    cv2.rectangle(gray, (80, 80), (420, 300), 30, 3)
    cv2.line(gray, (440, 190), (560, 190), 30, 3)
    return gray, gray < 128


@pytest.mark.parametrize("method", ["lsd", "hough"])
def test_both_detectors_find_the_box_edges(method):
    gray, mask = box_page()
    found = seg.detect(gray if method == "lsd" else mask, method=method)
    assert len(found) >= 4
    assert seg.axis_aligned_fraction(found) > 0.9


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        seg.detect(box_page()[0], method="radon")


def test_segments_are_ordered_deterministically():
    gray, _ = box_page()
    assert [s.to_dict() for s in seg.detect(gray)] == [s.to_dict() for s in seg.detect(gray)]


def test_endpoint_order_does_not_change_a_segment():
    a = seg.Segment(*seg._orient(10, 20, 90, 20))
    b = seg.Segment(*seg._orient(90, 20, 10, 20))
    assert a == b


@pytest.mark.parametrize(
    ("x2", "y2", "expected"), [(100, 0, 0.0), (0, 100, 90.0), (100, 100, 45.0), (-100, 100, 135.0)]
)
def test_angle_is_folded_to_a_half_turn(x2, y2, expected):
    assert seg.Segment(0, 0, x2, y2).angle == pytest.approx(expected, abs=0.1)


def test_length_and_midpoint():
    s = seg.Segment(0, 0, 30, 40)
    assert s.length == pytest.approx(50.0)
    assert s.midpoint == (15.0, 20.0)


def test_angle_histogram_is_normalised_and_peaks_where_the_lines_are():
    gray, _ = box_page()
    histogram = seg.angle_histogram(seg.detect(gray))
    assert histogram.sum() == pytest.approx(1.0)
    # Bins are centred on the axes, so a rectangle's length lands in exactly two of them.
    assert histogram[0] + histogram[9] > 0.95, histogram


def test_axis_bins_are_centred_not_split():
    """The bug this fixes: with bins starting at 0, horizontal and vertical each split in two."""
    horizontal = seg.angle_histogram([seg.Segment(0, 0, 100, 0)])
    vertical = seg.angle_histogram([seg.Segment(0, 0, 0, 100)])
    assert horizontal[0] == pytest.approx(1.0)
    assert vertical[9] == pytest.approx(1.0)


def test_dominant_direction_wraps_around_the_fold():
    """A horizontal line that wobbles to 179.5 degrees must not read as a different direction."""
    just_under = [seg.Segment(0, 0, 100, 1)]
    just_over = [seg.Segment(0, 0, 100, -1)]
    assert seg.dominant_direction(just_under)[1] == pytest.approx(1.0)
    assert seg.dominant_direction(just_over)[1] == pytest.approx(1.0)


def test_diagonals_are_not_counted_as_axis_aligned():
    diagonal = [seg.Segment(0, 0, 100, 100)]
    assert seg.axis_aligned_fraction(diagonal) == 0.0


def test_empty_input_is_safe():
    assert seg.summarise([]) == {"segments": 0}
    assert seg.angle_histogram([]).sum() == 0
    assert seg.axis_aligned_fraction([]) == 0.0


def test_short_segments_are_filtered():
    gray = np.full((400, 400), 240, np.uint8)
    cv2.line(gray, (200, 200), (203, 203), 30, 2)
    assert seg.detect(gray, min_length_frac=0.2) == []

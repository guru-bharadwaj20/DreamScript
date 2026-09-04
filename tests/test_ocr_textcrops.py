"""Phase 9.3.1 - the crop geometry, the derivation, and the control that scores it."""

from __future__ import annotations

import numpy as np
import pytest

from src.ocr import textcrops


def test_iou_of_a_box_with_itself_is_one():
    box = (10.0, 20.0, 30.0, 40.0)
    assert textcrops.iou(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert textcrops.iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_of_a_degenerate_box_is_zero_rather_than_an_error():
    assert textcrops.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_the_edge_label_box_lands_on_the_polyline_midpoint():
    """A straight 100px line has its arc-length midpoint at x = 50, and the box centres there."""
    box = textcrops.edge_label_box([[0, 50], [100, 50]], unit=10.0, page_size=(200, 200))
    assert box is not None
    x, y, w, h = box
    assert x + w / 2 == pytest.approx(50.0, abs=1.0)
    assert y + h / 2 == pytest.approx(50.0, abs=1.0)


def test_the_midpoint_is_by_arc_length_and_not_by_vertex_count():
    """Three vertices with a long first segment: the midpoint must fall inside that segment.

    A naive implementation takes the middle *vertex*, which for this polyline sits at x = 90 -
    nowhere near the halfway point of the 100 + 10 = 110px path.
    """
    box = textcrops.edge_label_box([[0, 0], [100, 0], [100, 10]], unit=5.0, page_size=(200, 200))
    assert box is not None
    x, _, w, _ = box
    assert x + w / 2 == pytest.approx(55.0, abs=1.5)


def test_a_zero_length_polyline_has_no_label_box():
    assert textcrops.edge_label_box([[7, 7], [7, 7]], 10.0, (100, 100)) is None
    assert textcrops.edge_label_box([[7, 7]], 10.0, (100, 100)) is None


def test_the_label_box_is_clipped_into_the_page():
    box = textcrops.edge_label_box([[0, 0], [4, 0]], unit=50.0, page_size=(60, 60))
    assert box is not None
    x, y, w, h = box
    assert x >= 0 and y >= 0
    assert x + w <= 60 and y + h <= 60


def test_the_label_box_scales_with_the_glyph_unit():
    small = textcrops.edge_label_box([[0, 500], [1000, 500]], 10.0, (2000, 1000))
    large = textcrops.edge_label_box([[0, 500], [1000, 500]], 40.0, (2000, 1000))
    assert large[2] == pytest.approx(4 * small[2])


def test_glyph_height_comes_from_the_page_and_not_from_a_constant():
    tall = {"nodes": [{"bbox": [0, 0, 10, 600]}], "meta": {"image_size": (1000, 1000)}}
    short = {"nodes": [{"bbox": [0, 0, 10, 60]}], "meta": {"image_size": (1000, 1000)}}
    assert textcrops.glyph_height(tall) == pytest.approx(100.0)
    assert textcrops.glyph_height(short) == pytest.approx(10.0)


def test_glyph_height_falls_back_to_the_page_when_no_node_has_a_box():
    diagram = {"nodes": [{"bbox": None}], "meta": {"image_size": (1200, 600)}}
    assert textcrops.glyph_height(diagram) == pytest.approx(10.0)


def test_a_crop_is_height_normalised_and_width_capped():
    image = np.full((400, 4000), 255, dtype=np.uint8)
    patch = textcrops.cut(image, (0, 0, 4000, 400))
    assert patch.shape[0] == textcrops.HEIGHT
    assert patch.shape[1] == textcrops.MAX_WIDTH


def test_a_box_smaller_than_the_floor_yields_no_crop():
    image = np.full((100, 100), 255, dtype=np.uint8)
    assert textcrops.cut(image, (0, 0, 5, 5)) is None


def test_the_inset_removes_the_border_from_both_sides():
    image = np.zeros((200, 200), dtype=np.uint8)
    image[20:180, 20:180] = 255  # a white interior inside a black frame
    patch = textcrops.cut(image, (0, 0, 200, 200), inset=0.15)
    assert patch.min() == 255  # the frame is gone


def test_blank_paper_has_no_ink():
    assert not textcrops.has_ink(np.full((64, 200), 255, dtype=np.uint8))


def test_a_written_crop_has_ink():
    patch = np.full((64, 200), 250, dtype=np.uint8)
    patch[20:44, 20:120] = 10
    assert textcrops.has_ink(patch)


def test_a_box_containing_only_the_connector_fails_the_off_line_test():
    """The control that makes 9.3.1's edge-label rate mean something.

    A horizontal rule drawn through the middle of the box is exactly what an edge-label crop
    contains when the writing was somewhere else, and the plain ink test passes it.
    """
    image = np.full((300, 300), 255, dtype=np.uint8)
    image[148:152, :] = 0
    box = (50, 120, 200, 60)
    assert textcrops.has_ink(image[120:180, 50:250])
    assert not textcrops.off_line_ink(image, box, [[50, 150], [250, 150]], unit=12.0)


def test_writing_beside_the_connector_passes_the_off_line_test():
    image = np.full((300, 300), 255, dtype=np.uint8)
    image[148:152, :] = 0  # the connector
    image[125:140, 100:180] = 0  # a word written above it
    box = (50, 120, 200, 60)
    assert textcrops.off_line_ink(image, box, [[50, 150], [250, 150]], unit=12.0)

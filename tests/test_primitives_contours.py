"""Phase 3.2.2 - contours and containment."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocess.primitives import contours as ct


def nested_scene() -> np.ndarray:
    canvas = np.zeros((500, 500), np.uint8)
    cv2.rectangle(canvas, (40, 40), (460, 460), 255, 4)  # outer container
    cv2.rectangle(canvas, (100, 100), (240, 220), 255, 3)  # a box inside it
    cv2.circle(canvas, (350, 350), 55, 255, 3)  # a circle inside it
    return canvas > 0


def test_containment_depth_is_recovered():
    found = ct.extract(nested_scene())
    depths = {c.depth for c in found}
    assert 0 in depths and 1 in depths
    outer = ct.outer(found)
    assert len(outer) == 1
    assert outer[0].bbox[2] > 400


def test_children_are_linked_to_their_parent():
    found = ct.extract(nested_scene())
    # The container's direct child is its own inner stroke edge; the two shapes hang off that.
    container = max(found, key=lambda c: c.area)
    inner_edge = next(c for c in found if c.index in container.children)
    assert len(inner_edge.children) >= 2


def test_a_drawn_outline_produces_two_contours_and_pairing_removes_one():
    """`RETR_TREE` traces both edges of a pen stroke, so raw depth double-counts nesting."""
    found = ct.extract(nested_scene())
    drawn = ct.drawn_outlines(found)
    assert len(found) == 2 * len(drawn), "each drawn outline should contribute exactly two"
    assert len(drawn) == 3  # container, box, circle


def test_pairing_keeps_a_genuinely_nested_shape():
    """A double circle must survive: its inner ring is not the outer ring's stroke edge."""
    canvas = np.zeros((300, 300), np.uint8)
    cv2.circle(canvas, (150, 150), 90, 255, 3)
    cv2.circle(canvas, (150, 150), 60, 255, 3)
    drawn = ct.drawn_outlines(ct.extract(canvas > 0))
    assert len(drawn) == 2


def test_nested_pairs_are_reported():
    assert len(ct.nested_pairs(ct.extract(nested_scene()))) >= 2


def test_a_double_circle_is_two_nested_rings():
    """The accepting state of a state machine; the nesting is what distinguishes it."""
    canvas = np.zeros((300, 300), np.uint8)
    cv2.circle(canvas, (150, 150), 90, 255, 3)
    cv2.circle(canvas, (150, 150), 70, 255, 3)
    found = ct.extract(canvas > 0)
    assert max(c.depth for c in found) >= 1


def test_a_stroke_is_not_closed():
    canvas = np.zeros((200, 400), np.uint8)
    cv2.line(canvas, (20, 100), (380, 110), 255, 3)
    found = ct.extract(canvas > 0, min_area_frac=0.0)
    assert found
    assert not any(c.closed for c in found)


def test_a_box_is_closed():
    canvas = np.zeros((200, 200), np.uint8)
    cv2.rectangle(canvas, (40, 40), (160, 160), 255, 3)
    assert any(c.closed for c in ct.extract(canvas > 0))


def test_contours_are_sorted_by_area():
    areas = [c.area for c in ct.extract(nested_scene())]
    assert areas == sorted(areas, reverse=True)


def test_empty_mask_gives_nothing():
    assert ct.extract(np.zeros((60, 60), bool)) == []
    assert ct.summarise([]) == {"contours": 0}


def test_contour_serialises_without_the_point_array():
    payload = ct.extract(nested_scene())[0].to_dict()
    assert "points" not in payload
    assert payload["n_points"] > 3

"""Phase 3.2.1 - connected components."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocess.primitives import components as comp


def scene() -> np.ndarray:
    canvas = np.zeros((400, 600), np.uint8)
    cv2.rectangle(canvas, (40, 40), (200, 160), 255, 3)  # a box: one hole
    cv2.circle(canvas, (400, 100), 60, 255, 3)  # a circle: one hole
    cv2.line(canvas, (60, 300), (240, 320), 255, 3)  # a stroke: no hole
    return canvas > 0


def test_finds_every_piece():
    found = comp.extract(scene())
    assert len(found) == 3


def test_components_are_returned_largest_first():
    areas = [c.area for c in comp.extract(scene())]
    assert areas == sorted(areas, reverse=True)


def test_closed_outlines_have_a_hole_and_strokes_do_not():
    found = {tuple(c.bbox[:2]): c for c in comp.extract(scene())}
    box = next(c for c in found.values() if c.bbox[0] < 60 and c.bbox[1] < 60)
    stroke = next(c for c in found.values() if c.bbox[1] > 250)
    assert box.holes == 1
    assert stroke.holes == 0


def test_statistics_are_sane():
    for component in comp.extract(scene()):
        assert component.area > 0
        assert 0 < component.extent <= 1
        assert 0 < component.solidity <= 1.01
        assert component.perimeter > 0
        assert component.aspect > 0


def test_a_circle_is_more_circular_than_a_box():
    found = comp.extract(scene())
    circle = next(c for c in found if c.bbox[0] > 300)
    box = next(c for c in found if c.bbox[0] < 60 and c.bbox[1] < 60)
    assert circle.circularity > box.circularity


def test_border_contact_is_reported():
    canvas = np.zeros((100, 100), np.uint8)
    cv2.rectangle(canvas, (0, 0), (50, 50), 255, 3)
    assert comp.extract(canvas > 0)[0].touches_border


def test_speckle_is_filtered_by_area():
    mask = scene()
    mask[380, 590] = True  # one stray pixel
    assert len(comp.extract(mask)) == 3


def test_empty_mask_gives_nothing():
    assert comp.extract(np.zeros((50, 50), bool)) == []
    assert comp.summarise([]) == {"components": 0}


def test_component_serialises():
    first = comp.extract(scene())[0].to_dict()
    assert set(first) >= {"area", "bbox", "holes", "solidity", "circularity"}

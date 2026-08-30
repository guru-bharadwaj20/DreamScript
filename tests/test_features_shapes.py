"""Phase 4.1.3 - shape mix."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx
from src.features import shapes as sh
from tests.test_features_context import shape_only


def region_of(kind: str):
    return ctx.build_regions(shape_only(kind))[0]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("box", "rectangle"), ("diamond", "diamond"), ("circle", "circle"), ("ellipse", "ellipse")],
)
def test_the_rule_is_exact_on_clean_renders(kind, expected):
    """It fails on photographs - see the docstring - but the geometry itself is right."""
    assert sh.classify(region_of(kind)) == expected


def rotated_box(degrees: float) -> ctx.Region:
    image = np.zeros((600, 600), np.uint8)
    corners = cv2.boxPoints(((300, 300), (300, 200), float(degrees))).astype(np.int32)
    cv2.polylines(image, [corners], True, 255, 3)
    return ctx.build_regions(image > 0)[0]


@pytest.mark.parametrize("degrees", [0, 5, 10])
def test_a_box_survives_the_tilt_a_rectified_page_can_have(degrees):
    """3.1.7 leaves a median 0.28 degrees of skew; the rule holds to about fifteen."""
    assert sh.classify(rotated_box(degrees)) == "rectangle"


def test_a_box_turned_far_enough_becomes_a_diamond():
    """Not a bug: a diamond *is* a rotated square, so this is the definition being consistent."""
    assert sh.classify(rotated_box(45)) == "diamond"
    assert rotated_box(45).rect_fill > 0.95  # still plainly four-sided


def test_rect_fill_alone_cannot_tell_the_two_apart():
    """Pinned so nobody 'simplifies' the rule to the rotation-invariant measure alone."""
    assert region_of("box").rect_fill == pytest.approx(region_of("diamond").rect_fill, abs=0.02)
    assert region_of("box").extent > 0.9 > region_of("diamond").extent


def test_the_five_fractions_sum_to_one():
    image = np.zeros((600, 900), np.uint8)
    cv2.rectangle(image, (40, 40), (240, 200), 255, 3)
    cv2.circle(image, (450, 120), 80, 255, 3)
    cv2.polylines(image, [np.array([[700, 40], [860, 130], [700, 220], [560, 130]])], True, 255, 3)
    mix = sh.mix(ctx.from_masks(image > 0, page_id="mixed"))
    assert sum(mix.values()) == pytest.approx(1.0)
    assert mix["shape_frac_rectangle"] == pytest.approx(1 / 3)
    assert mix["shape_frac_circle"] == pytest.approx(1 / 3)
    assert mix["shape_frac_diamond"] == pytest.approx(1 / 3)


def test_a_page_with_no_shapes_reports_nan_not_zero():
    """0.0 would claim "no rectangles here"; there is nothing here to have a mix of."""
    mix = sh.mix(ctx.from_masks(np.zeros((300, 300), bool), page_id="blank"))
    assert set(mix) == set(sh.NAMES)
    assert all(np.isnan(v) for v in mix.values())


def test_every_kind_is_in_the_frozen_vocabulary():
    from src.ir.vocab import SHAPES

    assert set(sh.KINDS) <= set(SHAPES)


def test_labels_map_onto_the_five_kinds():
    assert set(sh.LABEL_MAP.values()) <= set(sh.KINDS)
    assert sh.LABEL_MAP["rounded-rect"] == "rectangle"


def test_an_unnamed_outline_is_freeform_not_a_guess():
    star = np.zeros((400, 400), np.uint8)
    points = []
    for i in range(10):
        radius = 160 if i % 2 == 0 else 60
        angle = np.pi * i / 5
        points.append([200 + radius * np.cos(angle), 200 + radius * np.sin(angle)])
    cv2.polylines(star, [np.array(points, np.int32)], True, 255, 3)
    assert sh.classify(ctx.build_regions(star > 0)[0]) == "freeform"

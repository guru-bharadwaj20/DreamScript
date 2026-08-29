"""Phase 3.3.2 - the degradation sweeps.

The curves themselves are measured by `python -m src.preprocess.robustness`; these tests pin
the degradations, the warping that keeps image and mask registered, and the plotting.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import robustness as rb


@pytest.fixture
def page():
    canvas = np.full((300, 300), 245, np.uint8)
    cv2.rectangle(canvas, (60, 90), (240, 210), 35, 3)
    return canvas


def test_blur_softens_and_zero_is_a_no_op(page):
    assert np.array_equal(rb.blur(page, 0), page)
    softened = rb.blur(page, 9)
    assert softened.std() < page.std()


def test_an_even_blur_radius_is_made_odd(page):
    """`GaussianBlur` rejects an even kernel; the sweep must not depend on the caller knowing."""
    assert rb.blur(page, 4).shape == page.shape


def test_light_only_darkens(page):
    lit = rb.light(page, 0.8)
    assert lit.max() <= page.max()
    assert lit[:, 0].mean() > lit[:, -1].mean(), "the shadow falls to the right"
    assert np.array_equal(rb.light(page, 0.0), page)


def test_light_never_clips_the_bright_side(page):
    """Clipped highlights are unrecoverable, so the sweep would stop measuring lighting."""
    for strength in (0.2, 0.5, 0.9):
        assert rb.light(page, strength).max() <= 245


def test_rotation_keeps_image_and_mask_registered(page):
    mask = page < 128
    turned = rb.rotate(page, 12.0)
    turned_mask = rb.rotate(mask.astype(np.uint8), 12.0, nearest=True) > 0
    # The ink in the warped image must sit where the warped mask says it does.
    overlap = (turned < 128) & turned_mask
    assert overlap.sum() > 0.8 * turned_mask.sum()


def test_mask_rotation_stays_binary(page):
    mask = (page < 128).astype(np.uint8)
    turned = rb.rotate(mask, 7.0, nearest=True)
    assert set(np.unique(turned)) <= {0, 1}


def test_zero_rotation_is_a_no_op(page):
    assert np.array_equal(rb.rotate(page, 0.0), page)


def test_figure_and_report_are_written(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "FIGURE", tmp_path / "p3_robustness.png")
    monkeypatch.setattr(rb, "REPORT", tmp_path / "robustness.md")
    result = {
        "items": 3,
        "deskew_pages": 2,
        "axes": {"blur": rb.BLUR_RADII, "rotation": rb.ROTATIONS, "lighting": rb.LIGHTING},
        "curves": {
            "blur": [0.99, 0.98, 0.9, 0.75, 0.6, 0.5],
            "rotation": [0.99, 0.96, 0.95, 0.95, 0.94, 0.94],
            "deskew_error": [0.2, 0.3, 0.4, 0.3, 0.2, 0.5],
            "lighting": [0.99, 0.99, 0.99, 0.98, 0.89],
            "lighting_raw": [0.99, 0.99, 0.99, 0.99, 0.99],
            "lighting_flatten": [0.99, 0.99, 0.99, 0.99, 0.99],
        },
    }
    figure = rb.figure(result)
    report = rb.report(result)
    assert figure.is_file() and figure.stat().st_size > 5000
    text = report.read_text(encoding="utf-8")
    assert "Blur" in text and "Rotation" in text and "Lighting" in text
    assert "flatten only" in text


def test_every_axis_has_at_least_five_levels():
    for axis in (rb.BLUR_RADII, rb.ROTATIONS, rb.LIGHTING):
        assert len(axis) >= 5, "a curve needs points"

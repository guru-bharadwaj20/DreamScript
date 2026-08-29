"""Phase 3.1.6 - denoising."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import denoise as dn


def speckled(size=400, strokes=True) -> np.ndarray:
    mask = np.zeros((size, size), bool)
    if strokes:
        canvas = np.zeros((size, size), np.uint8)
        cv2.rectangle(canvas, (60, 60), (240, 200), 255, 3)
        cv2.line(canvas, (60, 300), (340, 320), 255, 3)
        mask |= canvas > 0
    rng = np.random.default_rng(0)
    for _ in range(300):
        y, x = rng.integers(0, size - 2, 2)
        mask[y : y + 2, x : x + 2] = True
    return mask


def test_speckle_is_removed_and_strokes_survive():
    mask = speckled()
    before = dn.count_components(mask)
    cleaned = dn.denoise(mask)
    after = dn.count_components(cleaned)
    assert after < before / 4
    # The rectangle and the line must still be there. Check regions, not exact pixels: the
    # line slopes, so the row it occupies at a given column is not the row it starts on.
    assert cleaned[58:64, 100:200].any(), "the rectangle's top edge was removed"
    assert cleaned[295:325, 80:340].sum() > 200, "the line was removed"


def test_removal_reports_how_many_it_dropped():
    _, dropped = dn.remove_small_components(speckled())
    assert dropped > 100


def test_nothing_to_remove_is_handled():
    empty = np.zeros((100, 100), bool)
    cleaned, dropped = dn.remove_small_components(empty)
    assert not cleaned.any() and dropped == 0


def test_median_kernel_is_forced_odd_and_one_is_a_no_op():
    gray = np.full((50, 50), 200, np.uint8)
    gray[25, 25] = 0
    assert np.array_equal(dn.median(gray, 1), gray)
    assert dn.median(gray, 4)[25, 25] == 200  # the outlier is gone


def test_component_count_is_not_a_restatement_of_the_threshold():
    """`count_speckle` before/after removal is always 100% by construction; this must not be."""
    mask = speckled()
    assert dn.count_speckle(dn.denoise(mask)) == 0  # tautological, as documented
    assert dn.count_components(dn.denoise(mask)) > 0  # meaningful: strokes survive


def test_a_filter_that_does_nothing_would_be_visible():
    mask = speckled()
    unchanged = dn.denoise(mask, min_area_frac=0.0)
    assert dn.count_components(unchanged) == dn.count_components(mask)


def test_real_photo_evaluation_meets_the_bar():
    result = dn.speckle_on_real_photos(12)
    if not result["photos"]:
        pytest.skip("hdBPMN not present")
    assert result["components_removed"] >= 0.80
    assert result["ink_pixels_kept"] >= 0.90

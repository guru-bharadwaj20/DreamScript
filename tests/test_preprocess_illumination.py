"""Phase 3.1.4 - illumination correction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import illumination as illum


def page_with_ramp(strength=90) -> tuple[np.ndarray, np.ndarray]:
    """A page with strokes, plus the same page under a left-to-right lighting ramp."""
    clean = np.full((400, 500), 235, np.uint8)
    cv2.rectangle(clean, (80, 90), (240, 190), 30, 4)
    cv2.circle(clean, (380, 280), 50, 30, 4)
    cv2.line(clean, (100, 300), (300, 340), 30, 3)
    ramp = np.linspace(-strength, strength, clean.shape[1], dtype=np.float32)
    lit = np.clip(clean.astype(np.float32) + ramp[None, :], 0, 255).astype(np.uint8)
    return clean, lit


def test_ramp_is_flattened():
    clean, lit = page_with_ramp()
    assert illum.quadrant_spread(lit) > illum.quadrant_spread(clean) + 20
    corrected = illum.correct(lit)
    assert illum.quadrant_spread(corrected) < illum.quadrant_spread(lit) / 2


def test_strokes_survive_the_correction():
    """Flattening must remove the paper, not the ink."""
    _, lit = page_with_ramp()
    corrected = illum.correct(lit)
    # The stroke pixels must still be much darker than the paper around them.
    ink = corrected[92, 80:240].min()
    paper = np.median(corrected[350:390, 400:480])
    assert paper - ink > 60


def test_background_estimate_contains_no_ink():
    clean, _ = page_with_ramp()
    paper = illum.background(clean)
    # The closing should have filled the drawn rectangle's outline back to paper level.
    assert paper[92, 160] > 200


def test_correct_accepts_colour_and_returns_grayscale():
    _, lit = page_with_ramp()
    colour = cv2.cvtColor(lit, cv2.COLOR_GRAY2BGR)
    assert illum.correct(colour).ndim == 2


def test_clahe_is_contrast_limited():
    """Unlimited equalisation turns paper texture into strokes; the clip limit is the guard."""
    noisy = np.clip(
        np.full((300, 300), 230, np.float32) + np.random.default_rng(0).normal(0, 3, (300, 300)),
        0,
        255,
    ).astype(np.uint8)
    mild = illum.clahe(noisy, clip_limit=1.0)
    wild = illum.clahe(noisy, clip_limit=40.0)
    assert mild.std() < wild.std()


def test_corpus_shadows_are_measurably_reduced():
    result = illum.evaluate("shadow")
    if not result["images"]:
        pytest.skip("chaos corpus not present")
    assert result["spread_after_median"] < result["spread_before_median"]
    assert result["still_above_shadow_threshold"] < result["were_above_shadow_threshold"] / 2

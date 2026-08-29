"""Phase 3.1.5 - binarization."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import binarize as bz
from src.preprocess.evalset import build, f1


def ramped_page() -> np.ndarray:
    page = np.full((300, 400), 230, np.uint8)
    cv2.rectangle(page, (60, 60), (200, 160), 40, 3)
    ramp = np.linspace(-70, 70, 400, dtype=np.float32)
    return np.clip(page.astype(np.float32) + ramp[None, :], 0, 255).astype(np.uint8)


@pytest.mark.parametrize("method", sorted(bz.METHODS))
def test_every_method_returns_a_boolean_mask(method):
    mask = bz.binarize(ramped_page(), method)
    assert mask.dtype == bool
    assert mask.shape == (300, 400)
    assert mask.any(), f"{method} found no ink at all"


def test_unknown_method_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="unknown binarization method"):
        bz.binarize(ramped_page(), "niblack")


def test_sauvola_does_not_hallucinate_ink_on_blank_dark_paper():
    """The property Sauvola is chosen for: low variance means paper, however dark."""
    blank = np.full((200, 200), 90, np.uint8)
    assert bz.sauvola(blank).mean() < 0.02


def test_otsu_does_hallucinate_on_blank_paper():
    """The complementary weakness, pinned so the trade-off stays visible."""
    blank = np.full((200, 200), 90, np.uint8)
    rng = np.random.default_rng(0)
    noisy = np.clip(blank + rng.normal(0, 4, blank.shape), 0, 255).astype(np.uint8)
    assert bz.otsu(noisy).mean() > 0.2


def test_f1_of_a_perfect_prediction_is_one():
    truth = np.zeros((50, 50), bool)
    truth[10:20, 10:40] = True
    scores = f1(truth, truth)
    assert scores["f1"] == 1.0 and scores["iou"] == 1.0


def test_f1_of_an_empty_prediction_is_zero():
    truth = np.zeros((50, 50), bool)
    truth[10:20, 10:40] = True
    assert f1(np.zeros_like(truth), truth)["f1"] == 0.0


def test_evaluation_set_masks_are_exact_not_thresholded():
    """The mask is drawn from the same polylines as the render, so it cannot drift from it."""
    items = build(4)
    if not items:
        pytest.skip("FA database not present")
    for item in items:
        assert item.mask.dtype == bool
        assert 0.001 < item.mask.mean() < 0.25


def test_measured_ranking_holds():
    results = bz.compare(20)
    if not results["items"]:
        pytest.skip("FA database not present")
    methods = results["methods"]
    assert methods["otsu"]["f1"] >= methods["adaptive"]["f1"]
    assert all(0.0 <= m["f1"] <= 1.0 for m in methods.values())


def test_comparison_is_deterministic_under_parallelism():
    if not bz.compare(10)["items"]:
        pytest.skip("FA database not present")
    assert bz.compare(20) == bz.compare(20)

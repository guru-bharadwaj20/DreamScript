"""Phase 9.3.8 - the two damage gestures, and that the control matches the positive on ink."""

from __future__ import annotations

import numpy as np
import pytest

from src.ocr import strikeout


def written_crop() -> np.ndarray:
    """Paper with a band of writing across the middle third."""
    image = np.full((48, 192), 245, dtype=np.uint8)
    image[18:30, 20:170] = 30
    return image


def test_blank_paper_and_written_paper_differ_in_ink_share():
    blank = np.full((48, 192), 245, dtype=np.uint8)
    assert strikeout.ink_share(written_crop()) > strikeout.ink_share(blank)


def test_a_strike_adds_ink():
    rng = np.random.default_rng(0)
    base = written_crop()
    marked, _ = strikeout.strike(base, rng)
    assert strikeout.ink_share(marked) > strikeout.ink_share(base)


def test_the_requested_gesture_is_the_one_drawn():
    """The held-out-gesture control needs to be able to ask for a specific gesture."""
    for style in (0, 1, 2):
        _, drawn = strikeout.strike(written_crop(), np.random.default_rng(0), style=style)
        assert drawn == style


def test_the_pen_takes_its_ink_from_the_crop_rather_than_a_constant():
    """A constant ink value is a watermark the classifier can key on instead of the geometry."""
    dark = np.full((48, 192), 245, dtype=np.uint8)
    dark[10:20] = 5
    light = np.full((48, 192), 245, dtype=np.uint8)
    light[10:20] = 80
    dark_value, _ = strikeout.pen(dark, np.random.default_rng(0))
    light_value, _ = strikeout.pen(light, np.random.default_rng(0))
    assert dark_value < light_value


def test_a_strike_crosses_the_writing_band():
    """The positive class is defined by where the stroke goes, so the geometry is pinned."""
    rng = np.random.default_rng(1)
    base = written_crop()
    marked, _ = strikeout.strike(base, rng)
    top_changed = (marked[:14] != base[:14]).any()
    bottom_changed = (marked[34:] != base[34:]).any()
    assert top_changed and bottom_changed  # the stroke spans the crop vertically


def test_the_control_leaves_the_writing_band_untouched():
    """`inked` must not cross the text, or it is a second positive class wearing a negative label."""
    rng = np.random.default_rng(2)
    base = written_crop()
    marked = strikeout.inked(base, rng, target=0.02)
    assert (marked[20:28, 60:140] == base[20:28, 60:140]).all()


def test_the_control_adds_at_least_as_much_ink_as_it_was_asked_for():
    rng = np.random.default_rng(3)
    base = written_crop()
    target = 0.02
    marked = strikeout.inked(base, rng, target)
    assert strikeout.ink_share(marked) - strikeout.ink_share(base) >= target * 0.5


def test_the_strike_is_deterministic_under_a_seed():
    a, style_a = strikeout.strike(written_crop(), np.random.default_rng(7))
    b, style_b = strikeout.strike(written_crop(), np.random.default_rng(7))
    assert (a == b).all() and style_a == style_b


def test_precision_and_recall_against_hand_counted_values():
    truth = np.array([1, 1, 1, 0, 0])
    predicted = np.array([1, 1, 0, 1, 0])
    result = strikeout.precision_recall(truth, predicted)
    assert result["tp"] == 2 and result["fp"] == 1 and result["fn"] == 1
    assert result["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_precision_is_nan_when_nothing_is_predicted_positive():
    result = strikeout.precision_recall(np.array([1, 0]), np.array([0, 0]))
    assert result["precision"] != result["precision"]


def test_a_perfect_prediction_scores_one():
    truth = np.array([1, 0, 1, 0])
    result = strikeout.precision_recall(truth, truth)
    assert result["precision"] == 1.0 and result["recall"] == 1.0 and result["f1"] == 1.0


def test_the_control_class_is_scored_as_a_negative():
    assert strikeout.POSITIVE == "struck"
    assert set(strikeout.CLASSES) == {"clean", "struck", "inked"}

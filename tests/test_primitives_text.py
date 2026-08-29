"""Phase 3.2.7 - text region proposal.

Each filter is tested on a case where the answer is obvious, and the whole proposal is tested
on a synthetic page whose two layers are known exactly.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocess.primitives import text as tx


def word(text: str = "hello", scale: float = 1.0, size: int = 400) -> np.ndarray:
    canvas = np.zeros((size, size), np.uint8)
    cv2.putText(canvas, text, (40, size // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 2)
    return canvas > 0


def box(size: int = 400) -> np.ndarray:
    canvas = np.zeros((size, size), np.uint8)
    cv2.rectangle(canvas, (40, 120), (360, 280), 255, 3)
    return canvas > 0


def test_stroke_width_of_a_known_line():
    """`cv2.line` with thickness 7 paints nine rows, and the transform reports width + 1."""
    canvas = np.zeros((100, 100), np.uint8)
    cv2.line(canvas, (10, 50), (90, 50), 255, 7)
    painted = int(np.count_nonzero(canvas[:, 50]))
    width, variation = tx.stroke_width_stats(canvas > 0, (0, 0, 100, 100))
    assert width == painted + 1
    assert variation < 0.2, "a line of constant width should not vary"


def test_a_word_is_proposed():
    proposed = tx.propose(word())
    assert proposed, "no text proposed for a written word"


def test_a_drawn_box_is_not_proposed():
    assert tx.propose(box()) == []


def test_a_box_with_a_word_inside_keeps_only_the_word():
    canvas = box().astype(np.uint8) * 255
    cv2.putText(canvas, "task", (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    mask = canvas > 0
    proposal = tx.refine(mask, tx.propose(mask))
    assert proposal.any()
    # The proposal must sit inside the box's interior, not on its outline.
    ys, xs = np.nonzero(proposal)
    assert xs.min() > 45 and xs.max() < 355
    assert ys.min() > 125 and ys.max() < 275


def test_a_straight_stroke_is_rejected_however_letter_sized():
    """A fragment of a drawn line is the size and stroke width of a character."""
    canvas = np.zeros((200, 200), np.uint8)
    cv2.line(canvas, (60, 100), (140, 100), 255, 3)
    assert tx.propose(canvas > 0) == []


def test_a_lone_candidate_is_not_writing():
    """One letter with nothing near it is discarded: text comes in words."""
    single = word("l", scale=1.0)
    assert tx.propose(single) == []


def test_grouping_joins_a_line_of_letters():
    boxes = [(10 + 20 * i, 50, 14, 20) for i in range(5)]
    groups = tx.group_into_lines(boxes)
    assert len(groups) == 1 and len(groups[0]) == 5


def test_grouping_separates_two_distant_words():
    boxes = [(10, 50, 14, 20), (30, 50, 14, 20), (900, 400, 14, 20), (920, 400, 14, 20)]
    groups = sorted((len(g) for g in tx.group_into_lines(boxes)), reverse=True)
    assert groups == [2, 2]


def test_refine_drops_a_connector_crossing_a_word():
    """A line passing behind a word is inside the word's box but is not part of it."""
    canvas = np.zeros((400, 800), np.uint8)
    cv2.putText(canvas, "label", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    cv2.line(canvas, (0, 195), (799, 195), 255, 3)
    mask = canvas > 0
    proposal = tx.refine(mask, tx.propose(mask))
    line_only = np.zeros_like(canvas)
    cv2.line(line_only, (0, 195), (799, 195), 255, 3)
    overlap = proposal & (line_only > 0)
    assert overlap.sum() < 0.15 * (line_only > 0).sum()


def test_synthetic_page_layers_are_disjoint_by_construction():
    page = tx.synthetic_page(3)
    assert page["shapes"].any() and page["text"].any()
    assert page["ink"].sum() == (page["shapes"] | page["text"]).sum()


def test_separation_on_a_synthetic_page():
    page = tx.synthetic_page(1)
    proposal = tx.refine(page["ink"], tx.propose(page["ink"], page["gray"]))
    text_score = tx.score(proposal, page["text"])
    shape_score = tx.score(page["ink"] & ~proposal, page["shapes"])
    assert text_score["recall"] >= 0.70
    assert text_score["precision"] >= 0.90
    assert shape_score["recall"] >= 0.95

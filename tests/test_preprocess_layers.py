"""Phase 3.2.8 - the shape/text split, and the partition property that makes it a split."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocess import layers as ly
from src.preprocess.primitives import text as tx


def page() -> tuple[np.ndarray, np.ndarray]:
    built = tx.synthetic_page(5)
    return built["gray"], built["ink"]


def test_the_split_is_a_partition():
    gray, mask = page()
    split = ly.separate(mask, gray)
    assert all(ly.check_partition(split, mask).values())


def test_no_pixel_lands_in_both_layers():
    gray, mask = page()
    split = ly.separate(mask, gray)
    assert not (split.text & split.shape).any()


def test_the_text_layer_finds_the_words():
    built = tx.synthetic_page(5)
    split = ly.separate(built["ink"], built["gray"])
    caught = (split.text & built["text"]).sum() / built["text"].sum()
    assert caught >= 0.70, f"text layer caught only {caught:.0%} of the writing"


def test_the_shape_layer_keeps_the_drawing():
    built = tx.synthetic_page(5)
    split = ly.separate(built["ink"], built["gray"])
    kept = (split.shape & built["shapes"]).sum() / built["shapes"].sum()
    assert kept >= 0.95, f"shape layer kept only {kept:.0%} of the drawing"


def test_both_files_are_written(tmp_path):
    gray, mask = page()
    split = ly.separate(mask, gray)
    written = split.write("sample", tmp_path)
    assert set(written) == {"text", "shape"}
    for path in written.values():
        assert path.is_file()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert image.shape == mask.shape
        # Ink is black on white, the same polarity as the photograph.
        assert image.min() == 0 and image.max() == 255


def test_an_empty_page_splits_into_two_empty_layers():
    mask = np.zeros((200, 200), bool)
    split = ly.separate(mask)
    assert not split.text.any() and not split.shape.any()
    assert all(ly.check_partition(split, mask).values())


def test_a_page_with_no_writing_keeps_everything_as_shape():
    canvas = np.zeros((400, 400), np.uint8)
    cv2.rectangle(canvas, (40, 40), (360, 360), 255, 3)
    mask = canvas > 0
    split = ly.separate(mask)
    assert not split.text.any()
    assert np.array_equal(split.shape, mask)


def test_stats_report_the_two_populations():
    gray, mask = page()
    stats = ly.separate(mask, gray).stats()
    assert stats["ink_pixels"] > 0
    assert 0.0 < stats["text_pixel_share"] < 1.0
    assert stats["text_components"] > stats["shape_components"]

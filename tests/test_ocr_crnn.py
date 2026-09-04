"""Phase 9.3.2 - the CTC alphabet, the collapse rule, and the aspect-preserving fit."""

from __future__ import annotations

import numpy as np
import pytest

from src.ocr import crnn


def test_the_alphabet_is_the_union_of_its_sources_and_is_sorted():
    chars = crnn.alphabet(["abc"], ["cde"])
    assert chars == "abcde"


def test_the_alphabet_is_built_from_normalised_text():
    """Uppercase must not become its own symbol - 9.3.6 lowercases before scoring."""
    assert crnn.alphabet(["ABC"]) == "abc"


def test_encoding_skips_characters_the_alphabet_does_not_contain():
    table = {"a": 1, "b": 2}
    assert crnn.encode("abz", table) == [1, 2]


def test_index_zero_is_reserved_for_blank():
    chars = crnn.alphabet(["ab"])
    table = {c: i + 1 for i, c in enumerate(chars)}
    assert min(table.values()) == 1
    assert crnn.BLANK == 0


def _columns(sequence, classes):
    """A one-hot posterior with `sequence` as the argmax of each column."""
    logits = np.full((len(sequence), classes), -20.0)
    for i, index in enumerate(sequence):
        logits[i, index] = 0.0
    return logits


def test_greedy_decode_collapses_repeats():
    # chars "ab": a=1, b=2. Columns a a b -> "ab"
    assert crnn.greedy_decode(_columns([1, 1, 2], 3), "ab") == "ab"


def test_a_blank_between_repeats_keeps_both_characters():
    """The property CTC exists for: `a-a` is `aa`, `aa` is `a`."""
    assert crnn.greedy_decode(_columns([1, 0, 1], 3), "ab") == "aa"
    assert crnn.greedy_decode(_columns([1, 1], 3), "ab") == "a"


def test_an_all_blank_posterior_decodes_to_the_empty_string():
    assert crnn.greedy_decode(_columns([0, 0, 0], 3), "ab") == ""


def test_fit_normalises_height_and_pads_rather_than_stretching():
    image = np.full((32, 64), 255, dtype=np.uint8)
    out = crnn.fit(image)
    assert out.shape == (crnn.HEIGHT, crnn.WIDTH)
    # 32x64 scales to 64x128; everything past column 128 must be paper, not stretched glyph.
    assert (out[:, 200:] == 255).all()


def test_fit_shrinks_a_crop_wider_than_the_width_budget():
    image = np.full((64, 4000), 128, dtype=np.uint8)
    assert crnn.fit(image).shape == (crnn.HEIGHT, crnn.WIDTH)


def test_fit_preserves_aspect_so_two_different_labels_stay_different_sizes():
    narrow = crnn.fit(np.zeros((64, 100), dtype=np.uint8))
    wide = crnn.fit(np.zeros((64, 300), dtype=np.uint8))
    assert int((narrow < 255).sum(axis=0).astype(bool).sum()) == 100
    assert int((wide < 255).sum(axis=0).astype(bool).sum()) == 300


@pytest.mark.parametrize("classes", [5, 30])
def test_the_model_emits_one_column_per_four_input_columns(classes):
    torch = pytest.importorskip("torch")

    model = crnn.build_model(classes)
    out = model(torch.zeros(2, 1, crnn.HEIGHT, crnn.WIDTH))
    assert out.shape == (2, crnn.WIDTH // crnn.DOWNSAMPLE, classes)


def test_there_are_more_output_columns_than_the_longest_label():
    """CTC cannot place more characters than it has columns; the corpus maximum is 90."""
    assert crnn.WIDTH // crnn.DOWNSAMPLE > 90


def _two_line_crop() -> np.ndarray:
    """A box with two bands of writing, which is what a BPMN node label actually looks like."""
    image = np.full((120, 200), 255, dtype=np.uint8)
    image[10:30, 20:180] = 0
    image[60:80, 20:120] = 0
    return image


def test_two_bands_of_writing_are_found_as_two_lines():
    assert len(crnn.text_lines(_two_line_crop())) == 2


def test_a_single_line_is_not_split():
    image = np.full((40, 200), 255, dtype=np.uint8)
    image[10:30, 20:180] = 0
    assert len(crnn.text_lines(image)) == 1


def test_a_small_gap_does_not_cut_a_line_in_half():
    """Descenders leave one-row holes; cutting on those would make three lines out of one."""
    image = np.full((60, 200), 255, dtype=np.uint8)
    image[10:30, 20:180] = 0
    image[21:23, 20:180] = 255  # a two-row hole inside the band
    assert len(crnn.text_lines(image)) == 1


def test_blank_paper_returns_itself_rather_than_nothing():
    blank = np.full((60, 200), 255, dtype=np.uint8)
    assert len(crnn.text_lines(blank)) == 1


def test_unwrapping_makes_a_stacked_crop_wider_and_shorter():
    """The point of the arm: the characters move onto the axis CTC actually reads."""
    crop = _two_line_crop()
    out = crnn.unwrap_lines(crop)
    assert out.shape[1] > crop.shape[1]
    assert out.shape[0] < crop.shape[0]


def test_unwrapping_a_single_line_crop_is_a_no_op():
    image = np.full((40, 200), 255, dtype=np.uint8)
    image[10:30, 20:180] = 0
    assert (crnn.unwrap_lines(image) == image).all()

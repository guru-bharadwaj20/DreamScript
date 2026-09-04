"""Phase 9.3.4 - the beam's collapse rule, the snap budget, and the vocabulary's OOV ceiling."""

from __future__ import annotations

import numpy as np
import pytest

from src.ocr import lexicon


def _columns(rows, classes):
    """A log-posterior whose columns put almost all mass on the given indices."""
    logits = np.full((len(rows), classes), -20.0)
    for i, index in enumerate(rows):
        logits[i, index] = 0.0
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_the_beam_agrees_with_greedy_on_an_unambiguous_posterior():
    from src.ocr.crnn import greedy_decode

    columns = _columns([1, 0, 2, 2], 4)  # a - b b  -> "ab"
    assert lexicon.prefix_beam(columns, "abc") == greedy_decode(columns, "abc") == "ab"


def test_the_beam_keeps_a_repeat_separated_by_a_blank():
    assert lexicon.prefix_beam(_columns([1, 0, 1], 3), "ab") == "aa"


def test_the_beam_collapses_an_unseparated_repeat():
    assert lexicon.prefix_beam(_columns([1, 1], 3), "ab") == "a"


def test_an_all_blank_posterior_beams_to_the_empty_string():
    assert lexicon.prefix_beam(_columns([0, 0, 0], 3), "ab") == ""


def test_the_beam_can_beat_greedy_when_mass_is_split_across_paths():
    """The reason a beam exists: the argmax path is not the most probable *label*.

    Column 0 slightly prefers blank, but the two non-blank paths through 'a' sum to more than
    the blank path, so the highest-probability collapsed string is 'a' and greedy misses it.
    """
    from src.ocr.crnn import greedy_decode

    probabilities = np.array([[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]])
    columns = np.log(probabilities)
    assert greedy_decode(columns, "ab") == ""
    assert lexicon.prefix_beam(columns, "ab", beam=8) == "a"


def test_the_vocabulary_holds_words_and_whole_strings():
    vocabulary = lexicon.build_vocabulary(["Send the order"])
    assert "send" in vocabulary and "order" in vocabulary
    assert "send the order" in vocabulary


def test_a_word_in_the_vocabulary_is_left_alone():
    vocabulary = lexicon.build_vocabulary(["submit"])
    assert lexicon.snap("submit", vocabulary) == "submit"


def test_a_near_miss_is_snapped_to_its_neighbour():
    vocabulary = lexicon.build_vocabulary(["submit"])
    assert lexicon.snap("subrnit", vocabulary) == "submit"


def test_a_word_far_from_everything_is_left_alone_rather_than_forced():
    """A hard trie constraint cannot do this; the budget is what makes the snap safe."""
    vocabulary = lexicon.build_vocabulary(["submit"])
    assert lexicon.snap("xylophone", vocabulary) == "xylophone"


def test_short_words_are_never_snapped():
    """`a`, `b`, `0`, `1` are the most frequent strings in the corpus and are all one edit apart."""
    vocabulary = lexicon.build_vocabulary(["a", "b", "q0", "q1"])
    assert lexicon.snap("c", vocabulary) == "c"
    assert lexicon.snap("q9", vocabulary) == "q9"


def test_the_snap_budget_scales_with_word_length():
    """Two wrong characters is most of a short word and very little of a long one.

    At `SNAP_RATIO` 0.34 a four-letter word gets a budget of 1 and a twelve-letter word gets 3,
    so the same distance-2 error is refused on the short word and accepted on the long one.
    """
    short = lexicon.build_vocabulary(["stop"])
    assert lexicon.snap("styp", short) == "stop"  # distance 1, inside the budget of 1
    assert lexicon.snap("styq", short) == "styq"  # distance 2, refused
    long_vocabulary = lexicon.build_vocabulary(["registration"])
    assert lexicon.snap("regisrrction", long_vocabulary) == "registration"  # distance 2


def test_ties_are_broken_by_frequency():
    # `cxaim` is exactly one edit from both `claim` and `cnaim`, so only frequency separates them.
    vocabulary = lexicon.build_vocabulary(["claim"] * 5 + ["cnaim"])
    assert lexicon.snap("cxaim", vocabulary) == "claim"


def test_oov_rate_is_the_share_of_test_words_the_vocabulary_lacks():
    vocabulary = lexicon.build_vocabulary(["send order"])
    assert lexicon.oov_rate(vocabulary, ["send order"]) == 0.0
    assert lexicon.oov_rate(vocabulary, ["send parcel"]) == pytest.approx(0.5)


def test_snapping_against_an_empty_vocabulary_is_a_no_op():
    from collections import Counter

    assert lexicon.snap("anything at all", Counter()) == "anything at all"

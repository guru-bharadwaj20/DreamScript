"""Phase 9.3.6 - the one evaluator, pinned against hand-computed values."""

from __future__ import annotations

import pytest

from src.ocr import metrics


def test_edit_distance_of_identical_strings_is_zero():
    assert metrics.edit_distance("submit", "submit") == 0


def test_edit_distance_against_the_textbook_pair():
    assert metrics.edit_distance("kitten", "sitting") == 3


def test_edit_distance_against_an_empty_string_is_the_length():
    assert metrics.edit_distance("", "abcd") == 4
    assert metrics.edit_distance("abcd", "") == 4


def test_edit_distance_is_symmetric():
    assert metrics.edit_distance("start", "strat") == metrics.edit_distance("strat", "start")


def test_edit_distance_works_on_token_lists_for_wer():
    assert metrics.edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_normalisation_lowercases_and_collapses_whitespace():
    assert metrics.normalise("  Send   The\nOrder ") == "send the order"


def test_a_perfect_prediction_scores_zero_cer():
    result = metrics.score(["start", "end"], ["start", "end"])
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0
    assert result["exact_match"] == 1.0


def test_cer_is_corpus_level_and_not_the_mean_of_per_crop_rates():
    """The definition this module commits to, on a pair chosen so the two disagree.

    `q0` -> `qq` is one error over two characters; a twenty-character label read perfectly is
    zero over twenty. Corpus CER is 1/22 = 0.0455. The mean of the per-crop rates is
    (0.5 + 0.0) / 2 = 0.25 - five times larger, because it weights a two-character label as
    heavily as a twenty-character one.
    """
    truths = ["q0", "evaluate the applica"]
    predictions = ["qq", "evaluate the applica"]
    result = metrics.score(truths, predictions)
    assert result["cer"] == pytest.approx(1 / 22, abs=1e-4)
    assert result["mean_crop_cer"] == pytest.approx(0.25, abs=1e-4)


def test_case_is_normalised_away_by_default_and_visible_with_raw():
    assert metrics.score(["Submit"], ["submit"])["cer"] == 0.0
    assert metrics.score(["Submit"], ["submit"], raw=True)["cer"] > 0.0


def test_the_per_crop_rate_is_capped_but_the_corpus_rate_is_not():
    """A runaway insertion is bounded in the mean column and honest in the corpus column."""
    result = metrics.score(["q0"], ["a" * 40])
    assert result["mean_crop_cer"] == 1.0
    assert result["cer"] > 1.0


def test_an_empty_prediction_costs_the_whole_truth():
    assert metrics.score(["abcd"], [""])["cer"] == 1.0


def test_mismatched_lengths_are_an_error_rather_than_a_silent_truncation():
    with pytest.raises(ValueError):
        metrics.score(["a", "b"], ["a"])


def test_an_empty_corpus_scores_nan_rather_than_zero():
    result = metrics.score([], [])
    assert result["crops"] == 0
    assert result["cer"] != result["cer"]  # nan


def test_the_target_gate_reads_the_corpus_rate():
    assert metrics.score(["abcdefghij"], ["abcdefghij"])["meets_target"]
    assert not metrics.score(["abcdefghij"], ["zzzzzzzzzz"])["meets_target"]


def test_wer_counts_words_and_not_characters():
    result = metrics.score(["send the order"], ["send an order"])
    assert result["wer"] == pytest.approx(1 / 3, abs=1e-4)

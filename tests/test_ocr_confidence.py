"""Phase 9.3.7 - the confidence read off a CTC path, its AUROC, and its calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.ocr import confidence


def _posterior(rows):
    """Log-probabilities from a list of per-column probability vectors."""
    return np.log(np.asarray(rows, dtype=float))


def test_a_certain_path_scores_near_one():
    columns = _posterior([[0.001, 0.999], [0.001, 0.999]])
    scores = confidence.path_scores(columns)
    assert scores["min_char"] == pytest.approx(0.999, abs=1e-3)
    assert scores["mean_char"] == pytest.approx(0.999, abs=1e-3)


def test_blank_columns_do_not_dilute_the_character_confidence():
    """The blanks between letters are most of the columns; averaging over them measures nothing."""
    certain = _posterior([[0.01, 0.99]])
    with_blanks = _posterior([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
    assert confidence.path_scores(with_blanks)["mean_char"] == pytest.approx(
        confidence.path_scores(certain)["mean_char"], abs=1e-6
    )


def test_the_minimum_is_driven_by_the_worst_character_and_the_mean_is_not():
    # Three classes - blank, a, b - so the columns emit `a b a` rather than collapsing into one
    # character, which is the case where a minimum and a mean can differ at all.
    columns = _posterior([[0.01, 0.98, 0.01], [0.2, 0.2, 0.6], [0.01, 0.98, 0.01]])
    scores = confidence.path_scores(columns)
    assert scores["min_char"] == pytest.approx(0.6, abs=1e-6)
    assert scores["mean_char"] > scores["min_char"]


def test_a_repeat_counts_once_because_ctc_collapses_it():
    columns = _posterior([[0.01, 0.99], [0.01, 0.99]])
    assert confidence.path_scores(columns)["min_char"] == pytest.approx(0.99, abs=1e-3)


def test_an_all_blank_path_has_no_emitting_column_and_does_not_crash():
    scores = confidence.path_scores(_posterior([[0.99, 0.01], [0.99, 0.01]]))
    assert scores["min_char"] == 0.0


def test_a_perfect_ranking_scores_auroc_one():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    wrong = np.array([False, False, True, True])
    assert confidence.auroc(scores, wrong) == pytest.approx(1.0)


def test_a_reversed_ranking_scores_auroc_zero():
    scores = np.array([0.9, 0.95, 0.1, 0.2])
    wrong = np.array([False, False, True, True])
    assert confidence.auroc(scores, wrong) == pytest.approx(0.0)


def test_all_ties_score_auroc_one_half():
    assert confidence.auroc(np.ones(6), np.array([True] * 3 + [False] * 3)) == pytest.approx(0.5)


def test_auroc_is_nan_when_one_class_is_absent():
    value = confidence.auroc(np.arange(4.0), np.zeros(4, dtype=bool))
    assert value != value


def test_calibration_bins_are_equal_count():
    result = confidence.calibration(np.linspace(0, 1, 20), np.ones(20, dtype=bool), bins=4)
    assert [row["n"] for row in result["bins"]] == [5, 5, 5, 5]


def test_a_perfectly_calibrated_score_has_zero_ece():
    """Confidence 1.0 on reads that are all correct, 0.0 on reads that are all wrong."""
    values = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([False, False, True, True])
    assert confidence.calibration(values, correct, bins=2)["ece"] == pytest.approx(0.0)


def test_a_confidently_wrong_score_has_ece_one():
    values = np.ones(4)
    correct = np.zeros(4, dtype=bool)
    assert confidence.calibration(values, correct, bins=2)["ece"] == pytest.approx(1.0)


def test_budget_capture_reports_the_lift_over_flagging_at_random():
    """All of the error sits in the least confident fifth: capture 1.0 at budget 0.2, lift 5x."""
    values = np.arange(10.0)
    distances = np.array([7.0, 3.0] + [0.0] * 8)
    result = confidence.budget_capture(values, distances, 0.2)
    assert result["flagged"] == 2
    assert result["error_captured"] == pytest.approx(1.0)
    assert result["lift"] == pytest.approx(5.0)


def test_a_useless_confidence_captures_its_budget_share_and_no_more():
    values = np.arange(10.0)
    result = confidence.budget_capture(values, np.ones(10), 0.2)
    assert result["error_captured"] == pytest.approx(0.2)
    assert result["lift"] == pytest.approx(1.0)

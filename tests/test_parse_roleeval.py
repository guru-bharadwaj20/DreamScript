"""Phase 7.3.9 - per-role scores, and the two references they are scored against."""

from __future__ import annotations

import pytest

from src.parse import roleeval
from src.parse.roles import DERIVED_STATES, STATES

# -- the declared target ---------------------------------------------------------------------------


def test_the_target_is_the_plans_number():
    assert roleeval.TARGET == 0.80


def test_the_annotated_states_are_everything_not_derived():
    """7.3.1 found the annotation carries four of the nine; this is that split, reused."""
    assert set(roleeval.ANNOTATED_STATES) == set(STATES) - set(DERIVED_STATES)


def test_the_two_references_partition_the_state_space():
    assert set(roleeval.ANNOTATED_STATES) | set(DERIVED_STATES) == set(STATES)


# -- per-role scores -------------------------------------------------------------------------------


def test_every_state_is_scored_even_one_that_never_occurs():
    """A missing row would silently drop a state out of the macro average."""
    scores = roleeval.per_role(["process"] * 4, ["process"] * 4)
    assert set(scores) == set(STATES)
    assert scores["decision"]["support"] == 0


def test_a_perfect_prediction_scores_one():
    truth = ["start", "process", "decision", "terminal"]
    scores = roleeval.per_role(list(truth), truth)
    assert all(scores[role]["f1"] == 1.0 for role in truth)


def test_support_is_counted_from_the_truth_not_the_prediction():
    scores = roleeval.per_role(["process", "process"], ["process", "decision"])
    assert scores["decision"]["support"] == 1
    assert scores["process"]["support"] == 1


def test_a_state_never_predicted_scores_zero_rather_than_raising():
    scores = roleeval.per_role(["process", "process"], ["process", "loop-back"])
    assert scores["loop-back"]["f1"] == 0.0
    assert scores["loop-back"]["recall"] == 0.0


def test_each_row_records_which_reference_it_was_scored_against():
    """The whole point of 7.3.9's two headline numbers."""
    scores = roleeval.per_role(["process"], ["process"])
    assert scores["process"]["reference"] == "annotation"
    assert scores["branch-false"]["reference"] == "derivation"


def test_precision_and_recall_are_not_interchanged():
    # One true `decision`, predicted twice: recall 1.0, precision 0.5.
    scores = roleeval.per_role(["decision", "decision"], ["decision", "process"])
    assert scores["decision"]["recall"] == 1.0
    assert scores["decision"]["precision"] == 0.5


# -- the confusion table ---------------------------------------------------------------------------


def test_the_matrix_is_square_over_the_whole_state_space():
    matrix = roleeval.confusion(["process"], ["process"])["matrix"]
    assert len(matrix) == len(STATES)
    assert all(len(row) == len(STATES) for row in matrix)


def test_a_perfect_prediction_has_no_confusions():
    assert roleeval.confusion(["process", "start"], ["process", "start"])["worst_confusions"] == []


def test_confusions_are_ranked_by_node_count():
    truth = ["branch-false"] * 5 + ["start"] * 2
    predicted = ["process"] * 5 + ["loop-back"] * 2
    worst = roleeval.confusion(predicted, truth)["worst_confusions"]
    assert (worst[0]["true"], worst[0]["predicted"], worst[0]["nodes"]) == (
        "branch-false",
        "process",
        5,
    )
    assert [row["nodes"] for row in worst] == sorted((row["nodes"] for row in worst), reverse=True)


def test_the_share_is_of_the_true_class_not_of_the_corpus():
    """`branch-false` losing 36% of itself is the finding; 36% of everything would not be."""
    truth = ["branch-false"] * 4 + ["process"] * 96
    predicted = ["process"] * 100
    worst = roleeval.confusion(predicted, truth)["worst_confusions"]
    assert worst[0]["share_of_true"] == pytest.approx(1.0)


def test_the_diagonal_is_never_reported_as_a_confusion():
    truth = ["process"] * 9 + ["start"]
    predicted = ["process"] * 10
    worst = roleeval.confusion(predicted, truth)["worst_confusions"]
    assert all(row["true"] != row["predicted"] for row in worst)


def test_at_most_eight_confusions_are_reported():
    truth, predicted = [], []
    for i, actual in enumerate(STATES):
        for j, guessed in enumerate(STATES):
            if i != j:
                truth.append(actual)
                predicted.append(guessed)
    assert len(roleeval.confusion(predicted, truth)["worst_confusions"]) == 8

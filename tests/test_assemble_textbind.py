"""Phase 10.1.2 - the text-to-node binding cascade, tested as behaviour."""

from __future__ import annotations

import math

import pytest

from src.assemble import textbind as tb


def box(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, w, h]


DIAGONAL = math.hypot(1000.0, 1000.0)


# -- stage 1 --------------------------------------------------------------------------------


def test_containment_claims_a_text_box_that_sits_inside_a_node():
    nodes = [box(0, 0, 100, 100)]
    assert tb.containment([box(10, 10, 20, 20)], nodes) == {0: 0}


def test_containment_leaves_a_text_box_that_sits_outside_every_node_unclaimed():
    assert tb.containment([box(500, 500, 20, 20)], [box(0, 0, 100, 100)]) == {}


def test_containment_resolves_nested_boxes_to_the_innermost_one():
    pool, lane, task = box(0, 0, 400, 400), box(0, 0, 200, 200), box(10, 10, 60, 60)
    assert tb.containment([box(20, 20, 20, 20)], [pool, lane, task]) == {0: 2}


def test_containment_refuses_a_text_box_that_only_half_overlaps_a_node():
    # 50% of the area inside is below CONTAIN, so this is a straddling region and not a label.
    assert tb.containment([box(90, 10, 20, 20)], [box(0, 0, 100, 100)]) == {}


# -- stage 2 --------------------------------------------------------------------------------


def test_nearest_centroid_picks_the_closer_of_two_nodes():
    nodes = [box(0, 0, 40, 40), box(900, 900, 40, 40)]
    assert tb.nearest([box(60, 0, 10, 10)], nodes, [0], DIAGONAL, 0.2) == {0: 0}


def test_nearest_centroid_rejects_a_text_box_beyond_the_tolerance():
    nodes = [box(0, 0, 40, 40)]
    assert tb.nearest([box(900, 900, 10, 10)], nodes, [0], DIAGONAL, 0.05) == {}


def test_nearest_centroid_lets_one_node_own_several_text_boxes():
    texts = [box(60, 0, 10, 10), box(60, 20, 10, 10), box(60, 40, 10, 10)]
    nodes = [box(0, 0, 40, 40)]
    assignment = tb.nearest(texts, nodes, [0, 1, 2], DIAGONAL, 0.2)
    assert assignment == {0: 0, 1: 0, 2: 0}


# -- stage 3 --------------------------------------------------------------------------------


def test_hungarian_gives_each_node_at_most_one_text_box():
    texts = [box(50, 0, 10, 10), box(55, 0, 10, 10)]
    nodes = [box(0, 0, 20, 20), box(200, 0, 20, 20)]
    assignment = tb.hungarian(texts, nodes, [0, 1], DIAGONAL, 1.0)
    assert sorted(assignment.values()) == [0, 1]


def test_hungarian_prefers_the_globally_cheaper_pairing_over_the_locally_nearest():
    # Both text boxes are nearest to node 0; the optimum pays a little on one to serve both.
    texts = [box(100, 0, 10, 10), box(140, 0, 10, 10)]
    nodes = [box(80, 0, 10, 10), box(300, 0, 10, 10)]
    assert tb.hungarian(texts, nodes, [0, 1], DIAGONAL, 1.0) == {0: 0, 1: 1}


def test_hungarian_drops_pairs_beyond_the_tolerance():
    texts = [box(0, 0, 10, 10)]
    nodes = [box(900, 900, 10, 10)]
    assert tb.hungarian(texts, nodes, [0], DIAGONAL, 0.01) == {}


def test_hungarian_on_nothing_returns_nothing():
    assert tb.hungarian([], [], [], DIAGONAL, 0.1) == {}
    assert tb.hungarian([box(0, 0, 1, 1)], [], [0], DIAGONAL, 0.1) == {}


# -- the cascade ----------------------------------------------------------------------------


def test_the_cascade_answers_once_for_every_text_box_in_order():
    texts = [box(10, 10, 5, 5), box(500, 500, 5, 5), box(60, 10, 5, 5)]
    bindings = tb.bind(texts, [box(0, 0, 100, 100)], DIAGONAL)
    assert [b.text for b in bindings] == [0, 1, 2]


def test_the_cascade_prefers_containment_over_distance():
    # The text sits inside the large node but its centre is nearer the small one's centre.
    texts = [box(95, 45, 6, 6)]
    nodes = [box(0, 0, 200, 200), box(105, 48, 4, 4)]
    binding = tb.bind(texts, nodes, DIAGONAL)[0]
    assert (binding.stage, binding.node) == ("containment", 0)


def test_the_cascade_reports_no_owner_when_nothing_is_near():
    binding = tb.bind([box(900, 900, 5, 5)], [box(0, 0, 20, 20)], DIAGONAL)[0]
    assert binding.node is None and binding.stage == "unbound"


def test_the_cascade_records_which_stage_decided_each_box():
    texts = [box(10, 10, 5, 5), box(100, 50, 5, 5)]
    nodes = [box(0, 0, 100, 100)]
    assert [b.stage for b in tb.bind(texts, nodes, DIAGONAL)] == ["containment", "nearest"]


def test_the_greedy_stage_three_is_a_supported_alternative_to_the_hungarian():
    texts = [box(400, 400, 5, 5), box(410, 400, 5, 5)]
    nodes = [box(380, 380, 5, 5)]
    greedy = tb.bind(texts, nodes, DIAGONAL, near_tol=0.0, stage3="greedy")
    assert [b.node for b in greedy] == [0, 0]
    optimal = tb.bind(texts, nodes, DIAGONAL, near_tol=0.0, stage3="hungarian")
    assert sorted(b.node is None for b in optimal) == [False, True]


def test_the_binder_never_reads_ground_truth_geometry(monkeypatch):
    # A binding computed from detections must be unchanged by anything the truth says.
    texts, nodes = [box(10, 10, 5, 5)], [box(0, 0, 100, 100)]
    monkeypatch.setattr(tb, "truth", lambda page: pytest.fail("bind() consulted the truth"))
    assert tb.bind(texts, nodes, DIAGONAL)[0].node == 0


# -- ground truth ---------------------------------------------------------------------------


def test_the_true_owner_is_the_smallest_ground_truth_node_containing_the_text():
    gt = [("pool", box(0, 0, 400, 400)), ("task", box(10, 10, 60, 60))]
    assert tb.true_owners([box(20, 20, 10, 10)], gt) == ["task"]


def test_an_edge_label_outside_every_node_has_no_owner():
    gt = [("a", box(0, 0, 50, 50)), ("b", box(200, 200, 50, 50))]
    assert tb.true_owners([box(120, 120, 10, 10)], gt) == [None]


def test_a_detection_is_credited_with_a_ground_truth_node_only_when_it_overlaps_enough():
    gt = [("a", box(0, 0, 100, 100))]
    assert tb.detection_identity([box(2, 2, 100, 100)], gt) == ["a"]
    assert tb.detection_identity([box(80, 80, 100, 100)], gt) == [None]


def test_two_detections_cannot_both_be_credited_with_the_same_node():
    gt = [("a", box(0, 0, 100, 100))]
    identity = tb.detection_identity([box(0, 0, 100, 100), box(3, 3, 100, 100)], gt)
    assert identity == ["a", None]


# -- aggregation ----------------------------------------------------------------------------


def _row(**over) -> dict:
    row = {
        "source": "hdbpmn",
        "texts": 10,
        "nodes": 4,
        "gt_nodes": 4,
        "unmatched_detections": 0,
        "owned": 4,
        "correct": 9,
        "owned_correct": 3,
        "stages": {
            s: {"resolved": 0, "correct": 0, "owned": 0, "owned_correct": 0}
            for s in (*tb.STAGES, "unbound")
        },
        "controls": {"reject_everything": {"correct": 6, "owned_correct": 0}},
    }
    row["stages"]["containment"] = {
        "resolved": 4,
        "correct": 3,
        "owned": 4,
        "owned_correct": 3,
    }
    row["stages"]["unbound"]["resolved"] = 6
    row["stages"]["unbound"]["correct"] = 6
    return {**row, **over}


def test_aggregation_reports_both_the_overall_and_the_owner_only_accuracy():
    result = tb.aggregate([_row(), _row()])
    assert result["accuracy"] == 0.9
    assert result["owner_only_accuracy"] == 0.75


def test_aggregation_says_plainly_whether_the_target_was_met():
    assert tb.aggregate([_row()], target=0.90)["target_met"] is True
    assert tb.aggregate([_row()], target=0.95)["target_met"] is False


def test_aggregation_reports_the_owned_share_that_makes_accuracy_a_rejection_metric():
    assert tb.aggregate([_row()])["owned_share"] == 0.4


def test_aggregation_of_an_empty_stage_reports_a_nan_accuracy_rather_than_a_zero():
    result = tb.aggregate([_row()])
    assert math.isnan(result["stages"]["hungarian"]["accuracy"])


# -- the recorded run -----------------------------------------------------------------------


def test_the_published_run_reports_the_target_honestly():
    import json

    if not tb.OUT.is_file():
        pytest.skip("run `python -m src.assemble.textbind` first")
    result = json.loads(tb.OUT.read_text(encoding="utf-8"))
    assert result["pages"] == 470
    assert result["target_met"] is (result["accuracy"] >= result["target"])
    # The finding the docstring rests on now: 3.2's proposer covers this corpus so densely that
    # the vast majority of text boxes DO have a ground-truth owner, which makes "reject
    # everything" a bad baseline rather than the dangerously good one an earlier, sparser
    # proposer produced.
    assert result["owned_share"] > 0.5
    assert result["controls"]["reject_everything"]["accuracy"] < 0.5
    assert result["controls"]["reject_everything"]["accuracy"] == pytest.approx(
        1.0 - result["owned_share"], abs=1e-3
    )
    # The container artefact this task checked for: real, but a minority of owned boxes.
    assert 0.0 <= result["owned_share_excl_containers"] < result["owned_share"]
    assert set(result["stages"]) == {*tb.STAGES, "unbound"}

"""Phase 7.3.10 - the three repair rules, and the gate the damage switches off."""

from __future__ import annotations

import pytest

from src.parse import repair


def node(node_id, x, y):
    return {"id": node_id, "shape": "rectangle", "bbox": [x, y, 10, 10], "text": ""}


def diagram(nodes, edges):
    """A page plus a far anchor node, so the 25%-of-diagonal span limit is not the thing
    under test. The anchor is never given a role, so no rule ever fires on it."""
    return {
        "id": "d",
        "diagram_type": "flowchart",
        "nodes": [*nodes, node("_anchor", 0, 1000)],
        "edges": [
            {"id": f"e{i}", "src": s, "dst": d, "directed": True} for i, (s, d) in enumerate(edges)
        ],
    }


def confident(*node_ids):
    return dict.fromkeys(node_ids, 1.0)


# -- geometry --------------------------------------------------------------------------------------


def test_the_page_diagonal_comes_from_the_nodes():
    """The IR carries no canvas size, so the nodes are the only source."""
    nodes = {"a": node("a", 0, 0), "b": node("b", 30, 40)}
    assert repair._diagonal(nodes) == pytest.approx(50.0)


def test_an_empty_page_has_a_nonzero_diagonal():
    """Zero would divide the distance limit by nothing."""
    assert repair._diagonal({}) == 1.0


def test_a_single_node_page_has_a_nonzero_diagonal():
    assert repair._diagonal({"a": node("a", 5, 5)}) == 1.0


def test_the_nearest_candidate_wins_not_the_first():
    nodes = {"a": node("a", 0, 0), "far": node("far", 0, 100), "near": node("near", 0, 10)}
    assert repair._nearest("a", ["far", "near"], nodes, 1000.0) == "near"


def test_a_candidate_beyond_the_limit_is_refused():
    """An unbounded nearest-node rule would connect opposite corners of the page."""
    nodes = {"a": node("a", 0, 0), "far": node("far", 0, 100)}
    assert repair._nearest("a", ["far"], nodes, 50.0) is None


def test_no_candidates_gives_none_rather_than_raising():
    assert repair._nearest("a", [], {"a": node("a", 0, 0)}, 100.0) is None


# -- rule 1: a decision needs two ways out -----------------------------------------------------------


def test_a_one_armed_decision_gets_a_second_branch():
    page = diagram([node("d", 0, 0), node("x", 0, 10), node("y", 0, 20)], [("d", "x")])
    made = repair.repair(page, {"d": "decision"}, confident("d"))
    assert made["count"] == 1
    assert made["repairs"][0]["rule"] == "missing_branch"


def test_a_decision_that_already_branches_is_left_alone():
    page = diagram([node("d", 0, 0), node("x", 0, 10), node("y", 0, 20)], [("d", "x"), ("d", "y")])
    assert repair.repair(page, {"d": "decision"}, confident("d"))["count"] == 0


def test_the_new_branch_does_not_duplicate_the_edge_that_exists():
    page = diagram([node("d", 0, 0), node("x", 0, 10), node("y", 0, 20)], [("d", "x")])
    repair.repair(page, {"d": "decision"}, confident("d"))
    assert [e["dst"] for e in page["edges"] if e["src"] == "d"] == ["x", "y"]


def test_the_repaired_edge_is_marked_with_the_rule_that_made_it():
    """Phase 10 has to be able to tell an inferred edge from a detected one."""
    page = diagram([node("d", 0, 0), node("x", 0, 10), node("y", 0, 20)], [("d", "x")])
    repair.repair(page, {"d": "decision"}, confident("d"))
    assert page["edges"][-1]["attrs"]["repaired_by"] == "missing_branch"


def test_the_gate_that_the_damage_switches_off():
    """The measured mechanism: deleting the branch lowers the decision's own confidence,
    and the rule that exists to repair it is gated on exactly that number."""
    page = diagram([node("d", 0, 0), node("x", 0, 10), node("y", 0, 20)], [("d", "x")])
    assert repair.repair(page, {"d": "decision"}, {"d": 0.4}, threshold=0.9)["count"] == 0
    assert repair.repair(page, {"d": "decision"}, {"d": 0.4}, threshold=0.3)["count"] == 1


# -- rule 2: a terminal does not continue ------------------------------------------------------------


def test_an_edge_leaving_a_terminal_is_removed():
    page = diagram([node("t", 0, 0), node("x", 0, 10)], [("t", "x")])
    made = repair.repair(page, {"t": "terminal"}, confident("t"))
    assert made["repairs"][0]["rule"] == "stray_terminal"
    assert page["edges"] == []


def test_the_least_confident_stray_edge_is_the_one_dropped():
    page = diagram([node("t", 0, 0), node("x", 0, 10), node("y", 0, 20)], [("t", "x"), ("t", "y")])
    page["edges"][0]["confidence"] = 0.9
    page["edges"][1]["confidence"] = 0.1
    repair.repair(page, {"t": "terminal"}, confident("t"))
    assert [e["dst"] for e in page["edges"]] == ["x"]


def test_a_terminal_with_no_outgoing_edge_is_left_alone():
    page = diagram([node("t", 0, 0), node("x", 0, 10)], [("x", "t")])
    assert repair.repair(page, {"t": "terminal"}, confident("t"))["count"] == 0


# -- rule 3: orphans ---------------------------------------------------------------------------------


def test_an_orphan_is_attached_to_the_nearest_node_above_it():
    page = diagram([node("a", 0, 0), node("orphan", 0, 10)], [])
    made = repair.repair(page, {"a": "process", "orphan": "process"}, confident("a", "orphan"))
    assert {"rule": "orphan", "src": "a", "dst": "orphan"} in made["repairs"]


def test_a_data_object_is_left_isolated():
    """The measured reason the orphan rule is the least accurate: an isolated node in this
    corpus is usually a data object that was always isolated."""
    page = diagram([node("a", 0, 0), node("data", 0, 10)], [])
    made = repair.repair(page, {"a": "process", "data": "input"}, confident("a", "data"))
    assert all(r["dst"] != "data" for r in made["repairs"])


def test_a_node_that_already_has_an_edge_is_not_an_orphan():
    page = diagram([node("a", 0, 0), node("b", 0, 10)], [("a", "b")])
    made = repair.repair(page, {"a": "process", "b": "process"}, confident("a", "b"))
    assert all(r["rule"] != "orphan" for r in made["repairs"])


def test_the_topmost_orphan_has_nothing_above_it_to_attach_to():
    page = diagram([node("first", 0, 0), node("b", 0, 10)], [])
    made = repair.repair(page, {"first": "process", "b": "process"}, confident("first", "b"))
    assert all(r["dst"] != "first" for r in made["repairs"])


# -- the damage model --------------------------------------------------------------------------------


def test_damage_deletes_and_returns_what_it_deleted():
    page = diagram(
        [node(str(i), 0, i * 10) for i in range(10)], [(str(i), str(i + 1)) for i in range(9)]
    )
    damaged, removed = repair.damage(page, 1.0)
    assert damaged["edges"] == []
    assert len(removed) == 9


def test_damage_at_zero_changes_nothing():
    page = diagram([node("a", 0, 0), node("b", 0, 10)], [("a", "b")])
    damaged, removed = repair.damage(page, 0.0)
    assert removed == []
    assert len(damaged["edges"]) == 1


def test_damage_leaves_the_original_untouched():
    """Scoring against the deleted edges needs the original to still exist."""
    page = diagram([node("a", 0, 0), node("b", 0, 10)], [("a", "b")])
    repair.damage(page, 1.0)
    assert len(page["edges"]) == 1


def test_damage_is_seeded_and_reproducible():
    page = diagram(
        [node(str(i), 0, i * 10) for i in range(20)], [(str(i), str(i + 1)) for i in range(19)]
    )
    first = [e["id"] for e in repair.damage(page, 0.5, seed=1)[1]]
    second = [e["id"] for e in repair.damage(page, 0.5, seed=1)[1]]
    assert first == second


# -- the declared defaults ---------------------------------------------------------------------------


def test_the_gate_and_the_span_limit_are_declared():
    assert repair.THRESHOLD == 0.9
    assert 0 < repair.MAX_SPAN < 1

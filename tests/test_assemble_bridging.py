"""Phase 10.1.4 - the damage model and the repairer, checked on hand-built diagrams.

The corpus sweep lives in `experiments/assemble/bridging.json`; these tests build small diagrams
with an exactly known answer, since that is the only way to tell "the repairer got the right
edge back" from "the repairer got lucky".
"""

from __future__ import annotations

import random

from src.assemble.bridging import (
    Stub,
    damage_diagram,
    nearest_control,
    repair,
    score_pair,
    score_repair,
)
from src.ir.model import Diagram, Edge, Node


def node(node_id, x, y, w=20.0, h=20.0):
    return Node(id=node_id, shape="rectangle", bbox=[x, y, w, h])


def straight_edge(edge_id, src, dst, x0, y0, x1, y1, n=20):
    """A straight polyline of `n` evenly spaced points from `(x0, y0)` to `(x1, y1)`."""
    points = [[x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)] for i in range(n)]
    return Edge(id=edge_id, src=src, dst=dst, directed=True, polyline=points)


# -- the damage model -----------------------------------------------------------------------


def test_a_straight_line_gap_leaves_two_stubs_that_point_at_each_other():
    edge = straight_edge("e1", "a", "b", 0, 0, 100, 0)
    diagram = Diagram(
        id="t",
        diagram_type="flowchart",
        nodes=[node("a", -20, -10), node("b", 100, -10)],
        edges=[edge],
    )
    stubs, truth_by_edge, skipped = damage_diagram(diagram, diagonal=100.0, gap_frac=0.2)
    assert skipped == 0
    assert truth_by_edge == {"e1": ("a", "b")}
    assert len(stubs) == 2
    src_stub = next(s for s in stubs if s.end == "src")
    dst_stub = next(s for s in stubs if s.end == "dst")
    assert src_stub.anchor == "a"
    assert dst_stub.anchor == "b"
    # A straight horizontal edge: both stubs travel in +x, and the free ends are left of right.
    assert src_stub.direction[0] > 0.9
    assert dst_stub.direction[0] > 0.9
    assert src_stub.point[0] < dst_stub.point[0]


def test_a_gap_that_would_leave_no_stub_is_skipped_not_forced():
    edge = straight_edge("e1", "a", "b", 0, 0, 10, 0)
    diagram = Diagram(
        id="t",
        diagram_type="flowchart",
        nodes=[node("a", -5, -5), node("b", 10, -5)],
        edges=[edge],
    )
    # A gap as large as the whole edge cannot leave two usable stubs.
    stubs, truth_by_edge, skipped = damage_diagram(diagram, diagonal=100.0, gap_frac=0.2)
    assert skipped == 1
    assert stubs == []
    assert truth_by_edge == {}


def test_an_edge_with_no_polyline_or_dangling_end_is_never_damaged():
    diagram = Diagram(
        id="t",
        diagram_type="flowchart",
        nodes=[node("a", 0, 0), node("b", 100, 0)],
        edges=[
            Edge(id="no_poly", src="a", dst="b", polyline=None),
            Edge(id="dangling", src="a", dst=None, polyline=[[0, 0], [10, 0]]),
        ],
    )
    stubs, truth_by_edge, skipped = damage_diagram(diagram, diagonal=100.0, gap_frac=0.1)
    assert stubs == []
    assert truth_by_edge == {}
    assert skipped == 0  # neither edge was even eligible to count as a skip


def test_gap_as_a_share_of_edge_length_scales_with_the_edge_not_the_page():
    short = straight_edge("short", "a", "b", 0, 0, 10, 0)
    long = straight_edge("long", "c", "d", 0, 100, 100, 100)
    diagram = Diagram(
        id="t",
        diagram_type="flowchart",
        nodes=[node("a", -5, -5), node("b", 10, -5), node("c", -5, 95), node("d", 100, 95)],
        edges=[short, long],
    )
    stubs, truth_by_edge, skipped = damage_diagram(
        diagram, diagonal=1000.0, gap_frac=0.3, by_edge=True
    )
    assert skipped == 0
    assert set(truth_by_edge) == {"short", "long"}
    gaps = {
        e: abs(
            next(s for s in stubs if s.edge_id == e and s.end == "dst").point[0]
            - next(s for s in stubs if s.edge_id == e and s.end == "src").point[0]
        )
        for e in truth_by_edge
    }
    # The long edge's gap (30% of 100) should dwarf the short edge's (30% of 10).
    assert gaps["long"] > 5 * gaps["short"]


# -- the repairer ----------------------------------------------------------------------------


def _stub(edge_id, end, anchor, point, direction):
    return Stub(edge_id, end, anchor, point, direction)


def test_two_stubs_that_point_straight_at_each_other_are_reunited():
    src = _stub("e1", "src", "a", (0.0, 0.0), (1.0, 0.0))
    dst = _stub("e1", "dst", "b", (10.0, 0.0), (1.0, 0.0))
    nodes = {"a": node("a", -20, -10), "b": node("b", 10, -10)}
    pairs = repair([src, dst], nodes, tolerance=50.0)
    assert len(pairs) == 1
    a, b, _ = pairs[0]
    assert {a.edge_id, b.edge_id} == {"e1"}


def test_a_stub_pair_further_apart_than_tolerance_is_left_unmatched():
    src = _stub("e1", "src", "a", (0.0, 0.0), (1.0, 0.0))
    dst = _stub("e1", "dst", "b", (1000.0, 0.0), (1.0, 0.0))
    pairs = repair([src, dst], {}, tolerance=5.0)
    assert pairs == []


def test_direction_continuity_prefers_the_colinear_partner_over_the_closer_one():
    # 'near' is closer to src but heads off at a right angle; 'far' continues src's own heading.
    src = _stub("far_edge", "src", "a", (0.0, 0.0), (1.0, 0.0))
    near = _stub("near_edge", "dst", "b", (5.0, 0.0), (0.0, 1.0))
    far = _stub("far_edge", "dst", "c", (9.0, 0.0), (1.0, 0.0))
    pairs = repair([src, near, far], {}, tolerance=50.0)
    assert len(pairs) == 1
    a, b, _ = pairs[0]
    assert {a.edge_id, b.edge_id} == {"far_edge"}


def test_removing_the_distance_gate_and_both_scores_reduces_to_the_control():
    src = _stub("e1", "src", "a", (0.0, 0.0), (1.0, 0.0))
    dst = _stub("e1", "dst", "b", (10.0, 0.0), (-1.0, 0.0))  # direction disagrees; ignored anyway
    pairs = repair(
        [src, dst],
        {},
        tolerance=5.0,
        use_direction=False,
        use_port=False,
        use_gate=False,
    )
    control_pairs = nearest_control([src, dst])
    assert [(a.edge_id, b.edge_id) for a, b, _ in pairs] == [
        (a.edge_id, b.edge_id) for a, b, _ in control_pairs
    ]


def test_score_pair_reports_a_lower_direction_score_for_a_reversed_stub():
    a = _stub("e1", "src", "n", (0.0, 0.0), (1.0, 0.0))
    aligned = _stub("e1", "dst", "m", (10.0, 0.0), (1.0, 0.0))
    reversed_ = _stub("e2", "dst", "m", (10.0, 0.0), (-1.0, 0.0))
    _, _, parts_aligned = score_pair(a, aligned, {}, tolerance=50.0)
    _, _, parts_reversed = score_pair(a, reversed_, {}, tolerance=50.0)
    assert parts_aligned["direction"] > parts_reversed["direction"]


def test_score_repair_counts_recovered_wrong_and_missed_correctly():
    a1 = _stub("e1", "src", "a", (0.0, 0.0), (1.0, 0.0))
    b1 = _stub("e1", "dst", "b", (10.0, 0.0), (1.0, 0.0))
    a2 = _stub("e2", "src", "c", (0.0, 100.0), (1.0, 0.0))
    b2 = _stub("e2", "dst", "d", (10.0, 100.0), (1.0, 0.0))
    truth_by_edge = {"e1": ("a", "b"), "e2": ("c", "d")}
    # One correct pair, one wrong cross-pair, matching every stub so nothing is left missing.
    pairs = [(a1, b1, 1.0), (a2, b2, 1.0)]
    counts = score_repair(pairs, truth_by_edge)
    assert counts == {"broken": 2, "recovered": 2, "wrong": 0, "missed": 0}

    wrong_pairs = [(a1, b2, 1.0), (a2, b1, 1.0)]
    counts = score_repair(wrong_pairs, truth_by_edge)
    assert counts == {"broken": 2, "recovered": 0, "wrong": 2, "missed": 2}


# -- a small end-to-end sanity check, on a synthetic page ------------------------------------


def test_a_lightly_damaged_synthetic_page_is_fully_recovered():
    rng = random.Random(0)
    nodes, edges = [], []
    for i in range(6):
        x = i * 100.0
        nodes.append(node(f"n{i}", x, 0.0))
        nodes.append(node(f"m{i}", x, 100.0))
        edges.append(straight_edge(f"e{i}", f"n{i}", f"m{i}", x + 10, 10, x + 10, 90))
    diagram = Diagram(id="t", diagram_type="flowchart", nodes=nodes, edges=edges)
    diagonal = 600.0
    stubs, truth_by_edge, skipped = damage_diagram(
        diagram, diagonal, gap_frac=0.01, seed=rng.randint(0, 1000)
    )
    assert skipped == 0
    node_map = {n.id: n for n in diagram.nodes}
    pairs = repair(stubs, node_map, tolerance=0.2 * diagonal)
    counts = score_repair(pairs, truth_by_edge)
    assert counts["recovered"] == counts["broken"]
    assert counts["wrong"] == 0

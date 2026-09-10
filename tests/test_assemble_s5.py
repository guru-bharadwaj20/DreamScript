"""S5's metric: the decomposition must add up to the GED, and identity must cost nothing."""

from __future__ import annotations

import pytest

from src.assemble import s5
from src.ir.model import Diagram, Edge, Node


def graph(edges: list[tuple[str, str]], text: dict[str, str] | None = None) -> Diagram:
    text = text or {}
    boxes = {
        "a": [0.0, 0.0, 10.0, 10.0],
        "b": [50.0, 0.0, 10.0, 10.0],
        "c": [0.0, 50.0, 10.0, 10.0],
        "d": [50.0, 50.0, 10.0, 10.0],
    }
    # Every graph carries all four boxes, so a node is never *missing* unless a test
    # means it to be and the edge kinds stay separable from the node kinds.
    used = sorted(boxes)
    return Diagram(
        id="t",
        diagram_type="flowchart",
        nodes=[Node(id=n, shape="rectangle", bbox=list(boxes[n]), text=text.get(n)) for n in used],
        edges=[Edge(id=f"e{i}", src=u, dst=v, directed=False) for i, (u, v) in enumerate(edges)],
    )


def test_identity_costs_nothing():
    """`diff(x, x)` under `geometric` is exact, so every edit kind must be zero."""
    g = graph([("a", "b"), ("b", "c")], {"a": "start"})
    row = s5.decompose(g, g)
    assert row["ged"] == 0.0
    assert all(row[kind] == 0 for kind in s5.KINDS)


def test_decomposition_sums_to_ged():
    """The reported kinds are the whole GED and not a subset of it - if they ever stop adding
    up, the error-mass shares this criterion is reasoned from are meaningless."""
    predicted = graph([("a", "b"), ("a", "c")], {"a": "begin"})
    actual = graph([("a", "b"), ("b", "c")], {"a": "start"})
    row = s5.decompose(predicted, actual)
    assert row["ged"] == pytest.approx(sum(row[kind] for kind in s5.KINDS))
    assert row["ged"] > 0


def test_text_and_shape_substitutions_are_separated():
    """36.7% of S5's error mass is text and 0.2% is shape; pooling them hides the whole story."""
    predicted = graph([("a", "b")], {"a": "begin"})
    actual = graph([("a", "b")], {"a": "start"})
    row = s5.decompose(predicted, actual)
    assert row["sub_text"] == 1
    assert row["sub_shape"] == 0


def test_duplicate_fragments_between_one_pair_cost_one_edit():
    """The heart of the correction to this row: the metric compares *sets* of endpoint pairs, so
    five traced fragments lying between the same two boxes are one wrong edge, not five. The old
    edge-only bound counted them with multiplicity and read 2.2x high because of it."""
    actual = graph([("a", "b")])
    once = graph([("a", "c")])
    many = graph([("a", "c")] * 5)
    assert s5.decompose(many, actual)["edge_insert"] == 1
    assert s5.decompose(many, actual)["ged"] == s5.decompose(once, actual)["ged"]


def test_dangling_ends_are_not_asserted_edges():
    """An end that attached to nothing is not a claim about the graph and is not charged as one -
    on either side, which is what keeps `diff(x, x)` exact for a truth that records open ends."""
    actual = graph([("a", "b")])
    predicted = graph([("a", "b")])
    predicted.edges.append(Edge(id="open", src="a", dst=None, directed=False))
    assert s5.decompose(predicted, actual)["ged"] == 0.0


def test_summarise_reports_the_ladder_and_the_bar():
    rows = [
        {**{k: 0 for k in s5.KINDS}, "ged": 2.0, "true_edges": 1},
        {
            **{k: 0 for k in s5.KINDS},
            "ged": 40.0,
            "sub_text": 20,
            "edge_delete": 20,
            "true_edges": 9,
        },
    ]
    out = s5.summarise(rows)
    assert out["median_ged"] == 21.0
    assert out["passes"] is False
    assert out["pass_share"] == 0.5
    # Removing text takes the second page from 40 to 20, so the median falls to 11.
    assert out["ladder"]["perfect_text"] == 11.0
    assert out["ladder"]["both_perfect"] == 1.0

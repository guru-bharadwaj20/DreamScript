"""Phase 10.2.3 - Tarjan's SCCs against the definition, and the loop classification codegen reads."""

from __future__ import annotations

import random

import pytest

from src.assemble import loops as L
from src.ir.model import Diagram, Edge, Node


def graph(edges: list[tuple[str, str]], nodes: list[str] | None = None) -> Diagram:
    """A Diagram with no geometry - only the arrows matter to anything in this module."""
    names = nodes or sorted({n for pair in edges for n in pair})
    return Diagram(
        id="t",
        diagram_type="flowchart",
        nodes=[Node(id=n, shape="rectangle", bbox=None) for n in names],
        edges=[Edge(id=f"e{i}", src=a, dst=b) for i, (a, b) in enumerate(edges)],
    )


def components(diagram: Diagram) -> list[list[str]]:
    return sorted(L.sccs(diagram))


# -- Tarjan itself ------------------------------------------------------------------------


def test_an_acyclic_chain_is_all_singleton_components():
    assert components(graph([("a", "b"), ("b", "c")])) == [["a"], ["b"], ["c"]]


def test_a_three_node_ring_is_one_component():
    assert components(graph([("a", "b"), ("b", "c"), ("c", "a")])) == [["a", "b", "c"]]


def test_a_self_edge_makes_a_component_of_one_that_is_still_a_loop():
    diagram = graph([("a", "a")])
    assert components(diagram) == [["a"]]
    assert [loop.kind for loop in L.loops(diagram)] == ["self_loop"]


def test_two_rings_joined_by_one_arrow_stay_two_components():
    diagram = graph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "d"), ("d", "c")])
    assert components(diagram) == [["a", "b"], ["c", "d"]]


def test_two_rings_sharing_a_node_are_one_component():
    """The case a naive cycle-walker gets wrong: an SCC is maximal, not a single cycle."""
    diagram = graph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")])
    assert components(diagram) == [["a", "b", "c"]]


def test_components_come_back_in_reverse_topological_order():
    diagram = graph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "d"), ("d", "c")])
    order = L.sccs(diagram)
    assert order.index(["c", "d"]) < order.index(["a", "b"])


def test_a_dangling_edge_does_not_invent_a_node():
    diagram = graph([("a", "b")])
    diagram.edges.append(Edge(id="open", src="b", dst=None))
    assert components(diagram) == [["a"], ["b"]]


def test_a_long_chain_does_not_exhaust_the_stack():
    """The explicit stack's reason for existing, at ten times the corpus's deepest path."""
    chain = [(f"n{i}", f"n{i + 1}") for i in range(300)]
    assert len(L.sccs(graph(chain))) == 301


# -- Tarjan against the definition ---------------------------------------------------------


def test_tarjan_agrees_with_brute_force_reachability_on_hand_built_graphs():
    cases = [
        [("a", "b"), ("b", "c")],
        [("a", "b"), ("b", "a")],
        [("a", "a")],
        [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "d")],
        [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "a")],
    ]
    for edges in cases:
        assert L.verify(graph(edges)), edges


@pytest.mark.parametrize("seed", range(200))
def test_tarjan_agrees_with_brute_force_on_random_graphs(seed):
    """200 random graphs, the correctness evidence the corpus sweep repeats on 5,497 files."""
    rng = random.Random(seed)
    names = [f"n{i}" for i in range(rng.randint(1, 9))]
    edges = [(a, b) for a in names for b in names if rng.random() < 0.25]
    assert L.verify(graph(edges, names))


# -- classification -----------------------------------------------------------------------


def test_a_ring_with_one_way_round_is_a_simple_cycle():
    diagram = graph([("a", "b"), ("b", "c"), ("c", "a")])
    assert [loop.kind for loop in L.loops(diagram)] == ["simple_cycle"]


def test_a_component_with_a_short_cut_is_complex_not_simple():
    diagram = graph([("a", "b"), ("b", "c"), ("c", "a"), ("a", "c")])
    assert [loop.kind for loop in L.loops(diagram)] == ["complex"]


def test_only_the_cyclic_components_are_reported_as_loops():
    diagram = graph([("s", "a"), ("a", "b"), ("b", "a"), ("b", "t")])
    found = L.loops(diagram)
    assert [loop.nodes for loop in found] == [["a", "b"]]


# -- entries, exits and reducibility --------------------------------------------------------


def test_a_single_entry_loop_is_reducible_and_names_its_header_and_exit():
    diagram = graph([("s", "a"), ("a", "b"), ("b", "a"), ("b", "t")])
    loop = L.loops(diagram)[0]
    assert loop.entries == ["a"]
    assert loop.reducible
    assert [diagram.edges[int(e[1:])].src for e in loop.exit_edges] == ["b"]
    assert loop.back_edges == ["e2"]


def test_two_entries_make_the_loop_irreducible():
    """Two ways into the ring, so no header dominates the body and codegen needs a jump."""
    diagram = graph([("s", "a"), ("s", "b"), ("a", "b"), ("b", "a")])
    loop = L.loops(diagram)[0]
    assert loop.entries == ["a", "b"]
    assert not loop.reducible


def test_a_loop_nothing_enters_is_treated_as_reducible_with_no_entry():
    loop = L.loops(graph([("a", "b"), ("b", "a")]))[0]
    assert loop.entries == []
    assert loop.reducible


def test_a_loop_with_no_way_out_reports_no_exit_edges():
    """27% of the corpus's loops look like this, and every one is a `while (true)`."""
    assert L.loops(graph([("s", "a"), ("a", "b"), ("b", "a")]))[0].exit_edges == []


# -- nesting ------------------------------------------------------------------------------


def test_an_inner_loop_is_found_one_level_deeper_than_the_outer_one():
    diagram = graph(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "b"), ("c", "a")],
    )
    found = L.loops(diagram)
    assert [(loop.nodes, loop.depth) for loop in found] == [
        (["a", "b", "c"], 1),
        (["b", "c"], 2),
    ]


def test_a_single_ring_has_no_nesting():
    assert {loop.depth for loop in L.loops(graph([("s", "a"), ("a", "b"), ("b", "a")]))} == {1}


# -- the attrs a later phase consumes -------------------------------------------------------


def test_mark_writes_the_documented_keys_onto_nodes_and_edges():
    diagram = L.mark(graph([("s", "a"), ("a", "b"), ("b", "a"), ("b", "t")]))
    header = diagram.node("a")
    assert header.attrs[L.NODE_LOOP_ROLE] == "entry"
    assert header.attrs[L.NODE_LOOP_KIND] == "simple_cycle"
    assert header.attrs[L.NODE_LOOP_DEPTH] == 1
    assert diagram.node("b").attrs[L.NODE_LOOP_ROLE] == "body"
    roles = {e.id: (e.attrs or {}).get(L.EDGE_LOOP_ROLE) for e in diagram.edges}
    assert roles == {"e0": "entry", "e1": "body", "e2": "back", "e3": "exit"}


def test_mark_leaves_nodes_outside_every_loop_untouched():
    diagram = L.mark(graph([("s", "a"), ("a", "b"), ("b", "a")]))
    assert diagram.node("s").attrs is None


def test_mark_keeps_existing_attrs():
    diagram = graph([("a", "b"), ("b", "a")])
    diagram.node("a").attrs = {"semantic": "kept"}
    L.mark(diagram)
    assert diagram.node("a").attrs["semantic"] == "kept"
    assert diagram.node("a").attrs[L.NODE_LOOP_ID]


def test_a_marked_diagram_still_satisfies_the_ir_schema():
    """9.3.7's lesson - a writer into shared state is validated against the consumer's loader."""
    diagram = graph([("s", "a"), ("a", "b"), ("b", "a")])
    for node in diagram.nodes:
        node.bbox = [0.0, 0.0, 1.0, 1.0]
    diagram.meta = {"source": "test", "geometry": "annotated"}
    assert Diagram.from_dict(L.mark(diagram).require_valid().to_dict()).node("a").attrs


def test_a_node_in_a_nest_carries_the_innermost_loop():
    diagram = L.mark(graph([("s", "a"), ("a", "b"), ("b", "c"), ("c", "b"), ("c", "a")]))
    assert diagram.node("b").attrs[L.NODE_LOOP_DEPTH] == 2
    assert diagram.node("a").attrs[L.NODE_LOOP_DEPTH] == 1


# -- the summary ---------------------------------------------------------------------------


def test_summarise_counts_reducible_and_irreducible_separately():
    rows = [
        {
            "source": "x",
            "loops": [loop.to_dict() for loop in L.loops(d)],
            "max_depth": 1,
            "brute_force_checked": True,
            "brute_force_agrees": True,
        }
        for d in (
            graph([("s", "a"), ("a", "b"), ("b", "a")]),
            graph([("s", "a"), ("s", "b"), ("a", "b"), ("b", "a")]),
        )
    ]
    summary = L.summarise(rows)
    assert (summary["reducible"], summary["irreducible"]) == (1, 1)
    assert summary["files_with_a_cycle"] == 2

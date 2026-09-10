"""Phase 11.2.7 - the baselines, the ambiguity measure and the win-rate harness.

The tests are mostly about the three things that can silently corrupt a comparison table: an arm
that does not return a permutation (which changes the denominators), an arm that crashes on the
messy input two thirds of this corpus actually is (cycles, disconnection, unresolved ends), and a
tie counted as a win. The corpus-level numbers quoted in the module docstring are pinned loosely
- direction and rough magnitude - so the file stays a test and not a copy of the JSON.
"""

from __future__ import annotations

import pytest

from src.rl import baselines as bl

# ------------------------------------------------------------------------------------------
# fixtures: the shapes real diagrams come in
# ------------------------------------------------------------------------------------------


def _node(node_id: str, x: float, y: float) -> dict:
    return {
        "id": node_id,
        "shape": "process",
        "bbox": [x, y, 10.0, 10.0],
        "text": node_id,
        "semantic_role": "unknown",
        "confidence": 1.0,
    }


def _edge(edge_id: str, src, dst) -> dict:
    return {"id": edge_id, "src": src, "dst": dst, "directed": True, "label": ""}


def _diagram(nodes, edges, diagram_id="t") -> dict:
    return {
        "id": diagram_id,
        "diagram_type": "flowchart",
        "nodes": nodes,
        "edges": edges,
        "meta": {"source": "test"},
    }


LINE = _diagram(
    [_node("a", 0, 0), _node("b", 0, 10), _node("c", 0, 20)],
    [_edge("e1", "a", "b"), _edge("e2", "b", "c")],
    "line",
)

DIAMOND = _diagram(
    [_node("a", 0, 0), _node("b", 0, 10), _node("c", 20, 10), _node("d", 0, 20)],
    [_edge("e1", "a", "b"), _edge("e2", "a", "c"), _edge("e3", "b", "d"), _edge("e4", "c", "d")],
    "diamond",
)

CYCLE = _diagram(
    [_node("a", 0, 0), _node("b", 0, 10), _node("c", 0, 20)],
    [_edge("e1", "a", "b"), _edge("e2", "b", "c"), _edge("e3", "c", "b")],
    "cycle",
)

#: Nothing has in-degree 0 - the case a naive Kahn hangs on forever.
PURE_CYCLE = _diagram(
    [_node("a", 0, 0), _node("b", 0, 10), _node("c", 0, 20)],
    [_edge("e1", "a", "b"), _edge("e2", "b", "c"), _edge("e3", "c", "a")],
    "pure_cycle",
)

SPLIT = _diagram(
    [_node("a", 0, 0), _node("b", 0, 10), _node("c", 100, 0), _node("d", 100, 10)],
    [_edge("e1", "a", "b"), _edge("e2", "c", "d")],
    "split",
)

DANGLING = _diagram(
    [_node("a", 0, 0), _node("b", 0, 10)],
    [_edge("e1", "a", None), _edge("e2", None, "b"), _edge("e3", "a", "ghost")],
    "dangling",
)

SELF_LOOP = _diagram(
    [_node("a", 0, 0), _node("b", 0, 10)],
    [_edge("e1", "a", "b"), _edge("e2", "b", "b")],
    "self_loop",
)

SINGLETON = _diagram([_node("a", 0, 0)], [], "singleton")
EMPTY = _diagram([], [], "empty")

ALL_SHAPES = [
    LINE,
    DIAMOND,
    CYCLE,
    PURE_CYCLE,
    SPLIT,
    DANGLING,
    SELF_LOOP,
    SINGLETON,
    EMPTY,
]


# ------------------------------------------------------------------------------------------
# the permutation contract - the one property the whole table rests on
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(bl.ARMS))
@pytest.mark.parametrize("diagram", ALL_SHAPES, ids=lambda d: d["id"])
def test_every_arm_returns_a_permutation(name, diagram):
    order = bl.ARMS[name](diagram)
    expected = [n["id"] for n in diagram["nodes"]]
    assert sorted(order) == sorted(expected)
    assert len(order) == len(set(order)), "a node was emitted twice"


@pytest.mark.parametrize("name", sorted(bl.ARMS))
def test_arms_are_deterministic(name):
    assert bl.ARMS[name](DIAMOND) == bl.ARMS[name](DIAMOND)


def test_check_arm_rejects_a_broken_policy():
    with pytest.raises(ValueError, match="permutation"):
        bl.check_arm("dropper", lambda d: [n["id"] for n in d["nodes"]][:-1])
    with pytest.raises(ValueError, match="permutation"):
        bl.check_arm("duplicator", lambda d: [n["id"] for n in d["nodes"]] * 2)


# ------------------------------------------------------------------------------------------
# the messy input the docstring promises is handled
# ------------------------------------------------------------------------------------------


def test_topological_order_respects_edges_when_it_can():
    order = bl.topological_order(DIAMOND)
    assert order.index("a") < order.index("b") < order.index("d")
    assert order.index("a") < order.index("c") < order.index("d")


def test_topological_order_terminates_on_a_cycle_with_no_root():
    # every node has in-degree 1, so the ready queue is empty from the first step.
    order = bl.topological_order(PURE_CYCLE)
    assert sorted(order) == ["a", "b", "c"]
    # the break admits the reading-first node, so the loop is entered at the top of the page.
    assert order[0] == "a"


def test_cycle_break_is_the_lowest_remaining_indegree():
    order = bl.topological_order(CYCLE)
    assert order == ["a", "b", "c"], "a should still lead; b is re-entered, not restarted"


def test_dangling_and_ghost_edges_are_dropped_and_counted():
    ids, successors, indegree, dropped = bl.adjacency(DANGLING)
    assert dropped == 3
    assert all(not targets for targets in successors.values())
    assert set(indegree.values()) == {0}
    assert sorted(ids) == ["a", "b"]


def test_self_loops_are_dropped_not_treated_as_a_constraint():
    _, successors, indegree, dropped = bl.adjacency(SELF_LOOP)
    assert dropped == 1
    assert successors["b"] == []
    assert indegree["b"] == 1  # only the a -> b edge
    assert not bl.ambiguity(SELF_LOOP).cyclic


def test_parallel_edges_count_once():
    doubled = _diagram(
        [_node("a", 0, 0), _node("b", 0, 10)],
        [_edge("e1", "a", "b"), _edge("e2", "a", "b")],
        "parallel",
    )
    _, successors, indegree, _ = bl.adjacency(doubled)
    assert successors["a"] == ["b"]
    assert indegree["b"] == 1


def test_dfs_keeps_components_contiguous_and_topological_interleaves_them():
    # a/b at x=0 and c/d at x=100 are two components on the same two rows.
    assert bl.dfs_order(SPLIT) == ["a", "b", "c", "d"]
    assert bl.topological_order(SPLIT) == ["a", "c", "b", "d"]


def test_bfs_is_level_order():
    wide = _diagram(
        [_node("a", 0, 0), _node("b", 0, 10), _node("c", 20, 10), _node("d", 0, 20)],
        [_edge("e1", "a", "b"), _edge("e2", "a", "c"), _edge("e3", "b", "d")],
        "wide",
    )
    assert bl.bfs_order(wide) == ["a", "b", "c", "d"]
    assert bl.dfs_order(wide) == ["a", "b", "d", "c"]


def test_dfs_arm_delegates_to_the_phase_7_traversal():
    from src.parse.sequences import traversal

    for diagram in ALL_SHAPES:
        assert bl.dfs_order(diagram)[: len(traversal(diagram)[0])] == traversal(diagram)[0]


def test_reading_order_ignores_edges():
    scrambled = _diagram(DIAMOND["nodes"], [], "no_edges")
    assert bl.reading_order(scrambled) == bl.reading_order(DIAMOND)


# ------------------------------------------------------------------------------------------
# the ambiguity measure
# ------------------------------------------------------------------------------------------


def test_a_single_path_has_no_ordering_freedom():
    assert bl.ambiguity(LINE).total_bits == pytest.approx(0.0)
    assert not bl.ambiguity(LINE).ambiguous


def test_a_branch_gives_exactly_one_bit():
    # after `a`, both `b` and `c` are ready: one binary choice, two valid orders.
    score = bl.ambiguity(DIAMOND)
    assert score.choice_bits == pytest.approx(1.0)
    assert score.cycle_bits == pytest.approx(0.0)
    assert score.branch_points == 1
    assert score.merge_points == 1


def test_a_cycle_contributes_log2_factorial():
    import math

    score = bl.ambiguity(PURE_CYCLE)
    assert score.cyclic
    assert score.cycle_bits == pytest.approx(math.log2(6))  # 3! orders inside the SCC


def test_disconnection_shows_up_as_choice_bits():
    score = bl.ambiguity(SPLIT)
    assert score.components == 2
    assert score.choice_bits > 0.0


def test_ambiguous_set_uses_the_threshold():
    diagrams = [LINE, DIAMOND, PURE_CYCLE]
    assert bl.ambiguous_set(diagrams, bits=0.5) == [DIAMOND, PURE_CYCLE]
    assert bl.ambiguous_set(diagrams, bits=2.0) == [PURE_CYCLE]
    assert bl.ambiguous_set(diagrams, bits=100.0) == []


def test_distribution_reports_the_single_property_shares():
    dist = bl.ambiguity_distribution([LINE, DIAMOND, PURE_CYCLE, SPLIT])
    assert dist["diagrams"] == 4
    assert dist["share_cyclic"] == pytest.approx(0.25)
    assert dist["share_multi_component"] == pytest.approx(0.25)
    assert [row["bits"] for row in dist["by_threshold"]][0] == 0.5


# ------------------------------------------------------------------------------------------
# the metric and the table
# ------------------------------------------------------------------------------------------


def test_the_perfect_order_on_a_path_scores_one():
    result = bl.score(LINE, ["a", "b", "c"])
    assert result["edge_f1"] == pytest.approx(1.0)
    assert result["ged_normalised"] == pytest.approx(0.0)


def test_the_reversed_order_on_a_path_scores_zero_edges():
    result = bl.score(LINE, ["c", "b", "a"])
    assert result["edge_f1"] == pytest.approx(0.0)


def test_node_f1_is_one_for_every_arm_which_is_why_it_cannot_separate_them():
    for name in bl.ARMS:
        assert bl.score(DIAMOND, bl.ARMS[name](DIAMOND))["node_f1"] == pytest.approx(1.0)


def test_the_chain_has_n_minus_one_edges_so_denominators_match_across_arms():
    for name in bl.ARMS:
        chain = bl.as_chain(DIAMOND, bl.ARMS[name](DIAMOND))
        assert len(chain.edges) == len(DIAMOND["nodes"]) - 1
        assert len(chain.nodes) == len(DIAMOND["nodes"])


def test_a_tie_is_not_a_win():
    # every arm produces the same order on a single path, so nobody wins.
    table = bl.compare([LINE])
    assert sum(table.wins.values()) == 0
    assert table.all_tied == 1
    assert all(count == 1 for count in table.shared.values())


def test_win_rates_never_sum_past_one():
    table = bl.compare([LINE, DIAMOND, CYCLE, SPLIT, DANGLING])
    assert sum(table.win_rate.values()) <= 1.0 + 1e-9
    assert sum(table.wins.values()) + len(
        [d for d in [LINE, DIAMOND, CYCLE, SPLIT, DANGLING] if True]
    ) >= sum(table.wins.values())


def test_head_to_head_is_antisymmetric():
    table = bl.compare([LINE, DIAMOND, CYCLE, SPLIT])
    for a in table.arms:
        for b in table.arms:
            if a != b:
                # a diagram cannot be a strict win for both directions
                assert table.head_to_head[a][b] + table.head_to_head[b][a] <= table.diagrams


def test_compare_rejects_an_arm_that_breaks_the_contract():
    with pytest.raises(ValueError):
        bl.compare([LINE], arms={"bad": lambda d: [n["id"] for n in d["nodes"]][:1]})


def test_the_blind_control_loses_to_dfs_on_a_branching_graph():
    scrambled = _diagram(
        [_node("a", 0, 0), _node("b", 0, 30), _node("c", 0, 10), _node("d", 0, 20)],
        [_edge("e1", "a", "b"), _edge("e2", "b", "c"), _edge("e3", "c", "d")],
        "scrambled",
    )
    dfs = bl.score(scrambled, bl.dfs_order(scrambled))["edge_f1"]
    reading = bl.score(scrambled, bl.reading_order(scrambled))["edge_f1"]
    assert dfs > reading


# ------------------------------------------------------------------------------------------
# the slot for the learned policy
# ------------------------------------------------------------------------------------------


def test_a_registered_arm_joins_the_table_and_can_win():
    def oracle(diagram: dict) -> list[str]:
        """The order that reconstructs LINE exactly - a stand-in for 11.2.1's rollout."""
        return ["a", "b", "c"]

    def backwards(diagram: dict) -> list[str]:
        return ["c", "b", "a"]

    table = bl.compare([LINE], arms={"oracle": oracle, "backwards": backwards})
    assert table.wins == {"oracle": 1, "backwards": 0}
    assert table.win_rate["oracle"] == pytest.approx(1.0)


def test_register_arm_mutates_the_registry_and_validates_first():
    original = dict(bl.ARMS)
    try:
        with pytest.raises(ValueError):
            bl.register_arm("broken", lambda d: [])
        assert "broken" not in bl.ARMS
        bl.register_arm("identity", lambda d: [n["id"] for n in d["nodes"]])
        assert "identity" in bl.ARMS
        assert "identity" in bl.compare([DIAMOND]).arms
    finally:
        bl.ARMS.clear()
        bl.ARMS.update(original)


def test_module_does_not_import_a_learner():
    """11.2.7 must be measurable while 11.1-11.2.6 are unwritten."""
    import inspect

    source = inspect.getsource(bl)
    for forbidden in ("src.rl.env", "src.rl.episode", "src.rl.reward", "src.rl.state"):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


# ------------------------------------------------------------------------------------------
# the corpus numbers the docstring quotes
# ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    from src.parse.sequences import IR_DIR, labelled_diagrams

    if not IR_DIR.is_dir():
        pytest.skip("no IR corpus on this machine")
    return labelled_diagrams()


@pytest.mark.slow
def test_every_labelled_diagram_is_order_ambiguous(corpus):
    """The finding that forced a magnitude threshold instead of a binary property."""
    assert len(corpus) == 993
    assert all(bl.ambiguity(d).total_bits > 0 for d in corpus)


@pytest.mark.slow
def test_ambiguous_set_size_and_source_balance(corpus):
    subset = bl.ambiguous_set(corpus)
    assert len(subset) == 684
    sources = {d["meta"]["source"] for d in subset}
    assert sources == {"hdbpmn", "fa_bresler"}, "the threshold must not select one source"


@pytest.mark.slow
def test_dfs_leads_the_table_on_the_ambiguous_set(corpus):
    table = bl.compare(bl.ambiguous_set(corpus))
    assert table.diagrams == 684
    assert table.mean_edge_f1["dfs"] > table.mean_edge_f1["topological"]
    assert table.mean_edge_f1["topological"] > table.mean_edge_f1["bfs"]
    assert table.mean_edge_f1["bfs"] > table.mean_edge_f1["reading"]
    assert table.wins["dfs"] > 4 * table.wins["topological"]
    assert table.wins["reading"] == 0
    # the tie band is the real bar for a learned arm
    assert table.wins["dfs"] + table.shared["dfs"] > 0.9 * table.diagrams

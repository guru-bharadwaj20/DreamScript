"""Tests for 11.1.2 - `src.rl.actions`: the flat action space and the hard legality mask.

The mask is the contract 11.2's agents code against, so most of these tests are about what is
*refused* rather than what is allowed. `test_max_branch_covers_the_measured_maximum` is the
regression test for the inherited `MAX_BRANCH = 10`, which was five columns wider than any
diagram in the corpus needs.
"""

from __future__ import annotations

import pytest

from src.rl import actions as A
from src.rl.state import DiagramGraph, TraversalState
from tests.test_rl_state import CYCLE, DIAMOND, LINE, ir


def test_action_space_is_flat_and_nine_wide():
    assert A.MAX_BRANCH == 5
    assert A.N_ACTIONS == 9
    assert len(A.ACTION_NAMES) == A.N_ACTIONS


def test_control_actions_sit_above_the_follow_slots():
    assert A.BACKTRACK == 5
    assert A.MARK_AS_LOOP == 6
    assert A.EMIT_NODE == 7
    assert A.TERMINATE == 8
    assert A.CONTROL_ACTIONS == (A.BACKTRACK, A.MARK_AS_LOOP, A.EMIT_NODE, A.TERMINATE)


def test_action_names_are_unique_and_descriptive():
    assert len(set(A.ACTION_NAMES)) == A.N_ACTIONS
    assert A.action_name(0) == "follow-edge-0"
    assert A.action_name(A.TERMINATE) == "terminate"


@pytest.mark.parametrize("action", [-1, A.N_ACTIONS, 999])
def test_action_name_rejects_out_of_range(action):
    with pytest.raises(ValueError):
        A.action_name(action)


def test_is_follow_covers_exactly_the_follow_slots():
    assert [a for a in range(A.N_ACTIONS) if A.is_follow(a)] == list(range(A.MAX_BRANCH))


def test_max_branch_covers_the_measured_maximum():
    """MAX_BRANCH must be >= the largest out-degree the environment will ever see.

    5 is the measured corpus maximum. A node with 5 successors must have all 5 reachable.
    """
    doc = ir([("a",)] + [(f"n{i}",) for i in range(5)], [("a", f"n{i}") for i in range(5)])
    graph = DiagramGraph.from_ir(doc)
    state = graph.initial_state()
    mask = A.action_mask(graph, state)
    assert sum(mask[: A.MAX_BRANCH]) == 5
    assert all(A.follow_target(graph, state, k) is not None for k in range(5))


# -- follow_target ----------------------------------------------------------------------


def test_follow_target_returns_the_kth_successor():
    graph = DiagramGraph.from_ir(DIAMOND)
    state = graph.initial_state()
    assert A.follow_target(graph, state, 0) == graph.successors[state.current][0]
    assert A.follow_target(graph, state, 1) == graph.successors[state.current][1]


def test_follow_target_is_none_past_the_out_degree():
    graph = DiagramGraph.from_ir(LINE)
    state = graph.initial_state()
    assert A.follow_target(graph, state, 3) is None


def test_follow_target_is_none_for_control_actions():
    graph = DiagramGraph.from_ir(LINE)
    state = graph.initial_state()
    for action in A.CONTROL_ACTIONS:
        assert A.follow_target(graph, state, action) is None


# -- the mask ---------------------------------------------------------------------------


def test_mask_length_is_the_action_space():
    graph = DiagramGraph.from_ir(DIAMOND)
    assert len(A.action_mask(graph, graph.initial_state())) == A.N_ACTIONS


def test_terminate_is_always_legal():
    """No state may be a dead end - this is what stops an episode from hanging."""
    for doc in (LINE, DIAMOND, CYCLE, {"nodes": [], "edges": []}):
        graph = DiagramGraph.from_ir(doc)
        for node in range(max(1, graph.n_nodes)):
            state = TraversalState(current=min(node, max(0, graph.n_nodes - 1)))
            assert A.action_mask(graph, state)[A.TERMINATE]


def test_legal_actions_is_never_empty():
    graph = DiagramGraph.from_ir({"nodes": [], "edges": []})
    assert A.legal_actions(graph, graph.initial_state()) == [A.TERMINATE]


def test_empty_diagram_allows_only_terminate():
    graph = DiagramGraph.from_ir({"nodes": [], "edges": []})
    mask = A.action_mask(graph, graph.initial_state())
    assert sum(mask) == 1 and mask[A.TERMINATE]


def test_follow_is_illegal_past_the_out_degree():
    graph = DiagramGraph.from_ir(LINE)
    mask = A.action_mask(graph, graph.initial_state())
    assert mask[0] is True
    assert all(not mask[k] for k in range(1, A.MAX_BRANCH))


def test_backtrack_needs_a_stack():
    graph = DiagramGraph.from_ir(LINE)
    empty = graph.initial_state()
    assert not A.action_mask(graph, empty)[A.BACKTRACK]
    with_stack = TraversalState(current=1, stack=(0,))
    assert A.action_mask(graph, with_stack)[A.BACKTRACK]


def test_emit_is_illegal_once_emitted():
    graph = DiagramGraph.from_ir(LINE)
    state = graph.initial_state()
    assert A.action_mask(graph, state)[A.EMIT_NODE]
    already = TraversalState(current=state.current, emitted=1 << state.current)
    assert not A.action_mask(graph, already)[A.EMIT_NODE]


def test_legal_actions_matches_the_mask():
    graph = DiagramGraph.from_ir(DIAMOND)
    state = graph.initial_state()
    mask = A.action_mask(graph, state)
    assert A.legal_actions(graph, state) == [i for i, ok in enumerate(mask) if ok]


def test_legal_actions_are_sorted():
    graph = DiagramGraph.from_ir(DIAMOND)
    actions = A.legal_actions(graph, graph.initial_state())
    assert actions == sorted(actions)


# -- mark-as-loop, the subtle one -------------------------------------------------------


def test_mark_as_loop_needs_a_target_on_the_stack():
    """7.3.3's definition: a back edge targets a node on the *current* stack, not any finished one.

    Re-convergence after a branch is not a loop, and the mask must refuse it rather than let the
    reward pay for it.
    """
    graph = DiagramGraph.from_ir(DIAMOND)
    join = graph.index_of["d"]
    state = TraversalState(current=join, visited=graph.full_mask)
    assert not A.action_mask(graph, state)[A.MARK_AS_LOOP]


def test_mark_as_loop_is_legal_on_a_real_back_edge():
    graph = DiagramGraph.from_ir(CYCLE)
    b, c = graph.index_of["b"], graph.index_of["c"]
    state = TraversalState(current=c, stack=(graph.index_of["a"], b), visited=graph.full_mask)
    assert A.loop_candidates(graph, state) == (b,)
    assert A.action_mask(graph, state)[A.MARK_AS_LOOP]


def test_mark_as_loop_is_illegal_twice_on_the_same_node():
    graph = DiagramGraph.from_ir(CYCLE)
    b, c = graph.index_of["b"], graph.index_of["c"]
    state = TraversalState(
        current=c, stack=(graph.index_of["a"], b), visited=graph.full_mask, loop_marked=1 << c
    )
    assert not A.action_mask(graph, state)[A.MARK_AS_LOOP]


def test_self_loop_is_a_loop_candidate():
    doc = ir([("a",), ("b",)], [("a", "a"), ("a", "b")])
    graph = DiagramGraph.from_ir(doc)
    a = graph.index_of["a"]
    state = TraversalState(current=a, visited=1 << a)
    assert a in A.loop_candidates(graph, state)


def test_loop_candidates_is_empty_on_an_acyclic_diagram():
    graph = DiagramGraph.from_ir(LINE)
    assert A.loop_candidates(graph, graph.initial_state()) == ()

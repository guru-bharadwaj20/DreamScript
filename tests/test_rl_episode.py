"""Tests for 11.1.4 - `src.rl.episode`: the transition function and the three terminals.

The row's definition of done is "env terminates correctly", so the load-bearing tests here are
`test_every_episode_stops_*`: a random policy and a TERMINATE-avoiding policy are both driven to
completion on every fixture, including cyclic and disconnected ones.
"""

from __future__ import annotations

import random

import pytest

from src.rl import actions as A
from src.rl.episode import STEP_CAP_INTERCEPT, STEP_CAP_SLOPE, Episode, Outcome, step_cap
from src.rl.state import DiagramGraph
from tests.test_rl_state import CYCLE, DIAMOND, LINE, ir

DISCONNECTED = ir([("a",), ("b",), ("c",), ("d",)], [("a", "b"), ("c", "d")])
SELF_LOOP = ir([("a",), ("b",)], [("a", "a"), ("a", "b")])
SINGLE = ir([("a",)], [])
FIXTURES = [LINE, DIAMOND, CYCLE, DISCONNECTED, SELF_LOOP, SINGLE]


def episode_of(doc):
    return Episode(DiagramGraph.from_ir(doc))


# -- the step cap -----------------------------------------------------------------------


def test_step_cap_scales_with_the_diagram():
    assert step_cap(DiagramGraph.from_ir(LINE)) == STEP_CAP_SLOPE * 3 + STEP_CAP_INTERCEPT
    assert step_cap(DiagramGraph.from_ir(DIAMOND)) == STEP_CAP_SLOPE * 4 + STEP_CAP_INTERCEPT


def test_step_cap_is_positive_on_an_empty_diagram():
    assert step_cap(DiagramGraph.from_ir({"nodes": [], "edges": []})) > 0


def test_explicit_cap_overrides_the_formula():
    graph = DiagramGraph.from_ir(LINE)
    assert Episode(graph, cap=7).cap == 7


# -- transitions ------------------------------------------------------------------------


def test_emit_sets_the_bit_and_extends_the_sequence():
    episode = episode_of(LINE)
    start = episode.state.current
    outcome = episode.apply(A.EMIT_NODE)
    assert outcome.legal and outcome.emitted_node == start
    assert episode.state.has_emitted(start)
    assert episode.state.emit_sequence == (start,)


def test_follow_moves_and_pushes_the_stack():
    episode = episode_of(LINE)
    start = episode.state.current
    target = episode.graph.successors[start][0]
    outcome = episode.apply(0)
    assert outcome.moved_to == target
    assert episode.state.current == target
    assert episode.state.stack == (start,)
    assert episode.state.has_visited(target)


def test_backtrack_pops_the_stack():
    episode = episode_of(LINE)
    start = episode.state.current
    episode.apply(0)
    outcome = episode.apply(A.BACKTRACK)
    assert outcome.backtracked_to == start
    assert episode.state.current == start
    assert episode.state.stack == ()


def test_mark_as_loop_sets_the_bit():
    graph = DiagramGraph.from_ir(CYCLE)
    episode = Episode(graph)
    while not episode.mask()[A.MARK_AS_LOOP]:
        legal = [a for a in episode.legal_actions() if A.is_follow(a)]
        if not legal:
            pytest.skip("no back edge reachable by following successors")
        episode.apply(legal[0])
    node = episode.state.current
    outcome = episode.apply(A.MARK_AS_LOOP)
    assert outcome.marked_loop == node
    assert episode.state.is_loop_marked(node)


def test_terminate_stops_the_episode():
    episode = episode_of(LINE)
    outcome = episode.apply(A.TERMINATE)
    assert outcome.terminated_by_policy
    assert episode.terminated() and not episode.truncated()
    assert episode.stopped_by == "policy"


def test_state_object_is_replaced_not_mutated():
    """Agents hold on to states; `apply` must not change one under them."""
    episode = episode_of(LINE)
    before = episode.state
    episode.apply(A.EMIT_NODE)
    assert episode.state is not before
    assert before.emitted == 0


def test_every_step_increments_the_counter():
    episode = episode_of(DIAMOND)
    for expected in (1, 2, 3):
        episode.apply(A.EMIT_NODE if expected == 1 else 0)
        assert episode.state.steps == expected


# -- illegal actions --------------------------------------------------------------------


def test_illegal_action_is_refused_not_raised():
    episode = episode_of(LINE)
    episode.apply(A.EMIT_NODE)
    outcome = episode.apply(A.EMIT_NODE)
    assert outcome.legal is False
    assert outcome.duplicate_emission is True


def test_illegal_action_leaves_the_state_alone_except_the_counter():
    episode = episode_of(LINE)
    episode.apply(A.EMIT_NODE)
    before = episode.state
    episode.apply(A.EMIT_NODE)
    after = episode.state
    assert (after.current, after.visited, after.emitted, after.loop_marked, after.stack) == (
        before.current,
        before.visited,
        before.emitted,
        before.loop_marked,
        before.stack,
    )
    assert after.steps == before.steps + 1


def test_duplicate_emissions_are_counted():
    episode = episode_of(LINE)
    episode.apply(A.EMIT_NODE)
    episode.apply(A.EMIT_NODE)
    episode.apply(A.EMIT_NODE)
    assert episode.state.duplicate_emissions == 2


def test_strict_mode_raises_on_an_illegal_action():
    episode = episode_of(LINE)
    episode.apply(A.EMIT_NODE)
    with pytest.raises(ValueError):
        episode.apply(A.EMIT_NODE, strict=True)


@pytest.mark.parametrize("action", [-1, A.N_ACTIONS, 500])
def test_out_of_range_actions_always_raise(action):
    with pytest.raises(ValueError):
        episode_of(LINE).apply(action)


def test_apply_after_done_raises():
    episode = episode_of(LINE)
    episode.apply(A.TERMINATE)
    with pytest.raises(RuntimeError):
        episode.apply(A.TERMINATE)


# -- termination ------------------------------------------------------------------------


def test_full_coverage_needs_emission_not_just_visiting():
    """The plan's "full coverage" is read as emitted; a walked-over node is not in the code."""
    episode = episode_of(LINE)
    for _ in range(2):
        episode.apply(0)
    assert episode.state.visited == episode.graph.full_mask
    assert not episode.full_coverage()


def test_full_coverage_terminates_the_episode():
    episode = episode_of(SINGLE)
    episode.apply(A.EMIT_NODE)
    assert episode.full_coverage()
    assert episode.terminated()
    assert episode.stopped_by == "coverage"


def test_truncation_is_not_termination():
    graph = DiagramGraph.from_ir(CYCLE)
    episode = Episode(graph, cap=3)
    while not episode.done():
        legal = [a for a in episode.legal_actions() if a != A.TERMINATE] or [A.TERMINATE]
        episode.apply(legal[0])
    if episode.truncated():
        assert episode.stopped_by == "cap"
        assert not episode.terminated()


def test_terminated_and_truncated_are_mutually_exclusive():
    rng = random.Random(0)
    for doc in FIXTURES:
        for _ in range(20):
            episode = episode_of(doc)
            while not episode.done():
                episode.apply(rng.choice(episode.legal_actions()))
            assert not (episode.terminated() and episode.truncated())


@pytest.mark.parametrize("seed", range(5))
def test_every_episode_stops_under_random_play(seed):
    rng = random.Random(seed)
    for doc in FIXTURES:
        episode = episode_of(doc)
        for _ in range(10_000):
            if episode.done():
                break
            episode.apply(rng.choice(episode.legal_actions()))
        assert episode.done(), "episode never stopped"
        assert episode.stopped_by in {"coverage", "policy", "cap"}


@pytest.mark.parametrize("seed", range(5))
def test_every_episode_stops_when_the_policy_avoids_terminate(seed):
    """The step cap is the only thing that stops a policy that refuses to terminate."""
    rng = random.Random(seed)
    for doc in FIXTURES:
        episode = episode_of(doc)
        for _ in range(10_000):
            if episode.done():
                break
            legal = [a for a in episode.legal_actions() if a != A.TERMINATE] or [A.TERMINATE]
            episode.apply(rng.choice(legal))
        assert episode.done()
        assert episode.state.steps <= episode.cap


def test_steps_never_exceed_the_cap():
    rng = random.Random(3)
    for doc in FIXTURES:
        episode = episode_of(doc)
        while not episode.done():
            episode.apply(rng.choice(episode.legal_actions()))
        assert episode.state.steps <= episode.cap


# -- gold play --------------------------------------------------------------------------


def test_gold_actions_do_not_disturb_the_live_episode():
    episode = episode_of(DIAMOND)
    before = episode.state
    episode.gold_actions()
    assert episode.state is before
    assert episode.state.steps == 0


def test_gold_actions_end_in_terminate():
    for doc in FIXTURES:
        assert episode_of(doc).gold_actions()[-1] == A.TERMINATE


def test_gold_actions_are_all_legal_when_replayed():
    for doc in FIXTURES:
        episode = episode_of(doc)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action, strict=True)


def test_gold_play_covers_a_connected_acyclic_diagram():
    for doc in (LINE, DIAMOND, SINGLE):
        episode = episode_of(doc)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action)
        assert episode.full_coverage(), "gold play must cover a reachable diagram"


def test_gold_play_cannot_cover_a_disconnected_diagram():
    """There is no jump action, so a second component is unreachable. This is the 63.06% ceiling.

    Recorded as a test rather than left implicit, because it is the reason 11.2's coverage numbers
    have a ceiling below 1 and must not be read as a policy failure.
    """
    episode = episode_of(DISCONNECTED)
    for action in episode.gold_actions():
        if episode.done():
            break
        episode.apply(action)
    assert not episode.full_coverage()
    assert episode.state.n_emitted() < episode.graph.n_nodes


def test_gold_play_never_hits_the_step_cap():
    for doc in FIXTURES:
        episode = episode_of(doc)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action)
        assert not episode.truncated()


def test_gold_play_handles_a_self_loop():
    episode = episode_of(SELF_LOOP)
    for action in episode.gold_actions():
        if episode.done():
            break
        episode.apply(action)
    assert episode.full_coverage()


# -- bookkeeping ------------------------------------------------------------------------


def test_history_records_every_action():
    episode = episode_of(LINE)
    episode.apply(A.EMIT_NODE)
    episode.apply(0)
    assert len(episode.history) == 2
    assert all(isinstance(o, Outcome) for o in episode.history)
    assert episode.history[0].name == "emit-node"


def test_outcome_name_matches_the_action():
    outcome = episode_of(LINE).apply(A.TERMINATE)
    assert outcome.name == A.action_name(A.TERMINATE)


def test_empty_diagram_terminates_immediately_on_terminate():
    episode = Episode(DiagramGraph.from_ir({"nodes": [], "edges": []}))
    assert episode.legal_actions() == [A.TERMINATE]
    episode.apply(A.TERMINATE)
    assert episode.done()

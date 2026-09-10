"""Tests for 11.1.3 - `src.rl.reward`: the terms, their signs, and the ranking they induce.

`test_gold_order_beats_its_own_reverse` is the regression test for the inherited bug: the reward
scored a reversed traversal *above* the gold one, because `src.rl.emit` counted successfully
built structure as an error. Everything else here exists to keep a term from silently changing
sign under a future sweep.
"""

from __future__ import annotations

import random

import pytest

from src.rl import actions as A
from src.rl.emit import emit_code, order_score
from src.rl.episode import Episode
from src.rl.reward import (
    DEFAULT,
    SEMANTIC_WEIGHTS,
    RewardConfig,
    loop_score,
    sandbox_hook,
    semantic_score,
    step_reward,
    structure_score,
    terminal_reward,
)
from src.rl.state import DiagramGraph, TraversalState
from tests.test_rl_episode import DISCONNECTED, SELF_LOOP, SINGLE
from tests.test_rl_state import CYCLE, DIAMOND, LINE

FIXTURES = [LINE, DIAMOND, CYCLE, DISCONNECTED, SELF_LOOP, SINGLE]


def played(doc):
    """Run gold play to its terminal and return `(graph, episode)`."""
    graph = DiagramGraph.from_ir(doc)
    episode = Episode(graph)
    for action in episode.gold_actions():
        if episode.done():
            break
        episode.apply(action)
    return graph, episode


def state_with(graph, sequence, visited=None):
    emitted = 0
    for node in sequence:
        emitted |= 1 << node
    return TraversalState(
        current=sequence[-1] if sequence else 0,
        visited=graph.full_mask if visited is None else visited,
        emitted=emitted,
        emit_sequence=tuple(sequence),
        steps=len(sequence),
    )


# -- configuration ----------------------------------------------------------------------


def test_semantic_weights_sum_to_one():
    """`semantic_score` must stay in [0, 1] so `semantic_bonus` is the whole range of the bonus."""
    assert sum(SEMANTIC_WEIGHTS.values()) == pytest.approx(1.0)


def test_penalties_are_negative_and_rewards_positive():
    assert DEFAULT.valid_code > 0 and DEFAULT.semantic_bonus > 0
    for field in (
        "unreachable_node",
        "duplicate_emission",
        "infinite_loop",
        "illegal_action",
        "step_cost",
    ):
        assert getattr(DEFAULT, field) < 0, field


def test_replace_produces_an_independent_config():
    tweaked = DEFAULT.replace(semantic_bonus=9.0)
    assert tweaked.semantic_bonus == 9.0
    assert DEFAULT.semantic_bonus == 4.0
    assert tweaked.valid_code == DEFAULT.valid_code


# -- step reward ------------------------------------------------------------------------


def test_a_legal_step_costs_only_the_step_cost():
    graph = DiagramGraph.from_ir(LINE)
    episode = Episode(graph)
    outcome = episode.apply(A.EMIT_NODE)
    assert step_reward(graph, outcome) == pytest.approx(DEFAULT.step_cost)


def test_an_illegal_step_costs_more_than_a_legal_one():
    graph = DiagramGraph.from_ir(LINE)
    episode = Episode(graph)
    legal = step_reward(graph, episode.apply(A.EMIT_NODE))
    illegal = step_reward(graph, episode.apply(A.EMIT_NODE))
    assert illegal < legal


def test_a_duplicate_emission_is_charged_on_top_of_the_illegal_penalty():
    graph = DiagramGraph.from_ir(LINE)
    episode = Episode(graph)
    episode.apply(A.EMIT_NODE)
    duplicate = step_reward(graph, episode.apply(A.EMIT_NODE))
    expected = DEFAULT.step_cost + DEFAULT.illegal_action + DEFAULT.duplicate_emission
    assert duplicate == pytest.approx(expected)


def test_step_reward_never_touches_the_sandbox():
    """A per-step sandbox call would cost 71.4 ms; the hook must not fire from `step_reward`."""
    calls = []

    def spy(code):
        calls.append(code)
        return {"passed": True}

    graph = DiagramGraph.from_ir(LINE)
    episode = Episode(graph)
    step_reward(graph, episode.apply(A.EMIT_NODE), RewardConfig(sandbox=spy))
    assert calls == []


# -- the component scores ---------------------------------------------------------------


def test_structure_score_is_zero_without_any_emission():
    graph = DiagramGraph.from_ir(LINE)
    assert structure_score(graph, TraversalState(current=0)) == 0.0


def test_structure_score_is_perfect_on_a_complete_emission():
    graph = DiagramGraph.from_ir(LINE)
    assert structure_score(graph, state_with(graph, list(graph.gold_order))) == 1.0


def test_structure_score_punishes_a_partial_emission():
    """An emitted node whose successor was never emitted leaves an undefined name."""
    graph = DiagramGraph.from_ir(LINE)
    partial = state_with(graph, [graph.index_of["a"]])
    assert structure_score(graph, partial) < 1.0


def test_structure_score_does_not_punish_building_structure():
    """The inherited emitter left `_cond`/`_again` undefined, so opening an `if` was an error.

    On the diamond, gold play builds a branch. Its structure score must be perfect, not degraded.
    """
    graph = DiagramGraph.from_ir(DIAMOND)
    assert structure_score(graph, state_with(graph, list(graph.gold_order))) == 1.0


def test_loop_score_is_one_when_there_are_no_loops():
    graph = DiagramGraph.from_ir(LINE)
    assert loop_score(graph, TraversalState(current=0)) == 1.0


def test_loop_score_rewards_closing_a_back_edge():
    graph = DiagramGraph.from_ir(CYCLE)
    sources = {a for a, _ in graph.back_edges}
    assert sources, "fixture must contain a back edge"
    unmarked = TraversalState(current=0)
    marked = TraversalState(current=0, loop_marked=sum(1 << s for s in sources))
    assert loop_score(graph, marked) > loop_score(graph, unmarked)
    assert loop_score(graph, marked) == 1.0


def test_order_score_ranks_gold_above_its_reverse():
    for doc in (LINE, DIAMOND):
        graph = DiagramGraph.from_ir(doc)
        gold = list(graph.gold_order)
        assert order_score(graph, gold) > order_score(graph, gold[::-1])


def test_order_score_is_one_when_no_forward_edge_is_spanned():
    graph = DiagramGraph.from_ir(SINGLE)
    assert order_score(graph, [0]) == 1.0


def test_semantic_score_is_in_the_unit_interval():
    rng = random.Random(0)
    for doc in FIXTURES:
        graph = DiagramGraph.from_ir(doc)
        for _ in range(15):
            episode = Episode(graph)
            while not episode.done():
                episode.apply(rng.choice(episode.legal_actions()))
            score, _detail = semantic_score(graph, episode.state)
            assert 0.0 <= score <= 1.0


def test_semantic_detail_reports_every_term():
    graph = DiagramGraph.from_ir(DIAMOND)
    _score, detail = semantic_score(graph, state_with(graph, list(graph.gold_order)))
    assert set(detail) >= {"coverage", "structure", "ordering", "loops", "source"}
    assert detail["source"] == "static-proxy"


# -- the terminal reward ----------------------------------------------------------------


def test_terminal_reward_returns_a_full_breakdown():
    graph, episode = played(LINE)
    reward, info = terminal_reward(graph, episode.state, episode.truncated())
    assert info["terminal_reward"] == pytest.approx(reward)
    for key in (
        "syntactic",
        "semantic_bonus",
        "semantic_score",
        "n_unreachable",
        "unreachable_penalty",
        "n_duplicate_emissions",
        "duplicate_penalty",
        "infinite_loop_penalty",
        "parse_ok",
    ):
        assert key in info


def test_syntactic_point_is_paid_on_valid_code():
    graph, episode = played(LINE)
    _reward, info = terminal_reward(graph, episode.state, False)
    assert info["parse_ok"] is True
    assert info["syntactic"] == pytest.approx(DEFAULT.valid_code)


def test_unreachable_nodes_are_charged():
    graph = DiagramGraph.from_ir(DISCONNECTED)
    covered = state_with(graph, [0, 1], visited=0b0011)
    _reward, info = terminal_reward(graph, covered, False)
    assert info["n_unreachable"] == 2
    assert info["unreachable_penalty"] == pytest.approx(2 * DEFAULT.unreachable_node)


def test_duplicate_emissions_are_charged():
    graph = DiagramGraph.from_ir(LINE)
    state = TraversalState(
        current=0, visited=graph.full_mask, emitted=0b001, emit_sequence=(0,), duplicate_emissions=3
    )
    _reward, info = terminal_reward(graph, state, False)
    assert info["n_duplicate_emissions"] == 3
    assert info["duplicate_penalty"] == pytest.approx(3 * DEFAULT.duplicate_emission)


def test_truncation_is_charged_as_an_infinite_loop():
    graph, episode = played(LINE)
    clean, _ = terminal_reward(graph, episode.state, False)
    looped, info = terminal_reward(graph, episode.state, True)
    assert info["infinite_loop_penalty"] == pytest.approx(DEFAULT.infinite_loop)
    assert looped < clean


def test_terminal_reward_is_finite_under_random_play():
    rng = random.Random(1)
    for doc in FIXTURES:
        graph = DiagramGraph.from_ir(doc)
        for _ in range(15):
            episode = Episode(graph)
            while not episode.done():
                episode.apply(rng.choice(episode.legal_actions()))
            reward, _info = terminal_reward(graph, episode.state, episode.truncated())
            assert reward == reward and abs(reward) < 1e6


def test_terminal_reward_on_an_empty_diagram_does_not_crash():
    graph = DiagramGraph.from_ir({"nodes": [], "edges": []})
    reward, _info = terminal_reward(graph, graph.initial_state(), False)
    assert reward == reward


# -- the ranking the reward must induce --------------------------------------------------


def test_gold_order_beats_its_own_reverse():
    """The inherited reward scored the reverse *above* gold (0.9997 vs 0.8915 on structure).

    Coverage is held identical between the two states, so this isolates ordering alone. This is
    the single most important property of the row: a reward that prefers a reversed traversal
    would teach 11.2's agents to emit backwards.
    """
    for doc in (LINE, DIAMOND):
        graph = DiagramGraph.from_ir(doc)
        gold = list(graph.gold_order)
        forward, _ = terminal_reward(graph, state_with(graph, gold), False)
        backward, _ = terminal_reward(graph, state_with(graph, gold[::-1]), False)
        assert forward > backward, doc


def test_full_emission_beats_a_partial_one():
    for doc in (LINE, DIAMOND):
        graph = DiagramGraph.from_ir(doc)
        gold = list(graph.gold_order)
        full, _ = terminal_reward(graph, state_with(graph, gold), False)
        half, _ = terminal_reward(graph, state_with(graph, gold[: max(1, len(gold) // 2)]), False)
        assert full > half


def test_gold_play_beats_random_play_on_average():
    rng = random.Random(0)
    for doc in (LINE, DIAMOND, CYCLE):
        graph, episode = played(doc)
        gold_reward, _ = terminal_reward(graph, episode.state, episode.truncated())
        rewards = []
        for _ in range(25):
            rollout = Episode(graph)
            while not rollout.done():
                rollout.apply(rng.choice(rollout.legal_actions()))
            reward, _info = terminal_reward(graph, rollout.state, rollout.truncated())
            rewards.append(reward)
        assert gold_reward > sum(rewards) / len(rewards)


def test_closing_a_loop_pays_more_than_ignoring_it():
    graph = DiagramGraph.from_ir(CYCLE)
    gold = list(graph.gold_order)
    sources = {a for a, _ in graph.back_edges}
    plain = state_with(graph, gold)
    marked = TraversalState(
        current=plain.current,
        visited=plain.visited,
        emitted=plain.emitted,
        emit_sequence=plain.emit_sequence,
        loop_marked=sum(1 << s for s in sources),
    )
    assert terminal_reward(graph, marked, False)[0] > terminal_reward(graph, plain, False)[0]


# -- the sandbox hook -------------------------------------------------------------------


def test_sandbox_hook_replaces_the_static_proxy():
    graph, episode = played(LINE)
    config = RewardConfig(sandbox=lambda code: {"passed": True})
    score, detail = semantic_score(graph, episode.state, config)
    assert score == 1.0
    assert detail["source"] == "sandbox"


def test_a_failing_sandbox_verdict_scores_zero():
    graph, episode = played(LINE)
    config = RewardConfig(sandbox=lambda code: {"passed": False, "kind": "exception"})
    score, detail = semantic_score(graph, episode.state, config)
    assert score == 0.0
    assert detail["sandbox"]["kind"] == "exception"


def test_a_raising_sandbox_does_not_kill_the_episode():
    """A sandbox failure mid-sweep must degrade to the static proxy, not crash training."""

    def boom(code):
        raise RuntimeError("sandbox died")

    graph, episode = played(LINE)
    score, detail = semantic_score(graph, episode.state, RewardConfig(sandbox=boom))
    assert 0.0 <= score <= 1.0
    assert "sandbox_error" in detail
    assert detail["source"] == "static-proxy"


def test_sandbox_receives_the_emitted_code():
    seen = {}

    def spy(code):
        seen["code"] = code
        return {"passed": True}

    graph, episode = played(LINE)
    semantic_score(graph, episode.state, RewardConfig(sandbox=spy))
    marked = {i for i in range(graph.n_nodes) if episode.state.is_loop_marked(i)}
    assert seen["code"] == emit_code(graph, list(episode.state.emit_sequence), marked)


@pytest.mark.slow
def test_real_sandbox_runs_complete_code_and_rejects_partial_code():
    """The wiring 11.1.3 asks for, end to end. ~2 subprocess spawns, 71.4 ms each."""
    graph, episode = played(LINE)
    hook = sandbox_hook(timeout_s=15)

    complete = hook(emit_code(graph, list(episode.state.emit_sequence)))
    assert complete["passed"] is True
    assert complete["kind"] == "ok"

    partial = hook(emit_code(graph, [graph.index_of["a"]]))
    assert partial["passed"] is False
    assert partial["kind"] == "exception"

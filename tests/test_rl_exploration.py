"""Phase 11.2.3 - the three exploration rules.

An exploration rule that returns an illegal action does not crash this environment: `apply`
charges the illegal-action penalty and carries on, so the damage shows up only as a worse number
much later. Legality is therefore asserted under thousands of draws rather than spot-checked.
The other two risks pinned here are a softmax that overflows on this corpus's real reward range
(terminal rewards run below -40) and a UCB that gives `inf` to every untried action and then
breaks the tie by index, which silently reintroduces the positional bias.
"""

from __future__ import annotations

import math
import random

import pytest

from src.rl import actions as A
from src.rl import exploration as X
from src.rl import qlearning as Q
from src.rl.episode import Episode
from src.rl.state import DiagramGraph
from tests.test_rl_qlearning import RAW


@pytest.fixture(scope="module")
def graphs() -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in RAW]


def _agent(rule: str, **kwargs) -> Q.TabularAgent:
    return Q.TabularAgent(Q.TrainConfig(explore=rule, **kwargs))


# ------------------------------------------------------------------------------------------
# legality - the property that fails silently
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rule", X.RULES)
def test_every_rule_only_ever_returns_a_legal_action(rule, graphs):
    agent = _agent(rule)
    agent.q[:] = 3.0  # a table with opinions, so no rule falls back to a uniform draw
    rng = random.Random(0)
    draws = 0
    for graph in graphs:
        episode = Episode(graph)
        while not episode.done():
            mask = episode.mask()
            key = Q.key_index(Q.abstract(graph, episode.state))
            for _ in range(200):
                assert mask[X.select(agent, key, mask, rng, 0.5)]
                draws += 1
            episode.apply(rng.choice(episode.legal_actions()))
    assert draws >= 1000, draws


def test_an_unknown_rule_is_an_error_and_not_a_silent_fallback(graphs):
    agent = _agent("egreedy")
    agent.cfg.explore = "boltzmann-ish"
    graph = graphs[0]
    state = Episode(graph).state
    with pytest.raises(ValueError):
        X.select(agent, 0, A.action_mask(graph, state), random.Random(0), 0.1)


def test_a_state_with_one_legal_action_returns_it_without_arithmetic(graphs):
    agent = _agent("ucb")
    mask = [False] * A.N_ACTIONS
    mask[A.TERMINATE] = True
    rng = random.Random(0)
    assert X.ucb_action(agent, 0, mask, rng, 1.0) == A.TERMINATE
    assert X.softmax_action(agent, 0, mask, rng, 1.0) == A.TERMINATE


# ------------------------------------------------------------------------------------------
# softmax
# ------------------------------------------------------------------------------------------


def test_softmax_survives_the_reward_range_this_corpus_actually_produces():
    """A 90-node page with everything unreachable is -45 before any other term."""
    agent = _agent("softmax", temperature=0.25)
    agent.q[0, A.EMIT_NODE] = 45.0
    agent.q[0, A.BACKTRACK] = -45.0
    mask = [True] * A.N_ACTIONS
    rng = random.Random(0)
    picks = [X.softmax_action(agent, 0, mask, rng, 0.25) for _ in range(200)]
    assert set(picks) == {A.EMIT_NODE}
    assert all(math.isfinite(v) for v in agent.q[0])


def test_softmax_prefers_the_higher_value_action_but_does_not_only_pick_it():
    agent = _agent("softmax")
    agent.q[0, A.EMIT_NODE] = 1.0
    agent.q[0, A.BACKTRACK] = 0.0
    mask = [False] * A.N_ACTIONS
    mask[A.EMIT_NODE] = mask[A.BACKTRACK] = True
    rng = random.Random(3)
    picks = [X.softmax_action(agent, 0, mask, rng, 1.0) for _ in range(2000)]
    share = picks.count(A.EMIT_NODE) / len(picks)
    assert 0.6 < share < 0.85  # exp(1) / (exp(1) + 1) = 0.731
    assert picks.count(A.BACKTRACK) > 0


def test_a_low_temperature_is_greedy_and_a_high_one_is_nearly_uniform():
    agent = _agent("softmax")
    agent.q[0, A.EMIT_NODE] = 1.0
    mask = [False] * A.N_ACTIONS
    mask[A.EMIT_NODE] = mask[A.BACKTRACK] = True
    cold = [X.softmax_action(agent, 0, mask, random.Random(s), 0.01) for s in range(100)]
    hot = [X.softmax_action(agent, 0, mask, random.Random(s), 50.0) for s in range(400)]
    assert set(cold) == {A.EMIT_NODE}
    assert 0.4 < hot.count(A.EMIT_NODE) / len(hot) < 0.6


# ------------------------------------------------------------------------------------------
# UCB
# ------------------------------------------------------------------------------------------


def test_ucb_tries_every_untried_legal_action_before_any_tried_one():
    agent = _agent("ucb")
    mask = [False] * A.N_ACTIONS
    mask[A.EMIT_NODE] = mask[A.BACKTRACK] = mask[A.TERMINATE] = True
    agent.counts[0, A.EMIT_NODE] = 50
    agent.q[0, A.EMIT_NODE] = 99.0
    rng = random.Random(0)
    picks = {X.ucb_action(agent, 0, mask, rng, 1.0) for _ in range(60)}
    assert picks == {A.BACKTRACK, A.TERMINATE}


def test_ucb_untried_actions_are_not_separated_by_index(graphs):
    agent = _agent("ucb")
    mask = [False] * A.N_ACTIONS
    mask[A.BACKTRACK] = mask[A.MARK_AS_LOOP] = mask[A.TERMINATE] = True
    picks = [X.ucb_action(agent, 0, mask, random.Random(s), 1.0) for s in range(120)]
    assert len(set(picks)) == 3


def test_the_bonus_lifts_a_rarely_tried_action_over_a_slightly_better_one():
    agent = _agent("ucb")
    mask = [False] * A.N_ACTIONS
    mask[A.EMIT_NODE] = mask[A.BACKTRACK] = True
    agent.counts[0, A.EMIT_NODE] = 10_000
    agent.counts[0, A.BACKTRACK] = 1
    agent.q[0, A.EMIT_NODE] = 1.0
    agent.q[0, A.BACKTRACK] = 0.5
    assert X.ucb_action(agent, 0, mask, random.Random(0), 2.0) == A.BACKTRACK
    assert X.ucb_action(agent, 0, mask, random.Random(0), 0.01) == A.EMIT_NODE


# ------------------------------------------------------------------------------------------
# the rules drive real training, and the study runs
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rule", X.RULES)
def test_each_rule_trains_end_to_end_and_reaches_keys(rule, graphs):
    agent = Q.train(graphs, Q.TrainConfig(explore=rule, episodes=200, seed=0))
    assert agent.report["reached_keys"] > 0
    assert len(agent.history["reward"]) == 200


def test_the_three_rules_do_not_produce_the_same_table(graphs):
    tables = [
        Q.train(graphs, Q.TrainConfig(explore=rule, episodes=250, seed=2)).q for rule in X.RULES
    ]
    assert not (tables[0] == tables[1]).all()
    assert not (tables[1] == tables[2]).all()


def test_the_greedy_probe_fires_on_schedule_and_on_the_last_episode(graphs):
    probe = X.GreedyProbe(graphs, every=40, episodes=100)
    Q.train(graphs, Q.TrainConfig(episodes=100, seed=0, on_episode=probe))
    assert [p["episode"] for p in probe.result()] == [40, 80, 100]


def test_the_probe_does_not_perturb_training(graphs):
    """It reads the table between episodes; the trained table must be bit-identical without it."""
    plain = Q.train(graphs, Q.TrainConfig(episodes=150, seed=3))
    probed = Q.train(
        graphs, Q.TrainConfig(episodes=150, seed=3, on_episode=X.GreedyProbe(graphs, every=25))
    )
    assert (plain.q == probed.q).all()


def test_the_final_probe_point_is_the_final_greedy_evaluation(graphs):
    probe = X.GreedyProbe(graphs, every=1000, episodes=120)
    agent = Q.train(graphs, Q.TrainConfig(episodes=120, seed=1, on_episode=probe))
    final = Q.evaluate(agent, graphs)
    assert probe.result()[-1]["mean_terminal_reward"] == final["mean_terminal_reward"]


def test_best_parameters_rank_on_reward_within_each_rule():
    def row(rule, value, reward):
        return {
            "spec": {"cfg": {"explore": rule, X.PARAMETER[rule]: value}},
            "eval": {"all": {"mean_terminal_reward": reward}},
        }

    rows = [
        row("egreedy", 0.01, -3.0),
        row("egreedy", 0.1, -2.0),
        row("softmax", 0.5, -9.0),
        row("ucb", 1.0, -4.0),
        row("ucb", 4.0, -5.0),
    ]
    assert X.best_parameters(rows) == {"egreedy": 0.1, "softmax": 0.5, "ucb": 1.0}


def test_the_sweep_covers_every_rule_grid():
    specs = X.sweep_specs(episodes=10)
    assert len(specs) == sum(len(X.GRID[r]) for r in X.RULES)
    for spec in specs:
        rule = spec["cfg"]["explore"]
        assert spec["cfg"][X.PARAMETER[rule]] in X.GRID[rule]


def test_the_study_reports_every_rule_with_curves_probes_and_seeds(graphs, tmp_path):
    sets = {"all": (RAW, graphs), "ambiguous": (RAW, graphs)}
    params = {"egreedy": 0.05, "softmax": 0.5, "ucb": 1.0}
    result = X.study(sets, params, episodes=120, seeds=(0, 1), probe_every=60, workers=1)
    assert set(result["rules"]) == set(X.RULES)
    for block in result["rules"].values():
        assert [p["episode"] for p in block["probe"]] == [60, 120]
        assert len(block["all"]["mean_terminal_reward"]["values"]) == 2
        assert block["reached_keys"]["mean"] > 0
    refs = {name: Q.references(g, raw) for name, (raw, g) in sets.items()}
    path = X.plot(result, refs, tmp_path / "p11_exploration.png")
    assert path.exists() and path.stat().st_size > 5000

"""Phase 11.2.1 - the tabular Q-learning agent and the training loop 11.2.2-11.2.6 share.

The tests are about the four things that make a learned RL number a lie: an update that
bootstraps off illegal actions, a policy that emits an order which is not a permutation, an
evaluation harness that scores the learner and its references differently, and a fast training
loop that has quietly stopped agreeing with the environment it claims to be a shortcut for. The
corpus-scale results are not pinned here - they live in `reports/rl_qlearning.md` - because a
test that asserts a learning outcome is a test that fails when the result is honestly negative.
"""

from __future__ import annotations

import random

import pytest

from src.rl import actions as A
from src.rl import qlearning as Q
from src.rl.episode import Episode
from src.rl.state import DiagramGraph

# ------------------------------------------------------------------------------------------
# fixtures
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


def _edge(edge_id: str, src: str, dst: str) -> dict:
    return {"id": edge_id, "src": src, "dst": dst, "directed": True, "label": ""}


LINE = {
    "id": "line",
    "diagram_type": "flowchart",
    "nodes": [_node("a", 0, 0), _node("b", 0, 10), _node("c", 0, 20)],
    "edges": [_edge("e1", "a", "b"), _edge("e2", "b", "c")],
    "meta": {"source": "test"},
}

DIAMOND = {
    "id": "diamond",
    "diagram_type": "flowchart",
    "nodes": [_node("a", 0, 0), _node("b", 0, 10), _node("c", 20, 10), _node("d", 0, 20)],
    "edges": [
        _edge("e1", "a", "b"),
        _edge("e2", "a", "c"),
        _edge("e3", "b", "d"),
        _edge("e4", "c", "d"),
    ],
    "meta": {"source": "test"},
}

#: Two components: nothing in this action space can get from the first to the second.
SPLIT = {
    "id": "split",
    "diagram_type": "flowchart",
    "nodes": [_node("a", 0, 0), _node("b", 0, 10), _node("c", 40, 0), _node("d", 40, 10)],
    "edges": [_edge("e1", "a", "b"), _edge("e2", "c", "d")],
    "meta": {"source": "test"},
}

RAW = [LINE, DIAMOND, SPLIT]


@pytest.fixture(scope="module")
def graphs() -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in RAW]


@pytest.fixture(scope="module")
def trained(graphs) -> Q.TabularAgent:
    return Q.train(graphs, Q.TrainConfig(episodes=400, seed=0))


# ------------------------------------------------------------------------------------------
# the epsilon schedule
# ------------------------------------------------------------------------------------------


def test_epsilon_decays_monotonically_from_start_to_floor():
    cfg = Q.TrainConfig(episodes=1000, epsilon=1.0, epsilon_min=0.05)
    values = [Q.epsilon_at(cfg, i) for i in range(0, 1000, 50)]
    assert values[0] == pytest.approx(1.0)
    assert Q.epsilon_at(cfg, 999) == pytest.approx(0.05, abs=1e-6)
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_epsilon_never_goes_below_the_floor():
    cfg = Q.TrainConfig(episodes=10, epsilon=1.0, epsilon_min=0.2)
    assert min(Q.epsilon_at(cfg, i) for i in range(50)) >= 0.2


def test_a_degenerate_schedule_is_the_floor_rather_than_a_zero_division():
    cfg = Q.TrainConfig(episodes=1, epsilon=0.3, epsilon_min=0.3)
    assert Q.epsilon_at(cfg, 0) == pytest.approx(0.3)


# ------------------------------------------------------------------------------------------
# action selection - the mask is the whole safety property
# ------------------------------------------------------------------------------------------


def test_greedy_never_returns_an_action_the_mask_refuses(graphs):
    agent = Q.TabularAgent()
    rng = random.Random(0)
    agent.q[:, A.TERMINATE] = 5.0  # make the tempting action the one we want masked out
    for graph in graphs:
        episode = Episode(graph)
        while not episode.done():
            mask = episode.mask()
            chosen = agent.greedy(Q.key_index(Q.abstract(graph, episode.state)), mask, rng)
            assert mask[chosen]
            episode.apply(chosen)


def test_epsilon_one_still_only_explores_legal_actions(graphs):
    agent = Q.TabularAgent(Q.TrainConfig(explore="egreedy"))
    rng = random.Random(1)
    graph = graphs[1]
    episode = Episode(graph)
    while not episode.done():
        mask = episode.mask()
        chosen = agent.select(Q.key_index(Q.abstract(graph, episode.state)), mask, rng, 1.0)
        assert mask[chosen]
        episode.apply(chosen)


def test_greedy_breaks_ties_at_random_rather_than_by_action_index(graphs):
    """On an untouched row every action ties at 0.0, so the tie rule *is* the initial policy."""
    agent = Q.TabularAgent()
    graph = graphs[1]
    state = Episode(graph).state
    key = Q.key_index(Q.abstract(graph, state))
    mask = A.action_mask(graph, state)
    seen = {agent.greedy(key, mask, random.Random(s)) for s in range(40)}
    assert len(seen) > 1


# ------------------------------------------------------------------------------------------
# the update
# ------------------------------------------------------------------------------------------


def test_the_bootstrap_ignores_illegal_actions(graphs):
    """An illegal `follow-edge-4` must not leak value backwards. The mask leaves 3.29 of 9."""
    graph = graphs[0]
    agent = Q.TabularAgent()
    poison = 1_000.0
    agent.q[:, A.MAX_BRANCH - 1] = poison  # follow-edge-E: legal almost nowhere in this corpus
    before = float(agent.q[:, A.MAX_BRANCH - 1].max())
    Q.train([graph], Q.TrainConfig(episodes=50, seed=0), agent=agent)
    touched = agent.counts[:, A.MAX_BRANCH - 1].sum()
    assert touched == 0, "a follow-edge slot with no edge behind it was updated"
    assert float(agent.q[:, A.MAX_BRANCH - 1].max()) == pytest.approx(before)


def test_training_moves_the_table_and_records_one_row_per_episode(graphs):
    agent = Q.train(graphs, Q.TrainConfig(episodes=200, seed=0))
    assert agent.counts.sum() > 0
    assert (agent.q != 0).any()
    for key in ("reward", "td_error", "full_coverage", "emitted_share", "length", "epsilon"):
        assert len(agent.history[key]) == 200
    assert agent.report["episodes"] == 200
    assert 0 < agent.report["reached_keys"] <= Q.BOUND


def test_only_reachable_keys_are_ever_touched(trained):
    """11.1.6's bound is a guarantee, not a size estimate: 128,000 cells, ~1,766 ever reached."""
    assert trained.report["reached_keys"] < Q.BOUND / 100


def test_sarsa_and_q_take_different_paths_through_the_same_loop(graphs):
    off = Q.train(graphs, Q.TrainConfig(algo="q", episodes=300, seed=3))
    on = Q.train(graphs, Q.TrainConfig(algo="sarsa", episodes=300, seed=3))
    assert not (off.q == on.q).all()


def test_training_is_reproducible_under_a_seed(graphs):
    a = Q.train(graphs, Q.TrainConfig(episodes=150, seed=7))
    b = Q.train(graphs, Q.TrainConfig(episodes=150, seed=7))
    assert (a.q == b.q).all()
    assert a.history["reward"] == b.history["reward"]


def test_potential_based_shaping_is_wired_but_off_by_default(graphs):
    plain = Q.train(graphs, Q.TrainConfig(episodes=150, seed=1))
    shaped = Q.train(
        graphs,
        Q.TrainConfig(episodes=150, seed=1, potential=lambda g, s: float(s.n_emitted())),
    )
    assert not (plain.q == shaped.q).all()
    assert Q.TrainConfig().potential is None


def test_train_refuses_an_empty_pool():
    with pytest.raises(ValueError):
        Q.train([], Q.TrainConfig(episodes=1))


# ------------------------------------------------------------------------------------------
# the fast loop must still be the environment
# ------------------------------------------------------------------------------------------


def test_the_direct_loop_and_the_env_agree_step_for_step(graphs):
    """`train` skips `DiagramTraversalEnv` for speed; this is the claim that it costs nothing."""
    from src.rl.env import DiagramTraversalEnv

    for graph in graphs:
        rng_a, rng_b = random.Random(11), random.Random(11)
        direct = Episode(graph)
        env = DiagramTraversalEnv([graph], order="sequential")
        env.reset(options={"index": 0})
        while not direct.done():
            action = rng_a.choice(direct.legal_actions())
            assert action == rng_b.choice(env.episode.legal_actions())
            direct.apply(action)
            env.step(action)
            assert direct.state == env.state
            assert Q.abstract(graph, direct.state) == env.abstract_state()
        assert env.episode.done()


# ------------------------------------------------------------------------------------------
# the emission order, and the filler that decides the metric
# ------------------------------------------------------------------------------------------


def test_the_order_is_always_a_permutation_even_when_the_policy_emits_nothing(graphs, trained):
    for graph in graphs:
        order = trained.order(graph)
        assert sorted(order) == sorted(graph.node_ids)


def test_a_disconnected_component_is_unreachable_so_the_tail_does_the_work(graphs, trained):
    """SPLIT's second component cannot be entered - there is no jump action. 11.1.4's ceiling."""
    split = graphs[2]
    episode = trained.rollout(split)
    assert not episode.full_coverage()
    assert set(episode.state.emit_sequence) <= {0, 1}
    assert sorted(Q.completed_order(split, episode)) == sorted(split.node_ids)


def test_the_tail_carries_no_flow_information(graphs):
    """Document order, deliberately - `gold_order` would hand an empty policy 7.3.3's DFS answer."""
    graph = graphs[0]
    empty = Episode(graph)
    empty.apply(A.TERMINATE)
    assert Q.completed_order(graph, empty) == list(graph.node_ids)


def test_the_agent_satisfies_the_11_2_7_arm_contract():
    from src.rl.baselines import check_arm

    agent = Q.TabularAgent()
    check_arm("q_learning", agent.arm, DIAMOND)
    check_arm("q_learning", agent.arm, SPLIT)


def test_the_agent_can_actually_join_the_baseline_table():
    from src.rl.baselines import compare

    agent = Q.TabularAgent()
    from src.rl.baselines import dfs_order

    table = compare(RAW, arms={"dfs": dfs_order, "q_learning": agent.arm})
    assert table.diagrams == 3
    assert set(table.arms) == {"dfs", "q_learning"}


# ------------------------------------------------------------------------------------------
# evaluation: one harness for the learner and its references
# ------------------------------------------------------------------------------------------


def test_coverage_is_reported_against_gold_and_never_against_one_hundred_percent(graphs, trained):
    report = Q.evaluate(trained, graphs, RAW)
    assert report["gold_full_coverage_ceiling"] <= 1.0
    assert report["full_coverage"] <= report["gold_full_coverage_ceiling"] + 1e-9
    assert "coverage_vs_ceiling" in report


def test_gold_cannot_reach_full_coverage_on_a_two_component_diagram(graphs):
    report = Q.play(Q.gold_player, [graphs[2]], [SPLIT])
    assert report["full_coverage"] == 0.0
    assert report["gold_full_coverage_ceiling"] == 0.0


def test_gold_beats_random_through_the_same_harness(graphs):
    refs = Q.references(graphs, RAW)
    assert refs["gold"]["mean_terminal_reward"] > refs["random"]["mean_terminal_reward"]
    assert refs["gold"]["mean_emitted_share"] > refs["random"]["mean_emitted_share"]


def test_the_empty_policy_floor_is_reported_alongside_the_learned_f1(graphs, trained):
    report = Q.evaluate(trained, graphs, RAW)
    assert "empty_policy_edge_f1" in report
    assert report["scored_diagrams"] == len(graphs)


def test_evaluation_without_raw_diagrams_skips_the_f1_rather_than_inventing_one(graphs, trained):
    report = Q.evaluate(trained, graphs)
    assert "mean_edge_f1" not in report


def test_the_sweep_returns_one_row_per_cell(graphs):
    rows = Q.sweep(graphs, RAW, alphas=(0.1, 0.4), gammas=(0.9, 1.0), episodes=60)
    assert len(rows) == 4
    assert {(r["alpha"], r["gamma"]) for r in rows} == {
        (0.1, 0.9),
        (0.1, 1.0),
        (0.4, 0.9),
        (0.4, 1.0),
    }
    assert all("convergence" in r for r in rows)

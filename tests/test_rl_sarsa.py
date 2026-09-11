"""Phase 11.2.2 - SARSA against Q-learning.

The comparison is only worth anything if the two runs differ in exactly one thing. These tests
pin that: the same loop, the same schedule, the same evaluation harness, a summary that carries
the spread across seeds, and a sweep that tunes each algorithm on reward rather than handing one
of them the other's cell. The *direction* of the difference is a measurement and lives in
`reports/rl_sarsa.md`, not here.
"""

from __future__ import annotations

import pytest

from src.rl import actions as A
from src.rl import qlearning as Q
from src.rl import sarsa as S
from src.rl.episode import Episode
from src.rl.reward import step_reward
from src.rl.state import DiagramGraph
from tests.test_rl_qlearning import RAW


@pytest.fixture(scope="module")
def graphs() -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in RAW]


@pytest.fixture(scope="module")
def sets(graphs) -> S.Sets:
    return {"all": (RAW, graphs), "ambiguous": (RAW, graphs)}


def test_truncation_rate_reads_the_step_cap_out_of_the_stop_reasons():
    assert S.truncation_rate({"stopped_by": {"cap": 1, "policy": 3}}) == 0.25
    assert S.truncation_rate({"stopped_by": {"coverage": 4}}) == 0.0
    assert S.truncation_rate({}) == 0.0


def test_smooth_is_a_trailing_mean_that_starts_at_the_first_value():
    out = S.smooth([1.0, 3.0, 5.0, 7.0], window=2)
    assert out == pytest.approx([1.0, 2.0, 4.0, 6.0])


def test_smooth_with_a_window_longer_than_the_run_is_the_running_mean():
    assert S.smooth([2.0, 4.0], window=1000)[-1] == pytest.approx(3.0)


def test_the_two_algorithms_are_actually_different_runs(graphs):
    off = Q.train(graphs, Q.TrainConfig(algo="q", episodes=400, seed=0))
    on = Q.train(graphs, Q.TrainConfig(algo="sarsa", episodes=400, seed=0))
    assert not (off.q == on.q).all()
    assert off.history["epsilon"] == on.history["epsilon"]  # the schedule is not what changed


def test_sarsa_bootstraps_off_the_chosen_action_and_q_off_the_legal_max(graphs):
    """One transition, by hand: the only line that differs between the algorithms.

    On LINE the behaviour is scripted to emit `a` and then terminate. With follow-edge-0 valued at
    +5 and terminate at -1 in the next state, Q-learning must back up the +5 (the legal max) and
    SARSA the -1 (the action it actually took next).
    """
    graph = graphs[0]
    first = Episode(graph)
    key0 = Q.key_index(Q.abstract(graph, first.state))
    outcome = first.apply(A.EMIT_NODE)
    key1 = Q.key_index(Q.abstract(graph, first.state))
    assert first.mask()[0] and key1 != key0
    r = step_reward(graph, outcome)
    for algo, bootstrap in (("q", 5.0), ("sarsa", -1.0)):
        cfg = Q.TrainConfig(algo=algo, alpha=1.0, gamma=1.0, episodes=1, seed=0)
        agent = Q.TabularAgent(cfg)
        agent.q[key1, 0] = 5.0
        agent.q[key1, A.TERMINATE] = -1.0
        script = iter([A.EMIT_NODE, A.TERMINATE])
        agent.select = lambda key, mask, rng, eps, _s=script: next(_s)
        Q.train([graph], cfg, agent=agent)
        assert agent.q[key0, A.EMIT_NODE] == pytest.approx(r + bootstrap, abs=1e-5)


def test_training_history_records_cap_truncation(graphs):
    agent = Q.train(graphs, Q.TrainConfig(episodes=50, seed=1))
    assert len(agent.history["truncated"]) == 50
    assert set(agent.history["truncated"]) <= {0.0, 1.0}


def test_the_epsilon_player_is_greedy_at_zero_and_legal_always(graphs):
    agent = Q.train(graphs, Q.TrainConfig(episodes=200, seed=0))
    for graph in graphs:
        greedy = agent.rollout(graph)
        frozen = S.epsilon_player(agent, 0.0, seed=0)(graph)
        assert frozen.state.emit_sequence == greedy.state.emit_sequence
        noisy = S.epsilon_player(agent, 1.0, seed=3)(graph)
        assert all(o.legal for o in noisy.history)


def test_the_sweep_covers_the_11_2_1_grid_for_both_algorithms():
    specs = S.sweep_specs(episodes=10)
    assert len(specs) == len(S.ALGOS) * len(Q.ALPHAS) * len(Q.GAMMAS)


def test_best_cells_rank_on_reward_per_algorithm_not_on_edge_f1():
    def row(algo, alpha, reward, f1):
        return {
            "spec": {"cfg": {"algo": algo, "alpha": alpha, "gamma": 1.0}},
            "eval": {"all": {"mean_terminal_reward": reward, "mean_edge_f1": f1}},
        }

    rows = [row("q", 0.1, -3.0, 0.9), row("q", 0.4, -2.0, 0.1), row("sarsa", 0.2, -1.0, 0.0)]
    assert S.best_cells(rows) == {
        "q": {"alpha": 0.4, "gamma": 1.0},
        "sarsa": {"alpha": 0.2, "gamma": 1.0},
    }


def test_the_comparison_carries_seeds_curves_and_the_online_evaluation(sets):
    result = S.compare_algorithms(sets, episodes=120, seeds=(0, 1), workers=1)
    assert set(result["algos"]) == {"q", "sarsa"}
    for block in result["algos"].values():
        for scope in ("all", "ambiguous"):
            reward = block[scope]["mean_terminal_reward"]
            assert len(reward["values"]) == 2
            assert reward["min"] <= reward["mean"] <= reward["max"]
            assert set(block[scope]["online"]) >= {"mean_terminal_reward", "truncation_rate"}
            ceiling = block[scope]["gold_full_coverage_ceiling"]
            assert block[scope]["full_coverage"]["max"] <= ceiling + 1e-9
        assert len(block["train_truncation_last_10pct"]["values"]) == 2
        assert len(block["curve_reward"]["x"]) == len(block["curve_reward"]["mean"])


def test_a_job_resolves_hooks_by_name(sets):
    out = S.run_job(
        {
            "cfg": {"episodes": 30, "seed": 0},
            "step_bonus": "src.rl.shaping:emit_bonus",
            "eval": ["all"],
        },
        sets,
    )
    assert set(out["eval"]) == {"all"}


def test_the_plot_writes_a_file(sets, tmp_path):
    result = S.compare_algorithms(sets, episodes=60, seeds=(0, 1), workers=1)
    refs = {name: Q.references(g, raw) for name, (raw, g) in sets.items()}
    path = S.plot(result, refs, tmp_path / "p11_sarsa.png")
    assert path.exists() and path.stat().st_size > 5000

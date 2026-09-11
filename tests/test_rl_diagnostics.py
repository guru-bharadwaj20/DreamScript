"""Phase 11.2.8 - convergence diagnostics.

What is pinned is that each number measures what its name says: churn is over the policy and not
over untaken zero-valued actions, the noise floor comes from a table that genuinely cannot move,
the key decoder inverts the Q-table index, and the figures are drawn from a real run. Whether the
corpus-scale table converges is a measurement and lives in `reports/rl_diagnostics.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rl import actions as A
from src.rl import diagnostics as D
from src.rl import qlearning as Q
from src.rl.abstraction import BOUND, abstract, key_index
from src.rl.episode import Episode
from src.rl.sarsa import run_job
from src.rl.state import DiagramGraph
from tests.test_rl_qlearning import RAW


@pytest.fixture(scope="module")
def graphs() -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in RAW]


@pytest.fixture(scope="module")
def sets(graphs):
    return {"all": (RAW, graphs), "ambiguous": (RAW, graphs)}


def test_greedy_ignores_untaken_actions_even_when_they_sit_at_zero():
    """The inherited bug: an untaken action at 0.0 beats a row of learned negative values."""
    q = np.zeros((3, A.N_ACTIONS), dtype=np.float32)
    counts = np.zeros((3, A.N_ACTIONS), dtype=np.int32)
    q[0, A.EMIT_NODE], counts[0, A.EMIT_NODE] = -1.0, 5
    q[0, A.TERMINATE], counts[0, A.TERMINATE] = -3.0, 5
    q[1, A.BACKTRACK], counts[1, A.BACKTRACK] = 2.0, 1
    out = D.greedy_over_taken(q, counts)
    assert out.tolist() == [A.EMIT_NODE, A.BACKTRACK, -1]
    assert q[0].argmax() != A.EMIT_NODE  # the unmasked argmax the draft used gets this wrong


def test_churn_counts_only_keys_reached_in_both_snapshots():
    snaps = D.PolicySnapshots(every=1)
    snaps.episodes = [1, 2]
    snaps.frames = [np.array([0, 1, -1, 3], np.int16), np.array([0, 2, 4, -1], np.int16)]
    assert snaps.churn() == [{"episode": 2, "keys": 2, "changed": 1, "churn": 0.5}]


def test_decode_key_inverts_key_index(graphs):
    for graph in graphs:
        episode = Episode(graph)
        while not episode.done():
            key = abstract(graph, episode.state)
            assert D.decode_key(key_index(key)) == tuple(key)
            episode.apply(episode.legal_actions()[0])
    assert D.decode_key(BOUND - 1) == tuple(s - 1 for s in D.FACTOR_SIZES)


def test_reward_slope_reads_the_second_half_only():
    flat_then_rising = [0.0] * 100 + [i / 1000 for i in range(100)]
    assert D.reward_slope(flat_then_rising) == pytest.approx(1.0, abs=1e-6)
    assert D.reward_slope([5.0] * 50) == pytest.approx(0.0)


def test_visit_concentration_on_a_known_table():
    counts = np.zeros((5, A.N_ACTIONS), dtype=np.int32)
    counts[0, 0] = 90
    counts[1, 0] = 5
    counts[2, 1] = 5
    out = D.visit_concentration(counts)
    assert out["reached_keys"] == 3
    assert out["keys_for_half_the_updates"] == 1
    assert out["share_in_top_10_keys"] == 1.0
    assert out["keys_visited_under_10_times"] == 2


def test_the_noise_floor_uses_a_frozen_table_and_leaves_the_agent_untouched(graphs):
    agent = Q.train(graphs, Q.TrainConfig(episodes=300, seed=0))
    before_q, before_counts = agent.q.copy(), agent.counts.copy()
    floor = D.td_noise_floor(agent, graphs, episodes=200)
    assert floor >= 0.0
    assert (agent.q == before_q).all() and (agent.counts == before_counts).all()


def test_a_run_produces_every_number_and_both_figures(sets, tmp_path):
    jobs = [
        run_job(spec, sets)
        for spec in D.specs(
            episodes=400, seeds=(0, 1), every=100, probe_every=200, noise_episodes=100
        )
    ]
    report = D.diagnose(jobs)
    for seed in report["seeds"]:
        assert seed["policy_churn_final"] is not None
        assert [p["episode"] for p in seed["probe"]] == [200, 400]
        assert seed["td_noise_floor"] > 0
        assert set(seed["evaluation"]) == {"all", "ambiguous"}
    refs = {name: Q.references(g, raw) for name, (raw, g) in sets.items()}
    assert D.plot_convergence(jobs, refs, tmp_path / "c.png").stat().st_size > 5000
    assert D.plot_qvalues(jobs[0], tmp_path / "q.png").stat().st_size > 5000
    table = jobs[0]["extra"]["table"]
    assert len(table["labels"]) == len(table["q"]) == len(table["visits"])
    assert table["visits"] == sorted(table["visits"], reverse=True)

"""Phase 11.2.2 - SARSA against Q-learning.

The comparison is only worth anything if the two runs differ in exactly one thing. These tests
pin that: the same loop, the same seeds, the same evaluation harness, and a summary that carries
the spread across seeds rather than a single number that a re-seed would move. The one behaviour
assertion - that the two algorithms produce different tables - is the check that `algo` is read
at all; the *direction* of the difference is a measurement and lives in the report, not here.
"""

from __future__ import annotations

import pytest

from src.rl import qlearning as Q
from src.rl import sarsa as S
from src.rl.state import DiagramGraph
from tests.test_rl_qlearning import RAW


@pytest.fixture(scope="module")
def graphs() -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in RAW]


def test_truncation_rate_reads_the_step_cap_out_of_the_stop_reasons():
    assert S.truncation_rate({"stopped_by": {"cap": 1, "policy": 3}}) == 0.25
    assert S.truncation_rate({"stopped_by": {"coverage": 4}}) == 0.0
    assert S.truncation_rate({}) == 0.0


def test_smooth_is_a_trailing_mean_that_starts_at_the_first_value():
    values = [1.0, 3.0, 5.0, 7.0]
    out = S.smooth(values, window=2)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(2.0)
    assert out[3] == pytest.approx(6.0)
    assert len(out) == len(values)


def test_smooth_with_a_window_longer_than_the_run_is_the_running_mean():
    assert S.smooth([2.0, 4.0], window=1000)[-1] == pytest.approx(3.0)


def test_the_comparison_runs_both_algorithms_over_both_sets(graphs):
    result = S.compare_algorithms(graphs, RAW, graphs, RAW, episodes=100, seeds=(0, 1))
    assert set(result["algos"]) == {"q", "sarsa"}
    for block in result["algos"].values():
        assert set(block) >= {"all", "ambiguous", "curve_reward", "curve_td"}
        assert len(block["curve_reward"]) == 100
        assert block["all"]["diagrams"] == len(graphs)


def test_the_summary_carries_the_spread_and_not_just_a_mean(graphs):
    result = S.compare_algorithms(graphs, RAW, graphs, RAW, episodes=80, seeds=(0, 1, 2))
    block = result["algos"]["sarsa"]["all"]
    for key in ("mean_terminal_reward", "full_coverage", "mean_emitted_share"):
        assert set(block[key]) == {"mean", "sd", "min", "max"}
        assert block[key]["min"] <= block[key]["mean"] <= block[key]["max"]
    assert len(block["truncation_rate"]["per_seed"]) == 3


def test_coverage_is_never_reported_above_the_gold_ceiling(graphs):
    result = S.compare_algorithms(graphs, RAW, graphs, RAW, episodes=80, seeds=(0,))
    for block in result["algos"].values():
        for scope in ("all", "ambiguous"):
            ceiling = block[scope]["gold_full_coverage_ceiling"]
            assert block[scope]["full_coverage"]["max"] <= ceiling + 1e-9


def test_the_two_algorithms_are_actually_different_runs(graphs):
    off = Q.train(graphs, Q.TrainConfig(algo="q", episodes=400, seed=0))
    on = Q.train(graphs, Q.TrainConfig(algo="sarsa", episodes=400, seed=0))
    assert not (off.q == on.q).all()
    assert off.history["epsilon"] == on.history["epsilon"]  # the schedule is not what changed


def test_an_unknown_algorithm_falls_through_to_the_off_policy_max(graphs):
    """`train` treats anything that is not "sarsa" as Q-learning; this pins that it is not silent
    about producing a *different* result - the same table as "q", so a typo cannot invent a third
    algorithm whose numbers would then be reported under its own name."""
    typo = Q.train(graphs, Q.TrainConfig(algo="Q-LEARNING", episodes=200, seed=5))
    off = Q.train(graphs, Q.TrainConfig(algo="q", episodes=200, seed=5))
    assert (typo.q == off.q).all()


def test_the_plot_writes_a_file(graphs, tmp_path):
    result = S.compare_algorithms(graphs, RAW, graphs, RAW, episodes=60, seeds=(0,))
    path = S.plot(result, tmp_path / "p11_sarsa_vs_q.png")
    assert path.exists() and path.stat().st_size > 5000

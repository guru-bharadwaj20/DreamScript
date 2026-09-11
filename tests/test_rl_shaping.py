"""Phase 11.2.4 - reward shaping and the exploit check.

The property that makes potential-based shaping worth using is that it telescopes: over a full
episode the added terms sum to `gamma^T Phi(s_T) - Phi(s_0)` regardless of the path taken, so no
policy can farm it. That is a checkable identity rather than a citation, and it is checked here
on real trajectories. The unconstrained bonuses are checked for the opposite property - that they
*do* depend on the path, which is exactly why they can be exploited and why they are in the study.
"""

from __future__ import annotations

import pytest

from src.rl import actions as A
from src.rl import qlearning as Q
from src.rl import shaping as S
from src.rl.episode import Episode
from src.rl.state import DiagramGraph
from tests.test_rl_qlearning import RAW


@pytest.fixture(scope="module")
def graphs() -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in RAW]


# ------------------------------------------------------------------------------------------
# potentials
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(S.POTENTIALS))
def test_every_potential_is_bounded_by_the_scale(name, graphs):
    phi = S.POTENTIALS[name]
    for graph in graphs:
        episode = Episode(graph)
        while not episode.done():
            value = phi(graph, episode.state)
            assert 0.0 <= value <= S.SCALE + 1e-9
            episode.apply(episode.legal_actions()[-1])


def test_the_emitted_potential_rises_only_when_a_node_is_emitted(graphs):
    graph = graphs[0]
    episode = Episode(graph)
    before = S.phi_emitted(graph, episode.state)
    episode.apply(A.EMIT_NODE)
    assert S.phi_emitted(graph, episode.state) > before


def test_potential_based_shaping_telescopes_over_an_episode(graphs):
    """The identity that makes it policy-invariant.

    The shaping terms enter the *return*, so each carries its own `gamma^t`; summed that way they
    telescope to `gamma^T Phi(s_T) - Phi(s_0)` no matter which path the episode took. Summing them
    undiscounted does not telescope, which is why the discount is written explicitly here.
    """
    gamma = 0.95
    for graph in graphs:
        for phi in S.POTENTIALS.values():
            episode = Episode(graph)
            total = 0.0
            phi_prev = phi(graph, episode.state)
            start = phi_prev
            steps = 0
            while not episode.done():
                episode.apply(episode.legal_actions()[0])
                phi_next = phi(graph, episode.state)
                total += (gamma**steps) * (gamma * phi_next - phi_prev)
                phi_prev = phi_next
                steps += 1
            assert total == pytest.approx(gamma**steps * phi_prev - start, abs=1e-9)


def test_the_shaped_and_unshaped_loops_differ_only_in_the_reward(graphs):
    plain = Q.train(graphs, Q.TrainConfig(episodes=200, seed=4))
    shaped = Q.train(graphs, Q.TrainConfig(episodes=200, seed=4, potential=S.phi_emitted))
    assert plain.history["epsilon"] == shaped.history["epsilon"]
    assert plain.history["length"] != shaped.history["length"] or not (plain.q == shaped.q).all()


# ------------------------------------------------------------------------------------------
# the unconstrained bonuses
# ------------------------------------------------------------------------------------------


def test_the_loop_bonus_pays_for_marking_and_nothing_else(graphs):
    graph = graphs[0]
    episode = Episode(graph)
    outcome = episode.apply(A.EMIT_NODE)
    assert S.loop_bonus(graph, outcome, episode.state) == 0.0
    assert S.emit_bonus(graph, outcome, episode.state) == S.BONUS


def test_the_emit_bonus_cannot_be_collected_twice_for_one_node(graphs):
    """The mask refuses a duplicate emission, so this bonus is bounded by n. `loop_bonus` is not."""
    graph = graphs[0]
    episode = Episode(graph)
    first = episode.apply(A.EMIT_NODE)
    second = episode.apply(A.EMIT_NODE)
    assert S.emit_bonus(graph, first, episode.state) == S.BONUS
    assert not second.legal
    assert S.emit_bonus(graph, second, episode.state) == 0.0


def test_a_step_bonus_is_actually_added_to_the_reward(graphs):
    plain = Q.train(graphs, Q.TrainConfig(episodes=150, seed=6))
    bribed = Q.train(graphs, Q.TrainConfig(episodes=150, seed=6, step_bonus=S.emit_bonus))
    assert not (plain.q == bribed.q).all()


def test_the_loop_bonus_is_inert_on_an_acyclic_diagram(graphs):
    """`mark-as-loop` is only legal at a back edge (11.1.2), so on these three fixtures the
    exploit cannot fire at all - which is why the study has to run on the real corpus, where
    34.79% of nodes sit behind one."""
    bribed = Q.train(graphs, Q.TrainConfig(episodes=150, seed=6, step_bonus=S.loop_bonus))
    plain = Q.train(graphs, Q.TrainConfig(episodes=150, seed=6))
    assert (plain.q == bribed.q).all()


def test_the_config_records_which_hooks_were_set():
    cfg = Q.TrainConfig(potential=S.phi_emitted, step_bonus=S.loop_bonus)
    assert cfg.to_dict()["potential"] == "phi_emitted"
    assert cfg.to_dict()["step_bonus"] == "loop_bonus"
    assert Q.TrainConfig().to_dict()["step_bonus"] is None


# ------------------------------------------------------------------------------------------
# indicators and the study
# ------------------------------------------------------------------------------------------


def test_behaviour_reports_the_indicators_an_exploit_would_move(graphs):
    agent = Q.train(graphs, Q.TrainConfig(episodes=100, seed=0))
    report = S.behaviour(agent, graphs)
    assert set(report) == {
        "marks_per_node",
        "emitted_share",
        "mean_length",
        "truncation_rate",
        "illegal_action_share",
    }
    assert 0.0 <= report["marks_per_node"] <= 1.0
    assert 0.0 <= report["emitted_share"] <= 1.0


def test_gold_behaviour_is_the_scale_marks_are_read_against(graphs):
    report = S.gold_behaviour(graphs)
    assert report["emitted_share"] > 0.0
    assert report["truncation_rate"] == 0.0  # 11.1.4: 0 of 3,993 gold episodes hit the cap


def test_the_study_covers_the_control_every_potential_and_every_bonus(graphs):
    result = S.study(graphs, RAW, episodes=80, seeds=(0,))
    expected = {"none"} | {f"potential_{n}" for n in S.POTENTIALS} | set(S.BONUSES)
    assert set(result["variants"]) == expected
    assert result["verdict"]["control_reward"] == result["variants"]["none"]["mean_terminal_reward"]


def test_every_variant_is_scored_on_the_unshaped_reward(graphs):
    """The audit: a shaped agent is priced by `RewardConfig()`, which knows nothing about it."""
    result = S.study(graphs, RAW, episodes=80, seeds=(0,))
    for block in result["variants"].values():
        assert block["mean_terminal_reward"] <= 12.0  # nothing inflated by its own bonus
        assert "behaviour" in block


def test_the_verdict_names_an_exploit_only_on_evidence(graphs):
    result = S.study(graphs, RAW, episodes=80, seeds=(0,))
    for row in result["verdict"]["rows"]:
        if row["exploit"]:
            assert not row["beats_control"]
            assert row["marks_per_node"] > 0.0

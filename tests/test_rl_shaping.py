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


@pytest.mark.parametrize("name", sorted(S.POTENTIALS))
def test_the_training_loop_pays_exactly_minus_phi_s0_per_episode(name, graphs):
    """The identity checked on the loop itself, not on a re-implementation of it.

    With alpha 0 the table never moves, so a shaped and an unshaped run take identical actions
    from the same seed, and at gamma 1 with Phi = 0 at the terminal each episode's return must
    differ by exactly -Phi(s_0), whatever path it took.
    """
    phi = S.POTENTIALS[name]
    common = {"alpha": 0.0, "gamma": 1.0, "episodes": 60, "seed": 9}
    plain = Q.train(graphs, Q.TrainConfig(**common))
    shaped = Q.train(graphs, Q.TrainConfig(**common, potential=phi))
    assert plain.history["length"] == shaped.history["length"]
    starts = {round(phi(g, g.initial_state()), 9) for g in graphs}
    for p, q in zip(plain.history["reward"], shaped.history["reward"], strict=True):
        assert any(abs((q - p) + s0) < 1e-6 for s0 in starts)


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


def test_the_visit_bonus_is_farmable_past_n_and_the_loop_bonus_is_not(graphs):
    """The inherited draft had these reversed. A follow/backtrack cycle collects the visit bonus
    on every follow until the cap; `mark-as-loop` is refused on an already-marked node."""
    graph = graphs[0]  # LINE: a -> b -> c
    episode = Episode(graph)
    paid = 0.0
    while not episode.done():
        action = 0 if not episode.state.stack else A.BACKTRACK
        outcome = episode.apply(action, strict=True)
        paid += S.visit_bonus(graph, outcome, episode.state)
    assert episode.truncated()
    assert paid > S.BONUS * graph.n_nodes * 2


def test_the_loop_bonus_is_bounded_by_the_mask(graphs):
    raw = {
        "id": "loop",
        "diagram_type": "flowchart",
        "nodes": [dict(RAW[0]["nodes"][0], id=i) for i in ("a", "b")],
        "edges": [
            {"id": "e1", "src": "a", "dst": "b", "directed": True, "label": ""},
            {"id": "e2", "src": "b", "dst": "a", "directed": True, "label": ""},
        ],
        "meta": {"source": "test"},
    }
    graph = DiagramGraph.from_ir(raw)
    episode = Episode(graph)
    episode.apply(0, strict=True)  # a -> b, b has a back edge to a
    assert episode.mask()[A.MARK_AS_LOOP]
    episode.apply(A.MARK_AS_LOOP, strict=True)
    assert not episode.mask()[A.MARK_AS_LOOP]


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
        "follows_per_node",
        "revisit_step_share",
        "emitted_share",
        "mean_length",
        "truncation_rate",
    }
    assert 0.0 <= report["marks_per_node"] <= 1.0
    assert 0.0 <= report["emitted_share"] <= 1.0


def test_gold_behaviour_never_hits_the_cap(graphs):
    report = S.gold_behaviour(graphs)
    assert report["emitted_share"] > 0.0
    assert report["truncation_rate"] == 0.0  # 11.1.4: 0 of 3,993 gold episodes hit the cap


def test_the_study_covers_every_variant_and_scores_them_unshaped(graphs, tmp_path):
    sets = {"all": (RAW, graphs), "ambiguous": (RAW, graphs)}
    result = S.study(sets, episodes=80, seeds=(0, 1), workers=1)
    expected = {"none"} | {f"potential_{n}" for n in S.POTENTIALS} | set(S.BONUSES)
    assert set(result["variants"]) == expected
    plain = Q.train(graphs, Q.TrainConfig(**S.BASE, episodes=80, seed=0))
    assert result["variants"]["none"]["all"]["mean_terminal_reward"]["values"][0] == (
        Q.evaluate(plain, graphs, RAW)["mean_terminal_reward"]
    )
    for block in result["variants"].values():
        assert set(block["behaviour"]) == {"all", "ambiguous"}
    refs = {name: Q.references(g, raw) for name, (raw, g) in sets.items()}
    assert S.plot(result, refs, tmp_path / "p11_shaping.png").stat().st_size > 5000


def _block(reward, sd, shaped):
    return {
        "all": {"mean_terminal_reward": {"mean": reward, "sd": sd}},
        "shaped_training_return_last_10pct": {"mean": shaped},
    }


def test_the_verdict_needs_the_gap_to_clear_the_seed_noise():
    result = {
        "variants": {
            "none": _block(-3.0, 0.2, -3.0),
            "small_win": _block(-2.9, 0.1, -3.0),
            "real_win": _block(-2.0, 0.1, -2.0),
            "farm": _block(-4.0, 0.3, 5.0),
            "honest_loss": _block(-4.0, 0.3, -4.0),
        }
    }
    rows = {r["variant"]: r for r in S.verdict(result)["rows"]}
    assert not rows["small_win"]["justified"]
    assert rows["real_win"]["justified"] and not rows["real_win"]["exploit"]
    assert rows["farm"]["exploit"]
    assert not rows["honest_loss"]["exploit"] and not rows["honest_loss"]["justified"]

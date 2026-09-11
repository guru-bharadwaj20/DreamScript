"""Phase 11.2.6 - the curriculum ablation.

Pinned: that a "clean" stage really is clean (the property the stage exists for), that the pools
are not a handful of graphs repeated (the inherited draft's 96 pages held 10 distinct graphs),
that every arm spends exactly the same episode budget, and that `flat_restarts` differs from
`curriculum` only in the data. Learning outcomes are reported, not asserted.
"""

from __future__ import annotations

import pytest

from src.rl import curriculum as C
from src.rl.state import DiagramGraph


@pytest.fixture(scope="module")
def structured():
    return C.synthetic_pool(C.STRUCTURED, per_cell=25)


def test_clean_pools_are_clean(structured) -> None:
    pages, drop = structured
    report = C.clean_report(pages)
    assert report["gold_full_coverage"] == 1.0
    assert report["multi_component"] == 0 and report["with_unresolved"] == 0
    assert drop["kept"] + drop["dropped_unclean"] == drop["drawn"] == 150


def test_structured_pool_is_not_a_few_graphs_repeated(structured) -> None:
    pages, _ = structured
    graphs = [DiagramGraph.from_ir(p) for p in pages]
    assert len({C.signature(g) for g in graphs}) > 0.5 * len(graphs)


def test_synthetic_roles_mostly_map_into_the_encoder_vocabulary(structured) -> None:
    pages, _ = structured
    assert C.clean_report(pages)["unknown_role_share"] < 0.25


def test_multi_component_pages_are_filtered_out() -> None:
    from src.synth.graphs import random_diagram

    split = DiagramGraph.from_ir(random_diagram("flowchart", "disconnected", 0))
    assert not C.is_clean(split)


@pytest.mark.parametrize("episodes", [1, 7, 999, 60_000])
def test_every_arm_spends_exactly_the_budget(episodes) -> None:
    for arm in C.ARMS:
        plan = C.arm_plan(arm, {})
        assert abs(sum(share for _, share in plan) - 1.0) < 1e-9
        assert sum(C.budgets(plan, episodes)) == episodes


def test_flat_restarts_has_the_curriculum_schedule_and_not_its_data() -> None:
    restarts = C.arm_plan("flat_restarts", {})
    curriculum = C.arm_plan("curriculum", {})
    assert [s for _, s in restarts] == [s for _, s in curriculum]
    assert {p for p, _ in restarts} == {"real_all"}
    assert [p for p, _ in C.arm_plan("reversed", {})] == [p for p, _ in curriculum][::-1]
    assert {p for p, _ in C.arm_plan("synthetic_only", {})} == {
        "synthetic_linear",
        "synthetic_structured",
    }


def test_unknown_arm_raises() -> None:
    with pytest.raises(ValueError):
        C.arm_plan("nope", {})


def test_paired_difference_is_per_seed() -> None:
    rows = {
        "a": [{"seed": 0, "all": {"x": 1.0}}, {"seed": 1, "all": {"x": 3.0}}],
        "b": [{"seed": 1, "all": {"x": 2.0}}, {"seed": 0, "all": {"x": 0.5}}],
    }
    out = C.paired(rows, "a", "b", "x")
    assert out["values"] == [0.5, 1.0] and out["mean"] == 0.75


def test_tabular_arm_trains_through_its_stages() -> None:
    pools = C.stages(limit=40)
    agent, history, rewards = C.train_arm("curriculum", pools, 200, seed=0)
    assert [h["pool"] for h in history] == [p for p, _ in C.PLAN]
    assert sum(h["episodes"] for h in history) == 200 == len(rewards)


def test_dqn_arm_carries_weights_across_slices_and_spends_the_budget() -> None:
    rows = C._run_dqn_arm("flat_restarts", 80, seeds=(0, 1), limit=30)
    assert [r["seed"] for r in rows] == [0, 1]
    for row in rows:
        assert [h["pool"] for h in row["stages"]] == ["real_all"] * 4
        assert sum(h["episodes"] for h in row["stages"]) == 80
        assert 0.0 <= row["all"]["mean_emitted_share"] <= 1.0

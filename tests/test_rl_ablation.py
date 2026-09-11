"""Phase 11.2.10 - the pipeline ablation.

Pinned: that every arm is scored by one function over the same state convention, that the hybrid
completion is a permutation that keeps the learned prefix, that `gold_in_action_space` is never
counted as a heuristic, and that loop marks change only the loop term. Corpus-level numbers live in
`reports/rl_ablation.md`.
"""

from __future__ import annotations

import pytest

from src.rl import ablation as AB
from src.rl.episode import Episode
from src.rl.state import DiagramGraph


def _node(node_id: str, x: float, y: float) -> dict:
    return {
        "id": node_id,
        "shape": "process",
        "bbox": [x, y, 10.0, 10.0],
        "text": node_id,
        "semantic_role": "process",
        "confidence": 1.0,
    }


def _edge(edge_id: str, src: str, dst: str) -> dict:
    return {"id": edge_id, "src": src, "dst": dst, "directed": True, "label": ""}


LOOP_SPLIT = {
    "id": "loop_split",
    "nodes": [
        _node("a", 0, 0),
        _node("b", 0, 10),
        _node("c", 0, 20),
        _node("d", 40, 0),
        _node("e", 40, 10),
    ],
    "edges": [
        _edge("1", "a", "b"),
        _edge("2", "b", "c"),
        _edge("3", "c", "b"),
        _edge("4", "d", "e"),
    ],
    "meta": {"source": "test"},
}


@pytest.fixture(scope="module")
def graph() -> DiagramGraph:
    return DiagramGraph.from_ir(LOOP_SPLIT)


def _emit_nothing(graph: DiagramGraph) -> Episode:
    episode = Episode(graph)
    episode.apply(8)  # terminate
    return episode


def test_hybrid_keeps_the_prefix_and_is_a_permutation(graph) -> None:
    order = AB.complete_with(graph, [2, 1], [0, 1, 2, 3, 4])
    assert order[:2] == [2, 1] and sorted(order) == list(range(graph.n_nodes))


def test_arms_cover_heuristics_gold_learned_and_hybrids(graph) -> None:
    arms = AB.arms_for(LOOP_SPLIT, graph, {"dqn": _emit_nothing})
    for name in AB.HEURISTICS:
        assert sorted(arms[name][0]) == list(range(graph.n_nodes))
        assert arms[name][1] == 0  # a heuristic produces no loop marks
    assert arms["dqn"][0] == []
    assert arms["dqn+dfs"][0] == arms["dfs"][0]
    gold, marks = arms["gold_in_action_space"]
    assert len(gold) < graph.n_nodes  # cannot cross into the second component
    assert arms["gold+dfs"][0][: len(gold)] == gold


def test_empty_emission_is_charged_for_every_node(graph) -> None:
    row = AB.code_quality(graph, [], 0)
    full = AB.code_quality(graph, list(range(graph.n_nodes)), 0)
    assert row["coverage"] == 0.0 and full["coverage"] == 1.0
    assert row["terminal_reward"] < full["terminal_reward"]


def test_loop_marks_move_only_the_loop_term(graph) -> None:
    order = AB.gold_episode(graph).state.emit_sequence
    assert AB.gold_episode(graph).state.loop_marked == 0  # gold play never marks a loop
    oracle = 0
    for source, _ in graph.back_edges:
        oracle |= 1 << source
    assert oracle, "the fixture has a back edge"
    marked = AB.code_quality(graph, order, oracle)
    bare = AB.code_quality(graph, order, 0)
    assert marked["loops"] > bare["loops"]
    for key in ("coverage", "ordering"):
        assert marked[key] == bare[key]


def test_contribution_never_treats_gold_as_a_heuristic() -> None:
    row = {m: 0.0 for m in AB.METRICS}
    tab = {name: dict(row) for name in AB.HEURISTICS}
    tab["dfs"]["edge_f1"] = 0.5
    tab["gold_in_action_space"] = {**row, "edge_f1": 0.9}
    tab["dqn"] = {**row, "edge_f1": 0.4}
    tab["dqn+dfs"] = {**row, "edge_f1": 0.55}
    out = AB.contribution([tab], "dqn", "edge_f1")
    assert out["best_heuristic"] == "dfs"
    assert out["learned_minus_best_heuristic"]["mean"] == pytest.approx(-0.1)
    assert out["hybrid_minus_dfs"]["mean"] == pytest.approx(0.05)
    assert out["learned_minus_gold_in_action_space"]["mean"] == pytest.approx(-0.5)


def test_measure_and_table_agree_on_counts(graph) -> None:
    rows = AB.measure([LOOP_SPLIT], [graph], {"dqn": _emit_nothing})
    assert set(rows) == {"own", "oracle"}
    tab = AB.table(rows["own"])
    assert tab["dfs"]["diagrams"] == 1 and tab["dqn"]["coverage"] == 0.0
    paired = AB.paired(rows["own"], "dqn+dfs", "dfs", "edge_f1", None)
    assert paired["ties"] == 1

"""Phase 11.2.10 - the ablation: what does the pipeline lose if the RL agent is removed and the
emission order comes from a heuristic instead?

    python -m src.rl.ablation            # every arm, end to end, with the sandbox verdict
    python -m src.rl.ablation --quick    # a short run, for the tests

11.2.7 compares *orders* by edge F1. That is a proxy. This row asks the end-to-end question the
plan actually poses - "pipeline quality without RL" - and answers it on the artefact the pipeline
produces, which is code: every arm's order is run through 11.1.3's `src.rl.emit`, the emitted
Python is parsed, checked for undefined names, priced by the four-term semantic score, and
executed in 11.2.9's sandbox. The contribution of RL is then a subtraction, not an argument.

## Why an ablation here is a subtraction and not a re-run

Nothing downstream of the traversal depends on *how* the order was produced: `emit_code` takes
`(graph, emitted, loop_marked)` and nothing else. So removing RL from the pipeline is exactly
replacing one list with another, and the ablation is a table of the same measurements over the
candidate lists. There is no retraining to do and no confound to control for.

## The one asymmetry, stated because it decides the result

A heuristic arm orders **every node**: it reads the whole graph and can walk into a second
connected component, because it never passes through the action space. A learned policy cannot -
11.1.4 measured the consequence, gold play in this action space reaches full coverage on 63.06%
of the corpus and there is no jump action. So the heuristics are not being compared with the
learner on equal footing; they are being handed a capability the action space withholds.

Rather than hide that, the table carries a fourth row for it: `gold_in_action_space` is 7.3.3's
DFS *replayed through the action space*, so the gap between it and the `dfs` arm is the price of
the action space, and the gap between the learned arm and `gold_in_action_space` is the price of
the learning. Attributing the whole difference to the learner would be wrong by the size of the
first gap.

## Loop marks

`emit_code` wraps a `while` around a region the policy closed with `mark-as-loop`, and a
heuristic order carries no marks at all. Every arm is therefore emitted with the marks a *gold*
replay produced on that diagram (`marks_from_gold`, on by default), so the comparison is of
orderings and not of one arm's missing loop structure. Passing `marks_from_gold=False` measures
the same table with no marks anywhere, which is the value of the `mark-as-loop` action itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from typing import Any

from src.rl.emit import check, emit_code
from src.rl.episode import Episode
from src.rl.reward import DEFAULT, TraversalState, semantic_score, terminal_reward
from src.rl.state import DiagramGraph

#: Sandbox calls cost ~71 ms each (11.2.9), so the execution column is measured on a sample.
SANDBOX_SAMPLE = 120


def state_for(graph: DiagramGraph, order: Sequence[int], marks: int = 0) -> TraversalState:
    """A `TraversalState` asserting that `order` was emitted, with `marks` as the loop mask.

    `visited` is set to exactly the emitted nodes. That is the honest reading for a heuristic
    arm: it never visited anything, so crediting it with a visit it did not make would cancel the
    unreachable-node penalty the learned arm pays. Both arms are therefore charged for every node
    absent from their order.
    """
    emitted = 0
    for node in order:
        emitted |= 1 << node
    return TraversalState(
        current=order[-1] if order else 0,
        visited=emitted,
        emitted=emitted,
        loop_marked=marks & emitted,
        steps=len(order),
        emit_sequence=tuple(order),
    )


def gold_episode(graph: DiagramGraph) -> Episode:
    episode = Episode(graph)
    for action in episode.gold_actions():
        if episode.done():
            break
        episode.apply(action)
    return episode


def code_quality(
    graph: DiagramGraph, order: Sequence[int], marks: int, sandbox=None
) -> dict[str, Any]:
    """Everything that can be said about the code one order produces."""
    state = state_for(graph, order, marks)
    marked = {i for i in range(graph.n_nodes) if state.is_loop_marked(i)}
    code = emit_code(graph, list(order), marked)
    verdict = check(code)
    score, detail = semantic_score(graph, state, DEFAULT)
    reward, _ = terminal_reward(graph, state, False, DEFAULT)
    out = {
        "parse_ok": bool(verdict["parse_ok"]),
        "undefined": bool(verdict["undefined_names"]),
        "semantic_score": round(score, 6),
        "coverage": round(detail.get("coverage", 0.0), 6),
        "ordering": round(detail.get("ordering", 0.0), 6),
        "structure": round(detail.get("structure", 0.0), 6),
        "loops": round(detail.get("loops", 0.0), 6),
        "terminal_reward": reward,
        "code_lines": code.count("\n") + 1,
    }
    if sandbox is not None:
        out["sandbox"] = sandbox(code)
    return out


def arms_for(diagram: dict, graph: DiagramGraph, agent=None) -> dict[str, list[int]]:
    """Every candidate emission order for one diagram, as node indices.

    The heuristics come from 11.2.7 unchanged - this module does not reimplement an ordering.
    """
    from src.rl.baselines import ARMS

    index = graph.index_of
    orders: dict[str, list[int]] = {}
    for name, policy in ARMS.items():
        try:
            orders[name] = [index[node_id] for node_id in policy(diagram) if node_id in index]
        except Exception:  # a heuristic that cannot order this page is a result, not a crash
            orders[name] = []
    orders["gold_in_action_space"] = list(gold_episode(graph).state.emit_sequence)
    if agent is not None:
        orders["learned"] = list(agent.rollout(graph).state.emit_sequence)
    return orders


def ablate(
    diagrams: Sequence[dict],
    graphs: Sequence[DiagramGraph],
    agent=None,
    sandbox_sample: int = SANDBOX_SAMPLE,
    marks_from_gold: bool = True,
) -> dict[str, Any]:
    """The table. One row per arm, one column per thing that can be measured about its code."""
    from src.rl.baselines import score as edge_score

    rows: dict[str, list[dict[str, Any]]] = {}
    f1: dict[str, list[float]] = {}
    sandbox = None
    if sandbox_sample:
        from src.rl.reward import sandbox_hook

        sandbox = sandbox_hook()

    for position, (diagram, graph) in enumerate(zip(diagrams, graphs, strict=True)):
        if not graph.n_nodes:
            continue
        marks = gold_episode(graph).state.loop_marked if marks_from_gold else 0
        use_sandbox = sandbox if position < sandbox_sample else None
        for name, order in arms_for(diagram, graph, agent).items():
            rows.setdefault(name, []).append(code_quality(graph, order, marks, use_sandbox))
            seen = set(order)
            completed = [
                graph.node_ids[i]
                for i in list(order) + [i for i in range(graph.n_nodes) if i not in seen]
            ]
            f1.setdefault(name, []).append(edge_score(diagram, completed)["edge_f1"])

    table: dict[str, Any] = {}
    for name, values in rows.items():
        passed = [v["sandbox"]["passed"] for v in values if "sandbox" in v]
        table[name] = {
            "diagrams": len(values),
            "mean_semantic_score": round(
                statistics.fmean([v["semantic_score"] for v in values]), 4
            ),
            "mean_terminal_reward": round(
                statistics.fmean([v["terminal_reward"] for v in values]), 4
            ),
            "mean_edge_f1": round(statistics.fmean(f1[name]), 4),
            "mean_coverage": round(statistics.fmean([v["coverage"] for v in values]), 4),
            "mean_ordering": round(statistics.fmean([v["ordering"] for v in values]), 4),
            "mean_structure": round(statistics.fmean([v["structure"] for v in values]), 4),
            "parse_ok_rate": round(statistics.fmean([float(v["parse_ok"]) for v in values]), 4),
            "undefined_name_rate": round(
                statistics.fmean([float(v["undefined"]) for v in values]), 4
            ),
            "sandbox_pass_rate": (
                round(statistics.fmean([float(p) for p in passed]), 4) if passed else None
            ),
            "sandbox_n": len(passed),
        }
    return table


def contribution(table: dict[str, Any], metric: str = "mean_semantic_score") -> dict[str, Any]:
    """The plan's "contribution quantified": learned minus the best arm that is not the learner.

    Reported against two references on purpose. `vs_best_heuristic` is the number that answers
    "should this pipeline ship the RL agent"; `vs_gold_in_action_space` is the number that answers
    "did the learning work", because it holds the action space fixed.
    """
    if "learned" not in table:
        return {"learned": None}
    learned = table["learned"][metric]
    heuristics = {k: v[metric] for k, v in table.items() if k not in {"learned"}}
    in_space = table.get("gold_in_action_space", {}).get(metric)
    best_name = max(heuristics, key=heuristics.get) if heuristics else None
    return {
        "metric": metric,
        "learned": learned,
        "best_arm": best_name,
        "best_arm_value": heuristics.get(best_name),
        "vs_best_heuristic": round(learned - heuristics[best_name], 4) if best_name else None,
        "vs_gold_in_action_space": (round(learned - in_space, 4) if in_space is not None else None),
        "rl_helps": bool(best_name and learned > heuristics[best_name]),
    }


def run(
    episodes: int = 60_000,
    limit: int | None = None,
    seed: int = 0,
    write: bool = False,
    sandbox_sample: int = SANDBOX_SAMPLE,
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS, TrainConfig, train, training_set

    raw_all, graphs_all = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    agent = train(graphs_all, TrainConfig(alpha=0.4, gamma=1.0, episodes=episodes, seed=seed))
    result: dict[str, Any] = {"episodes": episodes, "sandbox_sample": sandbox_sample}
    for label, (raw, graphs) in {
        "all": (raw_all, graphs_all),
        "ambiguous": (raw_amb, graphs_amb),
    }.items():
        table = ablate(raw, graphs, agent, sandbox_sample)
        result[label] = {
            "table": table,
            "contribution_semantic": contribution(table),
            "contribution_edge_f1": contribution(table, "mean_edge_f1"),
        }
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.10 pipeline ablation without RL")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sandbox", type=int, default=SANDBOX_SAMPLE)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    episodes = 800 if args.quick else args.episodes
    sample = 0 if args.quick else args.sandbox
    print(json.dumps(run(episodes, args.limit, args.seed, args.write, sample), indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

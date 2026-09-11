"""Phase 11.2.6 - curriculum: clean synthetic graphs first, messy real ones after, and the
ablation that says whether the order mattered.

    python -m src.rl.curriculum            # the four arms, evaluated on the real corpus
    python -m src.rl.curriculum --quick    # a short run, for the tests

## Why a curriculum is even a candidate here

11.2.1's failure has a specific shape: the agent plateaus with a |TD| tail/head ratio of 0.82 and
never reaches full coverage on the ambiguous set. One standard reading of that shape is that the
early episodes are almost all failure - a random policy on this corpus emits 6.27% of nodes and
terminates by its own choice 97.32% of the time (11.1.4) - so the table is mostly learning what
*not* to do, on pages where the right thing to do is 40 steps away. A curriculum answers that by
making the first pages ones where success is three steps away.

## The synthetic stages are generated here, not borrowed

`src/synth` builds *images*. This needs IR, and it needs IR with a property no real page has:
being clean. `generate` produces flowcharts with one connected component, no unresolved edges,
every node reachable from a single entry, real bboxes in reading order, and a shape vocabulary
drawn from the same `ROLE_VOCAB` the encoder uses - so the state encoder and the abstraction see
a distribution they will see again in stage 3, only easier. Four families:

    chain     n nodes, one path. The minimum task: emit, follow, emit, terminate.
    branch    a decision with two arms that rejoin. Introduces `follow-edge-B`.
    loop      a chain with one back edge. The only stage where `mark-as-loop` is legal at all.
    nested    a branch inside a branch. Out-degree 2 at two depths.

**The ceiling on every synthetic stage is 100%**, because they are single-component by
construction - that is what makes them "clean", and it is also why a number from a synthetic
stage may never be quoted as a corpus result.

## The ablation

Four arms, all with the **same total episode budget**, all evaluated on the real corpus:

    flat            all episodes on the real corpus. 11.2.1's setting, the control.
    curriculum      synthetic chains -> branch/loop/nested -> single-component real -> all real.
    synthetic_only  all episodes on synthetic pages. The check that the curriculum's gain, if
                    any, is transfer and not just "clean pages are easier to learn on".
    reversed        the curriculum backwards, hard to easy. If this matches `curriculum`, the
                    gain was extra data and not ordering, and the row should say so.

Equal budgets is the whole design. A curriculum that trains on more episodes than its control is
not a curriculum result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from typing import Any

from src.rl.state import DiagramGraph

FAMILIES: tuple[str, ...] = ("chain", "branch", "loop", "nested")


# ------------------------------------------------------------------------------------------
# clean synthetic IR
# ------------------------------------------------------------------------------------------


def _node(node_id: str, index: int, shape: str, text: str) -> dict:
    """Bboxes descend the page in reading order, so `reading` ordering is meaningful here too."""
    return {
        "id": node_id,
        "shape": shape,
        "bbox": [40.0 + 30.0 * (index % 3), 40.0 * index, 60.0, 30.0],
        "text": text,
        "semantic_role": shape,
        "confidence": 1.0,
    }


def _edge(index: int, src: str, dst: str, label: str = "") -> dict:
    return {"id": f"s{index}", "src": src, "dst": dst, "directed": True, "label": label}


def generate(family: str, size: int, index: int = 0) -> dict:
    """One clean synthetic flowchart. Single component, no unresolved edges, one entry point."""
    size = max(2, size)
    nodes: list[dict] = []
    edges: list[dict] = []

    def add(shape: str, text: str) -> str:
        node_id = f"n{len(nodes)}"
        nodes.append(_node(node_id, len(nodes), shape, text))
        return node_id

    if family == "chain":
        ids = [add("terminator", "start")]
        ids += [add("process", f"step {i}") for i in range(1, size - 1)]
        ids.append(add("terminator", "end"))
        edges = [_edge(i, ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    elif family == "branch":
        start, decision = add("terminator", "start"), add("decision", "ok?")
        left, right = add("process", "yes"), add("process", "no")
        tail = [add("process", f"after {i}") for i in range(max(0, size - 5))]
        end = add("terminator", "end")
        edges = [_edge(0, start, decision), _edge(1, decision, left, "yes")]
        edges += [_edge(2, decision, right, "no")]
        join = tail[0] if tail else end
        edges += [_edge(3, left, join), _edge(4, right, join)]
        chain = tail + [end]
        edges += [_edge(5 + i, chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    elif family == "loop":
        ids = [add("terminator", "start")]
        ids += [add("process", f"body {i}") for i in range(1, size - 1)]
        ids.append(add("terminator", "end"))
        edges = [_edge(i, ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
        edges.append(_edge(len(edges), ids[-2], ids[1], "again"))  # the back edge
    elif family == "nested":
        start = add("terminator", "start")
        outer = add("decision", "outer?")
        inner = add("decision", "inner?")
        a, b = add("process", "a"), add("process", "b")
        c = add("process", "c")
        end = add("terminator", "end")
        edges = [
            _edge(0, start, outer),
            _edge(1, outer, inner, "yes"),
            _edge(2, outer, c, "no"),
            _edge(3, inner, a, "yes"),
            _edge(4, inner, b, "no"),
            _edge(5, a, end),
            _edge(6, b, end),
            _edge(7, c, end),
        ]
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")

    return {
        "id": f"syn_{family}_{size}_{index}",
        "diagram_type": "flowchart",
        "nodes": nodes,
        "edges": edges,
        "meta": {"source": "synthetic"},
    }


def synthetic_pool(families: Sequence[str], sizes: Sequence[int], per_cell: int = 8) -> list[dict]:
    """A pool of clean pages. Deterministic - the whole point is a reproducible easy stage."""
    return [
        generate(family, size, index)
        for family in families
        for size in sizes
        for index in range(per_cell)
    ]


def graphs_of(diagrams: Sequence[dict]) -> list[DiagramGraph]:
    return [DiagramGraph.from_ir(d) for d in diagrams]


# ------------------------------------------------------------------------------------------
# the stages
# ------------------------------------------------------------------------------------------


def stages(limit: int | None = None) -> list[dict[str, Any]]:
    """The curriculum, easy to hard. Each stage is a pool and the share of the budget it gets."""
    from src.rl.qlearning import training_set

    raw, graphs = training_set(limit, ambiguous=False)
    simple = [g for g in graphs if g.n_components == 1 and g.n_nodes <= 12]
    return [
        {
            "name": "synthetic_chain",
            "share": 0.15,
            "graphs": graphs_of(synthetic_pool(("chain",), (3, 4, 6))),
        },
        {
            "name": "synthetic_structured",
            "share": 0.20,
            "graphs": graphs_of(synthetic_pool(("branch", "loop", "nested"), (5, 7, 9))),
        },
        {
            "name": "real_simple",
            "share": 0.25,
            "graphs": simple or graphs,
        },
        {"name": "real_all", "share": 0.40, "graphs": graphs},
    ]


def train_curriculum(
    plan: Sequence[dict[str, Any]],
    episodes: int,
    seed: int = 0,
    alpha: float = 0.4,
    gamma: float = 1.0,
):
    """Train one agent across the stages, carrying the table forward.

    Each stage gets its own `TrainConfig` whose `episodes` is that stage's slice, so **epsilon
    decays within a stage and restarts at the next one**. That is deliberate and it is the part a
    reader should argue with: a single decay across the whole run would leave the last stage - the
    only one that resembles the evaluation - played almost greedily by a table trained elsewhere.
    The `reversed` arm inherits exactly the same schedule, so the comparison is unaffected.
    """
    from src.rl.qlearning import TabularAgent, TrainConfig, train

    agent = TabularAgent()
    history: list[dict[str, Any]] = []
    for stage in plan:
        budget = max(1, int(episodes * stage["share"]))
        cfg = TrainConfig(alpha=alpha, gamma=gamma, episodes=budget, seed=seed)
        train(stage["graphs"], cfg, agent=agent)
        history.append(
            {
                "stage": stage["name"],
                "episodes": budget,
                "pool": len(stage["graphs"]),
                "reward_tail": agent.report["reward_tail"],
                "td_tail_ratio": agent.report["td_tail_ratio"],
                "reached_keys": agent.report["reached_keys"],
            }
        )
    agent.stages = history  # type: ignore[attr-defined]
    return agent


def ablation(
    episodes: int = 60_000,
    limit: int | None = None,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """The four arms at equal budget, all evaluated on the real corpus."""
    from src.rl.qlearning import TrainConfig, evaluate, references, train, training_set

    raw, graphs = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    plan = stages(limit)
    synthetic = [
        {"name": "synthetic_only", "share": 1.0, "graphs": plan[0]["graphs"] + plan[1]["graphs"]}
    ]

    arms: dict[str, list[dict[str, Any]]] = {}
    stage_logs: dict[str, Any] = {}
    for seed in seeds:
        agents = {
            "flat": train(graphs, TrainConfig(alpha=0.4, gamma=1.0, episodes=episodes, seed=seed)),
            "curriculum": train_curriculum(plan, episodes, seed),
            "reversed": train_curriculum(list(reversed(plan)), episodes, seed),
            "synthetic_only": train_curriculum(synthetic, episodes, seed),
        }
        for name, agent in agents.items():
            arms.setdefault(name, []).append(
                {
                    "all": evaluate(agent, graphs, raw),
                    "ambiguous": evaluate(agent, graphs_amb, raw_amb),
                    "convergence": agent.report,
                }
            )
            if seed == seeds[0] and hasattr(agent, "stages"):
                stage_logs[name] = agent.stages

    def summarise(rows: Sequence[dict[str, Any]], scope: str) -> dict[str, Any]:
        def stat(key: str) -> dict[str, float]:
            values = [r[scope][key] for r in rows]
            return {
                "mean": round(statistics.fmean(values), 4),
                "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            }

        return {
            "mean_terminal_reward": stat("mean_terminal_reward"),
            "full_coverage": stat("full_coverage"),
            "mean_emitted_share": stat("mean_emitted_share"),
            "mean_edge_f1": stat("mean_edge_f1"),
            "gold_full_coverage_ceiling": rows[0][scope]["gold_full_coverage_ceiling"],
            "empty_policy_edge_f1": rows[0][scope]["empty_policy_edge_f1"],
        }

    result: dict[str, Any] = {
        "episodes": episodes,
        "seeds": list(seeds),
        "stage_plan": [
            {"name": s["name"], "share": s["share"], "pool": len(s["graphs"])} for s in plan
        ],
        "stage_logs": stage_logs,
        "arms": {
            name: {"all": summarise(rows, "all"), "ambiguous": summarise(rows, "ambiguous")}
            for name, rows in arms.items()
        },
        "references": references(graphs, raw),
    }
    result["verdict"] = verdict(result)
    return result


def verdict(result: dict[str, Any]) -> dict[str, Any]:
    """Did ordering help, or was it just more data? The `reversed` arm is what separates them."""
    arms = result["arms"]
    flat = arms["flat"]["all"]["mean_terminal_reward"]["mean"]
    curriculum = arms["curriculum"]["all"]["mean_terminal_reward"]["mean"]
    backwards = arms["reversed"]["all"]["mean_terminal_reward"]["mean"]
    return {
        "curriculum_minus_flat": round(curriculum - flat, 4),
        "curriculum_minus_reversed": round(curriculum - backwards, 4),
        "curriculum_helps": curriculum > flat,
        "ordering_is_what_helped": curriculum > flat and curriculum > backwards,
        "synthetic_only_transfers": (
            arms["synthetic_only"]["all"]["mean_terminal_reward"]["mean"] > flat
        ),
    }


def run(
    episodes: int = 60_000,
    limit: int | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    write: bool = False,
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS

    result = ablation(episodes, limit, seeds)
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "curriculum.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.6 curriculum ablation")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    episodes = 1200 if args.quick else args.episodes
    seeds = (0,) if args.quick else (0, 1, 2)
    print(json.dumps(run(episodes, args.limit, seeds, write=args.write), indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

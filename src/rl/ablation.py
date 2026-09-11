"""Phase 11.2.10 - the ablation: what does the pipeline lose if the RL agent is removed and the
emission order comes from a heuristic instead?

    python -m src.rl.ablation --write    # every arm, end to end, with the sandbox verdict
    python -m src.rl.ablation --quick    # a short run, for the tests

11.2.7 compares *orders* by edge F1. This row asks the end-to-end question - "pipeline quality
without RL" - on the artefact the pipeline produces, which is code: every arm's order is run
through 11.1.3's `src.rl.emit`, the emitted Python is parsed, checked for undefined names, priced
by the four-term semantic score, and executed in 11.2.9's sandbox on a sample. The contribution of
RL is then a subtraction, not an argument.

## Why an ablation here is a subtraction and not a re-run

Nothing downstream of the traversal depends on *how* the order was produced: `emit_code` takes
`(graph, emitted, loop_marked)` and nothing else (checked: that is its whole signature). So
removing RL is exactly replacing one list with another, and the ablation is a table of the same
measurements over the candidate lists.

## The arms

    topological / dfs / bfs / reading   11.2.7's heuristics, unchanged (`baselines.ARMS`).
    gold_in_action_space                7.3.3's DFS replayed through the action space.
    dqn                                 11.2.5's DQN, greedy, one checkpoint per seed - the best
                                        learned policy on record by greedy terminal reward.
    tabular_shaped                      the best tabular policy on record: Q-learning at 11.2.2's
                                        cell (alpha 0.2, gamma 0.99) with 11.2.4's emitted-share
                                        potential, retrained here on the same seeds.
    <learned>+dfs, gold+dfs             **added**: the learned prefix, then every node it did not
                                        emit in the `dfs` arm's order. A learned policy in this
                                        action space cannot reach a second component (11.1.4), so
                                        a real pipeline would have to finish its order somehow;
                                        this hybrid is the only arm that asks whether RL adds
                                        anything *on top of* the best heuristic rather than
                                        instead of it.

## The one asymmetry, stated because it decides the result

A heuristic arm orders **every node**; a learned policy cannot cross components, and gold play in
the action space reaches full coverage on 41.69% of the 993 labelled diagrams. The pure learned
arms are therefore not compared with heuristics on equal footing, which is why `gold_in_action_
space` and the hybrids are in the table: the gap from `dfs` to `gold_in_action_space` is the price
of the action space, the gap from `gold_in_action_space` to a learned arm is the price of the
learning, and `<learned>+dfs` against `dfs` is the contribution a pipeline could actually bank.

## Loop marks

`emit_code` wraps a `while` around a region closed with `mark-as-loop`; a heuristic order carries
no marks. Two tables are measured. **own** - every arm with the marks it produced itself (none for
heuristics), the honest pipeline and the one the verdict reads. **oracle** - every arm with every
back-edge source of the diagram marked, the marks 11.1.3's loop term pays for, which isolates the
ordering from the loop decision.

**The draft's mode could not have done what it said.** It emitted every arm "with the marks a gold
replay produced", but `Episode.gold_actions` never issues `mark-as-loop`: measured, **0 of 993**
gold episodes carry a mark although 681 of the 993 diagrams have a back edge. Its "gold marks" were
no marks, for every arm, always.

## Defects in the inherited draft, fixed

    biased sandbox sample   the sandbox column used "the first 120 diagrams", and the labelled
                            set is ordered hdbpmn-then-fa_bresler, so all 120 were hdbpmn (gold
                            full coverage 20.06% there against 91.67% on fa_bresler). Now a seeded
                            random sample shared by every arm.
    gold as a heuristic     `contribution` took the best arm "that is not the learner" and that
                            included `gold_in_action_space`, which is not a heuristic.
    a private learner       `run` trained a fresh 60k-episode tabular agent at 11.2.1's cell, not
                            the policy any row reported. Learned arms now load 11.2.5's
                            checkpoints and retrain 11.2.4's recommended table on stated seeds.

The measured results are in `reports/rl_ablation.md` and `experiments/rl/ablation.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Any

from src.rl.emit import check, emit_code
from src.rl.episode import Episode
from src.rl.reward import DEFAULT, TraversalState, semantic_score, terminal_reward
from src.rl.state import DiagramGraph
from src.utils.config import ROOT

#: Sandbox calls cost ~71 ms each (11.2.9), so execution is measured on a seeded sample.
SANDBOX_SAMPLE = 150
SEEDS: tuple[int, ...] = (0, 1, 2)
FIGURE = ROOT / "reports" / "figures" / "p11_ablation.png"
HEURISTICS: tuple[str, ...] = ("topological", "dfs", "bfs", "reading")
METRICS: tuple[str, ...] = (
    "semantic_score",
    "terminal_reward",
    "edge_f1",
    "coverage",
    "ordering",
    "structure",
    "loops",
    "undefined",
    "sandbox_pass",
)


def state_for(graph: DiagramGraph, order: Sequence[int], marks: int = 0) -> TraversalState:
    """A `TraversalState` asserting that `order` was emitted, with `marks` as the loop mask.

    `visited` is exactly the emitted nodes, for every arm alike: a heuristic never visited
    anything, and crediting a learned arm with visits that did not reach the code would cancel the
    unreachable-node penalty for one side only.
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
        "parse_ok": float(verdict["parse_ok"]),
        "undefined": float(bool(verdict["undefined_names"])),
        "semantic_score": score,
        "coverage": detail.get("coverage", 0.0),
        "ordering": detail.get("ordering", 0.0),
        "structure": detail.get("structure", 0.0),
        "loops": detail.get("loops", 0.0),
        "terminal_reward": reward,
    }
    if sandbox is not None:
        out["sandbox_pass"] = float(bool(sandbox(code).get("passed")))
    return out


def complete_with(graph: DiagramGraph, prefix: Sequence[int], tail: Sequence[int]) -> list[int]:
    """`prefix`, then every node of `tail` not already in it, in `tail`'s order."""
    seen = set(prefix)
    return list(prefix) + [i for i in tail if i not in seen]


Player = Callable[[DiagramGraph], Episode]


def arms_for(
    diagram: dict, graph: DiagramGraph, players: dict[str, Player]
) -> dict[str, tuple[list[int], int]]:
    """Every candidate `(order, own loop marks)` for one diagram, as node indices."""
    from src.rl.baselines import ARMS

    index = graph.index_of
    orders: dict[str, tuple[list[int], int]] = {}
    for name in HEURISTICS:
        orders[name] = ([index[i] for i in ARMS[name](diagram) if i in index], 0)
    dfs = orders["dfs"][0]
    episodes = {"gold_in_action_space": gold_episode(graph)}
    episodes.update({name: player(graph) for name, player in players.items()})
    for name, episode in episodes.items():
        prefix = list(episode.state.emit_sequence)
        marks = episode.state.loop_marked
        orders[name] = (prefix, marks)
        short = "gold" if name == "gold_in_action_space" else name
        orders[f"{short}+dfs"] = (complete_with(graph, prefix, dfs), marks)
    return orders


def measure(
    diagrams: Sequence[dict],
    graphs: Sequence[DiagramGraph],
    players: dict[str, Player],
    sandbox_ids: set[str] = frozenset(),
    sandbox=None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Per-diagram rows, `rows[mode][arm] = [row per diagram]`, in `graphs` order."""
    from src.rl.baselines import score as edge_score

    rows: dict[str, dict[str, list[dict[str, Any]]]] = {"own": {}, "oracle": {}}
    for diagram, graph in zip(diagrams, graphs, strict=True):
        if not graph.n_nodes:
            continue
        oracle_marks = 0
        for source, _ in graph.back_edges:
            oracle_marks |= 1 << source
        use_sandbox = sandbox if graph.diagram_id in sandbox_ids else None
        for name, (order, marks) in arms_for(diagram, graph, players).items():
            completed = complete_with(graph, order, range(graph.n_nodes))
            f1 = edge_score(diagram, [graph.node_ids[i] for i in completed])["edge_f1"]
            for mode, mode_marks, box in (
                ("own", marks, use_sandbox),
                ("oracle", oracle_marks, None),
            ):
                row = code_quality(graph, order, mode_marks, box)
                row["edge_f1"] = f1
                row["diagram_id"] = graph.diagram_id
                rows[mode].setdefault(name, []).append(row)
    return rows


def table(rows: dict[str, list[dict[str, Any]]], keep: set[str] | None = None) -> dict[str, Any]:
    """Mean of every metric per arm, optionally restricted to the diagram ids in `keep`."""
    out: dict[str, Any] = {}
    for name, values in rows.items():
        values = [v for v in values if keep is None or v["diagram_id"] in keep]
        block: dict[str, Any] = {"diagrams": len(values)}
        for metric in METRICS:
            series = [v[metric] for v in values if metric in v]
            block[metric] = round(statistics.fmean(series), 4) if series else None
            if metric == "sandbox_pass":
                block["sandbox_n"] = len(series)
        out[name] = block
    return out


def paired(
    rows: dict[str, list[dict[str, Any]]], a: str, b: str, metric: str, keep: set[str] | None
) -> dict[str, Any]:
    """Per-diagram `a - b`: mean, wins / ties / losses, and a 2-SE band over diagrams."""
    diffs = [
        x[metric] - y[metric]
        for x, y in zip(rows[a], rows[b], strict=True)
        if (keep is None or x["diagram_id"] in keep) and metric in x and metric in y
    ]
    if not diffs:
        return {}
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return {
        "mean": round(statistics.fmean(diffs), 4),
        "two_se": round(2 * sd / len(diffs) ** 0.5, 4),
        "wins": sum(d > 1e-9 for d in diffs),
        "ties": sum(abs(d) <= 1e-9 for d in diffs),
        "losses": sum(d < -1e-9 for d in diffs),
    }


def contribution(tables: Sequence[dict[str, Any]], learned: str, metric: str) -> dict[str, Any]:
    """Learned minus the best *heuristic* (never gold), with the seed spread of the difference."""
    diffs, hybrid, vs_gold = [], [], []
    best = None
    needed = list(HEURISTICS) + [learned, f"{learned}+dfs", "gold_in_action_space"]
    if any(tab[name][metric] is None for tab in tables for name in needed):
        return {"metric": metric, "measured": False}
    for tab in tables:
        best = max(HEURISTICS, key=lambda h: tab[h][metric])
        diffs.append(tab[learned][metric] - tab[best][metric])
        hybrid.append(tab[f"{learned}+dfs"][metric] - tab["dfs"][metric])
        vs_gold.append(tab[learned][metric] - tab["gold_in_action_space"][metric])

    def stat(values: list[float]) -> dict[str, Any]:
        return {
            "mean": round(statistics.fmean(values), 4),
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "values": [round(v, 4) for v in values],
        }

    return {
        "metric": metric,
        "best_heuristic": best,
        "learned_minus_best_heuristic": stat(diffs),
        "hybrid_minus_dfs": stat(hybrid),
        "learned_minus_gold_in_action_space": stat(vs_gold),
    }


# ------------------------------------------------------------------------------------------
# the learned players
# ------------------------------------------------------------------------------------------


def _train_shaped(graphs: Sequence[DiagramGraph], episodes: int, seed: int):
    from src.rl.qlearning import TrainConfig, train
    from src.rl.shaping import phi_emitted

    cfg = TrainConfig(alpha=0.2, gamma=0.99, episodes=episodes, seed=seed, potential=phi_emitted)
    return train(graphs, cfg)


def players_for_seed(seed: int, tabular, graphs: Sequence[DiagramGraph]) -> dict[str, Player]:
    """The DQN checkpoint for `seed` (batched over `graphs`) and the shaped table for `seed`."""
    from src.rl.dqn import CHECKPOINTS, DQNAgent, batched_player

    path = CHECKPOINTS / f"p11-dqn-features-s{seed}.pt"
    # CPU on purpose: one batched forward over 993 diagrams per step is trivial for a 28-wide MLP,
    # and a CUDA context per worker process would cost hundreds of MB for nothing.
    dqn = DQNAgent.load(path, device="cpu")
    return {"dqn": batched_player(dqn, graphs), "tabular_shaped": tabular.rollout}


def _seed_job(
    seed: int, episodes: int, limit: int | None, sample: Sequence[str], use_sandbox: bool
) -> dict[str, Any]:
    """One seed end to end in its own process: train the shaped table, load the DQN, measure."""
    from src.rl.qlearning import training_set

    raw, graphs = training_set(limit, ambiguous=False)
    sandbox = None
    if use_sandbox:
        from src.rl.reward import sandbox_hook

        sandbox = sandbox_hook()
    tabular = _train_shaped(graphs, episodes, seed)
    return measure(raw, graphs, players_for_seed(seed, tabular, graphs), set(sample), sandbox)


def run(
    episodes: int = 60_000,
    limit: int | None = None,
    seeds: Sequence[int] = SEEDS,
    sandbox_sample: int = SANDBOX_SAMPLE,
    write: bool = False,
    workers: int = 3,
) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor

    from src.rl.qlearning import RUNS, training_set

    raw, graphs = training_set(limit, ambiguous=False)
    raw_amb, _ = training_set(limit, ambiguous=True)
    amb_ids = {str(d.get("id")) for d in raw_amb}
    ids = sorted(g.diagram_id for g in graphs)
    sample = set(random.Random("ablation-sandbox").sample(ids, min(sandbox_sample, len(ids))))
    jobs = [(seed, episodes, limit, sorted(sample), bool(sandbox_sample)) for seed in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(seeds)))) as pool:
            per_seed = dict(zip(seeds, pool.map(_seed_job, *zip(*jobs, strict=True)), strict=True))
    else:
        per_seed = {job[0]: _seed_job(*job) for job in jobs}
    if write:  # the per-diagram rows are the expensive part; keep them before aggregating
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "ablation_rows.json").write_text(json.dumps(per_seed), encoding="utf-8")

    result: dict[str, Any] = {
        "episodes_tabular": episodes,
        "seeds": list(seeds),
        "sandbox_sample": len(sample),
        "sandbox_sample_sources": _sources(graphs, sample),
        "diagrams": {"all": len(graphs), "ambiguous": len(amb_ids)},
    }
    for scope, keep in (("all", None), ("ambiguous", amb_ids)):
        block: dict[str, Any] = {}
        for mode in ("own", "oracle"):
            tables = [table(per_seed[s][mode], keep) for s in seeds]
            block[mode] = {
                "table_seed_mean": _mean_tables(tables),
                "contribution": {
                    learned: {m: contribution(tables, learned, m) for m in METRICS_VERDICT}
                    for learned in ("dqn", "tabular_shaped")
                },
                "paired_seed0": {
                    f"{a} - {b}": {
                        m: paired(per_seed[seeds[0]][mode], a, b, m, keep) for m in METRICS_VERDICT
                    }
                    for a, b in PAIRS
                },
            }
        result[scope] = block
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["figure"] = str(plot(result).relative_to(ROOT))
    return result


METRICS_VERDICT: tuple[str, ...] = ("semantic_score", "terminal_reward", "edge_f1", "sandbox_pass")
PAIRS: tuple[tuple[str, str], ...] = (
    ("dqn+dfs", "dfs"),
    ("tabular_shaped+dfs", "dfs"),
    ("gold+dfs", "dfs"),
    ("dqn", "gold_in_action_space"),
)


def _sources(graphs: Sequence[DiagramGraph], ids: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in graphs:
        if g.diagram_id in ids:
            out[g.source] = out.get(g.source, 0) + 1
    return out


def _mean_tables(tables: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Seed-mean of every cell, with the seed SD for arms that vary by seed."""
    out: dict[str, Any] = {}
    for arm in tables[0]:
        block: dict[str, Any] = {"diagrams": tables[0][arm]["diagrams"]}
        for metric in METRICS:
            values = [t[arm][metric] for t in tables if t[arm][metric] is not None]
            if not values:
                block[metric] = None
                continue
            block[metric] = round(statistics.fmean(values), 4)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            if sd > 0:
                block[f"{metric}_sd"] = round(sd, 4)
        block["sandbox_n"] = tables[0][arm].get("sandbox_n")
        out[arm] = block
    return out


ORDER: tuple[str, ...] = (
    "reading",
    "bfs",
    "topological",
    "dfs",
    "gold_in_action_space",
    "tabular_shaped",
    "dqn",
    "gold+dfs",
    "tabular_shaped+dfs",
    "dqn+dfs",
)


def plot(result: dict[str, Any], path=FIGURE):
    """Four metrics per arm on the 993, own marks; heuristics grey, learned red, hybrids purple."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = result["all"]["own"]["table_seed_mean"]
    names = [n for n in ORDER if n in tab]

    def colour(name: str) -> str:
        if name in HEURISTICS:
            return "#8c8c8c"
        if name.startswith("gold"):
            return "#2ca02c" if "+" not in name else "#98df8a"
        return "#9467bd" if "+" in name else "#d62728"

    panels = (
        ("semantic_score", "semantic score (11.1.3 static proxy)"),
        ("terminal_reward", "terminal reward"),
        ("edge_f1", "edge F1 (11.2.7)"),
        ("sandbox_pass", f"sandbox pass rate (n={result['sandbox_sample']} diagrams)"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        values = [tab[n][metric] for n in names]
        errors = [tab[n].get(f"{metric}_sd", 0.0) for n in names]
        ax.barh(range(len(names)), values, xerr=errors, color=[colour(n) for n in names], capsize=3)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
        ax.axvline(tab["dfs"][metric], color="black", lw=0.8, ls="--")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names), fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "Pipeline quality by ordering arm, 993 labelled diagrams, own loop marks "
        f"(learned arms: mean ± sd over {len(result['seeds'])} seeds; dashed = dfs)",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.10 pipeline ablation without RL")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--sandbox", type=int, default=SANDBOX_SAMPLE)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        out = run(300, limit=30, seeds=(0,), sandbox_sample=0, workers=1)
    else:
        out = run(args.episodes, args.limit, SEEDS[: args.seeds], args.sandbox, args.write)
    print(json.dumps(out, indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

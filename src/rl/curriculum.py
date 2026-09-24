"""Phase 11.2.6 - curriculum: clean synthetic graphs first, messy real ones after, and the
ablation that says whether the order mattered.

    python -m src.rl.curriculum --write     # five arms x SEEDS, evaluated on the real corpus
    python -m src.rl.curriculum --quick     # a short run, for the tests

## Why a curriculum is even a candidate here

11.2.1's agent plateaus: |TD| tail/head 0.82 at 60,000 episodes and 0.82 at 250,000, with greedy
quality falling over the longer run. One standard reading of that shape is that early episodes are
almost all failure - on the 993 labelled diagrams a uniform random policy emits 6.27% of nodes
and ends every one of its 993 episodes by choosing `terminate` - so the table spends its budget
learning what not to do on pages where the right thing is dozens of steps away. A curriculum
answers that by making the first pages ones where success is a few steps away.

## The synthetic stages come from `src.synth.graphs`, not from a private generator

The inherited draft carried its own four-family generator and said `src/synth` "builds images".
That was already false - `src.synth.graphs` builds IR, for 12.1.4 - and the private generator
had two measured defects, so it was deleted rather than kept alongside:

    duplicates    `generate(family, size, index)` used `index` only in the id, and `nested`
                  ignored `size` too, so the draft's two synthetic stages - 96 pages - held
                  **10 distinct graphs**, and its 24 nested pages were **one graph 24 times**.
    roles         start/end nodes carried the role `terminator`, which `src.parse.roles` does not
                  map, so **31.58% of its nodes encoded as `unknown`** against 8.96% on the real
                  labelled set - the opposite of the draft's claim that its vocabulary was the
                  encoder's.

`src.synth.graphs.random_diagram` samples a program and lowers it (flowcharts) or samples a DFA
(state machines, the shape of fa_bresler), seeded per page, with start/end roles the encoder
maps; its flowchart `io` role is still unmapped, so the synthetic stages sit at 15.7-16.2%
`unknown` - better than the draft, still above real, and stated rather than hidden. A page is
admitted to a *clean* stage only if it is single-component, has no unresolved edge and gold play
reaches full coverage on it; `clean_report` measures all three rather than trusting the
generator's intent, and the filter drops 104 of 1,200 structured pages (state machines whose
random transitions leave a state unreachable). Measured pools (`--pools`):

    stage                  pages  distinct  gold full cov  unresolved  multi-comp  max out-deg
    synthetic_linear         400        43        100.00%           0           0            2
    synthetic_structured   1,096       816        100.00%           0           0            3
    real_simple              422       170         94.31%         271           0            5
    real_all                 993       738         41.69%         530         555            5

## The stages

    synthetic_linear      linear flowcharts and linear state machines          15% of budget
    synthetic_structured  branching / looping / nested, both types, clean      20%
    real_simple           real labelled pages, single-component, <= 12 nodes   25%
    real_all              all 993 labelled pages                               40%

## The ablation, and the control the draft was missing

Five arms, **the same total episode budget**, all evaluated greedily on the real labelled set with
11.2.1's `play` harness, ranked on terminal reward (the objective), with edge F1 reported beside:

    flat            all episodes on real_all, one epsilon decay. 11.2.1's setting.
    flat_restarts   all episodes on real_all, but split into the curriculum's four slices with
                    epsilon restarting at each. **Added**: `train_curriculum` restarts epsilon at
                    every stage, so `curriculum` against `flat` confounds the ordering of the data
                    with a four-times-restarted exploration schedule. This arm has the schedule and
                    not the curriculum.
    curriculum      the four stages in order.
    reversed        the four stages backwards, same shares moving with their stages.
    synthetic_only  both synthetic stages only - the transfer check.

Both learners run the same five arms: 11.2.1's table (`--learner tabular`, 5 seeds, each arm-seed a
worker process) and 11.2.5's DQN (`--learner dqn`, 3 seeds as one ensemble per arm, weights
carried across slices while replay, Adam moments and epsilon restart). The measured results are in
`reports/rl_curriculum.md`, `experiments/rl/curriculum.json` and `curriculum_dqn.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from src.rl.state import DiagramGraph, role_index
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p11_curriculum.png"

#: Five seeds: the tabular runs are cheap, and a curriculum effect is a small difference.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: 11.2.1's swept cell. Not re-swept here: the question is the data order, not the step size.
ALPHA = 0.4
GAMMA = 1.0

SYNTHETIC_TYPES: tuple[str, ...] = ("flowchart", "state_machine")
LINEAR: tuple[str, ...] = ("linear",)
STRUCTURED: tuple[str, ...] = ("branching", "looping", "nested")

#: Pages drawn per (type, structure) cell before the clean filter.
PER_CELL = 200

#: (stage name, share of the episode budget), easy to hard.
PLAN: tuple[tuple[str, float], ...] = (
    ("synthetic_linear", 0.15),
    ("synthetic_structured", 0.20),
    ("real_simple", 0.25),
    ("real_all", 0.40),
)

ARMS: tuple[str, ...] = ("flat", "flat_restarts", "curriculum", "reversed", "synthetic_only")

#: 11.2.5's swept DQN cell, used unchanged when the curriculum is run on the DQN.
DQN_CELL: dict[str, Any] = {"inputs": "features", "lr": 3e-4, "gamma": 1.0}
DQN_SEEDS: tuple[int, ...] = (0, 1, 2)


# ------------------------------------------------------------------------------------------
# clean synthetic IR
# ------------------------------------------------------------------------------------------


def is_clean(graph: DiagramGraph) -> bool:
    """Single component, no unresolved edge, and gold play reaches full coverage."""
    from src.rl.qlearning import gold_player

    return (
        graph.n_nodes > 0
        and graph.n_components == 1
        and graph.n_unresolved == 0
        and gold_player(graph).full_coverage()
    )


def synthetic_pool(
    structures: Sequence[str],
    types: Sequence[str] = SYNTHETIC_TYPES,
    per_cell: int = PER_CELL,
    seed: int = 0,
) -> tuple[list[dict], dict[str, Any]]:
    """Clean synthetic pages from `src.synth.graphs`, and what the clean filter removed."""
    from src.synth.graphs import random_diagram

    kept: list[dict] = []
    drawn = 0
    dropped = 0
    for diagram_type in types:
        for structure in structures:
            for index in range(per_cell):
                diagram = random_diagram(diagram_type, structure, seed * 100_000 + index)
                drawn += 1
                if is_clean(DiagramGraph.from_ir(diagram)):
                    kept.append(diagram)
                else:
                    dropped += 1
    return kept, {"drawn": drawn, "kept": len(kept), "dropped_unclean": dropped}


def signature(graph: DiagramGraph) -> tuple:
    """Structure-only identity: roles and successor lists. Two pages with equal signatures are
    the same learning problem whatever their ids and texts say."""
    return (tuple(role_index(r) for r in graph.roles), graph.successors)


def clean_report(diagrams: Sequence[dict]) -> dict[str, Any]:
    """Measured, not assumed: the properties that make a pool "clean", and how varied it is."""
    from src.rl.qlearning import gold_player

    graphs = [DiagramGraph.from_ir(d) for d in diagrams]
    if not graphs:
        return {"diagrams": 0}
    nodes = sum(g.n_nodes for g in graphs)
    unknown = sum(role_index(r) == len_roles() - 1 for g in graphs for r in g.roles)
    return {
        "diagrams": len(graphs),
        "distinct_structures": len({signature(g) for g in graphs}),
        "gold_full_coverage": round(
            statistics.fmean(float(gold_player(g).full_coverage()) for g in graphs), 4
        ),
        "multi_component": sum(g.n_components > 1 for g in graphs),
        "with_unresolved": sum(g.n_unresolved > 0 for g in graphs),
        "with_back_edge": sum(bool(g.back_edges) for g in graphs),
        "nodes_median": statistics.median(g.n_nodes for g in graphs),
        "nodes_max": max(g.n_nodes for g in graphs),
        "max_out_degree": max(max((len(s) for s in g.successors), default=0) for g in graphs),
        "unknown_role_share": round(unknown / max(1, nodes), 4),
    }


def len_roles() -> int:
    from src.rl.state import ROLE_VOCAB

    return len(ROLE_VOCAB)


# ------------------------------------------------------------------------------------------
# the stages and the arms
# ------------------------------------------------------------------------------------------


def stages(limit: int | None = None) -> dict[str, dict[str, Any]]:
    """Every stage's pool, as raw IR and compiled graphs, plus its clean report."""
    from src.rl.qlearning import training_set

    raw, graphs = training_set(limit, ambiguous=False)
    simple = [(d, g) for d, g in zip(raw, graphs, strict=True) if g.n_components == 1]
    simple = [(d, g) for d, g in simple if g.n_nodes <= 12]
    linear, linear_drop = synthetic_pool(LINEAR)
    structured, structured_drop = synthetic_pool(STRUCTURED)
    pools = {
        "synthetic_linear": (linear, linear_drop),
        "synthetic_structured": (structured, structured_drop),
        "real_simple": ([d for d, _ in simple], None),
        "real_all": (raw, None),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, (diagrams, drop) in pools.items():
        compiled = (
            [g for _, g in simple]
            if name == "real_simple"
            else graphs if name == "real_all" else [DiagramGraph.from_ir(d) for d in diagrams]
        )
        out[name] = {
            "graphs": compiled,
            "report": {**clean_report(diagrams), **({"filter": drop} if drop else {})},
        }
    return out


def arm_plan(arm: str, pools: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    """The (pool, share) sequence an arm trains through. Shares always sum to 1."""
    if arm == "flat":
        return [("real_all", 1.0)]
    if arm == "flat_restarts":
        return [("real_all", share) for _, share in PLAN]
    if arm == "curriculum":
        return list(PLAN)
    if arm == "reversed":
        return list(reversed(PLAN))
    if arm == "synthetic_only":
        total = PLAN[0][1] + PLAN[1][1]
        return [(PLAN[0][0], PLAN[0][1] / total), (PLAN[1][0], PLAN[1][1] / total)]
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def budgets(plan: Sequence[tuple[str, float]], episodes: int) -> list[int]:
    """Integer slices of `episodes` that sum to it exactly - the draft's `int()` truncation
    could drop episodes, and an arm with fewer episodes than its control is not a result."""
    raw = [episodes * share for _, share in plan]
    slices = [int(v) for v in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - slices[i], reverse=True):
        if sum(slices) >= episodes:
            break
        slices[i] += 1
    return slices


def train_arm(
    arm: str, pools: dict[str, dict[str, Any]], episodes: int, seed: int
) -> tuple[Any, list[dict[str, Any]], list[float]]:
    """Train one tabular agent through an arm's plan, carrying the table forward.

    Each slice gets its own `TrainConfig`, so epsilon decays within a slice and restarts at the
    next. That is the draft's choice and it is kept, but it is now controlled for by
    `flat_restarts` rather than argued to be harmless.
    """
    from src.rl.qlearning import TabularAgent, TrainConfig, train

    agent = TabularAgent()
    history: list[dict[str, Any]] = []
    rewards: list[float] = []
    plan = arm_plan(arm, pools)
    for (pool, _), budget in zip(plan, budgets(plan, episodes), strict=True):
        if budget <= 0:
            continue
        cfg = TrainConfig(
            alpha=ALPHA, gamma=GAMMA, episodes=budget, seed=seed + 1000 * len(history)
        )
        train(pools[pool]["graphs"], cfg, agent=agent)
        rewards += agent.history["reward"]
        history.append(
            {
                "pool": pool,
                "episodes": budget,
                "reward_tail": agent.report["reward_tail"],
                "td_tail_ratio": agent.report["td_tail_ratio"],
                "reached_keys": agent.report["reached_keys"],
            }
        )
    return agent, history, rewards


def _run_one(arm: str, episodes: int, seed: int, limit: int | None) -> dict[str, Any]:
    """One (arm, seed) cell, self-contained so it can run in a worker process."""
    from src.rl.dqn import _trail
    from src.rl.qlearning import evaluate, training_set

    pools = stages(limit)
    raw, graphs = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    started = time.perf_counter()
    agent, history, rewards = train_arm(arm, pools, episodes, seed)
    return {
        "arm": arm,
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 1),
        "stages": history,
        "curve": _trail(rewards),
        "all": evaluate(agent, graphs, raw),
        "ambiguous": evaluate(agent, graphs_amb, raw_amb),
    }


def _run_dqn_arm(
    arm: str, episodes: int, seeds: Sequence[int], limit: int | None
) -> list[dict[str, Any]]:
    """One arm for 11.2.5's DQN: every seed is a member of one ensemble, carried across slices.

    Weights carry forward from slice to slice; the replay buffer, Adam moments and epsilon start
    afresh at each slice - the DQN analogue of the tabular restart, controlled by the same
    `flat_restarts` arm.
    """
    from dataclasses import replace

    from src.rl import dqn as D
    from src.rl.qlearning import training_set

    pools = stages(limit)
    raw, graphs = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    base = replace(D.DQNConfig(), **DQN_CELL)
    plan = arm_plan(arm, pools)
    started = time.perf_counter()
    nets = None
    histories: list[list[dict[str, Any]]] = [[] for _ in seeds]
    rewards: list[list[float]] = [[] for _ in seeds]
    agents = []
    for position, ((pool, _), budget) in enumerate(zip(plan, budgets(plan, episodes), strict=True)):
        cfgs = [replace(base, episodes=budget, seed=s + 1000 * position) for s in seeds]
        agents = D.train_many(pools[pool]["graphs"], cfgs, init=nets)
        nets = [a.net for a in agents]
        for k, agent in enumerate(agents):
            rewards[k] += agent.history["reward"]
            histories[k].append(
                {
                    "pool": pool,
                    "episodes": budget,
                    "reward_tail": agent.report.get("reward_tail"),
                    "td_tail_ratio": agent.report.get("td_tail_ratio"),
                }
            )
    seconds = round(time.perf_counter() - started, 1)
    out = []
    for k, seed in enumerate(seeds):
        agent = agents[k]
        agent.cfg = replace(agent.cfg, seed=seed)
        out.append(
            {
                "arm": arm,
                "seed": seed,
                "seconds": seconds,
                "stages": histories[k],
                "curve": D._trail(rewards[k]),
                "all": D.evaluate(agent, graphs, raw),
                "ambiguous": D.evaluate(agent, graphs_amb, raw_amb),
            }
        )
    return out


def summarise(rows: Sequence[dict[str, Any]], scope: str) -> dict[str, Any]:
    keys = ("mean_terminal_reward", "full_coverage", "mean_emitted_share", "mean_edge_f1")
    out: dict[str, Any] = {}
    for key in keys:
        values = [r[scope][key] for r in rows]
        out[key] = {
            "mean": round(statistics.fmean(values), 4),
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "values": values,
        }
    return out


def paired(rows: dict[str, list[dict[str, Any]]], a: str, b: str, key: str, scope: str = "all"):
    """Per-seed difference `a - b` (same seeds, so paired), with its mean, SD and 2-SE band."""
    by_seed = {r["seed"]: r for r in rows[b]}
    diffs = [
        r[scope][key] - by_seed[r["seed"]][scope][key] for r in rows[a] if r["seed"] in by_seed
    ]
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    band = 2 * sd / len(diffs) ** 0.5 if len(diffs) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "two_se": round(band, 4),
        "values": [round(d, 4) for d in diffs],
        "resolved": abs(mean) > band,
    }


def verdict(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Ranked on terminal reward. Each comparison is paired over seeds with a 2-SE band."""
    key = "mean_terminal_reward"
    out = {
        "curriculum_minus_flat": paired(rows, "curriculum", "flat", key),
        "curriculum_minus_flat_restarts": paired(rows, "curriculum", "flat_restarts", key),
        "flat_restarts_minus_flat": paired(rows, "flat_restarts", "flat", key),
        "curriculum_minus_reversed": paired(rows, "curriculum", "reversed", key),
        "synthetic_only_minus_flat": paired(rows, "synthetic_only", "flat", key),
        "curriculum_minus_flat_edge_f1": paired(rows, "curriculum", "flat", "mean_edge_f1"),
    }
    c = out["curriculum_minus_flat_restarts"]
    out["ordering_helps"] = bool(c["resolved"] and c["mean"] > 0)
    return out


def run(
    episodes: int = 60_000,
    limit: int | None = None,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[str] = ARMS,
    workers: int = 8,
    write: bool = False,
    learner: str = "tabular",
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS, references, training_set

    pools = stages(limit)
    raw, graphs = training_set(limit, ambiguous=False)
    jobs = [(arm, episodes, seed, limit) for arm in arms for seed in seeds]
    if learner == "dqn":
        # one process per arm, each an ensemble over the seeds (11.2.5's throughput finding)
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(arms)))) as pool:
            futures = [pool.submit(_run_dqn_arm, arm, episodes, seeds, limit) for arm in arms]
            results = [row for f in futures for row in f.result()]
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_one, *zip(*jobs, strict=True)))
    else:
        results = [_run_one(*job) for job in jobs]
    rows: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        rows.setdefault(r["arm"], []).append(r)
    result: dict[str, Any] = {
        "learner": learner,
        "episodes": episodes,
        "seeds": list(seeds),
        "dqn_cell": DQN_CELL if learner == "dqn" else None,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "pools": {name: block["report"] for name, block in pools.items()},
        "plans": {arm: arm_plan(arm, pools) for arm in arms},
        "arms": {
            arm: {"all": summarise(rs, "all"), "ambiguous": summarise(rs, "ambiguous")}
            for arm, rs in rows.items()
        },
        "runs": results,
        "references": references(graphs, raw),
    }
    if set(ARMS) <= set(rows):
        result["verdict"] = verdict(rows)
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        suffix = "" if learner == "tabular" else f"_{learner}"
        (RUNS / f"curriculum{suffix}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        figure = FIGURE.with_name(f"p11_curriculum{suffix}.png")
        result["figure"] = str(plot(result, figure).relative_to(ROOT))
    return result


def plot(result: dict[str, Any], path=FIGURE):
    """Left: behaviour return per arm (seed 0) with stage boundaries. Right: greedy results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "flat": "#1f77b4",
        "flat_restarts": "#17becf",
        "curriculum": "#d62728",
        "reversed": "#ff7f0e",
        "synthetic_only": "#9467bd",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    first = result["seeds"][0]
    for run_ in result["runs"]:
        if run_["seed"] != first or not run_["curve"]:
            continue
        xs, ys = zip(*run_["curve"], strict=True)
        axes[0].plot(xs, ys, color=colours[run_["arm"]], lw=1.1, label=run_["arm"])
    edge = 0
    for _, share in PLAN[:-1]:
        edge += share * result["episodes"]
        axes[0].axvline(edge, color="#999999", lw=0.8, ls=":")
    axes[0].set_title(
        f"training return (trailing mean 1,000, seed {first}); dotted = curriculum stage edges",
        fontsize=9,
    )
    axes[0].set_xlabel("episode", fontsize=9)
    axes[0].set_ylabel("return (on that stage's pool)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)

    names = [a for a in ARMS if a in result["arms"]]
    refs = result["references"]
    for ax, key, label in (
        (axes[1], "mean_terminal_reward", "greedy terminal reward"),
        (axes[2], "mean_edge_f1", "greedy edge F1"),
    ):
        block = [result["arms"][a]["all"][key] for a in names]
        ax.bar(
            range(len(names)),
            [b["mean"] for b in block],
            yerr=[b["sd"] for b in block],
            capsize=4,
            color=[colours[a] for a in names],
        )
        for i, b in enumerate(block):
            ax.scatter([i] * len(b["values"]), b["values"], color="black", s=8, zorder=3)
        ax.axhline(refs["gold"][key], color="#2ca02c", ls="--", lw=1, label="gold in action space")
        ax.axhline(refs["random"][key], color="#7f7f7f", ls="--", lw=1, label="random")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_title(
            f"{label}, 993 labelled (mean ± sd, dots = {len(result['seeds'])} seeds)",
            fontsize=9,
        )
        ax.legend(fontsize=8, frameon=False)
    if refs["random"]["mean_terminal_reward"] < 0:
        low = min(refs["random"]["mean_terminal_reward"], axes[1].get_ylim()[0])
        axes[1].set_ylim(low - 0.2, 0)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.6 curriculum ablation")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pools", action="store_true", help="only the clean reports of the pools")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--learner", choices=("tabular", "dqn"), default="tabular")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.pools:
        print(json.dumps({k: v["report"] for k, v in stages(args.limit).items()}, indent=2))
        return 0
    if args.quick:
        out = run(600, limit=60, seeds=(0,), workers=1)
    else:
        seeds = (SEEDS if args.learner == "tabular" else DQN_SEEDS)[: args.seeds]
        out = run(
            args.episodes,
            args.limit,
            seeds,
            workers=args.workers,
            write=args.write,
            learner=args.learner,
        )
    out.pop("runs", None)
    print(json.dumps(out, indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

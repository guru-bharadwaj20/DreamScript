"""Phase 11.2.2 - SARSA against Q-learning: on-policy versus off-policy on ambiguous diagrams.

    python -m src.rl.sarsa --write    # sweep both, 5 seeds x 60k at each one's best cell, plot
    python -m src.rl.sarsa --quick    # a short run, for the tests

The two algorithms share `src.rl.qlearning.train` and differ in one line of it - the bootstrap is
`max_a' Q[s',a']` over the legal actions for Q-learning and `Q[s',a']` for the action the
behaviour policy actually chose for SARSA. Duplicating the loop would have made every other
difference between the runs a candidate explanation, so it is not duplicated.

## The hypothesis, stated before the numbers

On-policy control learns the value of the policy *including its exploration*, so it should avoid
states where an exploratory step is expensive. This environment has a concrete cliff of that
kind: `RewardConfig.infinite_loop` charges -2.0 when an episode is truncated at the `4n + 8` step
cap, and a greedy policy *can* reach it here (a follow/backtrack cycle never emits and never
terminates). The falsifiable prediction is therefore **SARSA truncates less often than
Q-learning**, both during training (`history["truncated"]`) and in play, and - the textbook
cliff-walking signature - does relatively better when evaluated *with* its exploration (epsilon
at the 0.05 floor) than greedily.

## How it is judged

* Each algorithm is swept over 11.2.1's alpha x gamma grid (12,000 episodes, seed 0) and run at
  its own best cell, ranked on greedy mean terminal reward on the 993 labelled diagrams - never on
  edge F1, which is the transfer metric.
* Because the two best cells differ, a difference at "own cells" could be the cell's rather than
  the algorithm's, so `run` also re-runs **both algorithms at each cell** (`controls`). The
  algorithm comparison is read off the shared-cell runs.
* Five seeds per configuration; mean, sd and the per-seed values are kept.
* Trained on all 993 labelled diagrams, evaluated greedily and at the epsilon floor ("online") on
  the 993 and on 11.2.7's 684-diagram ambiguous subset, by `src.rl.qlearning.play` - the harness
  that scores gold and random - so the references and the learners share one metric.

Measured results are in `reports/rl_sarsa.md`.

`run_jobs` is the process-pool runner 11.2.3 / 11.2.4 / 11.2.8 reuse: one job is one
`TrainConfig` trained from scratch and evaluated, so jobs are independent and seeds parallelise.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Any

from src.rl.abstraction import abstract, key_index
from src.rl.episode import Episode
from src.rl.qlearning import (
    ALPHAS,
    GAMMAS,
    RUNS,
    TabularAgent,
    TrainConfig,
    evaluate,
    play,
    references,
    train,
    training_set,
)
from src.rl.state import DiagramGraph
from src.utils.config import ROOT

OUT = RUNS / "sarsa.json"
FIGURE = ROOT / "reports" / "figures" / "p11_sarsa.png"

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
ALGOS: tuple[str, ...] = ("q", "sarsa")
#: The epsilon the online (with-exploration) evaluation plays at: the schedule's floor.
ONLINE_EPS = 0.05
#: Fixed-size smoothing window for training curves.
WINDOW = 1000

Sets = dict[str, tuple[list[dict], list[DiagramGraph]]]

# ------------------------------------------------------------------------------------------
# the shared job runner
# ------------------------------------------------------------------------------------------

_SETS: Sets = {}


def load_sets(limit: int | None = None) -> Sets:
    """All 993 labelled diagrams (the training set) and 11.2.7's ambiguous subset."""
    return {"all": training_set(limit, ambiguous=False), "ambiguous": training_set(limit, True)}


def _resolve(ref: str | None) -> Callable | None:
    """`"module:attr"` -> the object. Hooks travel to worker processes by name, not by pickle."""
    if not ref:
        return None
    module, attr = ref.split(":")
    return getattr(importlib.import_module(module), attr)


def run_job(spec: dict[str, Any], sets: Sets | None = None) -> dict[str, Any]:
    """Train one `TrainConfig` from scratch on `sets["all"]` and evaluate it on every set.

    `spec` keys: `cfg` (TrainConfig fields), optional `potential` / `step_bonus` ("mod:attr"),
    `keep` (history keys to return), `extra` ("mod:attr" called as `f(agent, sets, spec)`),
    `eval` (set names to evaluate greedily, default all), `on_episode` ("mod:factory" called as
    `factory(sets, spec)` to build the training hook; its `result()` is returned).
    """
    sets = sets if sets is not None else _SETS
    fields = dict(spec["cfg"])
    for hook in ("potential", "step_bonus"):
        if spec.get(hook):
            fields[hook] = _resolve(spec[hook])
    hook = None
    if spec.get("on_episode"):
        hook = _resolve(spec["on_episode"])(sets, spec)
        fields["on_episode"] = hook
    raw_all, graphs_all = sets["all"]
    agent = train(graphs_all, TrainConfig(**fields))
    out: dict[str, Any] = {
        "spec": spec,
        "report": agent.report,
        "history": {k: agent.history[k] for k in spec.get("keep", ())},
        "eval": {
            name: evaluate(agent, sets[name][1], sets[name][0])
            for name in spec.get("eval", list(sets))
        },
    }
    if hook is not None and hasattr(hook, "result"):
        out["on_episode"] = hook.result()
    if spec.get("extra"):
        out["extra"] = _resolve(spec["extra"])(agent, sets, spec)
    return out


def _init(sets: Sets) -> None:
    _SETS.update(sets)


def run_jobs(
    specs: Sequence[dict[str, Any]], sets: Sets, workers: int | None = None
) -> list[dict[str, Any]]:
    """Every spec, in order. `workers <= 1` runs inline (the tests); otherwise a process pool."""
    workers = min(len(specs), workers or max(1, (os.cpu_count() or 2) // 3))
    if workers <= 1:
        return [run_job(s, sets) for s in specs]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(sets,)) as pool:
        return pool.map(run_job, specs, chunksize=1)


# ------------------------------------------------------------------------------------------
# measures
# ------------------------------------------------------------------------------------------


def truncation_rate(report: dict[str, Any]) -> float:
    """Share of evaluation episodes stopped by the step cap - the cliff, as a number."""
    stopped = report.get("stopped_by", {})
    total = sum(stopped.values()) or 1
    return round(stopped.get("cap", 0) / total, 4)


def smooth(values: Sequence[float], window: int = 200) -> list[float]:
    """A trailing mean. Per-episode reward is dominated by which diagram was drawn (2 to 40
    nodes), so the raw series is unreadable."""
    out: list[float] = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def stat(values: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values]
    return {
        "mean": round(statistics.fmean(values), 4),
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "values": [round(v, 4) for v in values],
    }


def epsilon_player(agent: TabularAgent, eps: float, seed: int) -> Callable[[DiagramGraph], Episode]:
    """The behaviour policy frozen at `eps`: what the learner actually does while it explores."""
    rng = random.Random(seed)

    def player(graph: DiagramGraph) -> Episode:
        episode = Episode(graph)
        while not episode.done():
            mask = episode.mask()
            if rng.random() < eps:
                action = rng.choice([a for a, ok in enumerate(mask) if ok])
            else:
                action = agent.greedy(key_index(abstract(graph, episode.state)), mask, rng)
            episode.apply(action)
        return episode

    return player


def online_eval(agent: TabularAgent, sets: Sets, spec: dict[str, Any]) -> dict[str, Any]:
    """`run_job` extra: play at the epsilon floor on every set, reward and truncation."""
    out = {}
    for name, (raw, graphs) in sets.items():
        row = play(epsilon_player(agent, ONLINE_EPS, 1000 + spec["cfg"]["seed"]), graphs, raw)
        out[name] = {
            "mean_terminal_reward": row["mean_terminal_reward"],
            "truncation_rate": truncation_rate(row),
            "mean_edge_f1": row.get("mean_edge_f1"),
            "full_coverage": row["full_coverage"],
        }
    return out


# ------------------------------------------------------------------------------------------
# the study
# ------------------------------------------------------------------------------------------


def sweep_specs(episodes: int, seed: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "cfg": {"algo": algo, "alpha": a, "gamma": g, "episodes": episodes, "seed": seed},
            "eval": ["all"],
        }
        for algo in ALGOS
        for a in ALPHAS
        for g in GAMMAS
    ]


def best_cells(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per algorithm, the cell with the highest greedy terminal reward on the labelled set."""
    out = {}
    for algo in ALGOS:
        mine = [r for r in rows if r["spec"]["cfg"]["algo"] == algo]
        top = max(mine, key=lambda r: r["eval"]["all"]["mean_terminal_reward"])
        out[algo] = {"alpha": top["spec"]["cfg"]["alpha"], "gamma": top["spec"]["cfg"]["gamma"]}
    return out


def summarise(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Across seeds: greedy and online play per set, convergence, and training truncation."""
    out: dict[str, Any] = {}
    for name in jobs[0]["eval"]:
        rows = [j["eval"][name] for j in jobs]
        block = {
            key: stat([r[key] for r in rows])
            for key in (
                "mean_terminal_reward",
                "full_coverage",
                "mean_emitted_share",
                "mean_episode_length",
                "mean_edge_f1",
            )
        }
        block["truncation_rate"] = stat([truncation_rate(r) for r in rows])
        block["empty_policy_edge_f1"] = rows[0]["empty_policy_edge_f1"]
        block["gold_full_coverage_ceiling"] = rows[0]["gold_full_coverage_ceiling"]
        if "extra" in jobs[0]:
            online = [j["extra"][name] for j in jobs]
            block["online"] = {
                k: stat([o[k] for o in online])
                for k in ("mean_terminal_reward", "truncation_rate", "mean_edge_f1")
            }
        out[name] = block
    for key in ("td_tail_ratio", "reached_keys"):
        out[key] = stat([j["report"][key] for j in jobs])
    if "truncated" in jobs[0]["history"]:
        tails = []
        for j in jobs:
            t = j["history"]["truncated"]
            tails.append(statistics.fmean(t[-max(1, len(t) // 10) :]))
        out["train_truncation_last_10pct"] = stat(tails)
    return out


def compare_algorithms(
    sets: Sets,
    episodes: int = 60_000,
    seeds: Sequence[int] = SEEDS,
    cells: dict[str, dict[str, float]] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Both algorithms at their cells over `seeds`. `cells` defaults to 11.2.1's (0.4, 1.0)."""
    cells = cells or {algo: {"alpha": 0.4, "gamma": 1.0} for algo in ALGOS}
    specs = [
        {
            "cfg": {"algo": algo, **cells[algo], "episodes": episodes, "seed": seed},
            "keep": ["reward", "td_error", "truncated"],
            "extra": "src.rl.sarsa:online_eval",
        }
        for algo in ALGOS
        for seed in seeds
    ]
    jobs = run_jobs(specs, sets, workers)
    result: dict[str, Any] = {"episodes": episodes, "seeds": list(seeds), "algos": {}}
    for algo in ALGOS:
        mine = [j for j in jobs if j["spec"]["cfg"]["algo"] == algo]
        block = summarise(mine)
        block["cell"] = cells[algo]
        window = min(WINDOW, max(1, episodes // 20))
        for key in ("reward", "td_error", "truncated"):
            curves = [smooth(j["history"][key], window) for j in mine]
            step = max(1, episodes // 400)
            block[f"curve_{key}"] = {
                "x": list(range(0, episodes, step)),
                "mean": [
                    round(statistics.fmean(c[i] for c in curves), 5)
                    for i in range(0, episodes, step)
                ],
                "sd": [
                    round(statistics.pstdev([c[i] for c in curves]), 5)
                    for i in range(0, episodes, step)
                ],
            }
        result["algos"][algo] = block
    return result


# ------------------------------------------------------------------------------------------
# plot
# ------------------------------------------------------------------------------------------

COLOURS = {"q": "#2a78d6", "sarsa": "#eb6834"}
LABELS = {"q": "Q-learning (off-policy)", "sarsa": "SARSA (on-policy)"}


def _cell_name(cell: dict[str, float]) -> str:
    return f"a{cell['alpha']} g{cell['gamma']}"


def plot(result: dict[str, Any], refs: dict[str, Any] | None = None, path=FIGURE):
    """Training curves, then greedy reward and truncation under every cell configuration run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    for algo in ALGOS:
        block = result["algos"][algo]
        for ax, key in ((axes[0, 0], "curve_reward"), (axes[0, 1], "curve_truncated")):
            c = block[key]
            keep = [i for i, x in enumerate(c["x"]) if x >= min(WINDOW, result["episodes"] // 20)]
            xs = [c["x"][i] for i in keep]
            mean = [c["mean"][i] for i in keep]
            sd = [c["sd"][i] for i in keep]
            lo = [m - v for m, v in zip(mean, sd, strict=True)]
            hi = [m + v for m, v in zip(mean, sd, strict=True)]
            ax.fill_between(xs, lo, hi, color=COLOURS[algo], alpha=0.2, lw=0)
            label = f"{LABELS[algo]} at {_cell_name(block['cell'])}"
            ax.plot(xs, mean, color=COLOURS[algo], lw=2, label=label)
    axes[0, 0].set_title("training return, each at its own best cell (band = ±sd over seeds)")
    axes[0, 0].set_ylabel("return per episode")
    axes[0, 1].set_title("training episodes truncated at the step cap")
    axes[0, 1].set_ylabel("share of episodes")
    for ax in axes[0]:
        ax.set_xlabel("episode")
        ax.legend(frameon=False)

    configs = [("own best\ncells", result["algos"])]
    for block in result.get("controls", {}).values():
        cell = next(iter(block["algos"].values()))["cell"]
        configs.append((f"both at\n{_cell_name(cell)}", block["algos"]))
    groups = [(scope, name, algos) for scope in ("all", "ambiguous") for name, algos in configs]
    for ax, metric, title, ylabel in (
        (axes[1, 0], "mean_terminal_reward", "greedy terminal reward", "mean terminal reward"),
        (axes[1, 1], "truncation_rate", "greedy episodes truncated at the cap", "share"),
    ):
        for gi, (scope, _, algos) in enumerate(groups):
            for k, algo in enumerate(ALGOS):
                s = algos[algo][scope][metric]
                x = gi + (k - 0.5) * 0.38
                ax.bar(
                    x, s["mean"], 0.36, color=COLOURS[algo], label=LABELS[algo] if gi == 0 else None
                )
                ax.scatter([x] * len(s["values"]), s["values"], color="#222222", s=9, zorder=3)
            if refs and metric == "mean_terminal_reward":
                ax.hlines(
                    refs[scope]["gold"]["mean_terminal_reward"],
                    gi - 0.45,
                    gi + 0.45,
                    color="#1baf7a",
                    lw=2,
                    label="gold play" if gi == 0 else None,
                )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{scope}\n{name}" for scope, name, _ in groups], fontsize=8)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_title(title + " (dots = seeds)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"11.2.2 - SARSA vs Q-learning, {len(result['seeds'])} seeds x "
        f"{result['episodes']:,} episodes, trained on 993 labelled diagrams"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def replot() -> Any:
    """Redraw the figure from the written JSON without retraining."""
    result = json.loads(OUT.read_text(encoding="utf-8"))
    return plot(result, result.get("references"))


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------


def run(
    episodes: int = 60_000,
    sweep_episodes: int = 12_000,
    limit: int | None = None,
    seeds: Sequence[int] = SEEDS,
    write: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    sets = load_sets(limit)
    rows = run_jobs(sweep_specs(sweep_episodes), sets, workers)
    cells = best_cells(rows)
    result: dict[str, Any] = {
        "sweep": [
            {
                **r["spec"]["cfg"],
                "mean_terminal_reward": r["eval"]["all"]["mean_terminal_reward"],
                "truncation_rate": truncation_rate(r["eval"]["all"]),
            }
            for r in rows
        ],
        "best_cells": cells,
    }
    result.update(compare_algorithms(sets, episodes, seeds, cells, workers))
    # The main comparison runs each algorithm at its own cell, so a difference there could be the
    # cell's rather than the algorithm's. The control re-runs both algorithms at each cell.
    result["controls"] = {}
    for owner in ALGOS:
        shared = {algo: cells[owner] for algo in ALGOS}
        if shared == cells:
            continue
        block = compare_algorithms(sets, episodes, seeds, shared, workers)
        for algo_block in block["algos"].values():
            for key in [k for k in algo_block if k.startswith("curve_")]:
                algo_block.pop(key)
        result["controls"][f"both_at_{owner}_cell"] = block
    result["references"] = {name: references(g, raw) for name, (raw, g) in sets.items()}
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["figure"] = str(plot(result, result["references"]).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.2 SARSA vs Q-learning")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--sweep-episodes", type=int, default=12_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the JSON")
    args = ap.parse_args(argv)
    if args.replot:
        print(replot())
        return 0
    if args.quick:
        out = run(600, 300, args.limit, (0,), workers=1)
    else:
        seeds = tuple(range(max(1, args.seeds)))
        out = run(args.episodes, args.sweep_episodes, args.limit, seeds, args.write, args.workers)
    for block in out.get("algos", {}).values():
        for key in [k for k in block if k.startswith("curve_")]:
            block.pop(key)
    print(json.dumps(out, indent=2)[:30000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

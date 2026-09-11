"""Phase 11.2.2 - SARSA against Q-learning: on-policy versus off-policy on ambiguous diagrams.

    python -m src.rl.sarsa            # both algorithms, several seeds, the plot, the JSON
    python -m src.rl.sarsa --quick    # a short run, for the tests

The two algorithms share `src.rl.qlearning.train` and differ in one line of it - the bootstrap is
`max_a' Q[s',a']` over the legal actions for Q-learning and `Q[s', a']` for the action the
behaviour policy actually chose for SARSA. Duplicating the loop to make the comparison would
have made every other difference between the runs a candidate explanation, so it is not
duplicated; `test_rl_qlearning.py` pins that the two paths really do diverge.

## The hypothesis this row exists to test, stated before the numbers

The textbook difference is that on-policy control learns the value of the policy *including its
exploration*, so it avoids states where an exploratory step is expensive. This environment has a
concrete cliff of that kind, and it is not a metaphor: `RewardConfig.infinite_loop` charges
**-2.0** when an episode is truncated at the `4n + 8` step cap. A policy that wanders near the
cap risks that penalty every time epsilon fires; an off-policy learner, bootstrapping off the
greedy action, does not see the risk.

So the prediction is specific and falsifiable: **SARSA should truncate less often than
Q-learning at the same epsilon**, and pay for it with a shorter, more cautious trajectory. The
truncation rate is reported as the primary comparison for that reason, next to the reward.

## Why "on ambiguous diagrams" is the right set, and what it costs

11.2.7's ambiguous set is 684 of 993 labelled diagrams, but it is also the corpus's hardest
slice: 610 of the 684 are multi-component hdbpmn pages, on which gold play itself reaches only
19.74% full coverage because the action space has no jump. Both sets are therefore reported. The
ambiguous set answers the plan's question; the full labelled set is what says whether a
difference between the algorithms is a property of the algorithms or of the slice.

## Reading the result

Every figure is a mean over `SEEDS` independent runs with the spread alongside, because a single
seed's gap between two tabular learners on a corpus this noisy is not a result. The comparison is
scored by `src.rl.qlearning.play`, the same harness that scores gold and random, so the two
learners and their references never pass through two different metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from typing import Any

from src.rl.qlearning import RUNS, TrainConfig, evaluate, references, train, training_set
from src.utils.config import ROOT

OUT = RUNS / "sarsa.json"
FIGURE = ROOT / "reports" / "figures" / "p11_sarsa_vs_q.png"

#: Independent seeds per algorithm. Five is enough to see a spread and cheap enough to re-run.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

ALGOS: tuple[str, ...] = ("q", "sarsa")


def truncation_rate(report: dict[str, Any]) -> float:
    """Share of greedy evaluation episodes stopped by the step cap - the cliff, as a number."""
    stopped = report.get("stopped_by", {})
    total = sum(stopped.values()) or 1
    return round(stopped.get("cap", 0) / total, 4)


def smooth(values: Sequence[float], window: int = 200) -> list[float]:
    """A trailing mean. Raw per-episode reward on this corpus is unreadable - the variance across
    diagrams (3 to 200 nodes) is far larger than the variance across training."""
    out: list[float] = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def compare_algorithms(
    graphs_all,
    raw_all,
    graphs_amb,
    raw_amb,
    episodes: int = 20_000,
    seeds: Sequence[int] = SEEDS,
    alpha: float = 0.2,
    gamma: float = 0.95,
) -> dict[str, Any]:
    """Train both algorithms on the full labelled set; evaluate on both sets, over `seeds`."""
    result: dict[str, Any] = {"episodes": episodes, "seeds": list(seeds), "algos": {}}
    curves: dict[str, list[list[float]]] = {}
    for algo in ALGOS:
        rows_all: list[dict[str, Any]] = []
        rows_amb: list[dict[str, Any]] = []
        reward_curves: list[list[float]] = []
        td_curves: list[list[float]] = []
        for seed in seeds:
            cfg = TrainConfig(algo=algo, alpha=alpha, gamma=gamma, episodes=episodes, seed=seed)
            agent = train(graphs_all, cfg)
            rows_all.append(evaluate(agent, graphs_all, raw_all))
            rows_amb.append(evaluate(agent, graphs_amb, raw_amb))
            reward_curves.append(agent.history["reward"])
            td_curves.append(agent.history["td_error"])
        curves[algo] = reward_curves
        result["algos"][algo] = {
            "all": _summarise(rows_all),
            "ambiguous": _summarise(rows_amb),
            "curve_reward": [
                round(v, 4)
                for v in smooth([statistics.fmean(x) for x in zip(*reward_curves, strict=False)])
            ],
            "curve_td": [
                round(v, 5)
                for v in smooth([statistics.fmean(x) for x in zip(*td_curves, strict=False)])
            ],
        }
    return result


def _summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and spread across seeds for the fields that carry the comparison."""

    def stat(key: str) -> dict[str, float]:
        values = [float(r[key]) for r in rows]
        return {
            "mean": round(statistics.fmean(values), 4),
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    out = {
        "diagrams": rows[0]["diagrams"],
        "mean_terminal_reward": stat("mean_terminal_reward"),
        "full_coverage": stat("full_coverage"),
        "mean_emitted_share": stat("mean_emitted_share"),
        "mean_episode_length": stat("mean_episode_length"),
        "gold_full_coverage_ceiling": rows[0]["gold_full_coverage_ceiling"],
        "truncation_rate": {
            "mean": round(statistics.fmean([truncation_rate(r) for r in rows]), 4),
            "per_seed": [truncation_rate(r) for r in rows],
        },
    }
    if "mean_edge_f1" in rows[0]:
        out["mean_edge_f1"] = stat("mean_edge_f1")
        out["empty_policy_edge_f1"] = rows[0]["empty_policy_edge_f1"]
    return out


def plot(result: dict[str, Any], path=FIGURE):
    """The plan's "comparison plot": smoothed episode reward and |TD error|, both algorithms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colours = {"q": "#1f77b4", "sarsa": "#d62728"}

    for algo in ALGOS:
        block = result["algos"][algo]
        axes[0].plot(block["curve_reward"], color=colours[algo], lw=1.2, label=algo)
        axes[1].plot(block["curve_td"], color=colours[algo], lw=1.2, label=algo)
    axes[0].set_title("episode reward (trailing mean, 200)", fontsize=9)
    axes[0].set_xlabel("episode", fontsize=8)
    axes[1].set_title("|TD error| per step (trailing mean, 200)", fontsize=9)
    axes[1].set_xlabel("episode", fontsize=8)
    for ax in axes[:2]:
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, frameon=False)

    labels = ["reward", "coverage", "emitted", "truncated"]
    width = 0.38
    for offset, algo in zip((-width / 2, width / 2), ALGOS, strict=False):
        block = result["algos"][algo]["ambiguous"]
        values = [
            block["mean_terminal_reward"]["mean"],
            block["full_coverage"]["mean"],
            block["mean_emitted_share"]["mean"],
            block["truncation_rate"]["mean"],
        ]
        errors = [
            block["mean_terminal_reward"]["sd"],
            block["full_coverage"]["sd"],
            block["mean_emitted_share"]["sd"],
            0.0,
        ]
        axes[2].bar(
            [i + offset for i in range(len(labels))],
            values,
            width,
            yerr=errors,
            capsize=3,
            color=colours[algo],
            label=algo,
        )
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].axhline(0.0, color="#444444", lw=0.8)
    axes[2].set_title("greedy play, ambiguous set (mean over seeds)", fontsize=9)
    axes[2].tick_params(labelsize=7)
    axes[2].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(
    episodes: int = 20_000,
    limit: int | None = None,
    seeds: Sequence[int] = SEEDS,
    write: bool = False,
) -> dict[str, Any]:
    raw_all, graphs_all = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    result = compare_algorithms(
        graphs_all, raw_all, graphs_amb, raw_amb, episodes=episodes, seeds=seeds
    )
    result["references"] = {
        "all": references(graphs_all, raw_all),
        "ambiguous": references(graphs_amb, raw_amb),
    }
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["figure"] = str(plot(result).relative_to(ROOT))
        result["path"] = str(OUT.relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.2 SARSA vs Q-learning")
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    episodes = 600 if args.quick else args.episodes
    seeds = SEEDS[: max(1, args.seeds)] if not args.quick else (0,)
    out = run(episodes, args.limit, seeds, write=args.write)
    out.pop("algos", None) if args.quick else None
    print(json.dumps(out, indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

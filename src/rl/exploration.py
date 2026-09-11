"""Phase 11.2.3 - the exploration study: epsilon-greedy against softmax against UCB.

    python -m src.rl.exploration --write   # sweep each rule, 5 seeds x 60k at its best, plot
    python -m src.rl.exploration --quick   # a short run, for the tests

`src.rl.qlearning.TabularAgent.select` dispatches here for everything except epsilon-greedy,
which stays inline because it is the default. Every rule takes the same
`(agent, key, mask, rng, eps)` and returns **a legal action** - the mask is not advisory:
`apply` charges an illegal action and carries on, so a rule that ignored the mask would show up
only as a worse number. `test_rl_exploration.py` asserts legality over thousands of draws.

    epsilon-greedy  uniform over the legal actions with probability eps, eps decaying 1.0 -> floor
                    geometrically (11.2.1's schedule). The floor is swept.
    softmax         Boltzmann over the legal Q-values at a fixed temperature. The temperature is
                    swept. The row is shifted by its max before `exp`, which is exact and stops an
                    overflow at low temperature on this corpus's reward range.
    UCB             `Q + c * sqrt(ln N(s) / N(s,a))` over the legal actions, untried legal
                    actions first and at random (an infinite bonus broken by index would
                    reintroduce positional bias). N counts updates of the abstract key, so it is
                    a count over a *mixture* of the raw states 11.1.6 aliases onto that key. c is
                    swept.

Neither softmax nor UCB is wrapped in an epsilon draw: stacking uniform noise on a rule that
already explores would turn the study into three epsilon-greedy agents with different tie-breaks.

## How it is judged

Every rule runs Q-learning at 11.2.1's selected cell (alpha 0.4, gamma 1.0), is swept over its
own parameter with seed 0, and is then run at **its own best parameter, ranked on greedy terminal
reward** on the 993 labelled diagrams (never on edge F1) for five seeds. Reported per rule: the
training-return curve (the plan's "reward curves"), a **greedy-probe curve** - the greedy policy
scored every `PROBE_EVERY` episodes - because training return mixes in each rule's own
exploration noise and is not comparable across rules, and the final greedy evaluation on the 993
and on 11.2.7's 684 ambiguous diagrams. Measured results are in `reports/rl_exploration.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Sequence
from typing import Any

from src.rl import actions as A
from src.utils.config import ROOT

RULES: tuple[str, ...] = ("egreedy", "softmax", "ucb")

#: The swept parameter per rule: the epsilon floor, the temperature, the UCB constant.
EPSILON_FLOORS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
TEMPERATURES: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
UCB_CONSTANTS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
PARAMETER = {"egreedy": "epsilon_min", "softmax": "temperature", "ucb": "ucb_c"}
GRID = {"egreedy": EPSILON_FLOORS, "softmax": TEMPERATURES, "ucb": UCB_CONSTANTS}

#: Episodes between greedy probes during training.
PROBE_EVERY = 5000

FIGURE = ROOT / "reports" / "figures" / "p11_exploration.png"

#: Exponent floor after the max-shift in `softmax_action`: weights below e^-60 are zero anyway.
_CLIP = 60.0


def legal_of(mask: Sequence[bool]) -> list[int]:
    return [a for a in range(A.N_ACTIONS) if mask[a]]


def softmax_action(agent, key: int, mask: Sequence[bool], rng: random.Random, temp: float) -> int:
    """Boltzmann over the legal Q-values at temperature `temp`; the max-shift cancels exactly."""
    legal = legal_of(mask)
    if len(legal) == 1:
        return legal[0]
    temp = max(1e-6, temp)
    row = agent.q[key]
    values = [float(row[a]) / temp for a in legal]
    top = max(values)
    weights = [math.exp(max(-_CLIP, v - top)) for v in values]
    draw = rng.random() * sum(weights)
    for action, weight in zip(legal, weights, strict=True):
        draw -= weight
        if draw <= 0:
            return action
    return legal[-1]


def ucb_action(agent, key: int, mask: Sequence[bool], rng: random.Random, c: float) -> int:
    """`argmax Q + c * sqrt(ln N(s) / N(s,a))`, untried legal actions first, ties at random."""
    legal = legal_of(mask)
    if len(legal) == 1:
        return legal[0]
    counts = agent.counts[key]
    untried = [a for a in legal if counts[a] == 0]
    if untried:
        return rng.choice(untried)
    total = float(sum(int(counts[a]) for a in legal))
    row = agent.q[key]
    scores = [float(row[a]) + c * math.sqrt(math.log(total) / int(counts[a])) for a in legal]
    best = max(scores)
    top = [a for a, s in zip(legal, scores, strict=True) if s >= best - 1e-12]
    return top[0] if len(top) == 1 else rng.choice(top)


def select(agent, key: int, mask: Sequence[bool], rng: random.Random, eps: float) -> int:
    """The dispatch `TabularAgent.select` falls through to. Always returns a legal action."""
    rule = agent.cfg.explore
    if rule == "softmax":
        return softmax_action(agent, key, mask, rng, agent.cfg.temperature)
    if rule == "ucb":
        return ucb_action(agent, key, mask, rng, agent.cfg.ucb_c)
    if rule == "egreedy":
        legal = legal_of(mask)
        return rng.choice(legal) if rng.random() < eps else agent.greedy(key, mask, rng)
    raise ValueError(f"unknown exploration rule {rule!r}; expected one of {RULES}")


# ------------------------------------------------------------------------------------------
# the greedy probe - a learning curve that is comparable across rules
# ------------------------------------------------------------------------------------------


class GreedyProbe:
    """`on_episode` hook: score the *greedy* policy on the labelled set every `every` episodes.

    Training return is the behaviour policy's return, so a rule that explores more pays for it
    in the curve even if it learns the better table. The probe removes that confound, and 11.2.8
    uses the same probe to test whether greedy quality actually stops moving.
    """

    def __init__(self, graphs, every: int = PROBE_EVERY, episodes: int | None = None) -> None:
        self.graphs = graphs
        self.every = max(1, every)
        self.episodes = episodes
        self.points: list[dict[str, float]] = []

    def __call__(self, agent, index: int) -> None:
        last = self.episodes is not None and index == self.episodes - 1
        if (index + 1) % self.every and not last:
            return
        from src.rl.qlearning import evaluate

        row = evaluate(agent, self.graphs)
        self.points.append(
            {
                "episode": index + 1,
                "mean_terminal_reward": row["mean_terminal_reward"],
                "full_coverage": row["full_coverage"],
                "mean_emitted_share": row["mean_emitted_share"],
                "truncated": row["stopped_by"].get("cap", 0) / max(1, row["diagrams"]),
            }
        )

    def result(self) -> list[dict[str, float]]:
        return self.points


def probe_factory(sets, spec) -> GreedyProbe:
    """`run_job` hook factory: the probe over the training set, at `spec["probe_every"]`."""
    return GreedyProbe(
        sets["all"][1], spec.get("probe_every", PROBE_EVERY), spec["cfg"].get("episodes")
    )


# ------------------------------------------------------------------------------------------
# the study
# ------------------------------------------------------------------------------------------

BASE = {"algo": "q", "alpha": 0.4, "gamma": 1.0}


def _spec(rule: str, value: float, episodes: int, seed: int, **extra) -> dict[str, Any]:
    cfg = {**BASE, "explore": rule, PARAMETER[rule]: value, "episodes": episodes, "seed": seed}
    return {"cfg": cfg, **extra}


def sweep_specs(episodes: int, seed: int = 0) -> list[dict[str, Any]]:
    return [
        _spec(rule, value, episodes, seed, eval=["all"]) for rule in RULES for value in GRID[rule]
    ]


def best_parameters(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per rule, the parameter with the highest greedy terminal reward on the labelled set."""
    out = {}
    for rule in RULES:
        mine = [r for r in rows if r["spec"]["cfg"]["explore"] == rule]
        top = max(mine, key=lambda r: r["eval"]["all"]["mean_terminal_reward"])
        out[rule] = top["spec"]["cfg"][PARAMETER[rule]]
    return out


def study(
    sets,
    params: dict[str, float],
    episodes: int = 60_000,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    probe_every: int = PROBE_EVERY,
    workers: int | None = None,
) -> dict[str, Any]:
    """Every rule at `params[rule]` over `seeds`: curves, probes, and the final greedy evaluation."""
    from src.rl.sarsa import WINDOW, run_jobs, smooth, stat, summarise

    specs = [
        _spec(
            rule,
            params[rule],
            episodes,
            seed,
            keep=["reward", "td_error", "truncated", "length"],
            on_episode="src.rl.exploration:probe_factory",
            probe_every=probe_every,
        )
        for rule in RULES
        for seed in seeds
    ]
    jobs = run_jobs(specs, sets, workers)
    out: dict[str, Any] = {"episodes": episodes, "seeds": list(seeds), "rules": {}}
    step = max(1, episodes // 400)
    window = min(WINDOW, max(1, episodes // 20))
    for rule in RULES:
        mine = [j for j in jobs if j["spec"]["cfg"]["explore"] == rule]
        block = summarise(mine)
        block["parameter"] = {PARAMETER[rule]: params[rule]}
        curves = [smooth(j["history"]["reward"], window) for j in mine]
        xs = list(range(0, episodes, step))
        block["curve_reward"] = {
            "x": xs,
            "mean": [round(statistics.fmean(c[i] for c in curves), 5) for i in xs],
            "sd": [round(statistics.pstdev([c[i] for c in curves]), 5) for i in xs],
        }
        probes = [j["on_episode"] for j in mine]
        block["probe"] = [
            {
                "episode": points[0]["episode"],
                **{
                    k: stat([p[k] for p in points])
                    for k in ("mean_terminal_reward", "full_coverage", "truncated")
                },
            }
            for points in zip(*probes, strict=True)
        ]
        block["mean_training_length"] = stat(
            [statistics.fmean(j["history"]["length"]) for j in mine]
        )
        out["rules"][rule] = block
    return out


# ------------------------------------------------------------------------------------------
# plot
# ------------------------------------------------------------------------------------------

COLOURS = {"egreedy": "#2a78d6", "softmax": "#eb6834", "ucb": "#1baf7a"}
NAMES = {"egreedy": "epsilon-greedy", "softmax": "softmax", "ucb": "UCB"}


def plot(result: dict[str, Any], refs: dict[str, Any] | None = None, path=FIGURE):
    """Training-return curves, greedy-probe curves, and the final greedy reward per set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for rule in RULES:
        block = result["rules"][rule]
        label = f"{NAMES[rule]} ({', '.join(f'{k}={v}' for k, v in block['parameter'].items())})"
        c = block["curve_reward"]
        start = min(1000, result["episodes"] // 20)
        keep = [i for i, x in enumerate(c["x"]) if x >= start]
        cx = [c["x"][i] for i in keep]
        lo = [c["mean"][i] - c["sd"][i] for i in keep]
        hi = [c["mean"][i] + c["sd"][i] for i in keep]
        axes[0].fill_between(cx, lo, hi, color=COLOURS[rule], alpha=0.2, lw=0)
        axes[0].plot(cx, [c["mean"][i] for i in keep], color=COLOURS[rule], lw=2, label=label)
        xs = [p["episode"] for p in block["probe"]]
        ys = [p["mean_terminal_reward"]["mean"] for p in block["probe"]]
        sd = [p["mean_terminal_reward"]["sd"] for p in block["probe"]]
        axes[1].errorbar(
            xs,
            ys,
            yerr=sd,
            color=COLOURS[rule],
            lw=2,
            marker="o",
            ms=5,
            capsize=3,
            label=NAMES[rule],
        )
    if refs:
        # Only on the probe panel: training return carries step costs and exploration, so gold's
        # greedy terminal reward is not on its scale.
        for ax in axes[1:2]:
            ax.axhline(
                refs["all"]["gold"]["mean_terminal_reward"],
                color="#555555",
                ls="--",
                lw=1.2,
                label="gold play",
            )
            ax.axhline(
                refs["all"]["random"]["mean_terminal_reward"],
                color="#999999",
                ls=":",
                lw=1.2,
                label="random policy",
            )
    axes[0].set_title("training return: the behaviour policy (trailing 1,000, ±sd)")
    axes[1].set_title("greedy policy, terminal reward on 993 labelled (probe)")
    for ax in axes[:2]:
        ax.set_xlabel("episode")
        ax.legend(frameon=False, fontsize=8)

    for gi, scope in enumerate(("all", "ambiguous")):
        for k, rule in enumerate(RULES):
            s = result["rules"][rule][scope]["mean_terminal_reward"]
            x = gi + (k - 1) * 0.26
            axes[2].bar(
                x, s["mean"], 0.24, color=COLOURS[rule], label=NAMES[rule] if gi == 0 else None
            )
            axes[2].scatter([x] * len(s["values"]), s["values"], color="#222222", s=10, zorder=3)
        if refs:
            axes[2].hlines(
                refs[scope]["gold"]["mean_terminal_reward"],
                gi - 0.42,
                gi + 0.42,
                color="#555555",
                ls="--",
                lw=1.2,
                label="gold play" if gi == 0 else None,
            )
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["all labelled (993)", "ambiguous (684)"])
    axes[2].axhline(0, color="#888888", lw=0.8)
    axes[2].set_ylabel("mean terminal reward")
    axes[2].set_title("final greedy evaluation (dots = seeds)")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------


def run(
    episodes: int = 60_000,
    sweep_episodes: int = 12_000,
    limit: int | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    write: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS, references
    from src.rl.sarsa import load_sets, run_jobs, truncation_rate

    sets = load_sets(limit)
    rows = run_jobs(sweep_specs(sweep_episodes), sets, workers)
    params = best_parameters(rows)
    result: dict[str, Any] = {
        "sweep": [
            {
                "explore": r["spec"]["cfg"]["explore"],
                "parameter": PARAMETER[r["spec"]["cfg"]["explore"]],
                "value": r["spec"]["cfg"][PARAMETER[r["spec"]["cfg"]["explore"]]],
                "mean_terminal_reward": r["eval"]["all"]["mean_terminal_reward"],
                "full_coverage": r["eval"]["all"]["full_coverage"],
                "truncation_rate": truncation_rate(r["eval"]["all"]),
                "reached_keys": r["report"]["reached_keys"],
            }
            for r in rows
        ],
        "best_parameters": params,
    }
    result.update(study(sets, params, episodes, seeds, workers=workers))
    result["references"] = {name: references(g, raw) for name, (raw, g) in sets.items()}
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "exploration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["figure"] = str(plot(result, result["references"]).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.3 exploration study")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--sweep-episodes", type=int, default=12_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the JSON")
    args = ap.parse_args(argv)
    if args.replot:
        from src.rl.qlearning import RUNS

        result = json.loads((RUNS / "exploration.json").read_text(encoding="utf-8"))
        print(plot(result, result["references"]))
        return 0
    if args.quick:
        out = run(600, 300, args.limit, (0,), workers=1)
    else:
        seeds = tuple(range(max(1, args.seeds)))
        out = run(args.episodes, args.sweep_episodes, args.limit, seeds, args.write, args.workers)
    for block in out.get("rules", {}).values():
        block.pop("curve_reward", None)
    print(json.dumps(out, indent=2)[:30000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

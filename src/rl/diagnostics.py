"""Phase 11.2.8 - convergence diagnostics: episode reward, TD error, Q-value heatmaps.

    python -m src.rl.diagnostics --write    # 3 seeds at 11.2.1's cell, figures + JSON
    python -m src.rl.diagnostics --quick    # a short run, for the tests

The plots are the output; the numbers attached to them are the claim. Per seed:

    td_tail_ratio      mean |TD error| over the last 10% of training against the first 10%.
    td_noise_floor     |TD error| of the *final table frozen* (alpha 0) and played at the epsilon
                       floor. The table cannot move, so whatever TD error remains is target noise -
                       diagrams drawn at random, exploration, and 11.1.6's aliasing of different
                       raw states onto one key - not learning that has yet to finish. Training TD
                       error near this floor means "not converging" is the wrong reading of a
                       large TD error.
    policy_churn       share of keys whose greedy action changed between snapshots, the greedy
                       action taken over the actions *ever updated at that key*. The inherited
                       draft took the argmax over all nine actions, where an untaken (often
                       illegal) action sits at exactly 0.0 and wins any row whose learned values
                       are all negative. Measured on seed 0 at 120k episodes, that picks an untaken
                       action on 5.0-5.8% of reached keys, yet moves the churn figure by at most
                       0.002 - a real bug with a negligible effect on this number, fixed anyway.
    greedy probe       the greedy policy scored on the training set during training (reward and
                       full coverage against the gold ceiling), because training return mixes in
                       epsilon and falls or rises with the schedule rather than with the policy.
    reward_slope       least-squares slope of the training return over the last half, per 1k.
    visits             how concentrated the updates are over reached keys.

## The heatmap

Rows are the most-visited reached keys, labelled by their decoded factors; columns are the nine
actions. **Cells never updated are grey, not zero-coloured** - the inherited draft drew them at
0.0 on a diverging scale, where "never taken" and "learned to be worth nothing" are the same
white. The colour limit is the 98th percentile of |Q| over updated cells so one outlier does not
wash the table out. Measured results are in `reports/rl_diagnostics.md`.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import statistics
import sys
from typing import Any

import numpy as np

from src.rl import actions as A
from src.rl.abstraction import FACTOR_SIZES
from src.rl.state import ROLE_VOCAB
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p11_convergence.png"
HEATMAP = ROOT / "reports" / "figures" / "p11_qvalues.png"
#: The raw per-seed jobs (histories included) so `--replot` can redraw without retraining.
JOBS = ROOT / "experiments" / "rl" / "diagnostics_jobs.pkl"

SNAPSHOT_EVERY = 5000
PROBE_EVERY = 10_000
HEATMAP_ROWS = 40
BASE = {"algo": "q", "alpha": 0.4, "gamma": 1.0}


# ------------------------------------------------------------------------------------------
# measures
# ------------------------------------------------------------------------------------------


def greedy_over_taken(q: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per key, argmax Q over the actions ever updated there; -1 for a key never updated."""
    taken = counts > 0
    masked = np.where(taken, q, -np.inf)
    out = masked.argmax(axis=1).astype(np.int16)
    out[~taken.any(axis=1)] = -1
    return out


class PolicySnapshots:
    """`on_episode` hook: the greedy-over-taken action of every key, every `every` episodes."""

    def __init__(self, every: int = SNAPSHOT_EVERY) -> None:
        self.every = max(1, every)
        self.episodes: list[int] = []
        self.frames: list[np.ndarray] = []

    def __call__(self, agent, index: int) -> None:
        if (index + 1) % self.every:
            return
        self.episodes.append(index + 1)
        self.frames.append(greedy_over_taken(agent.q, agent.counts))

    def churn(self) -> list[dict[str, Any]]:
        """Per snapshot pair: share of keys reached in *both* whose greedy action changed."""
        out: list[dict[str, Any]] = []
        for i in range(1, len(self.frames)):
            before, after = self.frames[i - 1], self.frames[i]
            shared = (before >= 0) & (after >= 0)
            n = int(shared.sum())
            changed = int((before[shared] != after[shared]).sum())
            out.append(
                {
                    "episode": self.episodes[i],
                    "keys": n,
                    "changed": changed,
                    "churn": round(changed / n, 4) if n else None,
                }
            )
        return out


class Hooks:
    """Snapshots and the greedy probe on one `on_episode`; `result()` returns both."""

    def __init__(self, graphs, episodes: int, every: int, probe_every: int) -> None:
        from src.rl.exploration import GreedyProbe

        self.snapshots = PolicySnapshots(every)
        self.probe = GreedyProbe(graphs, probe_every, episodes)

    def __call__(self, agent, index: int) -> None:
        self.snapshots(agent, index)
        self.probe(agent, index)

    def result(self) -> dict[str, Any]:
        return {"policy_churn": self.snapshots.churn(), "probe": self.probe.result()}


def hooks_factory(sets, spec) -> Hooks:
    return Hooks(
        sets["all"][1],
        spec["cfg"]["episodes"],
        spec.get("snapshot_every", SNAPSHOT_EVERY),
        spec.get("probe_every", PROBE_EVERY),
    )


def reward_slope(rewards, per: int = 1000) -> float:
    """Least-squares slope over the last half of training, per `per` episodes."""
    tail = np.asarray(list(rewards)[len(rewards) // 2 :], dtype=float)
    if tail.size < 2:
        return 0.0
    return round(float(np.polyfit(np.arange(tail.size), tail, 1)[0]) * per, 5)


def visit_concentration(counts: np.ndarray) -> dict[str, Any]:
    """How concentrated the updates are over the reached keys. `reached_keys` alone can lie."""
    per_key = counts.sum(axis=1)
    reached = per_key[per_key > 0]
    if not reached.size:
        return {"reached_keys": 0}
    ordered = np.sort(reached)[::-1]
    cumulative = np.cumsum(ordered) / float(ordered.sum())
    return {
        "reached_keys": int(reached.size),
        "updates": int(ordered.sum()),
        "share_in_top_10_keys": round(float(cumulative[min(9, cumulative.size - 1)]), 4),
        "keys_for_half_the_updates": int(np.searchsorted(cumulative, 0.5) + 1),
        "keys_visited_under_10_times": int((reached < 10).sum()),
        "median_visits_per_key": int(np.median(reached)),
    }


def td_noise_floor(agent, graphs, episodes: int = 10_000, seed: int = 12345) -> float:
    """Mean per-step |TD error| of the final table frozen (alpha 0), played at the epsilon floor."""
    from src.rl.qlearning import TabularAgent, TrainConfig, train

    frozen = TabularAgent(copy.copy(agent.cfg))
    frozen.q[:] = agent.q
    cfg = TrainConfig(
        algo=agent.cfg.algo,
        alpha=0.0,
        gamma=agent.cfg.gamma,
        episodes=episodes,
        epsilon=agent.cfg.epsilon_min,
        epsilon_min=agent.cfg.epsilon_min,
        seed=seed,
    )
    before = frozen.q.copy()
    train(graphs, cfg, agent=frozen)
    assert (frozen.q == before).all(), "a frozen table moved"
    return round(statistics.fmean(frozen.history["td_error"]), 6)


def decode_key(index: int) -> tuple[int, ...]:
    """Inverse of `abstraction.key_index`: the nine factor values."""
    values = []
    for size in reversed(FACTOR_SIZES):
        values.append(index % size)
        index //= size
    return tuple(reversed(values))


def key_label(index: int) -> str:
    role, deg, unv, vis, emi, cur, stack, loop, unres = decode_key(index)
    return (
        f"{ROLE_VOCAB[role][:9]:<9} out{deg} unv{unv} vis{vis} emi{emi} "
        f"{'E' if cur else '-'} stk{stack} {'L' if loop else '-'}{'U' if unres else '-'}"
    )


def table_slice(agent, rows: int = HEATMAP_ROWS) -> dict[str, Any]:
    """The most-visited keys' Q and counts, small enough to ship back from a worker."""
    per_key = agent.counts.sum(axis=1)
    order = [int(k) for k in np.argsort(per_key)[::-1][:rows] if per_key[k] > 0]
    greedy = greedy_over_taken(agent.q, agent.counts)
    reached = greedy >= 0
    weights = per_key[reached]
    by_action = [float(weights[greedy[reached] == a].sum()) for a in range(A.N_ACTIONS)]
    total = sum(by_action) or 1.0
    return {
        "keys": order,
        "labels": [key_label(k) for k in order],
        "visits": [int(per_key[k]) for k in order],
        "q": agent.q[order].round(4).tolist(),
        "counts": agent.counts[order].tolist(),
        "greedy_action_share_by_visits": [round(v / total, 4) for v in by_action],
        "greedy_action_share_by_keys": [
            round(float((greedy[reached] == a).mean()), 4) for a in range(A.N_ACTIONS)
        ],
    }


def diagnostics_extra(agent, sets, spec) -> dict[str, Any]:
    """`run_job` extra: every number the history cannot give after the agent is gone."""
    graphs = sets["all"][1]
    return {
        "td_noise_floor": td_noise_floor(agent, graphs, spec.get("noise_episodes", 10_000)),
        "reward_slope_per_1000": reward_slope(agent.history["reward"]),
        "visits": visit_concentration(agent.counts),
        "table": table_slice(agent),
    }


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def specs(
    episodes: int, seeds, every: int, probe_every: int, noise_episodes: int
) -> list[dict[str, Any]]:
    return [
        {
            "cfg": {**BASE, "episodes": episodes, "seed": seed},
            "keep": ["reward", "td_error", "epsilon", "truncated"],
            "on_episode": "src.rl.diagnostics:hooks_factory",
            "extra": "src.rl.diagnostics:diagnostics_extra",
            "snapshot_every": every,
            "probe_every": probe_every,
            "noise_episodes": noise_episodes,
            "eval": ["all", "ambiguous"],
        }
        for seed in seeds
    ]


def summarise_seed(job: dict[str, Any]) -> dict[str, Any]:
    churn = job["on_episode"]["policy_churn"]
    values = [c["churn"] for c in churn if c["churn"] is not None]
    third = values[-max(1, len(values) // 3) :] if values else []
    td = job["history"]["td_error"]
    k = max(1, len(td) // 10)
    return {
        "seed": job["spec"]["cfg"]["seed"],
        **job["report"],
        "td_tail": round(statistics.fmean(td[-k:]), 6),
        "td_noise_floor": job["extra"]["td_noise_floor"],
        "td_tail_over_noise_floor": round(
            statistics.fmean(td[-k:]) / job["extra"]["td_noise_floor"], 4
        ),
        "reward_slope_per_1000": job["extra"]["reward_slope_per_1000"],
        "policy_churn_final": values[-1] if values else None,
        "policy_churn_mean_last_third": round(statistics.fmean(third), 4) if third else None,
        "visits": job["extra"]["visits"],
        "probe": job["on_episode"]["probe"],
        "evaluation": job["eval"],
    }


def diagnose(jobs) -> dict[str, Any]:
    seeds = [summarise_seed(j) for j in jobs]
    return {"seeds": seeds, "churn": [j["on_episode"]["policy_churn"] for j in jobs]}


# ------------------------------------------------------------------------------------------
# plots
# ------------------------------------------------------------------------------------------

SEED_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def plot_convergence(jobs, refs: dict[str, Any] | None = None, path=FIGURE):
    """Six panels: training return, |TD| against its frozen-table floor, greedy reward, greedy
    coverage against the ceiling, policy churn, epsilon and cap truncation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.rl.sarsa import smooth

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for i, job in enumerate(jobs):
        colour = SEED_COLOURS[i % len(SEED_COLOURS)]
        seed = job["spec"]["cfg"]["seed"]
        h = job["history"]
        n = len(h["reward"])
        step = max(1, n // 600)
        xs = list(range(min(2000, n // 10), n, step))
        r = smooth(h["reward"], 2000)
        axes[0, 0].plot(xs, [r[x] for x in xs], color=colour, lw=1.6, label=f"seed {seed}")
        td = smooth(h["td_error"], 2000)
        axes[0, 1].plot(xs, [td[x] for x in xs], color=colour, lw=1.6, label=f"seed {seed}")
        axes[0, 1].axhline(job["extra"]["td_noise_floor"], color=colour, ls="--", lw=1.2)
        probe = job["on_episode"]["probe"]
        px = [p["episode"] for p in probe]
        axes[0, 2].plot(
            px, [p["mean_terminal_reward"] for p in probe], color=colour, marker="o", lw=1.6
        )
        axes[1, 0].plot(px, [p["full_coverage"] for p in probe], color=colour, marker="o", lw=1.6)
        churn = [c for c in job["on_episode"]["policy_churn"] if c["churn"] is not None]
        axes[1, 1].plot(
            [c["episode"] for c in churn], [c["churn"] for c in churn], color=colour, lw=1.6
        )
        tr = smooth(h["truncated"], 2000)
        axes[1, 2].plot(xs, [tr[x] for x in xs], color=colour, lw=1.2, ls=":")
        axes[1, 2].plot(px, [p["truncated"] for p in probe], color=colour, marker="o", lw=1.6)
    if refs:
        gold = refs["all"]["gold"]
        axes[0, 2].axhline(gold["mean_terminal_reward"], color="#555555", ls="--", label="gold")
        axes[0, 2].axhline(
            refs["all"]["random"]["mean_terminal_reward"], color="#999999", ls=":", label="random"
        )
        axes[1, 0].axhline(gold["full_coverage"], color="#555555", ls="--", label="gold ceiling")
    axes[1, 1].axhline(0.05, color="#999999", ls="--", lw=1)
    titles = (
        "training return (trailing 2,000)",
        "|TD error| per step (trailing 2,000)\ndashed: frozen-table noise floor",
        f"greedy terminal reward on 993 (every {PROBE_EVERY:,})",
        "greedy full coverage on 993",
        f"policy churn between snapshots ({SNAPSHOT_EVERY:,} apart)",
        "episodes truncated at the step cap\nsolid: greedy probe, dotted: training (exploring)",
    )
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title)
        ax.set_xlabel("episode")
    for ax in (axes[0, 0], axes[0, 2], axes[1, 0]):
        ax.legend(frameon=False, fontsize=8)
    cfg = jobs[0]["spec"]["cfg"]
    fig.suptitle(
        f"11.2.8 - tabular Q-learning convergence, alpha {cfg['alpha']} gamma {cfg['gamma']}, "
        f"epsilon 1.0 -> 0.05 over {cfg['episodes']:,} episodes, one colour per seed"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_qvalues(job, path=HEATMAP):
    """Q over the most-visited keys (never-updated cells grey), and the greedy action mix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = job["extra"]["table"]
    q = np.asarray(table["q"], dtype=float)
    counts = np.asarray(table["counts"])
    masked = np.ma.masked_where(counts == 0, q)
    limit = float(np.percentile(np.abs(q[counts > 0]), 98)) if (counts > 0).any() else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#cfcfcf")

    fig, axes = plt.subplots(1, 2, figsize=(17, 11), gridspec_kw={"width_ratios": [3, 1.1]})
    image = axes[0].imshow(
        masked, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest"
    )
    axes[0].set_xticks(range(A.N_ACTIONS))
    axes[0].set_xticklabels(A.ACTION_NAMES, rotation=40, ha="right")
    axes[0].set_yticks(range(len(table["labels"])))
    axes[0].set_yticklabels(
        [f"{lab}  n={v:,}" for lab, v in zip(table["labels"], table["visits"], strict=True)],
        fontfamily="monospace",
        fontsize=7.5,
    )
    axes[0].set_title(
        f"Q values, top {len(table['keys'])} keys by visits (grey = never updated)\n"
        "label: role, out-degree, unvisited succ., visited/emitted fraction bucket, "
        "E=current emitted, stack, L=open loop, U=unresolved"
    )
    fig.colorbar(image, ax=axes[0], fraction=0.03, label="Q")

    ys = np.arange(A.N_ACTIONS)
    axes[1].barh(
        ys - 0.2,
        table["greedy_action_share_by_keys"],
        0.4,
        color="#2a78d6",
        label="share of reached keys",
    )
    axes[1].barh(
        ys + 0.2,
        table["greedy_action_share_by_visits"],
        0.4,
        color="#eb6834",
        label="weighted by visits",
    )
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels(A.ACTION_NAMES)
    axes[1].invert_yaxis()
    axes[1].set_title("greedy action\n(over actions taken at the key)")
    axes[1].legend(frameon=False)
    fig.suptitle(f"11.2.8 - Q-table, seed {job['spec']['cfg']['seed']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------


def run(
    episodes: int = 120_000,
    limit: int | None = None,
    seeds=(0, 1, 2),
    write: bool = False,
    every: int = SNAPSHOT_EVERY,
    probe_every: int = PROBE_EVERY,
    noise_episodes: int = 10_000,
    workers: int | None = None,
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS, references
    from src.rl.sarsa import load_sets, run_jobs

    sets = load_sets(limit)
    jobs = run_jobs(specs(episodes, seeds, every, probe_every, noise_episodes), sets, workers)
    result: dict[str, Any] = {"config": {**BASE, "episodes": episodes}, **diagnose(jobs)}
    result["references"] = {name: references(g, raw) for name, (raw, g) in sets.items()}
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with JOBS.open("wb") as handle:
            pickle.dump(jobs, handle)
        result["figures"] = [
            str(plot_convergence(jobs, result["references"]).relative_to(ROOT)),
            str(plot_qvalues(jobs[0]).relative_to(ROOT)),
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.8 convergence diagnostics")
    ap.add_argument("--episodes", type=int, default=120_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw both figures from the pickle")
    args = ap.parse_args(argv)
    if args.replot:
        from src.rl.qlearning import RUNS

        with JOBS.open("rb") as handle:
            jobs = pickle.load(handle)
        refs = json.loads((RUNS / "diagnostics.json").read_text(encoding="utf-8"))["references"]
        print(plot_convergence(jobs, refs), plot_qvalues(jobs[0]))
        return 0
    if args.quick:
        out = run(2000, args.limit, (0,), False, 500, 1000, 500, workers=1)
    else:
        seeds = tuple(range(max(1, args.seeds)))
        out = run(args.episodes, args.limit, seeds, args.write, workers=args.workers)
    print(json.dumps(out, indent=2)[:30000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

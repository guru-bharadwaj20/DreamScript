"""Phase 11.2.8 - convergence diagnostics: episode reward, TD error, Q-value heatmaps.

    python -m src.rl.diagnostics            # train, diagnose, write the figures and the JSON
    python -m src.rl.diagnostics --quick    # a short run, for the tests

The plan asks for plots. Plots are the output, but they are not the claim, so every panel here
has a number attached and the numbers are what the row quotes. Four are computed:

    td_tail_ratio    mean |TD error| over the last 10% of training against the first 10%. This is
                     the honest reading of "converged": the *table* has stopped moving. It says
                     nothing about optimality, because 11.1.6's abstraction aliases 25.2% of raw
                     states onto a shared key and the optimal policy over the aliased state is not
                     the optimal policy over the real one.
    policy_churn     share of reached keys whose greedy action changed between two snapshots. A
                     table can have a small TD error and still flip its policy every few hundred
                     episodes when two actions sit within noise of each other, and that is a
                     different failure with a different fix, so it is measured separately.
    reward_slope     least-squares slope of the episode reward over the last half of training,
                     per 1,000 episodes. Sign matters more than magnitude - a negative slope late
                     in a decayed-epsilon run means the greedy policy is getting worse.
    coverage_by_key  how concentrated the visits are. `reached_keys` alone hides the case where
                     90% of the updates land on 20 keys, which is the case that makes a large
                     "explored" number meaningless.

`policy_churn` is why `TrainConfig.on_episode` exists. Snapshots have to be taken *during* one
run: chunking the training into separate `train` calls would restart the epsilon schedule at
every boundary, and the churn would then be measuring the schedule.

## The heatmap, and what it is allowed to show

A Q-value heatmap over a 128,000-cell bound is a picture of mostly zeros. The panel here is
restricted to the keys that were actually reached, ordered by visit count, and the colour scale
is symmetric about zero so that "never updated" (exactly 0.0) is visually distinct from "learned
to be worthless" (a small negative). Without that, an untouched table and a converged pessimistic
one look identical.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any

import numpy as np

from src.rl import actions as A
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p11_convergence.png"
HEATMAP = ROOT / "reports" / "figures" / "p11_qvalues.png"

#: Episodes between policy snapshots.
SNAPSHOT_EVERY = 2000


class PolicySnapshots:
    """`on_episode` hook that records the greedy action of every reached key, periodically."""

    def __init__(self, every: int = SNAPSHOT_EVERY) -> None:
        self.every = every
        self.episodes: list[int] = []
        self.frames: list[np.ndarray] = []

    def __call__(self, agent, index: int) -> None:
        if index % self.every:
            return
        reached = agent.counts.sum(axis=1) > 0
        greedy = np.full(agent.q.shape[0], -1, dtype=np.int16)
        if reached.any():
            greedy[reached] = agent.q[reached].argmax(axis=1).astype(np.int16)
        self.episodes.append(index)
        self.frames.append(greedy)

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


def reward_slope(rewards, per: int = 1000) -> float:
    """Least-squares slope over the last half of training, per `per` episodes."""
    tail = list(rewards)[len(rewards) // 2 :]
    n = len(tail)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = statistics.fmean(xs), statistics.fmean(tail)
    denominator = sum((x - mx) ** 2 for x in xs)
    if not denominator:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, tail, strict=True)) / denominator
    return round(slope * per, 5)


def visit_concentration(agent) -> dict[str, Any]:
    """How concentrated the updates are over the reached keys. `reached_keys` alone can lie."""
    per_key = agent.counts.sum(axis=1)
    reached = per_key[per_key > 0]
    if not reached.size:
        return {"reached_keys": 0}
    ordered = np.sort(reached)[::-1]
    total = float(ordered.sum())
    cumulative = np.cumsum(ordered) / total
    return {
        "reached_keys": int(reached.size),
        "updates": int(total),
        "share_in_top_10_keys": round(float(cumulative[min(9, len(cumulative) - 1)]), 4),
        "keys_for_half_the_updates": int(np.searchsorted(cumulative, 0.5) + 1),
        "median_visits_per_key": int(np.median(reached)),
    }


def diagnose(agent, snapshots: PolicySnapshots | None = None) -> dict[str, Any]:
    """Every convergence number in one dict. The plots are drawn from exactly this."""
    report = dict(agent.report)
    report["reward_slope_per_1000"] = reward_slope(agent.history["reward"])
    report["visits"] = visit_concentration(agent)
    if snapshots is not None:
        churn = snapshots.churn()
        report["policy_churn"] = churn
        report["policy_churn_final"] = churn[-1]["churn"] if churn else None
        report["policy_churn_mean_last_third"] = (
            round(
                statistics.fmean(
                    [
                        c["churn"]
                        for c in churn[-max(1, len(churn) // 3) :]
                        if c["churn"] is not None
                    ]
                ),
                4,
            )
            if churn
            else None
        )
    report["converged"] = bool(
        report.get("td_tail_ratio") is not None
        and report["td_tail_ratio"] < 0.25
        and (report.get("policy_churn_final") or 0.0) < 0.05
    )
    return report


# ------------------------------------------------------------------------------------------
# plots
# ------------------------------------------------------------------------------------------


def plot_convergence(agent, report: dict[str, Any], path=FIGURE):
    """Episode reward, |TD error|, epsilon, and policy churn. Four panels, four numbers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.rl.sarsa import smooth

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.0))
    axes[0].plot(smooth(agent.history["reward"]), color="#1f77b4", lw=1.1)
    axes[0].axhline(0.0, color="#999999", lw=0.7)
    axes[0].set_title(
        f"episode reward (trailing 200)\nslope {report['reward_slope_per_1000']:+.4f}/1k",
        fontsize=9,
    )
    axes[1].plot(smooth(agent.history["td_error"]), color="#d62728", lw=1.1)
    axes[1].set_title(f"|TD error| per step\ntail/head {report.get('td_tail_ratio')}", fontsize=9)
    axes[2].plot(agent.history["epsilon"], color="#2ca02c", lw=1.1)
    axes[2].set_title("epsilon", fontsize=9)
    churn = report.get("policy_churn") or []
    if churn:
        axes[3].plot(
            [c["episode"] for c in churn],
            [c["churn"] for c in churn],
            color="#9467bd",
            marker="o",
            ms=2.5,
            lw=1.1,
        )
        axes[3].axhline(0.05, color="#999999", ls="--", lw=0.8)
    axes[3].set_title(
        f"policy churn per {SNAPSHOT_EVERY} episodes\nfinal {report.get('policy_churn_final')}",
        fontsize=9,
    )
    for ax in axes:
        ax.set_xlabel("episode", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_qvalues(agent, path=HEATMAP, top: int = 400):
    """The Q-table over the keys that were actually reached, ordered by visit count.

    Symmetric colour scale about zero, so an untouched row and a learned-pessimistic row are
    distinguishable - on a one-sided scale they are the same colour and the picture is a lie.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_key = agent.counts.sum(axis=1)
    order = np.argsort(per_key)[::-1]
    keys = [int(k) for k in order[:top] if per_key[k] > 0]
    table = agent.q[keys] if keys else np.zeros((1, A.N_ACTIONS))
    limit = float(np.abs(table).max()) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    image = axes[0].imshow(
        table, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
    )
    axes[0].set_xticks(range(A.N_ACTIONS))
    axes[0].set_xticklabels(A.ACTION_NAMES, rotation=60, ha="right", fontsize=7)
    axes[0].set_ylabel(f"reached key, by visit count (top {len(keys)})", fontsize=8)
    axes[0].set_title("Q values", fontsize=9)
    fig.colorbar(image, ax=axes[0], fraction=0.046)

    if keys:
        greedy = agent.q[keys].argmax(axis=1)
        counts = [int((greedy == a).sum()) for a in range(A.N_ACTIONS)]
    else:
        counts = [0] * A.N_ACTIONS
    axes[1].barh(list(A.ACTION_NAMES), counts, color="#4c72b0")
    axes[1].set_title("greedy action over reached keys", fontsize=9)
    axes[1].tick_params(labelsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------


def run(
    episodes: int = 60_000,
    limit: int | None = None,
    seed: int = 0,
    write: bool = False,
    every: int = SNAPSHOT_EVERY,
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS, TrainConfig, evaluate, references, train, training_set

    raw, graphs = training_set(limit, ambiguous=False)
    snapshots = PolicySnapshots(every)
    cfg = TrainConfig(alpha=0.4, gamma=1.0, episodes=episodes, seed=seed, on_episode=snapshots)
    agent = train(graphs, cfg)
    report = diagnose(agent, snapshots)
    result = {
        "config": cfg.to_dict(),
        "diagnostics": report,
        "evaluation": evaluate(agent, graphs, raw),
        "references": references(graphs, raw),
    }
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["figures"] = [
            str(plot_convergence(agent, report).relative_to(ROOT)),
            str(plot_qvalues(agent).relative_to(ROOT)),
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.8 convergence diagnostics")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    episodes = 2000 if args.quick else args.episodes
    every = 200 if args.quick else SNAPSHOT_EVERY
    print(json.dumps(run(episodes, args.limit, args.seed, args.write, every), indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

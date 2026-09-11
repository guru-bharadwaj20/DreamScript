"""Phase 11.2.3 - the exploration study: epsilon-greedy against softmax against UCB.

    python -m src.rl.exploration           # all three, several seeds, curves, plot, JSON
    python -m src.rl.exploration --quick   # a short run, for the tests

`src.rl.qlearning.TabularAgent.select` dispatches here for everything except epsilon-greedy,
which stays inline because it is the default and the dispatch would be in the innermost loop.
Every rule below takes the same `(agent, key, mask, rng, eps)` and returns **a legal action** -
the mask is not advisory. `terminate` is legal in every state (11.1.2), so a rule that ignored
the mask would still produce a runnable episode and the bug would show up only as a bad number
weeks later; `test_rl_exploration.py` asserts legality for all three over a thousand draws.

## What each rule is being asked

    epsilon-greedy  uniform over the legal actions with probability eps. Ignores what it already
                    knows: an action tried 400 times and an action never tried are equally likely
                    to be the exploratory pick.
    softmax         Boltzmann over the legal Q-values. Explores *in proportion to value*, which
                    is the right shape when the actions are not equally bad - and here they are
                    not: emitting a node the policy is standing on is almost always better than
                    backtracking off it.
    UCB             an optimism bonus `c * sqrt(ln N(s) / N(s,a))`, untried legal actions first.
                    In principle the strongest of the three; in practice its assumption is that
                    the state is a bandit whose payoff is stationary, and 11.1.6's abstraction
                    aliases 25.2% of raw states onto a shared key, so N(s,a) is being counted
                    over a *mixture* of states whose values differ.

The last point is the one worth measuring rather than asserting, and it is why UCB is in this
study at all: the abstraction is exactly the condition under which a count-based bonus is
supposed to break down.

## The temperature and the constant are swept, not chosen

A study that compares one rule at its best hyper-parameter against two at an arbitrary one is
not a comparison. `sweep_parameters` runs softmax over a range of temperatures and UCB over a
range of constants, and the study reports each rule at its own best cell as well as the whole
grid, so a reader can see whether the ranking survives the choice.
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

#: Softmax temperatures and UCB constants swept by `sweep_parameters`.
TEMPERATURES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
UCB_CONSTANTS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

#: A Q-value range wide enough that `exp` would overflow before the shift in `softmax_action`.
_CLIP = 60.0


def legal_of(mask: Sequence[bool]) -> list[int]:
    return [a for a in range(A.N_ACTIONS) if mask[a]]


def softmax_action(agent, key: int, mask: Sequence[bool], rng: random.Random, temp: float) -> int:
    """Boltzmann over the legal Q-values, at temperature `temp`.

    The row is shifted by its own maximum before exponentiating. Without that shift this
    overflows for real: terminal rewards on this corpus run past +8 and below -40 (a 90-node page
    with every node unreachable is -45 from the unreachable term alone), and `exp(45 / 0.25)` is
    not a number. The shift is exact, not an approximation - it cancels in the ratio.
    """
    legal = legal_of(mask)
    if len(legal) == 1:
        return legal[0]
    temp = max(1e-6, temp)
    row = agent.q[key]
    values = [float(row[a]) / temp for a in legal]
    top = max(values)
    weights = [math.exp(max(-_CLIP, v - top)) for v in values]
    total = sum(weights)
    if not total or not math.isfinite(total):
        return rng.choice(legal)
    draw = rng.random() * total
    for action, weight in zip(legal, weights, strict=True):
        draw -= weight
        if draw <= 0:
            return action
    return legal[-1]


def ucb_action(agent, key: int, mask: Sequence[bool], rng: random.Random, c: float) -> int:
    """`argmax Q + c * sqrt(ln N(s) / N(s,a))`, untried legal actions first, ties at random.

    "Untried first" is not an optimisation - it is what makes the bonus defined. With N(s,a) = 0
    the bonus is infinite, and an implementation that computed it anyway would return `inf` for
    every untried action and then pick among them by index, which silently reintroduces the
    positional bias the random tie-break exists to remove.
    """
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
    """The dispatch `TabularAgent.select` falls through to. Always returns a legal action.

    Softmax and UCB do their own exploring, so neither is wrapped in an epsilon draw - stacking
    a uniform random action on top of a rule that is already exploring would make the comparison
    a comparison of three epsilon-greedy agents with different tie-breaks.
    """
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
# the study
# ------------------------------------------------------------------------------------------


def _config(rule: str, episodes: int, seed: int, temp: float, c: float):
    from src.rl.qlearning import TrainConfig

    return TrainConfig(explore=rule, episodes=episodes, seed=seed, temperature=temp, ucb_c=c)


def study(
    graphs,
    diagrams=None,
    episodes: int = 20_000,
    seeds: Sequence[int] = (0, 1, 2),
    temperature: float = 1.0,
    ucb_c: float = 1.0,
) -> dict[str, Any]:
    """Train one agent per (rule, seed) and report the reward curve and the greedy evaluation."""
    from src.rl.qlearning import evaluate, train
    from src.rl.sarsa import smooth

    out: dict[str, Any] = {"episodes": episodes, "seeds": list(seeds), "rules": {}}
    for rule in RULES:
        rows: list[dict[str, Any]] = []
        curves: list[list[float]] = []
        explored: list[float] = []
        for seed in seeds:
            agent = train(graphs, _config(rule, episodes, seed, temperature, ucb_c))
            rows.append(evaluate(agent, graphs, diagrams))
            curves.append(agent.history["reward"])
            explored.append(float(agent.report["reached_keys"]))
        mean_curve = [statistics.fmean(x) for x in zip(*curves, strict=True)]
        out["rules"][rule] = {
            "mean_terminal_reward": round(
                statistics.fmean([r["mean_terminal_reward"] for r in rows]), 4
            ),
            "sd_terminal_reward": (
                round(statistics.stdev([r["mean_terminal_reward"] for r in rows]), 4)
                if len(rows) > 1
                else 0.0
            ),
            "full_coverage": round(statistics.fmean([r["full_coverage"] for r in rows]), 4),
            "mean_emitted_share": round(
                statistics.fmean([r["mean_emitted_share"] for r in rows]), 4
            ),
            "mean_edge_f1": (
                round(statistics.fmean([r["mean_edge_f1"] for r in rows]), 4)
                if "mean_edge_f1" in rows[0]
                else None
            ),
            #: how much of the state space the rule actually visited - the point of exploring.
            "reached_keys": round(statistics.fmean(explored), 1),
            "curve": [round(v, 4) for v in smooth(mean_curve)],
        }
    return out


def sweep_parameters(
    graphs,
    diagrams=None,
    episodes: int = 6000,
    seed: int = 0,
    temperatures: Sequence[float] = TEMPERATURES,
    constants: Sequence[float] = UCB_CONSTANTS,
) -> dict[str, Any]:
    """Softmax over temperatures and UCB over constants, so neither is judged at one guess."""
    from src.rl.qlearning import evaluate, train

    out: dict[str, Any] = {"softmax": [], "ucb": []}
    for temp in temperatures:
        agent = train(graphs, _config("softmax", episodes, seed, temp, 1.0))
        row = {"temperature": temp}
        row.update(evaluate(agent, graphs, diagrams))
        out["softmax"].append(row)
    for c in constants:
        agent = train(graphs, _config("ucb", episodes, seed, 1.0, c))
        row = {"ucb_c": c}
        row.update(evaluate(agent, graphs, diagrams))
        out["ucb"].append(row)
    return out


def plot(result: dict[str, Any], path=None):
    """The plan's "reward curves": one smoothed curve per rule, plus the greedy bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or ROOT / "reports" / "figures" / "p11_exploration.png"
    colours = {"egreedy": "#1f77b4", "softmax": "#2ca02c", "ucb": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for rule in RULES:
        block = result["rules"][rule]
        axes[0].plot(block["curve"], color=colours[rule], lw=1.2, label=rule)
    axes[0].set_title("episode reward (trailing mean, 200; mean over seeds)", fontsize=9)
    axes[0].set_xlabel("episode", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].legend(fontsize=8, frameon=False)

    rules = list(RULES)
    axes[1].bar(
        rules,
        [result["rules"][r]["mean_terminal_reward"] for r in rules],
        yerr=[result["rules"][r]["sd_terminal_reward"] for r in rules],
        capsize=3,
        color=[colours[r] for r in rules],
    )
    axes[1].axhline(0.0, color="#444444", lw=0.8)
    axes[1].set_title("greedy mean terminal reward", fontsize=9)
    axes[1].tick_params(labelsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(
    episodes: int = 20_000,
    limit: int | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    write: bool = False,
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS, references, training_set

    raw, graphs = training_set(limit, ambiguous=False)
    raw_amb, graphs_amb = training_set(limit, ambiguous=True)
    result = {
        "study_all": study(graphs, raw, episodes=episodes, seeds=seeds),
        "study_ambiguous": study(graphs_amb, raw_amb, episodes=episodes, seeds=seeds),
        "parameters": sweep_parameters(graphs, raw, episodes=max(2000, episodes // 4)),
        "references": references(graphs, raw),
    }
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "exploration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["figure"] = str(plot(result["study_all"]).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.3 exploration study")
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    episodes = 600 if args.quick else args.episodes
    seeds = (0,) if args.quick else (0, 1, 2)
    print(json.dumps(run(episodes, args.limit, seeds, write=args.write), indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

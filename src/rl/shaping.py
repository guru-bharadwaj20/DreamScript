"""Phase 11.2.4 - reward shaping: intermediate rewards for valid partial structure, and the
exploit check that has to come with them.

    python -m src.rl.shaping --write    # every variant, 5 seeds x 60k, scored unshaped
    python -m src.rl.shaping --quick    # a short run, for the tests

11.1.3's reward is almost entirely terminal: `step_reward` is a flat -0.01 plus penalties, and
coverage, ordering, structure and loops are paid once at the end. Shaping is the standard answer
to that credit-assignment problem and the standard way to get a policy that scores well on a
reward nobody wanted, so this row is half construction and half audit.

## Two kinds of shaping

**Potential-based** (Ng, Harada & Russell 1999): the loop adds `gamma * Phi(s') - Phi(s)`, with
`Phi = 0` at the terminal. Over an episode the added terms telescope to `-Phi(s_0)`, a constant,
so in the MDP over raw states the optimal policy is unchanged. **That guarantee does not transfer
cleanly to this learner** and the row says so: `Phi` is a function of the raw `TraversalState`
while the Q-table is indexed by 11.1.6's abstract key, which aliases 25.2% of raw states, so the
shaping term is not a function of the learner's state. Invariance is therefore measured, not
assumed. Three potentials, each in [0, SCALE]:

    emitted     share of nodes emitted
    visited     share of nodes visited
    structure   11.1.3's own `structure_score` of the partial emission - "valid partial
                structure" in the plan's words, the term the terminal reward already prices,
                paid as it accrues

**Unconstrained per-event bonuses** (`TrainConfig.step_bonus`), no invariance guarantee, there
to measure the failure rather than warn about it:

    emit_bonus    + BONUS per emission. Bounded by n: the mask refuses a second emission.
    loop_bonus    + BONUS per `mark-as-loop`. Bounded too: the mask allows a mark only on an
                  unmarked node with a real back edge (11.1.2), so at most one per such node.
    visit_bonus   + BONUS per `follow-edge`, *including onto an already-visited node*. **Not
                  bounded**: a follow onto a back edge and a backtrack can be repeated until the
                  `4n + 8` cap. This is the one with a farm in it.

The inherited draft had the last two the wrong way round - it called `loop_bonus` "the exploit"
and `visit_bonus` "also bounded". The mask makes the first false and `Outcome.moved_to` (set on
every follow, revisits included) makes the second false.

## How a variant is judged

Every variant trains under its own shaped reward and is **evaluated greedily on the unshaped
one** - `src.rl.qlearning.evaluate` prices the finished episode with `RewardConfig()` and knows
nothing about the shaping. A variant is called an **exploit** when its training return (shaped,
what it was optimising) finishes above the control's while its unshaped greedy reward finishes
below the control's by more than two standard errors of the seed-mean difference - it was paid for something that costs the
real objective - and the behaviour indicators (`behaviour`) are reported to show *what* it was
paid for. A variant is **justified** only if its unshaped reward beats the control by more than
that same two-standard-error band. Measured results are in `reports/rl_shaping.md`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Any

from src.rl.reward import structure_score
from src.rl.state import DiagramGraph, TraversalState
from src.utils.config import ROOT

#: Potentials are in [0, 1] scaled by this, which puts a full-emission potential on the scale of
#: `RewardConfig.semantic_bonus` (4.0) so the shaping is not lost against the terminal reward.
SCALE = 4.0

#: Per-event bonus for the unconstrained variants: 25x the step cost, half an unreachable node.
BONUS = 0.25

FIGURE = ROOT / "reports" / "figures" / "p11_shaping.png"
#: 11.2.2's most repeatable Q-learning cell (greedy reward sd 0.10 over 5 seeds, against 0.58 at
#: 11.2.1's alpha 0.4 gamma 1.0). A first run at 11.2.1's cell had a control sd of 0.72 over 3
#: seeds, which no shaping effect in this study could clear.
BASE = {"algo": "q", "alpha": 0.2, "gamma": 0.99}


# ------------------------------------------------------------------------------------------
# potentials
# ------------------------------------------------------------------------------------------


def phi_emitted(graph: DiagramGraph, state: TraversalState) -> float:
    return SCALE * state.n_emitted() / max(1, graph.n_nodes)


def phi_visited(graph: DiagramGraph, state: TraversalState) -> float:
    return SCALE * state.n_visited() / max(1, graph.n_nodes)


def phi_structure(graph: DiagramGraph, state: TraversalState) -> float:
    """11.1.3's `structure_score` of the emission so far, scaled."""
    return SCALE * structure_score(graph, state)


POTENTIALS: dict[str, Callable[[DiagramGraph, TraversalState], float]] = {
    "emitted": phi_emitted,
    "visited": phi_visited,
    "structure": phi_structure,
}


# ------------------------------------------------------------------------------------------
# unconstrained bonuses
# ------------------------------------------------------------------------------------------


def emit_bonus(graph: DiagramGraph, outcome, state: TraversalState) -> float:
    return BONUS if outcome.emitted_node is not None else 0.0


def visit_bonus(graph: DiagramGraph, outcome, state: TraversalState) -> float:
    """Paid on every follow, revisits included - which is what makes it farmable."""
    return BONUS if outcome.moved_to is not None else 0.0


def loop_bonus(graph: DiagramGraph, outcome, state: TraversalState) -> float:
    return BONUS if outcome.marked_loop is not None else 0.0


BONUSES: dict[str, Callable[[DiagramGraph, Any, TraversalState], float]] = {
    "emit_bonus": emit_bonus,
    "visit_bonus": visit_bonus,
    "loop_bonus": loop_bonus,
}


# ------------------------------------------------------------------------------------------
# exploit indicators
# ------------------------------------------------------------------------------------------


def behaviour(agent, graphs: Sequence[DiagramGraph]) -> dict[str, float]:
    """What the greedy policy *does*, past what it scores. The shape of an exploit."""
    marks: list[float] = []
    emits: list[float] = []
    lengths: list[float] = []
    revisits: list[float] = []
    follows: list[float] = []
    truncated = 0
    for graph in graphs:
        episode = agent.rollout(graph)
        state = episode.state
        n = max(1, graph.n_nodes)
        marks.append(int(state.loop_marked).bit_count() / n)
        emits.append(state.n_emitted() / n)
        lengths.append(state.steps)
        steps = max(1, len(episode.history))
        follows.append(sum(1 for o in episode.history if o.moved_to is not None) / n)
        revisits.append(sum(1 for o in episode.history if o.revisit) / steps)
        truncated += int(episode.truncated())
    return {
        "marks_per_node": round(statistics.fmean(marks), 4),
        "follows_per_node": round(statistics.fmean(follows), 4),
        "revisit_step_share": round(statistics.fmean(revisits), 4),
        "emitted_share": round(statistics.fmean(emits), 4),
        "mean_length": round(statistics.fmean(lengths), 2),
        "truncation_rate": round(truncated / max(1, len(graphs)), 4),
    }


class _Gold:
    def rollout(self, graph):
        from src.rl.qlearning import gold_player

        return gold_player(graph)


def gold_behaviour(graphs: Sequence[DiagramGraph]) -> dict[str, float]:
    """The same indicators for gold play, the scale the learned ones are read against."""
    return behaviour(_Gold(), graphs)


def behaviour_extra(agent, sets, spec) -> dict[str, Any]:
    """`src.rl.sarsa.run_job` extra: the indicators on every set."""
    return {name: behaviour(agent, graphs) for name, (_, graphs) in sets.items()}


# ------------------------------------------------------------------------------------------
# the study
# ------------------------------------------------------------------------------------------


def variants() -> dict[str, dict[str, str]]:
    """name -> the `run_job` hook fields that define it. `none` is the unshaped control."""
    out: dict[str, dict[str, str]] = {"none": {}}
    for name, phi in POTENTIALS.items():
        out[f"potential_{name}"] = {"potential": f"src.rl.shaping:{phi.__name__}"}
    for name in BONUSES:
        out[name] = {"step_bonus": f"src.rl.shaping:{name}"}
    return out


def study(
    sets,
    episodes: int = 60_000,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    workers: int | None = None,
) -> dict[str, Any]:
    """Train every variant under its own reward; score every variant under the unshaped one."""
    from src.rl.sarsa import run_jobs, stat, summarise

    names = list(variants())
    specs = [
        {
            "cfg": {**BASE, "episodes": episodes, "seed": seed},
            **variants()[name],
            "variant": name,
            "keep": ["reward", "truncated", "length"],
            "extra": "src.rl.shaping:behaviour_extra",
        }
        for name in names
        for seed in seeds
    ]
    jobs = run_jobs(specs, sets, workers)
    out: dict[str, Any] = {"episodes": episodes, "seeds": list(seeds), "variants": {}}
    for name in names:
        mine = [j for j in jobs if j["spec"]["variant"] == name]
        block = summarise(_strip(mine))
        tail = max(1, episodes // 10)
        block["shaped_training_return_last_10pct"] = stat(
            [statistics.fmean(j["history"]["reward"][-tail:]) for j in mine]
        )
        block["training_length_last_10pct"] = stat(
            [statistics.fmean(j["history"]["length"][-tail:]) for j in mine]
        )
        block["behaviour"] = {
            scope: {k: stat([j["extra"][scope][k] for j in mine]) for k in mine[0]["extra"][scope]}
            for scope in mine[0]["extra"]
        }
        out["variants"][name] = block
    out["gold_behaviour"] = {name: gold_behaviour(graphs) for name, (_, graphs) in sets.items()}
    out["verdict"] = verdict(out)
    return out


def _strip(jobs):
    """`summarise` reads `extra` as 11.2.2's online evaluation; shaping's extra is not that."""
    return [{k: v for k, v in j.items() if k != "extra"} for j in jobs]


def verdict(result: dict[str, Any], scope: str = "all") -> dict[str, Any]:
    """Per variant: justified, exploit, or neither - each against 2 s.e. of the seed means."""
    control = result["variants"]["none"]
    c_reward = control[scope]["mean_terminal_reward"]
    c_return = control["shaped_training_return_last_10pct"]["mean"]
    rows = []
    for name, block in result["variants"].items():
        if name == "none":
            continue
        reward = block[scope]["mean_terminal_reward"]
        n = max(1, len(reward.get("values", [])) or 1)
        band = 2.0 * ((c_reward["sd"] ** 2 + reward["sd"] ** 2) / n) ** 0.5
        delta = reward["mean"] - c_reward["mean"]
        shaped_return = block["shaped_training_return_last_10pct"]["mean"]
        rows.append(
            {
                "variant": name,
                "delta_unshaped_reward": round(delta, 4),
                "two_standard_errors": round(band, 4),
                "justified": bool(delta > band),
                "exploit": bool(shaped_return > c_return and -delta > band),
                "shaped_return_minus_control": round(shaped_return - c_return, 4),
            }
        )
    return {"scope": scope, "control_reward": c_reward["mean"], "rows": rows}


# ------------------------------------------------------------------------------------------
# plot
# ------------------------------------------------------------------------------------------


def plot(result: dict[str, Any], refs: dict[str, Any] | None = None, path=FIGURE):
    """Unshaped greedy reward per variant (seeds as dots) next to the exploit indicators."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["variants"])
    colour = {"none": "#8a8a8a"}
    for n in names:
        colour.setdefault(n, "#2a78d6" if n.startswith("potential") else "#eb6834")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    panels = (
        (axes[0], lambda b: b["all"]["mean_terminal_reward"], "unshaped greedy terminal reward"),
        (axes[1], lambda b: b["shaped_training_return_last_10pct"], "shaped training return"),
        (
            axes[2],
            lambda b: b["behaviour"]["all"]["truncation_rate"],
            "greedy episodes truncated at the cap",
        ),
    )
    for ax, get, title in panels:
        for i, name in enumerate(names):
            s = get(result["variants"][name])
            ax.barh(i, s["mean"], 0.7, color=colour[name])
            ax.scatter(s["values"], [i] * len(s["values"]), color="#222222", s=12, zorder=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names))
        ax.axvline(0, color="#888888", lw=0.8)
        ax.invert_yaxis()
        ax.set_title(title + "\n(last 10% of training)" if "training" in title else title)
    if refs:
        axes[0].axvline(
            refs["all"]["gold"]["mean_terminal_reward"], color="#1baf7a", ls="--", lw=1.5
        )
        axes[0].text(
            refs["all"]["gold"]["mean_terminal_reward"], len(names) - 0.4, " gold", color="#333333"
        )
    control = result["variants"]["none"]["all"]["mean_terminal_reward"]["mean"]
    axes[0].axvline(control, color="#8a8a8a", ls=":", lw=1.5)
    fig.suptitle(
        f"11.2.4 - shaping variants on the 993 labelled diagrams, {len(result['seeds'])} seeds each "
        "(blue: potential-based, orange: unconstrained bonus, grey: unshaped control)"
    )
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
    limit: int | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    write: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    from src.rl.qlearning import RUNS, references
    from src.rl.sarsa import load_sets

    sets = load_sets(limit)
    result = study(sets, episodes, seeds, workers)
    result["references"] = {name: references(g, raw) for name, (raw, g) in sets.items()}
    result["scale"] = SCALE
    result["bonus"] = BONUS
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "shaping.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["figure"] = str(plot(result, result["references"]).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.4 reward shaping and its exploit check")
    ap.add_argument("--episodes", type=int, default=60_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        out = run(600, args.limit, (0,), workers=1)
    else:
        seeds = tuple(range(max(1, args.seeds)))
        out = run(args.episodes, args.limit, seeds, args.write, args.workers)
    print(json.dumps(out, indent=2)[:30000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

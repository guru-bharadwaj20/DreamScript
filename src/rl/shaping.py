"""Phase 11.2.4 - reward shaping: intermediate rewards for valid partial structure, and the
exploit check that has to come with them.

    python -m src.rl.shaping            # every shaping variant, evaluated unshaped
    python -m src.rl.shaping --quick    # a short run, for the tests

11.1.3's reward is almost entirely terminal: `step_reward` is a flat -0.01 with penalties, and
everything that matters - coverage, ordering, structure, loops - is paid once at the end. On a
corpus whose episodes run to 65 steps that is a hard credit-assignment problem, and shaping is
the standard answer. It is also the standard way to get a policy that scores well on a reward
nobody wanted, so this row is half construction and half audit.

## Two kinds of shaping, and only one of them is safe by construction

**Potential-based** (Ng, Harada & Russell): add `gamma * Phi(s') - Phi(s)` for any function Phi
of the state. The added terms telescope over an episode to `gamma^T Phi(s_T) - Phi(s_0)`, so the
*ordering* of policies by return is unchanged and the optimal policy is provably the same. Three
potentials are offered here - emitted fraction, visited fraction, and 11.1.3's own
`structure_score` - all in [0, 1] and scaled by `SCALE`.

**Unconstrained** ("just add a bonus when something good happens"): `TrainConfig.step_bonus`.
This has no invariance guarantee, and the point of including it is to *measure* the failure
rather than warn about it in a comment. Three are offered, and the third is designed to break:

    emit_bonus   + b for each node emitted. Bounded by n per episode; the mask forbids emitting
                 a node twice, so there is no obvious farm.
    visit_bonus  + b for each first visit. Also bounded.
    loop_bonus   + b for each `mark-as-loop`. **This is the exploit.** 11.1.2 lets a policy mark
                 any node with a back edge, marking is not restricted to nodes it will ever
                 emit, and 11.1.3's `loop_score` rewards marks it agrees with - so a bonus here
                 buys reward for an action that produces no code.

## How a variant is judged

Every variant is trained with its own shaped reward and then **evaluated on the unshaped one**.
That is the whole audit: `src.rl.qlearning.evaluate` prices the finished episode with
`RewardConfig()` and knows nothing about the shaping the agent trained under, so a variant that
only looks good under its own bonus shows up here as a loss. Alongside the reward, four exploit
indicators are reported per variant - marks per node, episode length, truncation rate and
emitted share - because a shaped agent that scores the same reward by doing something very
different is the case a single scalar hides.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Any

from src.rl.episode import Episode
from src.rl.reward import DEFAULT, structure_score
from src.rl.state import DiagramGraph, TraversalState

#: Potentials are in [0, 1]; this puts a full-coverage potential on the same scale as
#: `RewardConfig.semantic_bonus` (4.0), so the shaping is visible against the terminal reward
#: rather than lost in it.
SCALE = 4.0

#: The per-event bonus for the unconstrained variants. Deliberately the same size as the step
#: cost is small, so a variant that farms an action is farming something worth having.
BONUS = 0.25


# ------------------------------------------------------------------------------------------
# potentials - safe by construction
# ------------------------------------------------------------------------------------------


def phi_emitted(graph: DiagramGraph, state: TraversalState) -> float:
    """Progress towards emitting the diagram. The most direct statement of the goal."""
    return SCALE * state.n_emitted() / max(1, graph.n_nodes)


def phi_visited(graph: DiagramGraph, state: TraversalState) -> float:
    """Progress towards *reaching* the diagram - weaker, and reachable without emitting."""
    return SCALE * state.n_visited() / max(1, graph.n_nodes)


def phi_structure(graph: DiagramGraph, state: TraversalState) -> float:
    """11.1.3's own `structure_score`: the share of real edges the emission order got adjacent.

    This is "valid partial structure" in the plan's words, taken literally - it is the term the
    terminal reward already prices, moved earlier in the episode.
    """
    return SCALE * structure_score(graph, state)


POTENTIALS: dict[str, Callable[[DiagramGraph, TraversalState], float]] = {
    "emitted": phi_emitted,
    "visited": phi_visited,
    "structure": phi_structure,
}


# ------------------------------------------------------------------------------------------
# unconstrained bonuses - the audit
# ------------------------------------------------------------------------------------------


def emit_bonus(graph: DiagramGraph, outcome, state: TraversalState) -> float:
    return BONUS if outcome.emitted_node is not None else 0.0


def visit_bonus(graph: DiagramGraph, outcome, state: TraversalState) -> float:
    return BONUS if outcome.moved_to is not None else 0.0


def loop_bonus(graph: DiagramGraph, outcome, state: TraversalState) -> float:
    """The one built to be exploited: marking a loop costs a step and produces no code."""
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
    """What the greedy policy *does*, past what it scores. The shape of an exploit.

    `marks_per_node` is the one to watch: a policy that has learned to farm `mark-as-loop` shows
    up here at several times gold's rate while its emitted share stays flat.
    """
    marks: list[float] = []
    emits: list[float] = []
    lengths: list[float] = []
    truncated = 0
    illegal: list[float] = []
    for graph in graphs:
        episode = agent.rollout(graph)
        state = episode.state
        marks.append(int(state.loop_marked).bit_count() / max(1, graph.n_nodes))
        emits.append(state.n_emitted() / max(1, graph.n_nodes))
        lengths.append(state.steps)
        truncated += int(episode.truncated())
        illegal.append(
            sum(1 for o in episode.history if not o.legal) / max(1, len(episode.history))
        )
    return {
        "marks_per_node": round(statistics.fmean(marks), 4),
        "emitted_share": round(statistics.fmean(emits), 4),
        "mean_length": round(statistics.fmean(lengths), 2),
        "truncation_rate": round(truncated / max(1, len(graphs)), 4),
        "illegal_action_share": round(statistics.fmean(illegal), 4),
    }


def gold_behaviour(graphs: Sequence[DiagramGraph]) -> dict[str, float]:
    """The same indicators for gold play, so `marks_per_node` has a scale to be read against."""

    class _Gold:
        def rollout(self, graph):
            episode = Episode(graph)
            for action in episode.gold_actions():
                if episode.done():
                    break
                episode.apply(action)
            return episode

    return behaviour(_Gold(), graphs)


# ------------------------------------------------------------------------------------------
# the study
# ------------------------------------------------------------------------------------------


def variants() -> dict[str, dict[str, Any]]:
    """name -> the `TrainConfig` fields that define it. `none` is the unshaped control."""
    out: dict[str, dict[str, Any]] = {"none": {}}
    for name, phi in POTENTIALS.items():
        out[f"potential_{name}"] = {"potential": phi}
    for name, bonus in BONUSES.items():
        out[name] = {"step_bonus": bonus}
    return out


def study(
    graphs,
    diagrams=None,
    episodes: int = 20_000,
    seeds: Sequence[int] = (0, 1, 2),
    alpha: float = 0.4,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Train every variant under its own reward; score every variant under the unshaped one."""
    from src.rl.qlearning import TrainConfig, evaluate, train

    out: dict[str, Any] = {"episodes": episodes, "seeds": list(seeds), "variants": {}}
    for name, fields in variants().items():
        rows: list[dict[str, Any]] = []
        shapes: list[dict[str, float]] = []
        for seed in seeds:
            cfg = TrainConfig(alpha=alpha, gamma=gamma, episodes=episodes, seed=seed, **fields)
            agent = train(graphs, cfg)
            rows.append(evaluate(agent, graphs, diagrams, DEFAULT))
            shapes.append(behaviour(agent, graphs))
        out["variants"][name] = {
            "mean_terminal_reward": round(
                statistics.fmean([r["mean_terminal_reward"] for r in rows]), 4
            ),
            "sd_terminal_reward": (
                round(statistics.stdev([r["mean_terminal_reward"] for r in rows]), 4)
                if len(rows) > 1
                else 0.0
            ),
            "full_coverage": round(statistics.fmean([r["full_coverage"] for r in rows]), 4),
            "mean_edge_f1": (
                round(statistics.fmean([r["mean_edge_f1"] for r in rows]), 4)
                if "mean_edge_f1" in rows[0]
                else None
            ),
            "behaviour": {k: round(statistics.fmean([s[k] for s in shapes]), 4) for k in shapes[0]},
        }
    out["gold_behaviour"] = gold_behaviour(graphs)
    out["verdict"] = verdict(out)
    return out


def verdict(result: dict[str, Any]) -> dict[str, Any]:
    """Did any variant beat the unshaped control on the reward nobody shaped, and who cheated?

    An "exploit" here is a specific, checkable claim and not an impression: the variant marks
    loops at more than twice gold's rate *and* does not beat the control's unshaped reward - it
    is being paid for behaviour that buys nothing.
    """
    control = result["variants"]["none"]["mean_terminal_reward"]
    gold_marks = result["gold_behaviour"]["marks_per_node"]
    rows = []
    for name, block in result["variants"].items():
        if name == "none":
            continue
        marks = block["behaviour"]["marks_per_node"]
        rows.append(
            {
                "variant": name,
                "delta_unshaped_reward": round(block["mean_terminal_reward"] - control, 4),
                "beats_control": block["mean_terminal_reward"] > control,
                "marks_per_node": marks,
                "marks_vs_gold": round(marks / gold_marks, 2) if gold_marks else None,
                "exploit": bool(
                    marks > 2 * max(gold_marks, 0.01) and block["mean_terminal_reward"] <= control
                ),
            }
        )
    return {
        "control_reward": control,
        "best": max(rows, key=lambda r: r["delta_unshaped_reward"])["variant"] if rows else None,
        "rows": rows,
    }


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
        "all": study(graphs, raw, episodes=episodes, seeds=seeds),
        "ambiguous": study(graphs_amb, raw_amb, episodes=episodes, seeds=seeds),
        "references": references(graphs, raw),
        "scale": SCALE,
        "bonus": BONUS,
    }
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "shaping.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11.2.4 reward shaping and its exploit check")
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

"""Phase 11.1.3 - what a traversal is paid for, and what it is charged.

    python -m src.rl.reward          # the measured separation table below

The plan's line is *+1 syntactically valid code; bonus for semantic correctness (code runs and
passes tests); penalty for infinite loops, unreachable nodes, duplicate emission*.

## The +1, and why it is a floor rather than a signal

`src.rl.emit` turns the emitted order into real Python and `ast.parse` decides. **Measured over
3,993 diagrams under three different emission orders, the parse rate is 100.00% in all three**:
a template emitter is valid by construction. The +1 is therefore paid, exactly as the plan says,
but it carries no gradient, so the discriminating weight lives in `semantic_score`'s four terms.

## The semantic bonus, and the limitation that has to be recorded

**Phase 12 does not exist, so "code runs and passes tests" has no tests to run against, and this
module does not pretend otherwise.** `semantic_score` is measured against `src.rl.emit`'s
**template emitter**, not against an LLM and not against any task-level notion of correctness.
Nothing here is evidence for S6 or S7; it is evidence about traversal order only.

Two modes:

    static proxy (default)   `SEMANTIC_WEIGHTS` over coverage / structure / ordering / loops,
                             at 1.17 ms per terminal.
    sandbox (opt-in)         `RewardConfig(sandbox=sandbox_hook())` actually executes the emitted
                             module in `src.rl.sandbox` and pays only for `kind == "ok"`.

**The sandbox is wired and measured, and what it measures is narrower than it looks.** Over 200
random diagrams: gold play **126/200 (63.0%) execute cleanly**, and the other 74 raise `NameError`
because the emitted `EDGES` table names a node the policy could not reach; a half-emitted order is
**0/200**. That 63.0% is the same 63.06% as `src.rl.episode`'s coverage ceiling, which is the
honest finding: **on this emitter the sandbox verdict is very nearly a coverage indicator**, not
an independent measure of semantic quality. It is worth having as an executable check that the
emitted text is real Python that runs, and it is not worth reading as a semantic score.

**Cost, and why it is terminal-only**: `src.rl.sandbox.run` spawns a fresh interpreter at a
measured **71.4 ms per call**, against **1.17 ms** for the entire static proxy - **61x**. At the
corpus's p95 of 26 gold steps, calling it per step would cost 1.86 s of subprocess time per
episode. `terminal_reward` calls it exactly once, at the terminal; `step_reward` never calls it.

## The penalties

    infinite loop        the episode was truncated by the step cap.
    unreachable node     every node never visited when the episode ended, charged per node.
    duplicate emission   every refused re-emission of an already-emitted node, per attempt
                         (`src.rl.episode` counts these; the mask makes them illegal, so they are
                         charged once each and do not corrupt the emitted order).

## What was MEASURED - does the reward actually rank policies correctly?

Mean terminal reward over all 3,993 sequential diagrams, seed 0. The reversed and half rows are
built from **the gold episode's own emitted sequence**, so coverage and the unreachable penalty
are held fixed and only the emitted order varies:

    order                              mean terminal reward    std    semantic score   coverage
    gold replay                                    +2.8537   3.9705           0.8130     63.06%
    gold emission reversed                         +2.0102   3.6911           0.6022     63.06%
    first half of the gold emission                +1.6990   3.5285           0.5244      0.00%
    uniform random over legal actions              +0.4957   3.4909           0.4345      2.68%

The ordering is monotone and the reward now separates a correct traversal from its own reverse
(+0.84 with everything else held identical), which is the thing the inherited version could not
do. `full_coverage_rate` peaking at 63.06% rather than 100% is `src.rl.episode`'s reachability
ceiling, not a reward property.

## Two inherited claims that were false, and what they cost

**The inherited docstring said `structure_score` separated orders "0.00% of gold emissions have an
undefined name against 31.71% reversed and 61.33% half-emitted". Measured, it was 89.83% / 0.83%
/ 28.33% - the signal was inverted**, and `structure_score` scored gold **0.8915** against
reversed **0.9997**. The cause was in `src.rl.emit`: it emitted `_cond`/`_again` predicates and
deliberately left them undefined, so `undefined_names` counted *structure the policy successfully
built* rather than structure it failed to close. A policy trained against that reward would have
learned to avoid emitting branches. `src.rl.emit` now defines every predicate it calls and names
both endpoints of every edge in an `EDGES` table, so an undefined name means the one thing it
should: the policy referenced a node it never emitted.

**The inherited table's "depth-first, emit-on-arrival" row scored identically to gold (+4.4232 on
both) and the docstring drew a conclusion from the tie** - that the reward "does not currently
distinguish two sensible traversals from each other". No such policy exists in `measure()`; the
row could not be reproduced and is deleted. The conclusion it supported was also wrong, as the
gold-versus-reversed gap above shows.

## What was REJECTED

**A dense per-step shaping term for "made progress" was rejected at this row.** 11.2.4 owns reward
shaping and owns checking it for exploits; adding shaping here would mean the ablation in 11.2.4
had nothing to ablate against. `step_reward` charges a small `step_cost` and nothing else, so the
episode return is dominated by the terminal.

**Making `semantic_score` coverage-only was rejected**: coverage cannot see ordering at all, so a
reversed traversal scores identically to gold under it. That is precisely the failure the
`ordering` weight exists to fix, and it is why that weight is 0.25 rather than a token amount.

**Rewarding `mark-as-loop` unconditionally was rejected**: the mask already refuses a mark that is
not a real back edge, so paying for the action pays for a tautology. The loop credit is in
`semantic_score` and is per *closed* back edge, against the diagram's own back-edge set.

**The inherited "illegal action at -1.0 collapsed episode length from 18.4 to 2.1 steps"
rejection was deleted, not re-run.** It describes a sweep this module has no record of and
nothing here reproduces it. `illegal_action` stays at -0.05 because the mask already makes
illegal actions unreachable for a correct agent, so the coefficient only fires on a buggy one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from src.rl.emit import check, emit_code, order_score
from src.rl.episode import Episode, Outcome
from src.rl.state import DiagramGraph, TraversalState

#: Signature a sandbox hook must satisfy: `run(code) -> dict`. `src.rl.sandbox.run` satisfies it
#: via `sandbox_hook()` below, which is the wiring 11.1.3 asks for.
SandboxHook = Callable[[str], dict]


def sandbox_hook(timeout_s: float = 5.0, memory_limit_mb: int = 256) -> SandboxHook:
    """A `RewardConfig.sandbox` hook backed by `src.rl.sandbox.run`. **Terminal-only.**

    `src.rl.sandbox.run` starts a fresh interpreter, so it costs a process spawn per call.
    **Measured over 400 calls on emitted flowchart code: 71.4 ms per call**, against 1.17 ms for
    the whole static proxy - **61x**. At the corpus's p95 of 26 gold steps, calling it per step
    would cost 1.86 s of subprocess time per episode, so `terminal_reward` calls it exactly once,
    at the terminal, and `step_reward` never calls it at all.

    The verdict is mapped to the `passed` flag `semantic_score` reads: `passed` is true only when
    the sandbox returns `kind == "ok"`, i.e. the emitted module imported, ran `run()` and exited
    0. A partial emission raises `NameError` on the `EDGES` table and lands in `exception`.
    """

    def run(code: str) -> dict:
        from src.rl import sandbox

        verdict = sandbox.run(code, timeout_s=timeout_s, memory_limit_mb=memory_limit_mb)
        return {
            "passed": verdict.get("kind") == "ok",
            "kind": verdict.get("kind"),
            "exit_code": verdict.get("exit_code"),
            "wall_time_s": round(float(verdict.get("wall_time_s") or 0.0), 4),
            "stderr_tail": (verdict.get("stderr") or "")[-200:],
        }

    return run


#: Weights of the four static-proxy terms. They sum to 1.0, so `semantic_score` is in [0, 1] and
#: `semantic_bonus` is the whole range of the bonus. `ordering` exists because coverage and
#: structure are both blind to it - a reversed traversal emits every node and resolves every name
#: (measured: structure 1.0000 on both gold and reversed), so without this term the reward cannot
#: tell a correct order from its reverse at all.
SEMANTIC_WEIGHTS: dict[str, float] = {
    "coverage": 0.4,
    "structure": 0.2,
    "ordering": 0.25,
    "loops": 0.15,
}

#: Times a node may be re-entered before a trajectory is called a loop. Retained for 11.2.4's
#: shaping sweep; `terminal_reward` charges `infinite_loop` on truncation only, so nothing in
#: this module reads it today and no measurement here depends on it.
REVISIT_LIMIT = 3


@dataclass
class RewardConfig:
    """Every coefficient in one place, so 11.2.4 can sweep them without editing code."""

    valid_code: float = 1.0
    semantic_bonus: float = 4.0
    unreachable_node: float = -0.5
    duplicate_emission: float = -0.5
    infinite_loop: float = -2.0
    illegal_action: float = -0.05
    step_cost: float = -0.01
    #: 11.2.9's hook. When set, `run(code)` is called at the terminal and its `passed` flag
    #: replaces the static `semantic_score`. Left None here on purpose; see the docstring.
    sandbox: SandboxHook | None = field(default=None, repr=False)

    def replace(self, **kwargs) -> RewardConfig:
        """A copy with fields overridden, for sweeps."""
        merged = {**self.__dict__, **kwargs}
        return RewardConfig(**merged)


DEFAULT = RewardConfig()


def step_reward(
    graph: DiagramGraph,
    outcome: Outcome,
    config: RewardConfig = DEFAULT,
) -> float:
    """Per-step reward. Deliberately almost flat - see the shaping note in the docstring."""
    reward = config.step_cost
    if not outcome.legal:
        reward += config.illegal_action
        if outcome.duplicate_emission:
            reward += config.duplicate_emission
    return round(reward, 6)


def structure_score(graph: DiagramGraph, state: TraversalState) -> float:
    """Fraction of the emitted code's calls whose target the policy also defined, in [0, 1].

    1.0 means every branch and loop the emitter opened was closed by a node the policy emitted;
    below 1.0 means the emitted order references structure it never produced.
    """
    if not state.emit_sequence:
        return 0.0
    code = emit_code(graph, list(state.emit_sequence), _marked_set(graph, state))
    verdict = check(code)
    if not verdict["parse_ok"]:
        return 0.0
    calls = max(1, verdict["n_calls"])
    return max(0.0, 1.0 - len(verdict["undefined_names"]) / calls)


def _marked_set(graph: DiagramGraph, state: TraversalState) -> set[int]:
    return {i for i in range(graph.n_nodes) if state.is_loop_marked(i)}


def loop_score(graph: DiagramGraph, state: TraversalState) -> float:
    """Fraction of the diagram's real back edges the policy closed with `mark-as-loop`.

    Back edges are `src.parse.sequences.traversal`'s, reused - the same definition 7.3 labels
    `loop-back` with. 1.0 when the diagram has no back edges, so acyclic diagrams are not
    penalised for having no loops to find.
    """
    sources = {a for a, _ in graph.back_edges}
    if not sources:
        return 1.0
    return len(sources & _marked_set(graph, state)) / len(sources)


def semantic_score(
    graph: DiagramGraph,
    state: TraversalState,
    config: RewardConfig = DEFAULT,
) -> tuple[float, dict]:
    """The stand-in for "code runs and passes tests", in [0, 1], plus its breakdown.

    **Static proxy, measured against `src.rl.emit`'s template, not against an LLM or execution.**
    When `config.sandbox` is set, 11.2.9's verdict replaces it and the breakdown records that.
    """
    coverage = state.n_emitted() / max(1, graph.n_nodes)
    structure = structure_score(graph, state)
    loops = loop_score(graph, state)
    ordering = order_score(graph, list(state.emit_sequence))
    detail = {
        "coverage": round(coverage, 6),
        "structure": round(structure, 6),
        "ordering": round(ordering, 6),
        "loops": round(loops, 6),
        "source": "static-proxy",
    }
    if config.sandbox is not None:
        code = emit_code(graph, list(state.emit_sequence), _marked_set(graph, state))
        try:
            verdict = config.sandbox(code)
        except Exception as exc:  # a sandbox failure must not kill a training run
            detail["sandbox_error"] = repr(exc)
            verdict = {}
        if verdict:
            detail["source"] = "sandbox"
            detail["sandbox"] = verdict
            return float(bool(verdict.get("passed"))), detail
    return (
        round(
            SEMANTIC_WEIGHTS["coverage"] * coverage
            + SEMANTIC_WEIGHTS["structure"] * structure
            + SEMANTIC_WEIGHTS["ordering"] * ordering
            + SEMANTIC_WEIGHTS["loops"] * loops,
            6,
        ),
        detail,
    )


def terminal_reward(
    graph: DiagramGraph,
    state: TraversalState,
    truncated: bool,
    config: RewardConfig = DEFAULT,
) -> tuple[float, dict]:
    """The end-of-episode reward and its full breakdown. Returns `(reward, info)`.

    `info` carries every term separately so 11.2.8's diagnostics can plot them without re-deriving
    anything, and so a shaping change in 11.2.4 is visible term by term.
    """
    code = emit_code(graph, list(state.emit_sequence), _marked_set(graph, state))
    verdict = check(code)
    syntactic = config.valid_code if verdict["parse_ok"] else 0.0

    score, detail = semantic_score(graph, state, config)
    bonus = config.semantic_bonus * score

    unreachable = [i for i in range(graph.n_nodes) if not state.has_visited(i)]
    unreachable_penalty = config.unreachable_node * len(unreachable)
    duplicate_penalty = config.duplicate_emission * state.duplicate_emissions
    loop_penalty = config.infinite_loop if truncated else 0.0

    total = syntactic + bonus + unreachable_penalty + duplicate_penalty + loop_penalty
    info = {
        "syntactic": round(syntactic, 6),
        "semantic_bonus": round(bonus, 6),
        "semantic_score": score,
        "semantic_detail": detail,
        "n_unreachable": len(unreachable),
        "unreachable_penalty": round(unreachable_penalty, 6),
        "n_duplicate_emissions": state.duplicate_emissions,
        "duplicate_penalty": round(duplicate_penalty, 6),
        "infinite_loop_penalty": round(loop_penalty, 6),
        "parse_ok": verdict["parse_ok"],
        "undefined_names": verdict["undefined_names"][:8],
        "code_lines": code.count("\n") + 1,
        "terminal_reward": round(total, 6),
    }
    return round(total, 6), info


def measure(limit: int | None = None, seed: int = 0) -> dict:
    """The docstring's separation table: mean terminal reward for four emission orders.

    `reversed_order` and `half_emitted` are built from **the gold episode's own emitted sequence**
    rather than from `graph.gold_order`, so coverage, the unreachable-node penalty and legality
    are held fixed and only the order varies. Building them from `gold_order` instead - which the
    inherited draft did - hands them full coverage on diagrams where gold play cannot reach it,
    and the reversed row then scores *above* gold (measured: +3.90 against +2.85) purely because
    it was credited with nodes no policy in this action space can reach.
    """
    import random

    from src.parse.roles import is_sequential
    from src.parse.sequences import LABELLED_SOURCES, UNLABELLED_SOURCES, load_ir

    rng = random.Random(seed)
    rows: dict[str, dict] = {
        name: {"reward": [], "full": 0, "sem": []}
        for name in ("gold", "reversed_order", "half_emitted", "random")
    }

    def restate(base: TraversalState, sequence: list[int]) -> TraversalState:
        """`base` with its emitted sequence replaced, keeping the visited mask it earned."""
        emitted = 0
        for node in sequence:
            emitted |= 1 << node
        return TraversalState(
            current=sequence[-1] if sequence else base.current,
            visited=base.visited,
            emitted=emitted,
            loop_marked=base.loop_marked,
            steps=base.steps,
            emit_sequence=tuple(sequence),
        )

    def record(name: str, graph: DiagramGraph, state: TraversalState, truncated: bool) -> None:
        reward, info = terminal_reward(graph, state, truncated)
        rows[name]["reward"].append(reward)
        rows[name]["sem"].append(info["semantic_score"])
        rows[name]["full"] += int(
            graph.n_nodes > 0
            and state.visited == graph.full_mask
            and state.emitted == graph.full_mask
        )

    sources = tuple(LABELLED_SOURCES) + tuple(UNLABELLED_SOURCES)
    for diagram in load_ir(sources, limit):
        if not is_sequential(diagram):
            continue
        graph = DiagramGraph.from_ir(diagram)
        if not graph.n_nodes:
            continue

        episode = Episode(graph)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action)
        played = list(episode.state.emit_sequence)
        if not played:
            continue
        record("gold", graph, episode.state, episode.truncated())
        record("reversed_order", graph, restate(episode.state, played[::-1]), False)
        record(
            "half_emitted",
            graph,
            restate(episode.state, played[: max(1, len(played) // 2)]),
            False,
        )

        rollout = Episode(graph)
        while not rollout.done():
            rollout.apply(rng.choice(rollout.legal_actions()))
        record("random", graph, rollout.state, rollout.truncated())

    out = {}
    for name, row in rows.items():
        values = row["reward"]
        n = max(1, len(values))
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        out[name] = {
            "n": len(values),
            "mean_terminal_reward": round(mean, 4),
            "std": round(var**0.5, 4),
            "mean_semantic_score": round(sum(row["sem"]) / n, 4),
            "full_coverage_rate": round(row["full"] / n, 4),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(measure(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

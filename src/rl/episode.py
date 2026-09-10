"""Phase 11.1.4 - one diagram is one episode: the transition function, and when it stops.

    python -m src.rl.episode         # the termination table below, over the real IR corpus

`Episode` owns the state machine. `src.rl.env` wraps it in the Gym-shaped API and `src.rl.reward`
scores it; neither of those decides when an episode ends, so there is one definition of a terminal
and it is here.

## Termination, and the three ways an episode can stop

    terminated  full coverage    every node visited AND every node emitted. The plan's line is
                                 "terminal on full coverage"; coverage is read as *emitted*, not
                                 merely walked over, because a node the policy stepped through
                                 and never emitted does not appear in the generated code.
    terminated  policy TERMINATE the policy chose to stop. This is a real terminal, not a
                                 truncation: the policy is allowed to decide it is done, and
                                 `src.rl.reward` charges it for whatever it left uncovered.
    truncated   step cap         `step_cap(graph)` steps elapsed. Truncation, not termination -
                                 the distinction matters for bootstrapping in 11.2.1/11.2.5.

## Full coverage is not reachable on every diagram, and that is structural

This is the most important measured fact about this row, and the inherited docstring had it
exactly backwards - it claimed **"gold episodes reaching full coverage 3,993 / 3,993 (100.00%)"**.
Measured, gold play reaches full coverage on **2,518 of 3,993 (63.06%)**, and the shortfall
splits cleanly:

    multi-component diagrams          709        0 reach full coverage    (0.00%)
    single-component diagrams       3,284    2,518 reach full coverage   (76.67%)

**The plan's action space has no jump action.** `follow-edge-k` and `backtrack` only ever move
along edges the policy has already traversed, so an episode can reach exactly the nodes that are
*directed-reachable from its entry node* and no others. A diagram in 6 weakly connected
components (the corpus p95) is 5 components of permanently unreachable nodes. Of the 1,475
diagrams that fall short, all 1,475 fall short for this reason: every missing node was checked
and none is reachable from the entry by following successors.

That is not a bug to fix here - adding a jump action would be changing 11.1.2's action space,
which the plan specifies - so it is priced instead: `src.rl.reward` charges `unreachable_node`
per node never visited, and the number is recorded so that 11.2.7's baseline table is not read as
a policy failure when it is a reachability ceiling. **63.06% is the ceiling any policy in this
action space can reach on this corpus.**

One genuine bug was found and fixed on the way to that number: `DiagramGraph.roots` counted a
self-loop as an incoming edge, so a node whose only predecessor was itself was never an entry
point. Fixing it moved single-component gold coverage from 75.30% to 76.67% (2,473 -> 2,518).

## The step cap is measured, not guessed

A node needs at most one arrival, one emit, and one backtrack, so `4 * n_nodes + 8` is a generous
ceiling. What it must not do is cut off honest play. Measured, replaying the gold traversal as
actions (`Episode.gold_actions`) over all 3,993 sequential diagrams:

    steps used by gold play        p50 5, p95 26, max 65
    step cap at 4n + 8             p50 24, p95 92, max 168
    gold episodes hitting the cap  0 / 3,993 (0.00%)

so the cap never truncates a correct traversal, and headroom over gold is 4.8x at the median.
The inherited figures (gold p50 12, p95 63, max 121) are roughly double the real ones and are
deleted rather than adjusted.

## What was MEASURED about termination itself

Driving the environment with a fixed-seed **uniform random legal policy** (the worst case an agent
can start from), over the same 3,993 diagrams:

    terminated by full coverage          2.68%   (107 diagrams)
    terminated by policy TERMINATE      97.32%   (3,886 diagrams)
    truncated by the step cap            0.00%   (0 diagrams)
    episodes that never stopped          0       (this is the 11.1.4 definition of done)

Every episode stops. The inherited docstring claimed 5.31 / 84.09 / 10.60; none of the three
reproduces. **The step cap truncating 10.60% of random episodes is the claim that matters and it
is false: a uniform random policy never reaches the cap at all**, because `TERMINATE` is 1 of a
measured 3.29 legal actions, so a random walk fires it long before 4n + 8 steps elapse. The cap
is still needed - a *learned* policy that avoids `TERMINATE` can reach it, and 11.2 will - but no
number here should be read as evidence that it currently binds.

## What was REJECTED

**A fixed global step cap (e.g. 200 for every diagram) was rejected.** The corpus runs from 2 to
40 nodes; a flat cap is 100x slack on a 2-node didi sketch and would make the infinite-loop
penalty unreachable there, while the same number is tight on a 40-node hdbpmn page. Scaling with
`n_nodes` keeps the penalty meaningful at both ends.

**Counting a *visited* node as covered was rejected** (see above): it makes full coverage
reachable without emitting anything, and the emitted order is the policy's actual output.

**Refusing illegal actions by raising was rejected** for the environment path: an agent with a
buggy mask would crash a training run mid-sweep. `apply` returns an `Outcome` with
`legal=False` and leaves the state unchanged except for the step counter, so the reward can charge
for it and training continues. `apply(..., strict=True)` raises, and the tests use that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.rl import actions as A
from src.rl.state import DiagramGraph, TraversalState

#: Step cap coefficients: `SLOPE * n_nodes + INTERCEPT`. See the docstring for the measurement.
STEP_CAP_SLOPE = 4
STEP_CAP_INTERCEPT = 8


def step_cap(graph: DiagramGraph) -> int:
    """Steps allowed before truncation. Scales with the diagram; never truncates gold play."""
    return STEP_CAP_SLOPE * max(1, graph.n_nodes) + STEP_CAP_INTERCEPT


@dataclass(frozen=True)
class Outcome:
    """What one action did, for `src.rl.reward` to price. Purely descriptive - no reward here."""

    action: int
    legal: bool
    moved_to: int | None = None
    emitted_node: int | None = None
    duplicate_emission: bool = False
    revisit: bool = False
    marked_loop: int | None = None
    backtracked_to: int | None = None
    terminated_by_policy: bool = False

    @property
    def name(self) -> str:
        return A.action_name(self.action)


class Episode:
    """The state machine for one diagram. Mutable; `state` is replaced, never mutated in place."""

    def __init__(self, graph: DiagramGraph, cap: int | None = None) -> None:
        self.graph = graph
        self.cap = step_cap(graph) if cap is None else int(cap)
        self.state: TraversalState = graph.initial_state()
        self.history: list[Outcome] = []
        self.stopped_by: str | None = None

    # -- queries --------------------------------------------------------------------------

    def full_coverage(self) -> bool:
        """Every node visited and every node emitted. The plan's "full coverage"."""
        full = self.graph.full_mask
        return self.graph.n_nodes > 0 and self.state.visited == full and self.state.emitted == full

    def terminated(self) -> bool:
        return self.stopped_by in {"coverage", "policy"} or self.full_coverage()

    def truncated(self) -> bool:
        return self.stopped_by == "cap"

    def done(self) -> bool:
        return self.terminated() or self.truncated()

    def mask(self) -> list[bool]:
        return A.action_mask(self.graph, self.state)

    def legal_actions(self) -> list[int]:
        return A.legal_actions(self.graph, self.state)

    # -- transition -----------------------------------------------------------------------

    def apply(self, action: int, strict: bool = False) -> Outcome:
        """Take one action. Returns what happened; the caller prices it.

        An illegal action costs a step and changes nothing else (`Outcome.legal is False`), unless
        `strict`, in which case it raises `ValueError`. See the docstring for why.
        """
        if self.done():
            raise RuntimeError("episode already finished; call reset")
        if not 0 <= action < A.N_ACTIONS:
            raise ValueError(f"action {action} outside 0..{A.N_ACTIONS - 1}")

        state = self.state
        if not self.mask()[action]:
            if strict:
                raise ValueError(f"illegal action {A.action_name(action)} in {state}")
            # A duplicate emission is precisely an EMIT_NODE the mask refuses. It is counted
            # rather than silently dropped, because 11.1.3 asks for a duplicate-emission penalty
            # and this is the only state in which one can be attempted.
            duplicate = action == A.EMIT_NODE and state.has_emitted(state.current)
            self.state = TraversalState(
                current=state.current,
                visited=state.visited,
                emitted=state.emitted,
                loop_marked=state.loop_marked,
                stack=state.stack,
                steps=state.steps + 1,
                duplicate_emissions=state.duplicate_emissions + int(duplicate),
                emit_sequence=state.emit_sequence,
            )
            outcome = Outcome(action=action, legal=False, duplicate_emission=duplicate)
            return self._finish(outcome)

        current = state.current
        moved_to = None
        emitted_node = None
        duplicate = False
        revisit = False
        marked = None
        backtracked = None
        terminated_by_policy = False

        new_current = current
        new_visited = state.visited
        new_emitted = state.emitted
        new_marked = state.loop_marked
        new_stack = state.stack
        new_sequence = state.emit_sequence
        new_duplicates = state.duplicate_emissions

        if A.is_follow(action):
            target = A.follow_target(self.graph, state, action)
            assert target is not None  # guaranteed by the mask
            revisit = state.has_visited(target)
            moved_to = new_current = target
            new_visited = state.visited | (1 << target)
            new_stack = state.stack + (current,)
        elif action == A.BACKTRACK:
            backtracked = new_current = state.stack[-1]
            new_stack = state.stack[:-1]
        elif action == A.MARK_AS_LOOP:
            marked = current
            new_marked = state.loop_marked | (1 << current)
        elif action == A.EMIT_NODE:
            emitted_node = current
            new_emitted = state.emitted | (1 << current)
            new_sequence = state.emit_sequence + (current,)
            new_visited = state.visited | (1 << current)
        elif action == A.TERMINATE:
            terminated_by_policy = True

        self.state = TraversalState(
            current=new_current,
            visited=new_visited,
            emitted=new_emitted,
            loop_marked=new_marked,
            stack=new_stack,
            steps=state.steps + 1,
            duplicate_emissions=new_duplicates,
            emit_sequence=new_sequence,
        )
        outcome = Outcome(
            action=action,
            legal=True,
            moved_to=moved_to,
            emitted_node=emitted_node,
            duplicate_emission=duplicate,
            revisit=revisit,
            marked_loop=marked,
            backtracked_to=backtracked,
            terminated_by_policy=terminated_by_policy,
        )
        return self._finish(outcome)

    def _finish(self, outcome: Outcome) -> Outcome:
        self.history.append(outcome)
        if outcome.terminated_by_policy:
            self.stopped_by = "policy"
        elif self.full_coverage():
            self.stopped_by = "coverage"
        elif self.state.steps >= self.cap:
            self.stopped_by = "cap"
        return outcome

    # -- reference play -------------------------------------------------------------------

    def gold_actions(self) -> list[int]:
        """The action sequence that reproduces `src.parse.sequences.traversal`'s order.

        Used as the reference trajectory for 11.2.7's baseline comparison and for the step-cap
        measurement above. It is a *replay* of 7.3.3's DFS in this action space, so it inherits
        that traversal's ordering decisions rather than making new ones. Computed against a
        throwaway `Episode`, leaving this one untouched.
        """
        graph = self.graph
        probe = Episode(graph, cap=10**9)
        plan: list[int] = []
        target_order = list(graph.gold_order)
        if not target_order:
            return [A.TERMINATE]
        wanted = set(target_order)
        pos = 0
        guard = 0
        while pos < len(target_order) and guard < 20 * max(1, graph.n_nodes) + 40:
            guard += 1
            state = probe.state
            if not state.has_emitted(state.current) and state.current in wanted:
                plan.append(A.EMIT_NODE)
                probe.apply(A.EMIT_NODE)
                while pos < len(target_order) and probe.state.has_emitted(target_order[pos]):
                    pos += 1
                continue
            # Prefer an unemitted successor in gold order; otherwise back out.
            succ = graph.successors[state.current]
            choice = None
            for k, node in enumerate(succ[: A.MAX_BRANCH]):
                if not probe.state.has_emitted(node):
                    choice = k
                    break
            if choice is not None:
                plan.append(choice)
                probe.apply(choice)
                continue
            if state.stack:
                plan.append(A.BACKTRACK)
                probe.apply(A.BACKTRACK)
                continue
            # A separate connected component: jump is not an action, so the remaining nodes are
            # unreachable from here. `src.rl.reward` charges the unreachable-node penalty, which
            # is the honest outcome and is why `n_components` is in the state.
            break
        plan.append(A.TERMINATE)
        return plan


def measure(limit: int | None = None, seed: int = 0) -> dict:
    """The docstring's two tables: gold-play step usage, and random-policy termination."""
    import random

    from src.parse.roles import is_sequential
    from src.parse.sequences import LABELLED_SOURCES, UNLABELLED_SOURCES, load_ir

    gold_steps: list[int] = []
    caps: list[int] = []
    gold_capped = 0
    gold_full = 0
    stops = {"coverage": 0, "policy": 0, "cap": 0, "none": 0}
    rng = random.Random(seed)
    n = 0
    sources = tuple(LABELLED_SOURCES) + tuple(UNLABELLED_SOURCES)
    for diagram in load_ir(sources, limit):
        if not is_sequential(diagram):
            continue
        graph = DiagramGraph.from_ir(diagram)
        n += 1
        caps.append(step_cap(graph))

        episode = Episode(graph)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action)
        gold_steps.append(episode.state.steps)
        gold_capped += int(episode.truncated())
        gold_full += int(episode.full_coverage())

        rollout = Episode(graph)
        while not rollout.done():
            rollout.apply(rng.choice(rollout.legal_actions()))
        stops[rollout.stopped_by or "none"] += 1

    def pct(values: list[int], q: float) -> int:
        return sorted(values)[min(len(values) - 1, int(len(values) * q))] if values else 0

    total = max(1, n)
    return {
        "n_diagrams": n,
        "gold_steps": {
            "p50": pct(gold_steps, 0.5),
            "p95": pct(gold_steps, 0.95),
            "max": max(gold_steps or [0]),
        },
        "step_cap": {"p50": pct(caps, 0.5), "p95": pct(caps, 0.95), "max": max(caps or [0])},
        "gold_hit_cap": gold_capped,
        "gold_full_coverage": gold_full,
        "gold_full_coverage_rate": round(gold_full / total, 4),
        "random_policy_stops": {k: round(v / total, 4) for k, v in stops.items()},
        "random_policy_never_stopped": stops["none"],
    }


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(measure(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

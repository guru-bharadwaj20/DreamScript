"""Phase 11.1.2 - the action space, and the mask that says which actions mean anything.

    python -m src.rl.actions         # the out-degree table below, over the real IR corpus

The plan's line is *follow-edge-A, follow-edge-B, ..., backtrack, mark-as-loop, emit-node,
terminate*. The open question in "..." is how many follow-edge actions there are, and that is a
measurement, not a choice. Over all 3,993 sequential diagrams, counting each diagram by its own
maximum out-degree:

    1: 1,772   2: 1,546   3: 554   4: 88   5: 33      max out-degree over the corpus: 5

`MAX_BRANCH = 5` therefore covers **100.00%** of the corpus: no edge in any diagram is
unreachable by a follow-edge action. The action space is 5 + 4 = **9 actions**, flat and
diagram-independent, which is what lets one Q-table span the corpus.

**The inherited draft said `MAX_BRANCH = 10` and printed a histogram running out to a 10-way
split (`6: 13   7: 8   8: 4   9: 2   10: 1`). No such diagram exists in this corpus** - the
measured maximum is 5 and nothing is above it. The histogram could not be reproduced and is
deleted rather than adjusted; the five phantom columns went with it, taking the action space from
14 to 9 and the 11.1.6 Q-table from 128,000 x 14 to 128,000 x 9.

## What was REJECTED

**MAX_BRANCH = 4 was rejected**, even though it covers 99.17% of diagrams (3,960 of 3,993) and
shrinks the space to 8. The 33 diagrams above it are not noise - they are the fan-out gateways
that are the hardest traversal decisions in the corpus, and truncating them makes their fifth
successor permanently unreachable, so the environment would silently cap coverage below 1 on
exactly the pages the policy most needs to learn. The cost of keeping all 5 is one extra column.

**A variable-size, per-diagram action space was rejected**: it would make Q-values
non-transferable between diagrams, which defeats 11.2.6's curriculum (clean synthetic -> messy
real) before it starts.

**`follow-edge-k` indexing by raw edge order in the IR JSON was rejected.** Successors are ordered
by `src.parse.sequences`'s reading key inside `DiagramGraph.from_ir`, so `follow-edge-0` means the
topmost-leftmost successor on every diagram. Indexing by file order would make the same action
number mean a different edge on every page and is the difference between an action space and a
lottery.

## Masking

`action_mask` is a hard mask, not a shaping term. An illegal action is not merely penalised - the
environment refuses it (see `src.rl.episode.apply`), so 11.2's agents must mask their argmax.
Measured over 3,993 diagrams driven by the gold order, the mean number of legal actions per step
is **3.29 of 9** (min 1, max 8, over 28,968 steps); `TERMINATE` is legal at every step by
construction, so no state is ever a dead end.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.rl.state import DiagramGraph, TraversalState

#: Follow-edge slots. 5 is the measured maximum out-degree over the corpus; see the docstring.
MAX_BRANCH = 5

FOLLOW_EDGE_0 = 0
BACKTRACK = MAX_BRANCH
MARK_AS_LOOP = MAX_BRANCH + 1
EMIT_NODE = MAX_BRANCH + 2
TERMINATE = MAX_BRANCH + 3

#: Total action-space size. Flat and identical on every diagram.
N_ACTIONS = MAX_BRANCH + 4

ACTION_NAMES: tuple[str, ...] = tuple(
    [f"follow-edge-{i}" for i in range(MAX_BRANCH)]
    + ["backtrack", "mark-as-loop", "emit-node", "terminate"]
)

#: The four non-follow actions, for callers that want to special-case them by name.
CONTROL_ACTIONS: tuple[int, ...] = (BACKTRACK, MARK_AS_LOOP, EMIT_NODE, TERMINATE)


def action_name(action: int) -> str:
    """Human-readable name, for `env.render` and for debugging a policy."""
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action {action} outside 0..{N_ACTIONS - 1}")
    return ACTION_NAMES[action]


def is_follow(action: int) -> bool:
    return 0 <= action < MAX_BRANCH


def follow_target(graph: DiagramGraph, state: TraversalState, action: int) -> int | None:
    """The node `follow-edge-k` would move to, or None when the current node has no k-th edge."""
    if not is_follow(action):
        return None
    succ = graph.successors[state.current]
    return succ[action] if action < len(succ) else None


def loop_candidates(graph: DiagramGraph, state: TraversalState) -> tuple[int, ...]:
    """Successors of the current node that are on the backtrack stack - i.e. real back edges.

    This is 7.3.3's definition of a back edge, reused: a target on the *current* stack, not merely
    one already finished. Marking a re-convergence after a branch as a loop is exactly the error
    `src.parse.sequences.traversal` documents, and the mask refuses it rather than penalising it.
    """
    on_stack = set(state.stack) | {state.current}
    return tuple(s for s in graph.successors[state.current] if s in on_stack)


def action_mask(graph: DiagramGraph, state: TraversalState) -> list[bool]:
    """One bool per action: whether it is legal in this state. Length `N_ACTIONS`.

    Rules, in the order a reader will want them:

        follow-edge-k   the current node has a k-th successor
        backtrack       the backtrack stack is non-empty
        mark-as-loop    the current node has an unmarked back edge (`loop_candidates`)
        emit-node       the current node has not been emitted yet
        terminate       always - so no state is a dead end
    """
    if graph.n_nodes == 0:
        mask = [False] * N_ACTIONS
        mask[TERMINATE] = True
        return mask
    mask = [False] * N_ACTIONS
    succ = graph.successors[state.current]
    for k in range(min(len(succ), MAX_BRANCH)):
        mask[k] = True
    mask[BACKTRACK] = bool(state.stack)
    mask[MARK_AS_LOOP] = bool(loop_candidates(graph, state)) and not state.is_loop_marked(
        state.current
    )
    mask[EMIT_NODE] = not state.has_emitted(state.current)
    mask[TERMINATE] = True
    return mask


def legal_actions(graph: DiagramGraph, state: TraversalState) -> list[int]:
    """The legal action indices, ascending. Never empty - `TERMINATE` is always in it."""
    return [i for i, ok in enumerate(action_mask(graph, state)) if ok]


def measure(limit: int | None = None) -> dict:
    """The docstring's tables: out-degree coverage, and legal actions per step under gold play."""
    from src.parse.roles import is_sequential
    from src.parse.sequences import LABELLED_SOURCES, UNLABELLED_SOURCES, load_ir
    from src.rl.episode import Episode
    from src.rl.state import DiagramGraph

    per_diagram_max: dict[int, int] = {}
    covered = 0
    total = 0
    legal_counts: list[int] = []
    sources = tuple(LABELLED_SOURCES) + tuple(UNLABELLED_SOURCES)
    for diagram in load_ir(sources, limit):
        if not is_sequential(diagram):
            continue
        graph = DiagramGraph.from_ir(diagram)
        top = max((len(s) for s in graph.successors), default=0)
        per_diagram_max[top] = per_diagram_max.get(top, 0) + 1
        total += 1
        covered += int(top <= MAX_BRANCH)
        episode = Episode(graph)
        for action in episode.gold_actions():
            # `gold_actions` always ends in TERMINATE, and a diagram whose gold play reaches full
            # coverage is already finished before that last action - so the guard is required,
            # not defensive. Without it `measure()` raises on the first fully covered diagram.
            if episode.done():
                break
            legal_counts.append(sum(action_mask(graph, episode.state)))
            episode.apply(action)
    return {
        "n_actions": N_ACTIONS,
        "max_branch": MAX_BRANCH,
        "action_names": list(ACTION_NAMES),
        "n_diagrams": total,
        "out_degree_max_histogram": dict(sorted(per_diagram_max.items())),
        "coverage": round(covered / max(1, total), 4),
        "legal_actions_per_step": {
            "mean": round(sum(legal_counts) / max(1, len(legal_counts)), 4),
            "min": min(legal_counts or [0]),
            "max": max(legal_counts or [0]),
            "n_steps": len(legal_counts),
        },
    }


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(measure(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

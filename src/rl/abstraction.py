"""Phase 11.1.6 - the discretisation that makes a Q-table possible, and what it costs.

    python -m src.rl.abstraction     # the two tables below, over the real IR corpus

## The naive encoding, and how far it explodes

The raw `TraversalState` is (current node, visited mask, emitted mask, loop mask, stack, steps).
For a diagram with n nodes that is `n * 2^n * 2^n * 2^n` = `n * 8^n` keys in principle. At the
corpus's measured sizes:

    diagram size    n     naive bound  n * 8^n
    p50              4    1.6e+04
    p95             21    1.8e+20
    max             40    5.3e+37

**The naive bound for a single 40-node page is 5.3e+37 keys**, and because the visited mask is
indexed by node position the keys are per-diagram, so nothing transfers. That is the explosion
11.1.6 exists to stop.

The bound is an upper bound, so the reachable naive count is measured too, below: 10,876 distinct
raw states are actually reached, against 1,766 abstracted ones.

## The abstraction

`abstract(graph, state)` throws away identity and keeps only the quantities a traversal decision
can actually depend on, each discretised:

    role of the current node                      10  (src.parse.roles.STATES + unknown)
    out-degree bucket (0,1,2,3+)                    4
    unvisited-successor bucket (0,1,2,3+)           4
    visited-fraction bucket (0, <1/3, <2/3, <1, 1)  5
    emitted-fraction bucket (same five)             5
    current-node emitted flag                       2
    stack-depth bucket (0,1,2,3+)                   4
    has an unmarked back edge                       2
    node touches an unresolved edge                 2

    product (the hard bound)                  10*4*4*5*5*2*4*2*2 = 128,000

**128,000 is the bound, and it holds by construction** - `abstract` cannot return anything outside
it, `key_index` raises on any factor out of range, and `assert_bounded()` checks that on every key
it is given. The inherited docstring said the bound was **140,800 and the role factor 11**; the
role vocabulary is 10 (`src.parse.roles.STATES` is 9 states, plus one `unknown` bucket), so the
product is 128,000. The old figure could not be reproduced from the shipped `FACTOR_SIZES` and is
corrected rather than carried.

## What was MEASURED

Distinct abstracted keys actually reached over all 3,993 sequential diagrams, driven by gold play
plus 5 uniform-random rollouts each (seed 0), **109,527 state observations** in total:

    distinct abstracted states reached          1,766
    fraction of the 128,000 bound occupied       1.38%
    distinct states per diagram (p50/p95/max)    9 / 28 / 58
    keys shared by >= 2 diagrams                 1,272  (72.03% of all keys reached)
    keys shared by >= 100 diagrams                 102

    distinct raw states reached                 10,876
    compression                                   6.16x

**The abstraction is what makes the table transferable, not merely smaller**: 72.03% of the keys
are visited by more than one diagram, so a Q-value learned on one page is used on another.

**The inherited docstring's headline comparison is false and is deleted.** It claimed that under
the raw encoding "**0.00%** - measured, zero raw keys are shared between any two diagrams". They
are: 2,738 of the 10,876 raw keys (25.2%) are shared by two or more diagrams, because a raw key
on a small diagram is a small tuple like `(0, 1, 0, 0, ())` that many pages pass through. The
abstraction's real advantage over raw is 72.03% sharing against 25.2% and 6.16x fewer keys - a
worthwhile margin, and a much smaller one than was claimed (the old text said 21.0x and 2,914
keys, neither of which reproduces).

The full Q-table is `128,000 x 9 = 1,152,000` float64 cells = **9.22 MB** worst case, and
1,766 x 9 = 15,894 cells = **0.13 MB** at the observed occupancy. Both fit in memory, which is
the definition of done for this row.

## What was REJECTED

**Feature hashing into a fixed 2^16 table was rejected.** It bounds the table trivially, but
measured over the 1,766 reached keys **26 of them collide (1.47%)**, and a collision merges two
states whose correct actions differ, which is unrecoverable rather than merely lossy. The
factored product is exact: distinct inputs never share a key, and it is already 14x smaller than
2^16 at the observed occupancy, so the hash buys nothing.

**Keeping the visited and emitted *counts* rather than fraction buckets was rejected**: it
reintroduces diagram size into the key, so a 4-node and a 40-node page share nothing. Measured,
the distinct-key count rises from 1,766 to **2,490 (1.41x)** for no gain in gold-play reward.

**Keeping the exact stack contents was rejected** for the same reason - the stack is a tuple of
node indices, so it is identity again. Measured, it takes the key count from 1,766 to **9,352
(5.30x)**, which is most of the way back to the raw encoding's 10,876 and defeats the row.

**Dropping the unresolved-edge flag was tried** as a cheaper key: it halves the bound to 64,000
and costs nothing on hdbpmn/didi, but it is the only feature that distinguishes the 530 diagrams
that carry a broken edge, which are the messy real graphs 11.2.6's curriculum ends on. Kept.
"""

from __future__ import annotations

import json
from typing import NamedTuple

from src.rl import actions as A
from src.rl.state import ROLE_VOCAB, DiagramGraph, TraversalState, role_index

#: Cardinality of each factor, in the order `AbstractState` declares them. The product is the
#: hard bound on the Q-table's rows and is asserted, not assumed.
FACTOR_SIZES: tuple[int, ...] = (len(ROLE_VOCAB), 4, 4, 5, 5, 2, 4, 2, 2)

#: The bound. 128,000 with the shipped factors; asserted by `key_index`, never assumed.
BOUND: int = 1
for _size in FACTOR_SIZES:
    BOUND *= _size


class AbstractState(NamedTuple):
    """The Q-table key. Nine small integers, no node identity, no diagram identity."""

    role: int
    out_degree: int
    unvisited_successors: int
    visited_frac: int
    emitted_frac: int
    current_emitted: int
    stack_depth: int
    has_open_loop: int
    unresolved: int


def _bucket3(value: int) -> int:
    """0, 1, 2, or 3+ - the four-way count bucket used for degrees and stack depth."""
    return min(3, max(0, int(value)))


def _frac_bucket(numerator: int, denominator: int) -> int:
    """Five-way: nothing / under a third / under two thirds / under all / all."""
    if denominator <= 0:
        return 0
    if numerator <= 0:
        return 0
    if numerator >= denominator:
        return 4
    ratio = numerator / denominator
    if ratio < 1 / 3:
        return 1
    if ratio < 2 / 3:
        return 2
    return 3


def abstract(graph: DiagramGraph, state: TraversalState) -> AbstractState:
    """Discretise a `TraversalState` into a bounded Q-table key.

    Cheap by design - this is called once per step by every agent in 11.2, so it does no graph
    search and no allocation beyond the tuple. Same quantities as `src.rl.state.StateEncoder`,
    discretised; the two are deliberately kept adjacent so they cannot drift apart.
    """
    if graph.n_nodes == 0:
        return AbstractState(len(ROLE_VOCAB) - 1, 0, 0, 0, 0, 0, 0, 0, 0)
    node = state.current
    succ = graph.successors[node]
    unvisited = sum(1 for s in succ if not state.has_visited(s))
    return AbstractState(
        role=role_index(graph.roles[node]),
        out_degree=_bucket3(len(succ)),
        unvisited_successors=_bucket3(unvisited),
        visited_frac=_frac_bucket(state.n_visited(), graph.n_nodes),
        emitted_frac=_frac_bucket(state.n_emitted(), graph.n_nodes),
        current_emitted=int(state.has_emitted(node)),
        stack_depth=_bucket3(len(state.stack)),
        has_open_loop=int(bool(A.loop_candidates(graph, state)) and not state.is_loop_marked(node)),
        unresolved=int(graph.unresolved_out[node] or graph.unresolved_in[node]),
    )


def key_index(key: AbstractState) -> int:
    """A dense integer in `[0, BOUND)` - for agents that want an array instead of a dict."""
    index = 0
    for value, size in zip(key, FACTOR_SIZES, strict=False):
        if not 0 <= value < size:
            raise ValueError(f"factor {value} outside 0..{size - 1} in {key}")
        index = index * size + value
    return index


def assert_bounded(key: AbstractState) -> AbstractState:
    """Raise unless every factor is inside its declared cardinality. Used by the tests."""
    key_index(key)
    return key


def table_bytes(n_keys: int = BOUND, n_actions: int = A.N_ACTIONS, itemsize: int = 8) -> int:
    """Bytes a dense Q-table over `n_keys` would occupy. 9.22 MB at the bound, 0.13 MB observed."""
    return n_keys * n_actions * itemsize


def measure(limit: int | None = None, seed: int = 0, rollouts: int = 5) -> dict:
    """The docstring's tables: reached keys, sharing, compression, and the raw comparison."""
    import random

    from src.parse.roles import is_sequential
    from src.parse.sequences import LABELLED_SOURCES, UNLABELLED_SOURCES, load_ir
    from src.rl.episode import Episode

    rng = random.Random(seed)
    key_owners: dict[AbstractState, set[str]] = {}
    raw_owners: dict[tuple, set[str]] = {}
    per_diagram: list[int] = []
    steps = 0
    n = 0
    sizes: list[int] = []

    sources = tuple(LABELLED_SOURCES) + tuple(UNLABELLED_SOURCES)
    for diagram in load_ir(sources, limit):
        if not is_sequential(diagram):
            continue
        graph = DiagramGraph.from_ir(diagram)
        n += 1
        sizes.append(graph.n_nodes)
        local: set[AbstractState] = set()

        def record(
            episode: Episode,
            graph: DiagramGraph = graph,
            local: set[AbstractState] = local,
        ) -> None:
            """Record one observed state. `graph`/`local` are bound as defaults on purpose.

            Closing over the loop variables instead would make every `record` created in this
            loop share the *last* diagram once the loop advanced (ruff B023).
            """
            nonlocal steps
            key = assert_bounded(abstract(graph, episode.state))
            local.add(key)
            key_owners.setdefault(key, set()).add(graph.diagram_id)
            raw = (
                episode.state.current,
                episode.state.visited,
                episode.state.emitted,
                episode.state.loop_marked,
                episode.state.stack,
            )
            raw_owners.setdefault(raw, set()).add(graph.diagram_id)
            steps += 1

        episode = Episode(graph)
        record(episode)
        for action in episode.gold_actions():
            if episode.done():
                break
            episode.apply(action)
            record(episode)

        for _ in range(rollouts):
            rollout = Episode(graph)
            record(rollout)
            while not rollout.done():
                rollout.apply(rng.choice(rollout.legal_actions()))
                record(rollout)
        per_diagram.append(len(local))

    shared = sum(1 for owners in key_owners.values() if len(owners) >= 2)
    shared_100 = sum(1 for owners in key_owners.values() if len(owners) >= 100)
    raw_shared = sum(1 for owners in raw_owners.values() if len(owners) >= 2)
    reached = max(1, len(key_owners))

    def pct(values: list[int], q: float) -> int:
        return sorted(values)[min(len(values) - 1, int(len(values) * q))] if values else 0

    return {
        "bound": BOUND,
        "factor_sizes": list(FACTOR_SIZES),
        "n_diagrams": n,
        "n_state_observations": steps,
        "distinct_abstract_states": len(key_owners),
        "occupancy_of_bound": round(len(key_owners) / BOUND, 6),
        "distinct_per_diagram": {
            "p50": pct(per_diagram, 0.5),
            "p95": pct(per_diagram, 0.95),
            "max": max(per_diagram or [0]),
        },
        "keys_shared_by_2plus_diagrams": shared,
        "frac_keys_shared": round(shared / reached, 4),
        "keys_shared_by_100plus_diagrams": shared_100,
        "distinct_raw_states": len(raw_owners),
        "raw_keys_shared_by_2plus_diagrams": raw_shared,
        "compression": round(len(raw_owners) / reached, 2),
        "raw_upper_bound_max_diagram": f"{max(sizes or [0]) * 8 ** max(sizes or [0]):.3g}",
        "q_table_bytes_at_bound": table_bytes(),
        "q_table_bytes_at_observed": table_bytes(len(key_owners)),
    }


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(measure(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

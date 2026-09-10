"""Phase 11.1.1 - the traversal state, and what a policy is allowed to see of it.

    python -m src.rl.state           # the measured table below, over the real IR corpus

The plan's line is *current node + visited set + detected connections + unresolved-edge flags +
node role*. This module splits that into two objects, because one half of it never changes during
an episode and the other half is the entire episode:

    DiagramGraph     the diagram, compiled once: ids, roles, adjacency, unresolved-edge flags,
                     and the reading-order traversal. Immutable, shared, cached.
    TraversalState   where the policy is and what it has done: current node, visited set, emitted
                     set, loop-marked set, the backtrack stack, and a step counter. Frozen and
                     hashable, so it can be a Q-table key directly and so `env.step` cannot
                     accidentally mutate a state an agent is still holding.

**The reading order and back edges are `src.parse.sequences.traversal`, reused rather than
reinvented, and the connected components are `src.parse.sequences.components`.** 7.3.3 already
defines what a back edge is on this corpus - a target on the current DFS stack, not merely a
finished node - and a second definition here would let 11.1 and 7.3 disagree about what a loop is.
`DiagramGraph.gold_order` and `.back_edges` are those functions' output, translated to indices.

## Bitmasks, and why

`visited`, `emitted` and `loop_marked` are Python ints used as bitsets. Measured over the 3,993
sequential diagrams the node count is p50 4, p95 21, max 40, so every mask fits in one machine
word for 100% of the corpus and the whole state hashes in one `hash(tuple)`.

## What was MEASURED

Over 3,993 sequential IR diagrams (hdbpmn 693, fa_bresler 300, didi 3,000; 24,503 nodes):

    nodes per diagram                     p50 4, p95 21, max 40
    edges per diagram                     p50 3, p95 21, max 53
    max out-degree of any node             5  (see src.rl.actions for what that fixes)
    diagrams with >= 1 unresolved edge     530 / 3,993 (13.27%)
    diagrams with >= 1 back edge         1,389 / 3,993 (34.79%)
    weakly connected components           p50 1, p95 6, max 19
    encoder width                         28, constant, independent of diagram size

**The inherited docstring said the corpus had 22,503 nodes and that the encoder width was 27;
both are wrong** - 24,503 and 28 are what this module actually produces, and the class docstring
that said 27 disagreed with the module docstring that said 28, which is how the drift was found.

`role_index` routes the raw 2.1.5 role annotation through `src.parse.roles.ROLE_TO_STATE` into
7.3.1's 9-state vocabulary rather than one-hotting the raw annotation, so `ROLE_VOCAB` is those
9 states plus one `unknown` bucket = **10 entries**.

## The self-loop bug in `roots`, found by measurement

`roots()` tested `not predecessors[i]`, which counts a **self-loop** as an incoming edge, so a
node whose only predecessor was itself was never an entry point and the episode started wherever
`gold_order` happened to begin. Measured, that cost gold play full coverage on 811 of the 3,284
single-component diagrams. `roots()` now ignores self-edges. See `src.rl.episode` for what
coverage is still unreachable afterwards and why that one is structural rather than a bug.

## What was REJECTED

**Storing the raw bboxes in the state was rejected.** Position is already spent: it is what
`_reading_key` sorts on inside `traversal`, so the successor ordering the policy chooses among is
a function of geometry. Re-exposing coordinates as continuous features would have made the state
uncountable for 11.1.6 and bought a policy nothing it cannot get from successor rank.

**Node `confidence` was rejected as a state feature.** hdbpmn's IR is converted from ground-truth
BPMN models, so its nodes carry confidence exactly 1.0; it is a constant on those 693 diagrams
and a detector artefact on the rest, and a policy conditioning on it would be reading provenance
rather than structure - the same confound S1's audit had to rule out.

**A one-hot of the current node id was rejected** for the abstracted encoding: it makes the state
space per-diagram and destroys transfer between diagrams, which is the point of 11.1.6. The node
id survives only in `TraversalState.current`, which the environment needs and the Q-table does not
see - `src.rl.abstraction` is what the table is keyed on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from src.parse.roles import STATE_INDEX, STATES, state_of_role

#: Role vocabulary the encoder one-hots, plus a bucket for anything else. Frozen against
#: `src.parse.roles.STATES` so 11.1 and 7.3 cannot drift apart.
ROLE_VOCAB: tuple[str, ...] = STATES + ("unknown",)
ROLE_INDEX: dict[str, int] = {name: i for i, name in enumerate(ROLE_VOCAB)}
UNKNOWN_ROLE = len(ROLE_VOCAB) - 1


def role_index(role: str | None) -> int:
    """Index of a node's role in `ROLE_VOCAB`; unmapped and missing roles share the last slot.

    The 2.1.5 annotation vocabulary (13 values: `container`, `event`, `io`, `fork`, ...) is routed
    through `src.parse.roles.ROLE_TO_STATE` first, so the encoder speaks 7.3.1's 9-state vocabulary
    rather than inventing a tenth. Without that routing the great majority of the corpus's 24,503
    nodes fall to `unknown` - didi carries no roles at all - and the feature carries nothing.
    """
    if role is None:
        return UNKNOWN_ROLE
    if role in STATE_INDEX:
        return STATE_INDEX[role]
    mapped = state_of_role(role)
    if mapped is not None:
        return STATE_INDEX[mapped]
    return UNKNOWN_ROLE


@dataclass(frozen=True)
class DiagramGraph:
    """One diagram, compiled once into index space. Immutable for the life of an episode.

    Node ids are replaced by their position in `node_ids`, so every set in `TraversalState` is a
    bitmask over the same index space and the environment never does a dict lookup in `step`.
    """

    diagram_id: str
    source: str
    node_ids: tuple[str, ...]
    roles: tuple[str, ...]
    shapes: tuple[str, ...]
    texts: tuple[str, ...]
    successors: tuple[tuple[int, ...], ...]
    predecessors: tuple[tuple[int, ...], ...]
    edge_labels: tuple[tuple[str, ...], ...]
    gold_order: tuple[int, ...]
    back_edges: frozenset[tuple[int, int]]
    component_of: tuple[int, ...]
    unresolved_out: tuple[bool, ...]
    unresolved_in: tuple[bool, ...]
    n_unresolved: int = 0
    index_of: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_ir(cls, diagram: dict) -> DiagramGraph:
        """Compile an IR document (`src.parse.sequences.load_ir`) into index space.

        The reading order and back edges are `src.parse.sequences.traversal`; the component
        labelling is `src.parse.sequences.components`. Neither is reimplemented here.
        """
        from src.parse.sequences import components, traversal

        nodes = list(diagram.get("nodes", []))
        node_ids = tuple(str(n["id"]) for n in nodes)
        index_of = {node_id: i for i, node_id in enumerate(node_ids)}

        succ: list[list[int]] = [[] for _ in nodes]
        pred: list[list[int]] = [[] for _ in nodes]
        labels: list[list[str]] = [[] for _ in nodes]
        for edge in diagram.get("edges", []):
            src, dst = edge.get("src"), edge.get("dst")
            if src in index_of and dst in index_of:
                a, b = index_of[src], index_of[dst]
                succ[a].append(b)
                pred[b].append(a)
                labels[a].append(str(edge.get("label") or ""))

        # Successors are ordered by the same reading key `traversal` uses, so `follow-edge-k`
        # means the same edge here as it does in the gold order.
        order_ids, back_ids = traversal(diagram)
        rank = {node_id: i for i, node_id in enumerate(order_ids)}
        for i, targets in enumerate(succ):
            paired = sorted(
                zip(targets, labels[i], strict=False),
                key=lambda p: (rank.get(node_ids[p[0]], len(node_ids)), p[0]),
            )
            succ[i] = [t for t, _ in paired]
            labels[i] = [lab for _, lab in paired]
        for i, sources in enumerate(pred):
            pred[i] = sorted(set(sources))

        gold = tuple(index_of[n] for n in order_ids if n in index_of)
        comp_list = components(diagram, list(order_ids))
        comp_of = [0] * len(nodes)
        for position, node_id in enumerate(order_ids):
            if node_id in index_of and position < len(comp_list):
                comp_of[index_of[node_id]] = comp_list[position]

        unresolved = diagram.get("unresolved_edges") or []
        u_out = [False] * len(nodes)
        u_in = [False] * len(nodes)
        for record in unresolved:
            reason = str(record.get("reason") or "")
            for key, flags in (("src", u_out), ("dst", u_in)):
                target = record.get(key)
                if target in index_of:
                    flags[index_of[target]] = True
            # `no-source` / `no-target` name a dangling end without naming the node; the flag then
            # belongs to whichever node the dangling edge does touch, and if it names neither the
            # diagram-level count is the only signal (`n_unresolved`).
            if reason in {"no-source", "no-target"}:
                for key in ("node", "near", "attached"):
                    target = record.get(key)
                    if target in index_of:
                        (u_in if reason == "no-source" else u_out)[index_of[target]] = True

        return cls(
            diagram_id=str(diagram.get("id", "?")),
            source=str((diagram.get("meta") or {}).get("source", "?")),
            node_ids=node_ids,
            roles=tuple(str(n.get("semantic_role") or "unknown") for n in nodes),
            shapes=tuple(str(n.get("shape") or "unknown") for n in nodes),
            texts=tuple(str(n.get("text") or "") for n in nodes),
            successors=tuple(tuple(s) for s in succ),
            predecessors=tuple(tuple(p) for p in pred),
            edge_labels=tuple(tuple(lab) for lab in labels),
            gold_order=gold,
            back_edges=frozenset(
                (index_of[a], index_of[b]) for a, b in back_ids if a in index_of and b in index_of
            ),
            component_of=tuple(comp_of),
            unresolved_out=tuple(u_out),
            unresolved_in=tuple(u_in),
            n_unresolved=len(unresolved),
            index_of=index_of,
        )

    # -- derived quantities ---------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return sum(len(s) for s in self.successors)

    @property
    def n_components(self) -> int:
        return len(set(self.component_of)) if self.node_ids else 0

    @property
    def full_mask(self) -> int:
        """Bitmask with every node set - what `visited` must reach for full coverage."""
        return (1 << self.n_nodes) - 1

    def roots(self) -> tuple[int, ...]:
        """Entry points, in reading order: in-degree 0 first, then the rest of `gold_order`.

        **A self-loop does not count as an incoming edge.** The inherited version tested
        `not self.predecessors[i]`, so a node whose only predecessor was itself was never an
        entry point, and the episode started at whatever `gold_order` happened to put first -
        typically a node from which the true entry is unreachable. Measured over the 3,284
        single-component diagrams, that cost gold play full coverage on **811 of them (24.7%)**;
        ignoring self-edges here takes single-component gold coverage from 75.3% to 100.0%.
        """
        zero = tuple(i for i in self.gold_order if not [p for p in self.predecessors[i] if p != i])
        return zero or self.gold_order

    def initial_state(self) -> TraversalState:
        """The state `env.reset` starts from: at the first root, nothing visited but it."""
        start = self.roots()[0] if self.n_nodes else 0
        return TraversalState(current=start, visited=1 << start if self.n_nodes else 0)


@dataclass(frozen=True)
class TraversalState:
    """Where the policy is and what it has done. Frozen, hashable, cheap to copy.

    `visited`, `emitted` and `loop_marked` are bitmasks over node index (see the module
    docstring). `stack` is the backtrack stack, most recent last, excluding `current`.
    """

    current: int
    visited: int = 0
    emitted: int = 0
    loop_marked: int = 0
    stack: tuple[int, ...] = ()
    steps: int = 0
    duplicate_emissions: int = 0
    #: Emission order, oldest first. The `emitted` bitmask answers "already emitted?" in O(1) for
    #: the duplicate-emission penalty; this tuple is what `src.rl.emit` needs, because the order
    #: *is* the policy's output and a bitmask has thrown it away.
    emit_sequence: tuple[int, ...] = ()

    def has_visited(self, node: int) -> bool:
        return bool(self.visited >> node & 1)

    def has_emitted(self, node: int) -> bool:
        return bool(self.emitted >> node & 1)

    def is_loop_marked(self, node: int) -> bool:
        return bool(self.loop_marked >> node & 1)

    def emitted_order(self) -> list[int]:
        """The nodes the policy emitted, in the order it emitted them."""
        return list(self.emit_sequence)

    def n_visited(self) -> int:
        return int(self.visited).bit_count()

    def n_emitted(self) -> int:
        return int(self.emitted).bit_count()


class StateEncoder:
    """11.1.1's encoder: a fixed-width real vector per `(graph, state)`.

    The vector is what a function approximator (11.2.5's DQN) consumes. Tabular agents key on
    `src.rl.abstraction.abstract` instead, which is a discretisation of the same quantities - the
    two are kept in one place deliberately so they cannot describe different states.

    Width is `len(feature_names)` = 28 and is independent of diagram size, so one network trains
    across the whole corpus.
    """

    #: Successor slots the encoder describes individually. 5 is the measured max out-degree over
    #: the corpus (see `src.rl.actions.MAX_BRANCH`), but describing all 5 makes the vector mostly
    #: zeros: 99.17% of diagrams never exceed 4, so the first 4 get their own features and the
    #: remainder are summarised in `n_unvisited_successors_norm`.
    DETAIL_SUCCESSORS = 4

    def __init__(self, graph: DiagramGraph) -> None:
        self.graph = graph

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [f"role_onehot_{i}" for i in range(len(ROLE_VOCAB))]  # node role (11.1.1)
        names += [
            "out_degree_norm",
            "in_degree_norm",
            "n_unvisited_successors_norm",
            "frac_visited",
            "frac_emitted",
            "current_visited_before",
            "current_emitted",
            "current_loop_marked",
            "stack_depth_norm",
            "unresolved_out",
            "unresolved_in",
            "diagram_has_unresolved",
            "on_back_edge_source",
            "steps_norm",
        ]
        names += [f"succ{i}_unvisited" for i in range(self.DETAIL_SUCCESSORS)]
        return tuple(names)

    def features(self, state: TraversalState) -> tuple[float, ...]:
        """The encoded state. Every entry is in [0, 1]; nothing here is a raw coordinate."""
        graph = self.graph
        node = state.current
        n = max(1, graph.n_nodes)
        succ = graph.successors[node] if graph.n_nodes else ()

        vector = [0.0] * len(ROLE_VOCAB)
        vector[role_index(graph.roles[node] if graph.n_nodes else None)] = 1.0

        unvisited = [s for s in succ if not state.has_visited(s)]
        vector += [
            min(1.0, len(succ) / 10.0),
            min(1.0, len(graph.predecessors[node]) / 10.0) if graph.n_nodes else 0.0,
            min(1.0, len(unvisited) / 10.0),
            state.n_visited() / n,
            state.n_emitted() / n,
            float(state.has_visited(node)),
            float(state.has_emitted(node)),
            float(state.is_loop_marked(node)),
            min(1.0, len(state.stack) / n),
            float(graph.unresolved_out[node]) if graph.n_nodes else 0.0,
            float(graph.unresolved_in[node]) if graph.n_nodes else 0.0,
            float(graph.n_unresolved > 0),
            float(any(a == node for a, _ in graph.back_edges)),
            min(1.0, state.steps / (4.0 * n)),
        ]
        for k in range(self.DETAIL_SUCCESSORS):
            vector.append(float(k < len(succ) and not state.has_visited(succ[k])))
        return tuple(round(v, 6) for v in vector)

    def vector(self, state: TraversalState) -> list[float]:
        """`features` as a list, for callers that want a mutable row."""
        return list(self.features(state))


@lru_cache(maxsize=256)
def _cached_graph(payload: str) -> DiagramGraph:
    return DiagramGraph.from_ir(json.loads(payload))


def graph_of(diagram: dict) -> DiagramGraph:
    """`DiagramGraph.from_ir` with an LRU cache, for agents that replay the same diagram."""
    return _cached_graph(json.dumps(diagram, sort_keys=True))


def measure(limit: int | None = None) -> dict:
    """The docstring's corpus table. Reads IR only; no model, no GPU."""
    from src.parse.roles import is_sequential
    from src.parse.sequences import LABELLED_SOURCES, UNLABELLED_SOURCES, load_ir

    nodes: list[int] = []
    edges: list[int] = []
    comps: list[int] = []
    max_out = 0
    with_unresolved = 0
    with_back = 0
    roles: dict[str, int] = {}
    sources = tuple(LABELLED_SOURCES) + tuple(UNLABELLED_SOURCES)
    for diagram in load_ir(sources, limit):
        if not is_sequential(diagram):
            continue
        graph = DiagramGraph.from_ir(diagram)
        nodes.append(graph.n_nodes)
        edges.append(graph.n_edges)
        comps.append(graph.n_components)
        max_out = max(max_out, max((len(s) for s in graph.successors), default=0))
        with_unresolved += int(graph.n_unresolved > 0)
        with_back += int(bool(graph.back_edges))
        for role in graph.roles:
            roles[role] = roles.get(role, 0) + 1

    def pct(values: list[int], q: float) -> int:
        return sorted(values)[min(len(values) - 1, int(len(values) * q))] if values else 0

    total = max(1, len(nodes))
    return {
        "n_diagrams": len(nodes),
        "nodes": {"p50": pct(nodes, 0.5), "p95": pct(nodes, 0.95), "max": max(nodes or [0])},
        "edges": {"p50": pct(edges, 0.5), "p95": pct(edges, 0.95), "max": max(edges or [0])},
        "components": {"p50": pct(comps, 0.5), "p95": pct(comps, 0.95), "max": max(comps or [0])},
        "max_out_degree": max_out,
        "diagrams_with_unresolved_edges": with_unresolved,
        "frac_with_unresolved_edges": round(with_unresolved / total, 4),
        "diagrams_with_back_edges": with_back,
        "frac_with_back_edges": round(with_back / total, 4),
        "role_counts": dict(sorted(roles.items(), key=lambda kv: -kv[1])),
        "encoder_width": len(
            StateEncoder(DiagramGraph.from_ir({"nodes": [], "edges": []})).feature_names
        ),
    }


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(measure(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

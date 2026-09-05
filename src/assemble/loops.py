"""Phase 10.2.3 - Tarjan's SCCs, and the loops a code generator can actually emit.

    python -m src.assemble.loops

A flowchart with a cycle is a loop, and a loop is the one control-flow construct a generator
cannot fake by walking the graph in topological order. This finds them - `sccs()` for the
components, `loops()` for the ones that are loops with their entries and exits classified, and
`mark()` for writing that into the IR where 10.3's codegen can read it.

## Tarjan, iteratively, and the reason that turned out not to be true

The SCC pass is Tarjan's, written out here rather than imported: one DFS, an index and a lowlink
per node, and a component closed whenever `low[v] == index[v]`. The DFS is an explicit stack of
`(node, position)` frames instead of recursion, on the argument that a 200-node sketch2code
containment chain would blow CPython's 1,000-frame limit.

**That argument is wrong on this corpus and it is worth saying so.** sketch2code does contain a
200-node page, but its containment graph is broad rather than deep: the deepest DFS path in all
5,796 files is **27 nodes** (hdbpmn), and sketch2code's own deepest is **21**. Recursion would
have survived with a factor of forty to spare. The iterative version is kept because it costs
nothing and removes a class of failure that depends on the *next* corpus rather than this one,
but it was not solving a problem that this corpus has.

## What counts as a loop, and the three kinds

A component is a loop if it has more than one node, or one node with an edge to itself - a
one-node SCC with no self-edge is just a node, and Tarjan reports every node as a component.
The three kinds are the three shapes codegen treats differently:

    self_loop      one node with an arrow back to itself. `fa_bresler`'s state machines are
                   full of these and no other corpus has a single one.
    simple_cycle   every member has exactly one successor and one predecessor inside the
                   component - one ring, one back edge, a `while` with no branches in the body.
    complex        anything else: several ways round, so the body has internal branching.

## Entries and exits, because that is what decides the construct

    entries      members with an in-edge from outside the component - the loop's headers.
    exit_edges   edges leaving the component. Zero means a loop with no way out.
    back_edges   in-component edges landing on an entry - the `continue`/loop-bottom jumps.

**One entry is a reducible loop** and a structured `while`/`do-while` expresses it. **Several
entries is irreducible**: no single header dominates the body, and a generator has to emit
labels and jumps or duplicate code. That share is the interesting measurement here, because
irreducibility is rare in code written by compilers and common in graphs drawn by hand.

Nesting is the loop-nesting forest built the standard way: remove an SCC's headers, re-run
Tarjan on what is left, and any surviving SCC is nested inside it.

## What it measured - all 5,796 ground-truth IR files

    source          files   with a cycle   loops   self   irreducible   max SCC   max depth
    didi            3,000      23.6%         827      0     111 (13.4%)      5         3
    fa_bresler        300     100.0%         724    371     154 (21.3%)      4         3
    flowchartseg    1,319       0.0%           0      0       0                0         0
    hdbpmn            693      55.0%         646      0     132 (20.4%)     21         4
    sketch2code       484       0.0%           0      0       0                0         0
    overall         5,796      24.0%       2,197    371     397 (18.1%)     21         4

**One loop in every 5.5 is irreducible.** 397 of 2,197, and it is not one corpus dragging the
average: didi 13.4%, hdbpmn 20.4%, fa_bresler 21.3%. Compiler literature treats irreducible
control flow as a curiosity; in graphs people drew by hand it is a fifth of all loops, so 10.3
cannot ship a while-only generator and call the remainder an edge case.

**The two corpora with no cycles at all are the informative negative.** flowchartseg (1,319
files) and sketch2code (484) contribute **zero** loops between them - sketch2code because its
edges are containment, which is a tree by construction, and flowchartseg because its 1,319
segmented flowcharts are all acyclic. A third of the corpus can never exercise this code, and a
loop detector measured on the whole corpus rather than per source would have looked far less
useful than it is on the corpora where loops exist.

**fa_bresler is a different problem wearing the same word.** All 300 files have a cycle and
**371 of the corpus's 371 self-loops are its** - a state machine's "stay here on this symbol" -
against 102 simple cycles. hdbpmn is the opposite: no self-loops at all, the **largest SCC in
the corpus at 21 nodes**, the deepest nest at **4**, and 318 of its 646 loops complex. Codegen
for a state chart and codegen for a BPMN diagram share an algorithm and nothing else.

**585 loops - 27% - have no exit edge**, and 641 have no entry. Those are not detector noise:
this is ground truth, so a quarter of the hand-drawn loops in the corpus are genuinely
non-terminating as drawn, and every one of them is a generated `while (true)` that the author
did not intend. It is the strongest argument in these numbers for 10.3 emitting a warning rather
than trusting the graph.

## Correctness

Tarjan is checked against the definition - `u` reaches `v` and `v` reaches `u`, by flood fill -
on **5,497 of the 5,796 files** (every file at or under `BRUTE_FORCE_MAX_NODES`; the 299
skipped are all sketch2code pages too wide for the quadratic check). **Zero disagreements.** The
same comparison runs in `tests/test_assemble_loops.py` on hand-built graphs and on 200 random
ones, which is where the interesting shapes - two cycles sharing a node, a two-entry loop, a
component with no exit - are constructed rather than hoped for.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.ir.model import Diagram
from src.utils.config import ROOT
from src.utils.parallel import pmap

IR = ROOT / "data" / "processed" / "ir"
RUNS = ROOT / "experiments" / "assemble"
OUT = RUNS / "loops.json"

#: The five corpora, in the order they are reported everywhere else in Phase 10.
SOURCES = ("didi", "fa_bresler", "flowchartseg", "hdbpmn", "sketch2code")

#: Workers for the corpus sweep. Six sibling agents share this 32-core box.
N_JOBS = 4

#: Above this many nodes the brute-force reachability check is quadratic in the closure and
#: pointless to run; `verify()` is correctness evidence on small graphs, not a second
#: implementation for production.
BRUTE_FORCE_MAX_NODES = 60

#: Attribute keys written by `mark()`. Named here so a code generator can import them rather
#: than spell them, and so a rename is one edit.
NODE_LOOP_ID = "loop_id"
NODE_LOOP_ROLE = "loop_role"  # "entry" | "body"
NODE_LOOP_KIND = "loop_kind"  # "self_loop" | "simple_cycle" | "complex"
NODE_LOOP_DEPTH = "loop_depth"
EDGE_LOOP_ID = "loop_id"
EDGE_LOOP_ROLE = "loop_role"  # "back" | "body" | "exit" | "entry"


@dataclass
class Loop:
    """One strongly-connected component that is a loop, described for a code generator.

    `entries` and `exit_edges` are the whole point: a single entry whose exit leaves the header
    is a `while`, an exit on the back-edge tail is a `do/while`, an exit from the middle is a
    `break`, and several entries is none of those.
    """

    id: str
    nodes: list[str]
    kind: str
    depth: int
    entries: list[str] = field(default_factory=list)
    entry_edges: list[str] = field(default_factory=list)
    exit_edges: list[str] = field(default_factory=list)
    back_edges: list[str] = field(default_factory=list)

    @property
    def reducible(self) -> bool:
        """One entry (or none, for a loop nothing outside reaches) means a structured `while`.

        Several entries is an irreducible loop: no single header dominates the body, so no
        while/do-while nest expresses it and codegen has to emit labels and jumps.
        """
        return len(self.entries) <= 1

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "reducible": self.reducible, "size": len(self.nodes)}


# ------------------------------------------------------------------------------------------
# Tarjan, iteratively
# ------------------------------------------------------------------------------------------


def adjacency(diagram: Diagram, subset: set[str] | None = None) -> dict[str, list[str]]:
    """`{node: [successors]}` over real nodes, in a fixed order.

    Dangling edges (2.1.6 records plenty) and edges naming a node that is not in the diagram are
    dropped rather than invented into a phantom vertex, and successors are de-duplicated because
    a doubled arrow is not a cycle.
    """
    ids = diagram.node_ids if subset is None else diagram.node_ids & subset
    out: dict[str, list[str]] = {n.id: [] for n in diagram.nodes if n.id in ids}
    seen: set[tuple[str, str]] = set()
    for edge in diagram.edges:
        if edge.src in ids and edge.dst in ids and (edge.src, edge.dst) not in seen:
            seen.add((edge.src, edge.dst))
            out[edge.src].append(edge.dst)
    return out


def tarjan(graph: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's strongly-connected components, with an explicit stack instead of recursion.

    The explicit stack is insurance rather than a fix: the deepest DFS path in the whole 5,796
    file corpus is 27 nodes, so recursion would not in fact have overflowed here (see the module
    docstring). It costs one tuple per level and takes the question off the table for whatever
    graph arrives next.

    Components come back in Tarjan's own order, which is a reverse topological order of the
    condensation - every component is emitted before any component that reaches it.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in graph:
        if root in index:
            continue
        # Each frame is (node, how far through its successor list the DFS has got).
        work: list[tuple[str, int]] = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, position = work[-1]
            successors = graph[node]
            if position < len(successors):
                work[-1] = (node, position + 1)
                nxt = successors[position]
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


def sccs(diagram: Diagram) -> list[list[str]]:
    """Every strongly-connected component of the diagram's node graph, members sorted."""
    return [sorted(c) for c in tarjan(adjacency(diagram))]


def is_cyclic(graph: dict[str, list[str]], component: list[str]) -> bool:
    """A component is a loop if it has more than one node, or one node with an edge to itself."""
    if len(component) > 1:
        return True
    only = component[0]
    return only in graph.get(only, [])


# ------------------------------------------------------------------------------------------
# classification, entries and exits
# ------------------------------------------------------------------------------------------


def classify(graph: dict[str, list[str]], members: set[str]) -> str:
    """`self_loop`, `simple_cycle` or `complex` - the three shapes codegen treats differently."""
    if len(members) == 1:
        return "self_loop"
    inner_out = {n: [s for s in graph[n] if s in members] for n in members}
    inner_in = Counter(s for outs in inner_out.values() for s in outs)
    if all(len(v) == 1 for v in inner_out.values()) and all(inner_in[n] == 1 for n in members):
        return "simple_cycle"
    return "complex"


def _describe(diagram: Diagram, members: set[str], loop_id: str, depth: int) -> Loop:
    graph = adjacency(diagram)
    entries: set[str] = set()
    entry_edges: list[str] = []
    exit_edges: list[str] = []
    back_edges: list[str] = []
    for edge in diagram.edges:
        inside_src, inside_dst = edge.src in members, edge.dst in members
        if inside_dst and not inside_src and edge.src is not None:
            entries.add(edge.dst)
            entry_edges.append(edge.id)
        elif inside_src and not inside_dst and edge.dst is not None:
            exit_edges.append(edge.id)
    ordered_entries = sorted(entries)
    # A back edge is an in-loop edge landing on an entry - the arrow a generated `continue` or
    # loop-bottom jump corresponds to. With no entry at all (a loop nothing outside reaches) the
    # lowest-id member stands in as the header, so the field is never empty for a real loop.
    targets = set(ordered_entries) or {min(members)}
    for edge in diagram.edges:
        if edge.src in members and edge.dst in targets:
            back_edges.append(edge.id)
    return Loop(
        id=loop_id,
        nodes=sorted(members),
        kind=classify(graph, members),
        depth=depth,
        entries=ordered_entries,
        entry_edges=entry_edges,
        exit_edges=exit_edges,
        back_edges=back_edges,
    )


def loops(diagram: Diagram) -> list[Loop]:
    """Every loop in the diagram, outermost first, each with its entries, exits and depth.

    Nesting is found the way a loop-nesting forest is: an SCC is a loop; remove its entry nodes
    (its headers) and re-run Tarjan on what is left, and any SCC that survives is a loop nested
    inside it. That terminates because each round removes at least one node - when an SCC has no
    entry at all, the lowest-id member is removed instead, which is arbitrary but has to be
    something, and only decides which node is called the header of an unreachable loop.
    """
    found: list[Loop] = []
    agenda: list[tuple[set[str], int, str]] = [(set(diagram.node_ids), 1, "")]
    while agenda:
        subset, depth, prefix = agenda.pop(0)
        graph = adjacency(diagram, subset)
        for component in tarjan(graph):
            members = set(component)
            if not is_cyclic(graph, component):
                continue
            loop_id = f"{prefix}L{len(found)}"
            loop = _describe(diagram, members, loop_id, depth)
            found.append(loop)
            headers = set(loop.entries) or {min(members)}
            remaining = members - headers
            if remaining:
                agenda.append((remaining, depth + 1, f"{loop_id}."))
    return found


def mark(diagram: Diagram) -> Diagram:
    """Write the loop structure into the IR's `attrs`, and return the same diagram.

    Node `attrs`: `loop_id`, `loop_role` (`entry`/`body`), `loop_kind`, `loop_depth`.
    Edge `attrs`: `loop_id`, `loop_role` (`back`/`exit`/`entry`/`body`).

    Deeper loops are written last, so a node in a nest carries the innermost loop containing it -
    which is the one a generator emits around it. Nothing else in `attrs` is disturbed.
    """
    found = loops(diagram)
    by_id = {n.id: n for n in diagram.nodes}
    by_edge = {e.id: e for e in diagram.edges}
    for loop in sorted(found, key=lambda item: item.depth):
        entries = set(loop.entries)
        for node_id in loop.nodes:
            node = by_id.get(node_id)
            if node is None:
                continue
            node.attrs = dict(node.attrs or {})
            node.attrs[NODE_LOOP_ID] = loop.id
            node.attrs[NODE_LOOP_ROLE] = "entry" if node_id in entries else "body"
            node.attrs[NODE_LOOP_KIND] = loop.kind
            node.attrs[NODE_LOOP_DEPTH] = loop.depth
        members = set(loop.nodes)
        for edge in diagram.edges:
            if edge.src in members and edge.dst in members:
                edge.attrs = dict(edge.attrs or {})
                edge.attrs[EDGE_LOOP_ID] = loop.id
                edge.attrs[EDGE_LOOP_ROLE] = "body"
        # Order matters: a back edge is also an in-loop edge, and `back` is the more useful of
        # the two labels, so the specific roles are written over the body pass.
        for role, ids in (
            ("entry", loop.entry_edges),
            ("exit", loop.exit_edges),
            ("back", loop.back_edges),
        ):
            for edge_id in ids:
                edge = by_edge.get(edge_id)
                if edge is None:
                    continue
                edge.attrs = dict(edge.attrs or {})
                edge.attrs[EDGE_LOOP_ID] = loop.id
                edge.attrs[EDGE_LOOP_ROLE] = role
    return diagram


# ------------------------------------------------------------------------------------------
# the correctness evidence
# ------------------------------------------------------------------------------------------


def reachability(graph: dict[str, list[str]]) -> dict[str, set[str]]:
    """`{node: everything it reaches}`, by flood fill from each node. Deliberately naive."""
    closure: dict[str, set[str]] = {}
    for start in graph:
        seen: set[str] = set()
        stack = list(graph[start])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(graph[node])
        closure[start] = seen
    return closure


def brute_force_sccs(graph: dict[str, list[str]]) -> list[list[str]]:
    """SCCs straight from the definition: u and v are together iff u reaches v and v reaches u."""
    closure = reachability(graph)
    components: list[list[str]] = []
    placed: set[str] = set()
    for node in graph:
        if node in placed:
            continue
        group = {node} | {
            other
            for other in graph
            if other != node and other in closure[node] and node in closure[other]
        }
        placed |= group
        components.append(sorted(group))
    return sorted(components)


def verify(diagram: Diagram) -> bool:
    """Do Tarjan and the definition agree on this diagram's components?"""
    graph = adjacency(diagram)
    return sorted(sorted(c) for c in tarjan(graph)) == brute_force_sccs(graph)


# ------------------------------------------------------------------------------------------
# the corpus sweep
# ------------------------------------------------------------------------------------------


def _measure(path: Path) -> dict[str, Any]:
    try:
        diagram = Diagram.load(path)
    except Exception as exc:  # one unreadable file in 5,796 is a row, not a crashed sweep
        return {"source": path.parent.name, "error": f"{type(exc).__name__}: {exc}"[:120]}
    found = loops(diagram)
    graph = adjacency(diagram)
    checked = len(graph) <= BRUTE_FORCE_MAX_NODES
    return {
        "source": path.parent.name,
        "nodes": len(diagram.nodes),
        "edges": len(diagram.edges),
        "loops": [loop.to_dict() for loop in found],
        "max_depth": max((loop.depth for loop in found), default=0),
        "brute_force_checked": checked,
        "brute_force_agrees": verify(diagram) if checked else None,
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-file measurements into the numbers the write-up quotes."""
    files = [r for r in rows if "error" not in r]
    with_loops = [r for r in files if r["loops"]]
    every = [loop for r in files for loop in r["loops"]]
    kinds = Counter(loop["kind"] for loop in every)
    sizes = Counter(loop["size"] for loop in every)
    entries = Counter(len(loop["entries"]) for loop in every)
    exits = Counter(len(loop["exit_edges"]) for loop in every)
    irreducible = [loop for loop in every if not loop["reducible"]]
    checked = [r for r in files if r["brute_force_checked"]]
    return {
        "files": len(files),
        "errors": len(rows) - len(files),
        "files_with_a_cycle": len(with_loops),
        "share_with_a_cycle": round(len(with_loops) / max(1, len(files)), 4),
        "loops": len(every),
        "loops_per_cyclic_file": round(len(every) / max(1, len(with_loops)), 3),
        "kind": dict(kinds),
        "size_distribution": {str(k): sizes[k] for k in sorted(sizes)},
        "largest_scc": max(sizes) if sizes else 0,
        "self_loops": kinds.get("self_loop", 0),
        "multi_node_loops": len(every) - kinds.get("self_loop", 0),
        "reducible": len(every) - len(irreducible),
        "irreducible": len(irreducible),
        "irreducible_share": round(len(irreducible) / max(1, len(every)), 4),
        "entry_count_distribution": {str(k): entries[k] for k in sorted(entries)},
        "exit_edge_distribution": {str(k): exits[k] for k in sorted(exits)},
        "no_exit_loops": exits.get(0, 0),
        "max_nesting_depth": max((r["max_depth"] for r in files), default=0),
        "depth_distribution": {
            str(d): sum(1 for loop in every if loop["depth"] == d)
            for d in sorted({loop["depth"] for loop in every})
        },
        "brute_force_files_checked": len(checked),
        "brute_force_disagreements": sum(1 for r in checked if not r["brute_force_agrees"]),
    }


def run(ir_dir: Path = IR, n_jobs: int = N_JOBS) -> dict[str, Any]:
    paths = sorted(p for s in SOURCES for p in (ir_dir / s).glob("*.ir.json"))
    rows = pmap(_measure, paths, n_jobs=n_jobs, desc="loops")
    return {
        "corpus": str(ir_dir.relative_to(ROOT)),
        "overall": summarise(rows),
        "by_source": {s: summarise([r for r in rows if r["source"] == s]) for s in SOURCES},
        "attrs_written": {
            "node": [NODE_LOOP_ID, NODE_LOOP_ROLE, NODE_LOOP_KIND, NODE_LOOP_DEPTH],
            "edge": [EDGE_LOOP_ID, EDGE_LOOP_ROLE],
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    result = run(n_jobs=args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

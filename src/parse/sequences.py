"""Phase 7.3.3 - turning a graph into an ordered sequence, and the diagrams that refuse to be one.

    python -m src.parse.sequences --build     # writes data/processed/sequences.json

An HMM consumes a **sequence**. A diagram is a **graph**. This task is the conversion, and the
conversion is lossy in a way that is easy to hide and worth stating first: a graph with a branch
has no single order, so every ordering choice throws away part of the structure. What is kept is
what 7.3.4's transition matrix will be estimated from, so the choice made here decides what the
model can learn.

## The ordering: DFS, not topological sort

The plan names "topological/DFS ordering". They are not interchangeable here:

    topological   requires a DAG, and 681 of the 993 labelled diagrams contain a cycle - all 300
                  state machines and 381 of the 693 BPMN flowcharts. A topological sort is
                  undefined for **69% of this corpus**, which makes it not a default with an
                  occasional fallback but the fallback itself.
    DFS           always defined, and it produces the **back edge** as a by-product. A back edge
                  is an edge to a node already on the current stack, which is precisely the
                  "flow returns here" that 7.3.1's `loop-back` state means.

So: DFS from every source (in-degree 0), then from any unvisited node in id order, with children
visited in the order their target appears geometrically - top to bottom, then left to right,
which is how a reader's eye moves and is stable under the id renumbering the converters do.

**The DFS is what defines a back edge for the whole of 7.3**, and 7.3.1's `derive_states` takes
the set from here rather than recomputing it, so the two cannot disagree.

## What gets excluded, and why that is not cherry-picking

    5,012 flowcharts + 300 state machines in the IR
    - 484 wireframes           a containment tree, not a traversal (7.3.1's `is_sequential`)
    - 1,319 flowchartseg pages node polygons with no connectors at all - a "sequence" over them
                               would be a sort by position wearing a flow's clothes
    - 3,000 didi pages         no role annotations; kept separately as 7.3.6's unlabelled set
    = 993 labelled sequences   693 hdbpmn flowcharts, 300 fa_bresler state machines

The exclusions are structural, not score-driven: each is a diagram for which the *input* to the
model is undefined, and none is dropped for being hard.

## The multi-component problem, and the decision made about it

A hand-drawn page is often not one connected flow - hdbpmn's pools are separate lanes, and a
missed connector splits a flow in two. DFS from every source concatenates the components into one
sequence, which invents a transition between the last node of one component and the first of the
next. Those transitions are **recorded and excluded from 7.3.4's counts** (`component_breaks`),
because a model that learns them is learning the order the components happened to be visited in.

## What it measured

993 labelled sequences (693 hdbpmn, 300 fa_bresler) holding **14,056 observations**, plus 3,000
unlabelled didi sequences kept for 7.3.6. Median length 14 nodes, maximum 40.

**681 of the 993 sequences contain a back edge (68.6%)** - every one of the 300 state machines
and 381 of the 693 BPMN flowcharts. That is the number that justifies choosing DFS over a
topological sort: on a corpus where two pages in three contain a cycle, "topological order" is
not a default with an occasional fallback, it is undefined for most of the data.

## The multi-component number is the uncomfortable one

    multi-component sequences        555 of 993   (55.9%)
    component breaks                 2,524
    share of all transitions         19.3%

**A fifth of the transitions in the raw sequence set are jumps between disconnected pieces of the
page.** That is much higher than the "occasional missed connector" the design note anticipated,
and it is **almost entirely hdbpmn**: 554 of its 693 pages are multi-component against 1 of
fa_bresler's 300. The cause is structural rather than damage - a BPMN diagram is drawn as several
pools, and the message flows between them are frequently not edges in the converted IR at all, so
each pool arrives as its own component.

Had those 2,524 transitions been counted, they would have been **19% of 7.3.4's training data and
pure noise** - the model would have learned the order in which unrelated lanes happen to be
visited. They are recorded per sequence and excluded there, which is the whole reason the field
exists.

## What the ordering keeps and what it throws away

A DFS linearisation of a branching graph puts one whole branch before the other, so a decision's
two successors are never adjacent in the sequence: the model sees `decision -> branch-true` and
then, many steps later, an unexplained arrival at `branch-false`. That is the structural loss
this conversion makes, and it is why 7.3.4's transition matrix is expected to have a strong
`decision -> branch-true` cell and a weak `decision -> branch-false` one, despite the two being
equally real in the graph.

The alternative - BFS - has the mirror problem: it keeps siblings adjacent and destroys the
sequential path a reader actually follows, which is the thing Phase 12 needs. DFS is chosen
because the flow through a branch is what becomes code; the plan says "topological/DFS" and this
is the reading of that instruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import ROOT

IR_DIR = ROOT / "data" / "processed" / "ir"
OUT = ROOT / "data" / "processed" / "sequences.json"

#: Sources whose IR carries `semantic_role` annotations. Everything else is unlabelled and is
#: 7.3.6's material rather than 7.3.4's.
LABELLED_SOURCES: tuple[str, ...] = ("hdbpmn", "fa_bresler")
UNLABELLED_SOURCES: tuple[str, ...] = ("didi",)


def load_ir(sources, limit: int | None = None) -> list[dict]:
    """Every IR document from the named sources, in a stable order."""
    if not IR_DIR.is_dir():
        raise FileNotFoundError(f"no IR corpus at {IR_DIR}; run the Phase 2 converters first")
    out: list[dict] = []
    for source in sources:
        paths = sorted((IR_DIR / source).glob("*.ir.json"))
        if limit is not None:
            paths = paths[: max(1, limit // len(sources))]
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            document.setdefault("meta", {})["source"] = source
            out.append(document)
    return out


def labelled_diagrams(limit: int | None = None) -> list[dict]:
    from src.parse.roles import is_sequential

    return [d for d in load_ir(LABELLED_SOURCES, limit) if is_sequential(d)]


def unlabelled_diagrams(limit: int | None = None) -> list[dict]:
    from src.parse.roles import is_sequential

    return [d for d in load_ir(UNLABELLED_SOURCES, limit) if is_sequential(d)]


def _reading_key(node: dict) -> tuple[float, float]:
    """Top-to-bottom, then left-to-right. A node with no bbox sorts last, deterministically."""
    bbox = node.get("bbox")
    if not bbox:
        return (float("inf"), float("inf"))
    x, y, _, _ = bbox
    return (round(y, 3), round(x, 3))


def traversal(diagram: dict) -> tuple[list[str], set[tuple[str, str]]]:
    """(node ids in DFS order, back edges). Iterative - hdbpmn has 200-node pages.

    A back edge is an edge whose target is on the current stack. Edges to a node that is merely
    *finished* are cross or forward edges, not loops, and calling them back edges would label
    every re-convergence after a branch a `loop-back`.
    """
    nodes = {node["id"]: node for node in diagram.get("nodes", [])}
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for edge in diagram.get("edges", []):
        src, dst = edge.get("src"), edge.get("dst")
        if src in children and dst in nodes:
            children[src].append(dst)
    for node_id in children:
        children[node_id].sort(key=lambda target: _reading_key(nodes[target]))

    incoming = {node_id: 0 for node_id in nodes}
    for targets in children.values():
        for target in targets:
            incoming[target] += 1

    roots = [n for n in sorted(nodes, key=lambda i: _reading_key(nodes[i])) if incoming[n] == 0]
    roots += [n for n in sorted(nodes, key=lambda i: _reading_key(nodes[i])) if incoming[n]]

    order: list[str] = []
    visited: set[str] = set()
    back_edges: set[tuple[str, str]] = set()
    on_stack: set[str] = set()

    for root in roots:
        if root in visited:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        visited.add(root)
        on_stack.add(root)
        order.append(root)
        while stack:
            node_id, index = stack[-1]
            if index >= len(children[node_id]):
                stack.pop()
                on_stack.discard(node_id)
                continue
            stack[-1] = (node_id, index + 1)
            target = children[node_id][index]
            if target in on_stack:
                back_edges.add((node_id, target))
            elif target not in visited:
                visited.add(target)
                on_stack.add(target)
                order.append(target)
                stack.append((target, 0))
    return order, back_edges


def components(diagram: dict, order: list[str]) -> list[int]:
    """Component index per position in `order` - which weakly connected piece each node is in."""
    parent = {node["id"]: node["id"] for node in diagram.get("nodes", [])}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in diagram.get("edges", []):
        a, b = edge.get("src"), edge.get("dst")
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = {}
    out = []
    for node_id in order:
        root = find(node_id)
        out.append(roots.setdefault(root, len(roots)))
    return out


def sequence_of(diagram: dict, labelled: bool = True) -> dict | None:
    """One diagram -> one observation sequence, with states when the roles are annotated."""
    from src.parse.observations import observe
    from src.parse.roles import degrees, derive_states

    order, back_edges = traversal(diagram)
    if len(order) < 2:
        return None

    nodes = {node["id"]: node for node in diagram["nodes"]}
    incoming, outgoing = degrees(diagram)
    piece = components(diagram, order)

    observations = [observe(diagram, nodes[i], incoming[i], outgoing[i]) for i in order]
    record = {
        "id": diagram["id"],
        "source": diagram.get("meta", {}).get("source", "unknown"),
        "diagram_type": diagram["diagram_type"],
        "node_ids": order,
        "observations": observations,
        # Positions where the sequence jumps to a different connected component. 7.3.4 drops the
        # transition *into* each of these, because nothing in the diagram justifies it.
        "component_breaks": [i for i in range(1, len(order)) if piece[i] != piece[i - 1]],
        "components": len(set(piece)),
        "back_edges": sorted(back_edges),
    }
    if labelled:
        states = derive_states(diagram, order, back_edges)
        record["states"] = [states[i] for i in order]
    return record


def build(limit: int | None = None, write: bool = False) -> dict:
    labelled = [s for d in labelled_diagrams(limit) if (s := sequence_of(d, True))]
    unlabelled = [s for d in unlabelled_diagrams(limit) if (s := sequence_of(d, False))]

    lengths = [len(s["observations"]) for s in labelled]
    breaks = sum(len(s["component_breaks"]) for s in labelled)
    with_loops = sum(1 for s in labelled if s["back_edges"])
    result = {
        "sequences": labelled,
        "unlabelled": unlabelled,
        "summary": {
            "labelled_sequences": len(labelled),
            "unlabelled_sequences": len(unlabelled),
            "observations": sum(lengths),
            "median_length": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "sequences_with_a_back_edge": with_loops,
            "share_with_a_back_edge": round(with_loops / len(labelled), 4) if labelled else 0.0,
            "component_breaks": breaks,
            "share_of_transitions_that_are_breaks": round(
                breaks / max(1, sum(lengths) - len(labelled)), 4
            ),
            "multi_component_sequences": sum(1 for s in labelled if s["components"] > 1),
            "by_source": {
                source: sum(1 for s in labelled if s["source"] == source)
                for source in LABELLED_SOURCES
            },
        },
    }
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result), encoding="utf-8")
        result["summary"]["path"] = str(OUT.relative_to(ROOT))
    return result


def load(path: Path = OUT) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src.parse.sequences --build`")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        result = build(args.limit, write=args.build)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

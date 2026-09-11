"""Phase 12.1.4 (first half) - random graphs that are structured on purpose, not uniformly random.

    python -m src.synth.graphs --summary

12.1.4 asks for "random graphs -> rendered as diagrams -> reference code emitted
programmatically". The trap in that sentence is the word *random*: a uniformly random edge set
over N nodes is, with overwhelming probability, an irreducible tangle that no structured program
corresponds to. Generating 10K of those and calling the fallback dispatch loop their "reference
code" would produce 10K pairs teaching a model one degenerate template.

So the flowchart generator does not sample a graph. **It samples a program and lowers it into a
graph**, which is the direction that makes the reference code exact rather than reconstructed:

    stmt := action | io | if cond then STMT else STMT | while cond do STMT | STMT ; STMT

`_Lowering.lower` walks that tree emitting nodes and edges, and the `while` case is the only
place a back edge is ever created. The result is reducible by construction, so
`targets.emit_flowchart_python` rebuilds real `if`/`while` nesting for it instead of falling back
to its dispatch-loop template.

## The five types are not generated the same way, because they are not the same object

    flowchart      program tree, lowered (above)
    state_machine  random DFA: states x alphabet -> transitions, one initial, k accepting.
                   Cycles here are ordinary rather than a special case, so no program tree.
    er             entity set + typed relationships (1-N, N-M); N-M is lowered to a join table
                   at emit time, not here
    wireframe      containment tree only - sketch2code's IR has exactly one edge kind
                   (`contains`, 43,649 of 43,649 edges over its 484 pages), so a wireframe
                   carrying a non-containment edge would not match the real schema
    circuit        series/parallel two-terminal network over named nets

## Structure is a controlled variable, not an accident

Each diagram is stamped with `meta.structure`, one of `linear`, `branching`, `looping`, `nested`,
`disconnected`, and the corpus is generated to a requested mix rather than to whatever the
sampler happened to produce. That is what stops the set being trivially uniform, and it is what
`pairs.build_dataset` reports its distribution over.

`disconnected` is deliberate rather than a defect: 55.9% of the real hdbpmn pages are
multi-component (7.3.3's number), so a synthetic corpus in which every page is one connected flow
would be *less* like the target domain, not more.

## What was rejected

**Uniform random digraphs** - the literal reading of the row. Sampled at p = 2/N they are almost
never reducible, and the structured emitter falls back on the overwhelming majority of them.
Rejected: it optimises the corpus for the fallback template rather than for code.

**Reusing one program tree with renamed identifiers** to reach the 10K count cheaply. This is the
padding the row invites: it drives the exact-duplicate rate to 0 while leaving the *structural*
duplicate rate near 1.0. Rejected, and `pairs.py` measures the near-duplicate rate on a
structure-only signature precisely so that this cannot be done by accident.

**bbox = None** for generated nodes. `sequences.traversal` sorts children by reading order and a
null bbox sorts last, so every generated diagram would have had a traversal determined by
insertion order alone. A layered layout is assigned instead (BFS depth -> y, sibling index -> x),
which is what makes the traversal a *reading* order rather than a record of how this file
happened to append nodes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, deque

IR_VERSION = 1.0

#: The structural families the corpus is generated over. `pairs.build_dataset` requests a mix of
#: these; nothing here samples a "typical" graph without naming which of these it is.
STRUCTURES: tuple[str, ...] = ("linear", "branching", "looping", "nested", "disconnected")

DIAGRAM_TYPES: tuple[str, ...] = ("flowchart", "state_machine", "er", "wireframe", "circuit")

_VERBS = (
    "validate",
    "fetch",
    "compute",
    "normalise",
    "persist",
    "notify",
    "retry",
    "archive",
    "score",
    "merge",
    "dispatch",
    "reconcile",
)
_NOUNS = (
    "record",
    "invoice",
    "payload",
    "account",
    "batch",
    "token",
    "order",
    "profile",
    "ticket",
    "shipment",
)
_CONDS = (
    "is_valid",
    "has_balance",
    "is_expired",
    "needs_review",
    "in_stock",
    "is_duplicate",
    "over_limit",
)
_ENTITIES = (
    "customer",
    "order",
    "product",
    "invoice",
    "shipment",
    "payment",
    "supplier",
    "warehouse",
    "review",
    "category",
)
_COLUMNS = (
    ("name", "TEXT"),
    ("created_at", "TEXT"),
    ("amount", "REAL"),
    ("quantity", "INTEGER"),
    ("status", "TEXT"),
    ("code", "TEXT"),
    ("notes", "TEXT"),
    ("rating", "REAL"),
)
_UI_TAGS = ("div", "section", "nav", "header", "footer", "ul", "li", "span", "p", "h1", "h2", "h3")
_UI_LEAVES = (
    ("ui-label", "text-block", "span"),
    ("ui-button", "rounded-rect", "button"),
    ("ui-image", "rectangle", "img"),
    ("ui-input", "rectangle", "input"),
)
_PARTS = (("R", "resistor", "1k"), ("C", "capacitor", "10n"), ("L", "inductor", "1m"))


# ---------------------------------------------------------------------------------------------
# the IR envelope
# ---------------------------------------------------------------------------------------------


def _node(node_id: str, shape: str, text: str, role: str, **attrs: object) -> dict:
    """One IR node. `bbox` is filled by `_layout` afterwards and is never left None."""
    return {
        "id": node_id,
        "shape": shape,
        "bbox": [0.0, 0.0, 160.0, 90.0],
        "text": text,
        "semantic_role": role,
        "confidence": 1.0,
        "source_id": node_id,
        "attrs": {"shape_basis": "synthetic", **attrs},
    }


def _edge(edge_id: str, src: str, dst: str, label: str = "", **attrs: object) -> dict:
    return {
        "id": edge_id,
        "src": src,
        "dst": dst,
        "directed": True,
        "label": label,
        "polyline": None,
        "confidence": 1.0,
        "source_id": edge_id,
        "attrs": dict(attrs),
    }


def _document(diagram_id: str, diagram_type: str, nodes: list, edges: list, structure: str) -> dict:
    return {
        "ir_version": IR_VERSION,
        "id": diagram_id,
        "diagram_type": diagram_type,
        "nodes": nodes,
        "edges": edges,
        "unresolved_edges": [],
        "crossed_out": [],
        "low_conf_text": [],
        "meta": {
            "source": "synthetic",
            "geometry": "derived",
            "image": "",
            "image_size": [1024, 1024],
            "scribe_id": "synthetic",
            "producer": "src.synth.graphs",
            "structure": structure,
            "notes": "generated for 12.1.4; structure is a controlled variable",
        },
    }


def _layout(nodes: list, edges: list) -> None:
    """Assign a layered bbox in place: BFS depth -> y, sibling index within depth -> x.

    Without this every bbox would be identical and `sequences.traversal` would order children by
    nothing at all, making the traversal an artefact of the append order in this file.
    """
    index = {node["id"]: node for node in nodes}
    if not index:
        return
    children: dict[str, list[str]] = {node_id: [] for node_id in index}
    incoming = dict.fromkeys(index, 0)
    for edge in edges:
        if edge["src"] in children and edge["dst"] in index:
            children[edge["src"]].append(edge["dst"])
            incoming[edge["dst"]] += 1

    depth = dict.fromkeys(index, 0)
    roots = [node_id for node_id in index if incoming[node_id] == 0] or [next(iter(index))]
    seen = set(roots)
    queue = deque(roots)
    while queue:
        current = queue.popleft()
        for child in children[current]:
            if child not in seen:
                seen.add(child)
                depth[child] = depth[current] + 1
                queue.append(child)

    per_depth: Counter = Counter()
    for node in nodes:
        level = depth[node["id"]]
        column = per_depth[level]
        per_depth[level] += 1
        node["bbox"] = [80.0 + 220.0 * column, 80.0 + 150.0 * level, 160.0, 90.0]


# ---------------------------------------------------------------------------------------------
# flowchart: sample a program, lower it into a graph
# ---------------------------------------------------------------------------------------------


def _program(rng: random.Random, budget: int, depth: int, structure: str) -> dict:
    """A random statement. `structure` decides which constructs are reachable at all."""
    allow_if = structure in ("branching", "nested", "disconnected")
    allow_loop = structure in ("looping", "nested", "disconnected")

    if budget <= 1 or depth > 3 or (not allow_if and not allow_loop):
        return {"kind": rng.choice(("action", "io"))}

    choices = ["action", "io"]
    if allow_if:
        choices += ["if"] * (3 if structure in ("branching", "nested") else 1)
    if allow_loop:
        choices += ["while"] * (3 if structure == "looping" else 1)
    kind = rng.choice(choices)

    if kind == "if":
        left = max(1, (budget - 1) // 2)
        return {
            "kind": "if",
            "cond": rng.choice(_CONDS),
            "then": _seq(rng, left, depth + 1, structure),
            "else": _seq(rng, budget - 1 - left, depth + 1, structure),
        }
    if kind == "while":
        return {
            "kind": "while",
            "cond": rng.choice(_CONDS),
            "body": _seq(rng, budget - 1, depth + 1, structure),
        }
    return {"kind": kind}


def _seq(rng: random.Random, budget: int, depth: int, structure: str) -> dict:
    """One or more statements in sequence."""
    parts = []
    remaining = max(1, budget)
    while remaining > 0:
        take = rng.randint(1, remaining)
        parts.append(_program(rng, take, depth, structure))
        remaining -= take
    return {"kind": "seq", "parts": parts} if len(parts) > 1 else parts[0]


class _Lowering:
    """Lowers a statement tree into IR nodes and edges.

    The invariant that matters downstream: `while` is the *only* case that draws an edge back to
    a node already created above it on the path, so a back edge exists exactly where the source
    program had a loop, and `targets` can rebuild that loop from the graph alone.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._n = 0
        self._e = 0

    def node(self, shape: str, text: str, role: str, **attrs: object) -> str:
        node_id = f"n{self._n}"
        self._n += 1
        self.nodes.append(_node(node_id, shape, text, role, **attrs))
        return node_id

    def edge(self, src: str, dst: str, label: str = "", **attrs: object) -> None:
        self.edges.append(_edge(f"e{self._e}", src, dst, label, **attrs))
        self._e += 1

    def lower(self, stmt: dict, nxt: str) -> str:
        """Emit `stmt` so that it falls through to `nxt`; return its entry node id."""
        kind = stmt["kind"]
        if kind == "seq":
            entry = nxt
            for part in reversed(stmt["parts"]):
                entry = self.lower(part, entry)
            return entry
        if kind == "if":
            decision = self.node("diamond", f"{stmt['cond']}?", "decision", cond=stmt["cond"])
            self.edge(decision, self.lower(stmt["then"], nxt), "yes")
            self.edge(decision, self.lower(stmt["else"], nxt), "no")
            return decision
        if kind == "while":
            decision = self.node("diamond", f"{stmt['cond']}?", "decision", cond=stmt["cond"])
            # the body returns to the decision: this is the corpus's only source of back edges
            self.edge(decision, self.lower(stmt["body"], decision), "yes")
            self.edge(decision, nxt, "no")
            return decision
        noun = self.rng.choice(_NOUNS)
        if kind == "io":
            verb = self.rng.choice(("read", "write"))
            current = self.node("freeform", f"{verb} {noun}", "io", verb=verb, noun=noun)
        else:
            verb = self.rng.choice(_VERBS)
            current = self.node(
                "rounded-rect", f"{verb} the {noun}", "process", verb=verb, noun=noun
            )
        self.edge(current, nxt)
        return current


def _flowchart(rng: random.Random, diagram_id: str, structure: str) -> dict:
    lowering = _Lowering(rng)
    end = lowering.node("circle", "end", "end")
    entry = lowering.lower(_seq(rng, rng.randint(2, 9), 0, structure), end)
    start = lowering.node("circle", "start", "start")
    lowering.edge(start, entry)

    if structure == "disconnected":
        # a second, unattached flow - 55.9% of real hdbpmn pages are multi-component
        orphan_end = lowering.node("circle", "end", "end")
        orphan = lowering.lower(_seq(rng, rng.randint(1, 3), 0, "linear"), orphan_end)
        lowering.edge(lowering.node("circle", "start", "start"), orphan)

    _layout(lowering.nodes, lowering.edges)
    return _document(diagram_id, "flowchart", lowering.nodes, lowering.edges, structure)


# ---------------------------------------------------------------------------------------------
# state machine, ER, wireframe, circuit
# ---------------------------------------------------------------------------------------------


def _state_machine(rng: random.Random, diagram_id: str, structure: str) -> dict:
    base = {"linear": 3, "branching": 5, "looping": 4, "nested": 6, "disconnected": 5}[structure]
    count = max(2, base + rng.randint(-1, 1))
    alphabet = ["a", "b", "c"][: 2 if structure == "linear" else rng.randint(2, 3)]
    accepting = set(rng.sample(range(count), k=max(1, count // 3)))

    nodes, edges = [], []
    for index in range(count):
        is_final = index in accepting
        role = "initial-state" if index == 0 else ("final-state" if is_final else "state")
        nodes.append(
            _node(
                f"s{index}",
                "double-circle" if is_final else "circle",
                f"q{index}",
                role,
                accepting=is_final,
                initial=index == 0,
            )
        )
    counter = 0
    for index in range(count):
        for symbol in alphabet:
            if structure == "linear":
                target = min(index + 1, count - 1)
            elif structure == "looping" and rng.random() < 0.4:
                target = max(0, index - 1)
            else:
                target = rng.randrange(count)
            edges.append(
                _edge(f"t{counter}", f"s{index}", f"s{target}", symbol, self_loop=target == index)
            )
            counter += 1

    if structure == "disconnected":
        # an unreachable state, which is a real defect a drawn FA often has
        nodes.append(
            _node(f"s{count}", "circle", f"q{count}", "state", accepting=False, initial=False)
        )
        edges.append(_edge(f"t{counter}", f"s{count}", f"s{count}", alphabet[0], self_loop=True))

    _layout(nodes, edges)
    return _document(diagram_id, "state_machine", nodes, edges, structure)


def _er(rng: random.Random, diagram_id: str, structure: str) -> dict:
    base = {"linear": 2, "branching": 4, "looping": 3, "nested": 5, "disconnected": 4}[structure]
    names = rng.sample(_ENTITIES, k=min(base, len(_ENTITIES)))
    nodes, edges = [], []
    for index, name in enumerate(names):
        columns = rng.sample(_COLUMNS, k=rng.randint(1, 4))
        nodes.append(
            _node(
                f"E{index}",
                "rectangle",
                name,
                "entity",
                table=name,
                columns=[list(column) for column in columns],
            )
        )
    counter = 0
    for index in range(1, len(names)):
        parent = 0 if structure == "linear" else rng.randrange(index)
        many = structure in ("nested", "branching") and rng.random() < 0.4
        relation = _node(
            f"R{counter}",
            "diamond",
            f"{names[parent]}_{names[index]}",
            "relationship",
            cardinality="n-m" if many else "1-n",
        )
        nodes.append(relation)
        edges.append(_edge(f"re{counter}a", f"E{parent}", relation["id"], "1", side="left"))
        edges.append(_edge(f"re{counter}b", relation["id"], f"E{index}", "n", side="right"))
        counter += 1
    if structure == "looping" and names:
        # a self-referencing FK, the manager_id case - a cycle that SQL handles natively
        relation = _node(
            f"R{counter}", "diamond", f"{names[0]}_parent", "relationship", cardinality="1-n"
        )
        nodes.append(relation)
        edges.append(_edge(f"re{counter}a", "E0", relation["id"], "1", side="left"))
        edges.append(_edge(f"re{counter}b", relation["id"], "E0", "n", side="right"))

    _layout(nodes, edges)
    return _document(diagram_id, "er", nodes, edges, structure)


def _wireframe(rng: random.Random, diagram_id: str, structure: str) -> dict:
    max_depth = {"linear": 1, "branching": 2, "looping": 2, "nested": 4, "disconnected": 2}[
        structure
    ]
    nodes: list[dict] = []
    edges: list[dict] = []
    counter = [0]

    def build(parent: str | None, level: int) -> str:
        node_id = f"w{counter[0]}"
        counter[0] += 1
        if level >= max_depth or (level > 0 and rng.random() < 0.35):
            role, shape, tag = rng.choice(_UI_LEAVES)
            text = "" if role == "ui-image" else rng.choice(_NOUNS)
            nodes.append(_node(node_id, shape, text, role, html_tag=tag))
        else:
            tag = "body" if parent is None else rng.choice(_UI_TAGS)
            nodes.append(_node(node_id, "rectangle", "", "container", html_tag=tag))
            for _ in range(rng.randint(1, 2 if structure == "linear" else 3)):
                child = build(node_id, level + 1)
                edges.append(_edge(f"c{len(edges)}", node_id, child, kind="contains"))
        return node_id

    build(None, 0)
    if structure == "disconnected":
        build(None, max_depth)

    _layout(nodes, edges)
    return _document(diagram_id, "wireframe", nodes, edges, structure)


def _circuit(rng: random.Random, diagram_id: str, structure: str) -> dict:
    count = {"linear": 2, "branching": 4, "looping": 3, "nested": 5, "disconnected": 4}[structure]
    nodes = [
        _node(
            "P0",
            "circle",
            "V1",
            "source",
            component="V",
            ref="V1",
            value="5",
            net_pos="1",
            net_neg="0",
        )
    ]
    edges: list[dict] = []
    net = 1
    previous = "P0"
    for index in range(count):
        kind, name, value = rng.choice(_PARTS)
        if structure == "branching" and index and rng.random() < 0.5:
            pos, neg = str(net), "0"  # a shunt leg to ground rather than another series part
        else:
            net += 1
            pos, neg = str(net - 1), str(net) if index < count - 1 else "0"
        part_id = f"P{index + 1}"
        nodes.append(
            _node(
                part_id,
                "rectangle",
                f"{kind}{index + 1}",
                name,
                component=kind,
                ref=f"{kind}{index + 1}",
                value=value,
                net_pos=pos,
                net_neg=neg,
            )
        )
        edges.append(_edge(f"w{index}", previous, part_id, kind="wire"))
        previous = part_id
    if structure in ("looping", "nested"):
        edges.append(_edge(f"w{count}", previous, "P0", kind="wire"))
    if structure == "disconnected":
        nodes.append(
            _node(
                "P99",
                "rectangle",
                "R99",
                "resistor",
                component="R",
                ref="R99",
                value="1k",
                net_pos="90",
                net_neg="91",
            )
        )

    _layout(nodes, edges)
    return _document(diagram_id, "circuit", nodes, edges, structure)


_GENERATORS = {
    "flowchart": _flowchart,
    "state_machine": _state_machine,
    "er": _er,
    "wireframe": _wireframe,
    "circuit": _circuit,
}


# ---------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------


def random_diagram(diagram_type: str, structure: str, seed: int) -> dict:
    """One IR document of `diagram_type` in the named structural family.

    Deterministic in the triple: the same (type, structure, seed) always yields identical IR,
    which is what lets a 10K corpus be reproduced from a one-line command rather than shipped.
    """
    if diagram_type not in _GENERATORS:
        raise ValueError(f"unknown diagram_type {diagram_type!r}; expected one of {DIAGRAM_TYPES}")
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure {structure!r}; expected one of {STRUCTURES}")
    rng = random.Random(f"{diagram_type}/{structure}/{seed}")
    return _GENERATORS[diagram_type](rng, f"syn_{diagram_type}_{structure}_{seed:06d}", structure)


def generate(
    count: int,
    diagram_types: tuple[str, ...] = DIAGRAM_TYPES,
    structures: tuple[str, ...] = STRUCTURES,
    seed: int = 0,
):
    """Yield `count` diagrams, round-robin over the diagram_type x structure grid.

    Round-robin rather than random choice so that the per-type and per-structure counts are
    exact: 12.1.4 has to report a distribution, and a distribution the generator merely
    approximated would need error bars before it could be read.
    """
    grid = [(t, s) for t in diagram_types for s in structures]
    if not grid:
        return
    for index in range(count):
        diagram_type, structure = grid[index % len(grid)]
        yield random_diagram(diagram_type, structure, seed + index // len(grid))


def summary(count: int = 2000, seed: int = 0) -> dict:
    """Per-type node/edge/back-edge statistics over a sample, for the CLI."""
    from src.parse.sequences import traversal

    by_type: dict[str, Counter] = {}
    for diagram in generate(count, seed=seed):
        stats = by_type.setdefault(diagram["diagram_type"], Counter())
        order, back = traversal(diagram)
        stats["diagrams"] += 1
        stats["nodes"] += len(diagram["nodes"])
        stats["edges"] += len(diagram["edges"])
        stats["traversed"] += len(order)
        stats["with_back_edge"] += 1 if back else 0
    return {
        diagram_type: {
            "diagrams": stats["diagrams"],
            "mean_nodes": round(stats["nodes"] / stats["diagrams"], 2),
            "mean_edges": round(stats["edges"] / stats["diagrams"], 2),
            "back_edge_rate": round(stats["with_back_edge"] / stats["diagrams"], 4),
        }
        for diagram_type, stats in sorted(by_type.items())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="synthetic diagram generation (12.1.4)")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.summary:
        json.dump(summary(args.count, args.seed), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

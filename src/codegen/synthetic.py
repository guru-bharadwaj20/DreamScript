"""Phase 12.1.4 - synthetic pairs: random graphs -> rendered diagrams -> programmatic reference code.

    python -m src.codegen.synthetic build             # >=10K pairs + PNGs, then pairs merge
    python -m src.codegen.synthetic stats             # distribution + diversity of the shard

Output: `data/processed/codegen/pairs/synthetic.jsonl` (12.1.1 schema, 12.1.7-filtered, train
only by 12.1.8), rendered images under `data/processed/codegen/synthetic/images/<type>/`, and
`data/processed/codegen/synthetic/report.json`. Load with
`src.codegen.pairs.load_pairs("train", sources=["synthetic"])`.

## Why this does not use `src.synth.graphs` as its generator

The inherited `src.synth.graphs` was measured before anything was built on it, and it cannot
produce a training set, for four independent reasons:

1. **Near-duplicates.** Over its 12,500-diagram default run, circuits had **5 distinct
   structures** and 406 distinct IR texts, ER diagrams **33 structures**; flowcharts 1,015.
   Every target nonetheless looked unique, because the diagram id is spliced into the code.
2. **Targets not recoverable from the input.** 12.1.2's IR text carries shape, text, role and
   edges, not `attrs`. The draft kept ER columns and every circuit value and net only in
   `attrs`, so the SQL columns and the entire netlist were invisible to the model it trains.
3. ~~**Invalid IR.**~~ **Fixed.** This read "0 of 500 of its diagrams pass
   `schemas/ir.schema.json`: `ir_version` is a float, ER is typed `er`, and circuit roles
   `source`/`resistor` are not in the vocabulary". All three are corrected in `src.synth.graphs`
   and `tests/test_synth_schema.py` validates the whole type x structure x seed grid against the
   schema, so 200 of 200 pass. Kept in this list, struck through, because the other three reasons
   are what decide the question and striking one out is more honest than deleting it.
4. **Unsolvable circuits.** 926 of 2,500 draft circuits fail ngspice's operating point (12.1.7).

**On "left unchanged".** That sentence said `src.synth.graphs` was left alone because the 11.2
curriculum shares it - which read as "it is broken and we cannot touch it", and was the reason
its schema failures survived three phases. It is not true and it was not a good reason: a
generator shared by two consumers is a generator whose defects reach both, and 11.2's curriculum
was being fed invalid IR the entire time. Reason 3 is fixed in `graphs` itself, for both callers.

What is still true is the *design* half: this module generates its own graphs, because reasons
1, 2 and 4 are about what `graphs` generates rather than about bugs in it, and fixing those
would mean replacing the generator rather than repairing it. It reuses only the 11.x-independent
pieces (`graphs.STRUCTURES`, `graphs._layout`).

## How each type is generated so that code is a function of `ir_text`

    flowchart      a random statement tree (action / io / if / while / seq) lowered into a graph
                   - reducible by construction, so 12.1.6 rebuilds real if/while nesting
    state_machine  2-8 states, one of three alphabets (letters, bits, event words), partial or
                   total transition function; initial and accepting states are roles, not attrs
    er_diagram     2-7 entities; every column is an `attribute` node (`name: type`) wired to its
                   entity, and relationship edges are labelled with their cardinality (1/n, n/m)
    wireframe      a containment tree of `div` containers and label / button / image / input
                   leaves - containers are `div` only because a `nav` or `ul` tag lives in attrs
                   and would be invisible in `ir_text`
    circuit        a source plus 2-10 R/C/L parts over named nets, each net a `wire` node and each
                   terminal an edge labelled `+`/`-`; built so every net has a DC path to ground
                   and no loop is made of inductors and the source alone (a singular DC matrix)

Every diagram is validated against `schemas/ir.schema.json`, emitted through 12.1.6 against the
canonical id (12.1.1), deduplicated, filtered by 12.1.7 (ngspice, node, compile) and only then
rendered. The module docstring's numbers are in `reports/codegen_synthetic.md`.

## Deduplication and diversity

    exact duplicate    identical `ir_text` - dropped
    near-duplicate     identical *structure signature* (the IR with every text and label
                       removed) and a label-token Jaccard >= `NEAR_JACCARD` with a kept member -
                       dropped; the same drawing with one word changed is not a new example
    structure cap      at most `SIGNATURE_CAP` pairs share one structure signature, so a small
                       graph shape cannot be repeated hundreds of times under new names

Determinism: every candidate is a pure function of `(diagram_type, seed)` through
`random.Random(f"{type}/{seed}")`, candidates are processed in seed order, and parallel work is
`Pool.map`, which preserves order - so the shard is byte-identical across runs and worker counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from src.codegen import pairs, schema
from src.synth import graphs
from src.utils.config import ROOT

OUT = ROOT / "data" / "processed" / "codegen" / "synthetic"
IMAGES = OUT / "images"
TYPES: tuple[str, ...] = ("flowchart", "state_machine", "er_diagram", "wireframe", "circuit")
PER_TYPE = 2400
SEED = 1204
NEAR_JACCARD = 0.8
SIGNATURE_CAP = 12

# -- vocabularies -------------------------------------------------------------------------------

_VERBS = (
    "validate compute fetch normalise persist notify retry archive score merge dispatch "
    "reconcile parse enrich approve reject log flag route charge refund encrypt compress "
    "schedule cancel assign sync export import render resize tag index publish audit"
).split()
_NOUNS = (
    "record invoice payload account batch token order profile ticket shipment user session "
    "report image file message cart payment claim booking device sensor reading quote lead "
    "contract refund review coupon"
).split()
_CONDS = (
    "is_valid has_balance is_expired needs_review in_stock is_duplicate over_limit is_empty "
    "has_errors is_admin retry_left is_paid timed_out is_verified has_more is_open is_ready "
    "matches_rule is_flagged within_budget"
).split()
_STATE_WORDS = (
    "idle running paused stopped waiting ready locked open closed error done armed "
    "charging active standby booting"
).split()
_ALPHABETS = (
    ("a", "b", "c", "d"),
    ("0", "1"),
    ("coin", "push", "start", "stop", "reset", "timeout", "ack", "fail", "retry", "tick"),
)
_ENTITIES = (
    "customer order product invoice shipment payment supplier warehouse review category "
    "employee department project task course student teacher enrollment author book loan "
    "member account address vehicle driver route ticket event venue"
).split()
_COLUMNS = (
    ("name", "TEXT"), ("email", "TEXT"), ("created_at", "TEXT"), ("amount", "REAL"),
    ("quantity", "INTEGER"), ("status", "TEXT"), ("code", "TEXT"), ("notes", "TEXT"),
    ("rating", "REAL"), ("price", "REAL"), ("title", "TEXT"), ("phone", "TEXT"),
    ("age", "INTEGER"), ("score", "REAL"), ("due_date", "TEXT"), ("city", "TEXT"),
    ("country", "TEXT"), ("weight", "REAL"), ("capacity", "INTEGER"), ("sku", "TEXT"),
    ("level", "INTEGER"), ("balance", "REAL"), ("description", "TEXT"), ("priority", "INTEGER"),
)  # fmt: skip
_REL_VERBS = (
    "places has contains owns manages writes teaches holds ships belongs_to supplies "
    "reviews assigns books"
).split()
_BUTTONS = (
    "Submit|Cancel|Save|Login|Sign up|Search|Next|Back|Buy now|Add|Delete|Edit|Share|"
    "Download|Continue|Checkout|Subscribe|Reset|Apply|Confirm"
).split("|")
_LABELS = (
    "Welcome|Your profile|Latest news|Contact us|Price|Username|Password|Total|About|"
    "Shopping cart|Recent orders|Settings|Email address|Features|Pricing plans|"
    "Terms of service|Order summary|Notifications|Dashboard|Help center|Search results|"
    "Team|Blog|Comments|Photos|Upcoming events|Account details|Billing|Reviews|FAQ"
).split("|")
_E12 = ("1", "1.2", "1.5", "2.2", "3.3", "4.7", "6.8")
_PART_UNITS = {"R": ("", "k", "k", "meg"), "C": ("p", "n", "u"), "L": ("u", "m")}


# -- IR envelope -------------------------------------------------------------------------------


def _node(node_id: str, shape: str, text: str, role: str, **attrs: object) -> dict:
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


def _document(diagram_id: str, diagram_type: str, nodes: list, edges: list, family: str) -> dict:
    return {
        "ir_version": "1.0",
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
            "producer": "src.codegen.synthetic",
            "structure": family,
        },
    }


# -- flowchart ---------------------------------------------------------------------------------


def _program(rng: random.Random, budget: int, depth: int, family: str) -> dict:
    allow_if = family in ("branching", "nested", "disconnected")
    allow_loop = family in ("looping", "nested", "disconnected")
    if budget <= 1 or depth > 3 or not (allow_if or allow_loop):
        return {"kind": rng.choice(("action", "action", "io"))}
    choices = ["action", "action", "io"]
    choices += ["if"] * (3 if allow_if and family != "disconnected" else int(allow_if))
    choices += ["while"] * (3 if family == "looping" else int(allow_loop))
    kind = rng.choice(choices)
    if kind == "if":
        left = rng.randint(1, max(1, budget - 1))
        return {
            "kind": "if",
            "cond": rng.choice(_CONDS),
            "then": _seq(rng, left, depth + 1, family),
            "else": _seq(rng, max(1, budget - 1 - left), depth + 1, family),
        }
    if kind == "while":
        return {
            "kind": "while",
            "cond": rng.choice(_CONDS),
            "body": _seq(rng, budget - 1, depth + 1, family),
        }
    return {"kind": kind}


def _seq(rng: random.Random, budget: int, depth: int, family: str) -> dict:
    parts, remaining = [], max(1, budget)
    while remaining > 0:
        take = rng.randint(1, remaining)
        parts.append(_program(rng, take, depth, family))
        remaining -= take
    return {"kind": "seq", "parts": parts} if len(parts) > 1 else parts[0]


class _Lowering:
    """Statement tree -> nodes/edges. `while` is the only construct that draws a back edge."""

    def __init__(self, rng: random.Random) -> None:
        self.rng, self.nodes, self.edges = rng, [], []

    def node(self, shape: str, text: str, role: str, **attrs: object) -> str:
        node_id = f"n{len(self.nodes)}"
        self.nodes.append(_node(node_id, shape, text, role, **attrs))
        return node_id

    def edge(self, src: str, dst: str, label: str = "") -> None:
        self.edges.append(_edge(f"e{len(self.edges)}", src, dst, label))

    def lower(self, stmt: dict, nxt: str) -> str:
        kind = stmt["kind"]
        if kind == "seq":
            entry = nxt
            for part in reversed(stmt["parts"]):
                entry = self.lower(part, entry)
            return entry
        if kind in ("if", "while"):
            decision = self.node("diamond", f"{stmt['cond']}?", "decision", cond=stmt["cond"])
            if kind == "if":
                self.edge(decision, self.lower(stmt["then"], nxt), "yes")
                self.edge(decision, self.lower(stmt["else"], nxt), "no")
            else:
                self.edge(decision, self.lower(stmt["body"], decision), "yes")
                self.edge(decision, nxt, "no")
            return decision
        noun = self.rng.choice(_NOUNS)
        if kind == "io":
            verb = self.rng.choice(("read", "write"))
            current = self.node("parallelogram", f"{verb} {noun}", "io", verb=verb, noun=noun)
        else:
            verb = self.rng.choice(_VERBS)
            current = self.node("rectangle", f"{verb} {noun}", "process", verb=verb, noun=noun)
        self.edge(current, nxt)
        return current


def gen_flowchart(rng: random.Random, diagram_id: str, family: str) -> dict:
    low = _Lowering(rng)
    end = low.node("ellipse", "end", "end")
    entry = low.lower(_seq(rng, rng.randint(2, 12), 0, family), end)
    low.edge(low.node("ellipse", "start", "start"), entry)
    if family == "disconnected":
        orphan_end = low.node("ellipse", "end", "end")
        orphan = low.lower(_seq(rng, rng.randint(1, 4), 0, "branching"), orphan_end)
        low.edge(low.node("ellipse", "start", "start"), orphan)
    return _document(diagram_id, "flowchart", low.nodes, low.edges, family)


# -- state machine -----------------------------------------------------------------------------


def gen_state_machine(rng: random.Random, diagram_id: str, family: str) -> dict:
    count = rng.randint(2, 4) if family == "linear" else rng.randint(3, 8)
    alphabet = list(rng.choice(_ALPHABETS))
    alphabet = rng.sample(
        alphabet, k=rng.randint(1 if family == "linear" else 2, min(4, len(alphabet)))
    )
    words = rng.random() < 0.4 and count <= len(_STATE_WORDS)
    names = rng.sample(_STATE_WORDS, k=count) if words else [f"q{i}" for i in range(count)]
    accepting = set(rng.sample(range(1, count), k=rng.randint(1, max(1, (count - 1) // 2))))
    total = rng.random() < 0.5
    nodes = []
    for index in range(count):
        final = index in accepting
        role = "initial-state" if index == 0 else ("final-state" if final else "state")
        shape = "double-circle" if final else "circle"
        nodes.append(
            _node(f"s{index}", shape, names[index], role, accepting=final, initial=index == 0)
        )
    edges = []
    for index in range(count):
        for symbol in alphabet:
            if not total and rng.random() < 0.35:
                continue
            if family == "linear":
                target = min(index + 1, count - 1)
            elif family == "looping" and rng.random() < 0.5:
                target = rng.randrange(0, index + 1)
            elif family == "branching":
                target = rng.randrange(index, count)
            else:
                target = rng.randrange(count)
            edges.append(_edge(f"t{len(edges)}", f"s{index}", f"s{target}", symbol))
    if family == "disconnected":
        extra = count
        nodes.append(_node(f"s{extra}", "circle", f"q{extra}" if not words else "orphan", "state"))
        edges.append(_edge(f"t{len(edges)}", f"s{extra}", f"s{rng.randrange(count)}", alphabet[0]))
    return _document(diagram_id, "state_machine", nodes, edges, family)


# -- ER ----------------------------------------------------------------------------------------


def gen_er(rng: random.Random, diagram_id: str, family: str) -> dict:
    k = {"linear": (2, 3), "branching": (3, 6), "looping": (2, 5), "nested": (4, 7)}.get(
        family, (3, 6)
    )
    names = rng.sample(_ENTITIES, k=rng.randint(*k))
    nodes, edges = [], []
    for index, name in enumerate(names):
        columns = rng.sample(_COLUMNS, k=rng.randint(1, 5))
        nodes.append(
            _node(f"E{index}", "rectangle", name, "entity", columns=[list(c) for c in columns])
        )
        for j, (column, sql_type) in enumerate(columns):
            attr = f"A{index}_{j}"
            nodes.append(_node(attr, "ellipse", f"{column}: {sql_type.lower()}", "attribute"))
            edges.append(_edge(f"a{index}_{j}", f"E{index}", attr))

    links: list[tuple[int, int]] = []
    start = 1
    if family == "disconnected" and len(names) >= 4:
        split = len(names) // 2
        links += [(rng.randrange(i), i) for i in range(1, split)]
        links += [(rng.randrange(split, i), i) for i in range(split + 1, len(names))]
        start = len(names)
    for i in range(start, len(names)):
        links.append((0 if family == "linear" else rng.randrange(i), i))
    if family in ("nested", "looping") and len(names) >= 3:
        a, b = rng.sample(range(len(names)), 2)
        links.append((a, b))
    if family == "looping":
        links.append((rng.randrange(len(names)),) * 2)

    for r, (parent, child) in enumerate(links):
        many = family in ("nested", "branching", "disconnected") and rng.random() < 0.35
        rel = f"R{r}"
        verb = rng.choice(_REL_VERBS)
        nodes.append(
            _node(rel, "diamond", verb, "relationship", cardinality="n-m" if many else "1-n")
        )
        edges.append(_edge(f"r{r}a", f"E{parent}", rel, "n" if many else "1"))
        edges.append(_edge(f"r{r}b", rel, f"E{child}", "m" if many else "n"))
    return _document(diagram_id, "er_diagram", nodes, edges, family)


# -- wireframe ---------------------------------------------------------------------------------

_LEAVES = (
    ("ui-label", "text-block", "span"),
    ("ui-label", "text-block", "span"),
    ("ui-button", "rounded-rect", "button"),
    ("ui-image", "rectangle", "img"),
    ("ui-input", "rectangle", "input"),
)


def gen_wireframe(rng: random.Random, diagram_id: str, family: str) -> dict:
    max_depth = {"linear": 1, "branching": 2, "looping": 3, "nested": 4}.get(family, 2)
    width = 2 if family == "linear" else 4
    nodes: list[dict] = []
    edges: list[dict] = []

    def build(level: int) -> str:
        node_id = f"w{len(nodes)}"
        leaf = level >= max_depth or (level > 0 and rng.random() < 0.4)
        if leaf:
            role, shape, tag = rng.choice(_LEAVES)
            text = {"ui-label": rng.choice(_LABELS), "ui-button": rng.choice(_BUTTONS)}.get(
                role, ""
            )
            nodes.append(_node(node_id, shape, text, role, html_tag=tag))
            return node_id
        nodes.append(_node(node_id, "rectangle", "", "container", html_tag="div"))
        for _ in range(rng.randint(1, width)):
            child = build(level + 1)
            edges.append(_edge(f"c{len(edges)}", node_id, child, kind="contains"))
        return node_id

    build(0)
    if family == "disconnected":
        build(max(1, max_depth - 1))
    return _document(diagram_id, "wireframe", nodes, edges, family)


# -- circuit -----------------------------------------------------------------------------------


class _Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def _value(rng: random.Random, letter: str) -> str:
    return f"{rng.choice(_E12)}{rng.choice(_PART_UNITS[letter])}"


def gen_circuit(rng: random.Random, diagram_id: str, family: str) -> dict:
    """Parts over nets, constructed so the DC operating point exists (see module docstring)."""
    parts: list[tuple[str, str, str, str]] = []  # (letter, pos, neg, value)
    nets = ["0", "1"]
    parts.append(("V", "1", "0", str(rng.choice((1, 3, 5, 9, 12)))))
    zero_loop = _Union()  # connectivity through inductors and the source only
    zero_loop.union("1", "0")
    count = {"linear": (2, 4), "branching": (3, 7), "looping": (3, 7), "nested": (5, 10)}.get(
        family, (4, 8)
    )
    tail = "1"
    for _ in range(rng.randint(*count)):
        letter = rng.choices(("R", "C", "L"), weights=(6, 2, 2))[0]
        if family == "linear" or rng.random() < 0.5:
            pos, neg = tail, str(len(nets))
            nets.append(neg)
            tail = neg
        elif family in ("branching", "nested") and rng.random() < 0.6:
            pos, neg = rng.choice(nets[1:]), "0"
        else:
            pos, neg = rng.sample(nets, 2)
        if letter == "L" and zero_loop.find(pos) == zero_loop.find(neg):
            letter = "R"  # an inductor across already-shorted nets closes a zero-resistance loop
        if letter == "L":
            zero_loop.union(pos, neg)
        parts.append((letter, pos, neg, _value(rng, letter)))
    parts.append(("R", tail, "0", _value(rng, "R")))  # close the chain

    if family == "disconnected":
        base = len(nets)
        nets += [str(base), str(base + 1)]
        parts.append(("V", str(base), "0", str(rng.choice((3, 5)))))
        parts.append(("R", str(base), str(base + 1), _value(rng, "R")))
        parts.append(("R", str(base + 1), "0", _value(rng, "R")))

    # every net needs a DC path to ground through R, L or V, and at least two terminals
    for _ in range(3):
        dc = _Union()
        uses: Counter = Counter()
        for letter, pos, neg, _v in parts:
            uses[pos] += 1
            uses[neg] += 1
            if letter != "C":
                dc.union(pos, neg)
        missing = sorted(
            {n for n in uses if dc.find(n) != dc.find("0")} | {n for n in uses if uses[n] < 2}
        )
        if not missing:
            break
        for net in missing:
            if net != "0":
                parts.append(("R", net, "0", _value(rng, "R")))

    nodes: list[dict] = []
    edges: list[dict] = []
    counters: Counter = Counter()
    used_nets = sorted({n for _l, p, q, _v in parts for n in (p, q)}, key=int)
    for net in used_nets:
        nodes.append(_node(f"net{net}", "circle", net, "wire"))
    for index, (letter, pos, neg, value) in enumerate(parts):
        counters[letter] += 1
        ref = f"{letter}{counters[letter]}"
        part_id = f"P{index}"
        shape = "circle" if letter == "V" else "rectangle"
        nodes.append(
            _node(part_id, shape, f"{ref} {value}", "component", component=letter, ref=ref,
                  value=value, net_pos=pos, net_neg=neg)
        )  # fmt: skip
        edges.append(_edge(f"w{index}p", part_id, f"net{pos}", "+"))
        edges.append(_edge(f"w{index}n", part_id, f"net{neg}", "-"))
    return _document(diagram_id, "circuit", nodes, edges, family)


GENERATORS = {
    "flowchart": gen_flowchart,
    "state_machine": gen_state_machine,
    "er_diagram": gen_er,
    "wireframe": gen_wireframe,
    "circuit": gen_circuit,
}


def random_diagram(diagram_type: str, seed: int) -> dict:
    """One schema-valid IR document; a pure function of `(diagram_type, seed)`."""
    rng = random.Random(f"{diagram_type}/{seed}")
    family = graphs.STRUCTURES[seed % len(graphs.STRUCTURES)]
    diagram = GENERATORS[diagram_type](rng, f"syn_{diagram_type}_{seed:06d}", family)
    layout(diagram)
    return diagram


# -- layout and rendering ----------------------------------------------------------------------


def _circuit_layout(diagram: dict) -> None:
    """Nets on a row; each part below, centred between its two nets, on the first free row.

    BFS layering put every part on one row and every net on the next, so each wire crossed most
    of the drawing (seen when the rendered sample was inspected).
    """
    nets = [n for n in diagram["nodes"] if n["semantic_role"] == "wire"]
    xs = {}
    for i, net in enumerate(nets):
        net["bbox"] = [80.0 + 200.0 * i, 60.0, 40.0, 40.0]
        xs[net["id"]] = 100.0 + 200.0 * i
    terminals: dict[str, list[str]] = defaultdict(list)
    for edge in diagram["edges"]:
        terminals[edge["src"]].append(edge["dst"])
    rows: list[list[float]] = []
    for node in diagram["nodes"]:
        if node["semantic_role"] == "wire":
            continue
        ends = [xs[t] for t in terminals[node["id"]] if t in xs] or [100.0]
        cx = sum(ends) / len(ends)
        row = next(
            (r for r, taken in enumerate(rows) if all(abs(cx - t) > 175 for t in taken)), None
        )
        if row is None:
            rows.append([])
            row = len(rows) - 1
        rows[row].append(cx)
        node["bbox"] = [cx - 80.0, 190.0 + 130.0 * row, 160.0, 70.0]


def layout(diagram: dict) -> None:
    """Assign bboxes in place: nested boxes for wireframes, BFS layers for everything else."""
    if diagram["diagram_type"] == "circuit":
        _circuit_layout(diagram)
        return
    if diagram["diagram_type"] != "wireframe":
        graphs._layout(diagram["nodes"], diagram["edges"])
        return
    nodes = {n["id"]: n for n in diagram["nodes"]}
    children: dict[str, list[str]] = defaultdict(list)
    has_parent = set()
    for edge in diagram["edges"]:
        children[edge["src"]].append(edge["dst"])
        has_parent.add(edge["dst"])
    roots = [n["id"] for n in diagram["nodes"] if n["id"] not in has_parent]

    def place(node_id: str, x: float, y: float, w: float, h: float, depth: int) -> None:
        nodes[node_id]["bbox"] = [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]
        kids = children.get(node_id, [])
        if not kids:
            return
        pad = 12.0
        horizontal = depth % 2 == 1
        span = (w if horizontal else h) - pad * (len(kids) + 1)
        step = span / len(kids)
        for i, kid in enumerate(kids):
            if horizontal:
                place(kid, x + pad + i * (step + pad), y + pad, step, h - 2 * pad, depth + 1)
            else:
                place(kid, x + pad, y + pad + i * (step + pad), w - 2 * pad, step, depth + 1)

    height = 900.0 / max(1, len(roots))
    for i, root in enumerate(roots):
        place(root, 40.0, 40.0 + i * height, 900.0, height - 20.0, 0)


def render(diagram: dict, path: Path) -> tuple[int, int]:
    """Draw the IR as a hand-drawn-looking PNG with 1.3.7's wobbled-stroke primitives."""
    import cv2
    import numpy as np

    from src.ingest import synthetic as ink

    seed = int(hashlib.sha1(diagram["id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    boxes = {n["id"]: n["bbox"] for n in diagram["nodes"]}
    # `ink._text` draws into a 240x40 patch and silently skips one that overflows the canvas,
    # which left every label off narrow flowcharts until the rendered sample was inspected.
    right = max(x + w for x, _y, w, _h in boxes.values()) + 300
    bottom = max(y + h for _x, y, _w, h in boxes.values()) + 60
    img = np.full((int(bottom), int(right), 3), ink.PAPER, np.uint8)
    nodes = {n["id"]: n for n in diagram["nodes"]}

    def centre(node_id: str) -> tuple[float, float]:
        x, y, w, h = boxes[node_id]
        return x + w / 2, y + h / 2

    if diagram["diagram_type"] != "wireframe":
        for edge in diagram["edges"]:
            (x0, y0), (x1, y1) = centre(edge["src"]), centre(edge["dst"])
            if edge["src"] == edge["dst"]:
                x, y, w, _h = boxes[edge["src"]]
                ink._ellipse(img, x + w, y, 26, 18, rng)
            else:
                d = math.hypot(x1 - x0, y1 - y0) or 1.0
                ux, uy = (x1 - x0) / d, (y1 - y0) / d
                r0 = min(boxes[edge["src"]][2:]) / 2 + 4
                r1 = min(boxes[edge["dst"]][2:]) / 2 + 6
                ink._arrow(img, (x0 + ux * r0, y0 + uy * r0), (x1 - ux * r1, y1 - uy * r1), rng)
            if edge.get("label"):
                ink._text(img, (x0 + x1) / 2 + 6, (y0 + y1) / 2 - 26, edge["label"], rng, 0.45)

    for node_id, (x, y, w, h) in boxes.items():
        node = nodes[node_id]
        shape, text = node["shape"], node.get("text") or ""
        cx, cy = x + w / 2, y + h / 2
        if shape in ("rectangle", "rounded-rect"):
            ink._rect(img, x, y, w, h, rng)
            if node["semantic_role"] == "ui-image":
                ink._line(img, (x, y), (x + w, y + h), rng)
                ink._line(img, (x + w, y), (x, y + h), rng)
        elif shape == "diamond":
            ink._diamond(img, cx, cy, w, h, rng)
        elif shape in ("ellipse", "circle", "double-circle"):
            r = min(w, h) / 2 if shape != "ellipse" else None
            rx, ry = (r, r) if r else (w / 2, h / 2)
            ink._ellipse(img, cx, cy, rx, ry, rng)
            if shape == "double-circle":
                ink._ellipse(img, cx, cy, rx - 7, ry - 7, rng)
        elif shape == "parallelogram":
            skew = 18
            pts = [(x + skew, y), (x + w, y), (x + w - skew, y + h), (x, y + h)]
            for i in range(4):
                ink._line(img, pts[i], pts[(i + 1) % 4], rng)
        if text:
            tx = (
                x + 8
                if diagram["diagram_type"] == "wireframe"
                else max(0.0, cx - min(w, 7.5 * len(text)) / 2)
            )
            ty = y + 4 if node["semantic_role"] == "container" else cy - 20
            ink._text(img, tx, ty, text, rng, 0.5)

    scale = min(1.0, 1600.0 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), gray, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return gray.shape[1], gray.shape[0]


# -- candidates, dedup, build ------------------------------------------------------------------


def structure_signature(ir_text: str) -> str:
    """The IR with every text field and edge label removed: shape, role, wiring, order."""
    out = []
    for line in ir_text.split("\n"):
        parts = line.split("|")
        if parts[0] == "N":
            parts += [""] * (5 - len(parts))
            out.append(f"N|{parts[1]}|{parts[2]}|{parts[4]}")
        elif parts[0] == "E":
            parts += [""] * (5 - len(parts))
            out.append(f"E|{parts[1]}|{parts[2]}|{parts[4]}")
        else:
            out.append(line)
    return hashlib.sha1("\n".join(out).encode()).hexdigest()


def label_tokens(ir_text: str) -> frozenset[str]:
    """Words carried by node texts and edge labels - what a near-duplicate differs in."""
    words: list[str] = []
    for line in ir_text.split("\n"):
        parts = line.split("|")
        if parts[0] == "N" and len(parts) > 3:
            words += re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?[a-z]*", parts[3].lower())
        elif parts[0] == "E" and len(parts) > 3:
            words.append(f"edge:{parts[3].lower()}")
    counts = Counter(words)
    return frozenset(f"{w}#{i}" for w, c in counts.items() for i in range(c))


def _candidate(job: tuple[str, int]) -> dict:
    """Generate, validate and emit one candidate. Runs in a worker; pure in `job`."""
    from src.ir.model import Diagram

    diagram_type, seed = job
    diagram = random_diagram(diagram_type, seed)
    problems = Diagram.from_dict(diagram).problems()
    record = pairs.make_pair(
        diagram,
        "synthetic",
        meta={
            "structure": diagram["meta"]["structure"],
            "seed": seed,
            "nodes": len(diagram["nodes"]),
        },
    )
    return {"record": record, "schema_problems": problems[:3]}


def _render_job(job: tuple[str, int, str]) -> tuple[int, int]:
    diagram_type, seed, rel = job
    return render(random_diagram(diagram_type, seed), ROOT / rel)


def dedupe(records: list[dict], state: dict) -> tuple[list[dict], Counter]:
    """Drop exact and near duplicates against everything already kept in `state`."""
    kept: list[dict] = []
    why: Counter = Counter()
    for record in records:
        text = record["ir_text"]
        digest = hashlib.sha1(text.encode()).hexdigest()
        if digest in state["exact"]:
            why["exact_duplicate"] += 1
            continue
        signature = structure_signature(text)
        bucket = state["buckets"][signature]
        if len(bucket) >= SIGNATURE_CAP:
            why["structure_cap"] += 1
            continue
        tokens = label_tokens(text)
        if any(
            len(tokens & other) / max(1, len(tokens | other)) >= NEAR_JACCARD for other in bucket
        ):
            why["near_duplicate"] += 1
            continue
        state["exact"].add(digest)
        bucket.append(tokens)
        kept.append(record)
    return kept, why


def build(
    per_type: int = PER_TYPE, seed: int = SEED, workers: int | None = None, images: bool = True
) -> dict:
    """Generate until each type has `per_type` deduplicated, filtered pairs; write the shard."""
    from src.codegen import quality

    started = time.time()
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    report: dict = {"per_type": per_type, "seed": seed, "types": {}}
    chosen: list[dict] = []
    with Pool(workers) as pool:
        for diagram_type in TYPES:
            state = {"exact": set(), "buckets": defaultdict(list)}
            stats: Counter = Counter()
            kept_type: list[dict] = []
            cursor = seed * 1000
            while len(kept_type) < per_type:
                batch = int((per_type - len(kept_type)) * 1.6) + 64
                jobs = [(diagram_type, s) for s in range(cursor, cursor + batch)]
                cursor += batch
                results = pool.map(_candidate, jobs, chunksize=32)
                stats["candidates"] += len(results)
                valid = []
                for result in results:
                    if result["schema_problems"]:
                        stats["ir_schema_invalid"] += 1
                    else:
                        valid.append(result["record"])
                unique, why = dedupe(valid, state)
                stats.update(why)
                verdicts = quality.check_records(pairs.assign_splits(unique))
                for record, verdict in zip(unique, verdicts, strict=True):
                    if not verdict["ok"]:
                        stats[f"rejected:{verdict['kind']}"] += 1
                    elif len(kept_type) < per_type:
                        kept_type.append(record)
            stats["kept"] = len(kept_type)
            stats["distinct_structures"] = len(state["buckets"])
            report["types"][diagram_type] = dict(stats)
            chosen += kept_type

        if images:
            jobs = []
            for record in chosen:
                rel = f"data/processed/codegen/synthetic/images/{record['diagram_type']}/{record['diagram_id']}.png"
                record["meta"]["image"] = rel
                jobs.append((record["diagram_type"], record["meta"]["seed"], rel))
            t_render = time.time()
            sizes = pool.map(_render_job, jobs, chunksize=16)
            report["render_seconds"] = round(time.time() - t_render, 1)
            report["images"] = len(sizes)

    shard = pairs.write_shard("synthetic", pairs.assign_splits(chosen))
    report["shard"] = shard
    report["diversity"] = diversity(chosen)
    report["seconds"] = round(time.time() - started, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def diversity(records: list[dict]) -> dict:
    """Per type: node-count spread, structure families, distinct structures and their entropy."""
    out = {}
    by_type: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_type[record["diagram_type"]].append(record)
    for diagram_type, group in sorted(by_type.items()):
        signatures = Counter(structure_signature(r["ir_text"]) for r in group)
        total = sum(signatures.values())
        entropy = -sum(c / total * math.log(c / total) for c in signatures.values())
        sizes = sorted(len(r["traversal"]) for r in group)
        vocab = set().union(*(label_tokens(r["ir_text"]) for r in group))
        out[diagram_type] = {
            "pairs": total,
            "families": dict(Counter(r["meta"]["structure"] for r in group)),
            "distinct_structures": len(signatures),
            "max_per_structure": max(signatures.values()),
            "effective_structures": round(math.exp(entropy), 1),
            "nodes_p10_p50_p90_max": [
                sizes[len(sizes) // 10],
                sizes[len(sizes) // 2],
                sizes[len(sizes) * 9 // 10],
                sizes[-1],
            ],
            "label_vocabulary": len({t.split("#")[0] for t in vocab}),
            "flowchart_modes": dict(
                Counter(
                    r["meta"].get("flowchart_mode")
                    for r in group
                    if r["meta"].get("flowchart_mode")
                )
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.4 synthetic pairs")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--per-type", type=int, default=PER_TYPE)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--no-images", action="store_true")
    sub.add_parser("stats")
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build(args.per_type, args.seed, args.workers, not args.no_images)
        pairs.merge()
    else:
        report = diversity(list(schema.read_jsonl(pairs.PAIRS_DIR / "synthetic.jsonl")))
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

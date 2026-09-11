"""Phase 12.3.4 - structural fidelity: does the generated code's control flow match the IR?

    python -m src.eval.structural            # corpus sweep + corruption-detection study

A number called "structural fidelity" is undefined until somebody names the rule that decides
which AST construct *corresponds to* which IR node or edge, exactly as `src.assemble.irdiff`
argues for node F1: correspondence is a choice, and the choice moves the answer. This module
names one rule, defends it, and measures what it detects on code whose damage is known.

## The rule: node-anchored, representation-disjunctive correspondence

**Anchors.** Every IR node carries an anchor set - its `id`, and `src.ir.targets.slug()` of its
text (falling back to its role). A code symbol *is* that node iff it equals one of its anchors.
Nothing softer: no substring matching, no edit-distance nearest neighbour. A soft rule would
make "process" match nine different boxes on one page, which is the false-positive mode 12.3.8
measures directly.

**Where a symbol counts.** Only at a *definition or table site*: `def`/`class` names, assignment
targets, and string constants used as keys of a dict literal. A name in a docstring or a comment
is not a realisation of a node - the emitters write the node id into the module docstring, and a
metric that counted that would score a program with no code at all a perfect 1.0. Docstrings are
excised by node identity before any constant is read.

**The disjunction, which is the whole design.** Correct code for a cyclic labelled graph comes in
two shapes and a metric that assumes one scores the other zero:

    structured   `if`/`elif` per outgoing edge, `while` per cycle, one `def` per node.
    tabular      a `TRANSITIONS` dict literal holding the edges, one bounded driver loop.

This repo's own reference targets (`src/ir/targets.py`, 1,477 pairs) are **tabular**. Measured on
the 993 Python targets: the mean count of `ast.If` chains with arity >= 2 is **0.0** against a
mean of 1.02 gateways per diagram, so a structured-only branch rule scores the ground truth
**0.0** - not because the reference is wrong, but because the rule assumed a representation. So
each sub-metric accepts *either* realisation and takes the better of the two, and `detect_style()`
reports which one the code used, so no headline can be quoted without it.

## The three sub-metrics

    node_f1        realisation of nodes as definitions/table keys. Roles `container` and
                   `unknown` are excluded from recall - the emitters deliberately never emit
                   them (`src/ir/targets.py`), so counting them would measure a decision, not a
                   failure.
    branch_score   per gateway (out-degree >= 2, or role `decision`/`fork`), the arity the code
                   actually expresses vs the out-degree drawn, as min/max so both an under- and
                   an over-branching generator lose points. A gateway with no branch point at
                   all scores 0.
    loop_recall    cyclic components from `src.assemble.loops.loops()` - Tarjan, entries, and
                   the `self_loop`/`simple_cycle`/`complex` classification - **reused, not
                   reimplemented**, so this metric and 10.2.3 can never disagree about what a
                   loop is. Structured code must show one `While`/`For` per reducible loop;
                   tabular code must show at least one bounded driver loop when the IR has any
                   cycle, because one driver walks every cycle in the graph and demanding n
                   loops of it would be demanding a different program.

`structural_fidelity()` is the unweighted mean of the sub-metrics that are *defined* for the
diagram - a graph with no gateway contributes no branch term rather than a free 1.0, and `nan`
marks the undefined ones so a corpus mean skips them instead of averaging in a fiction.

## What it measured - all 993 Python reference targets, each against its own IR

    metric                  mean
    parsed                 1.0000
    style_tabular          1.0000
    node_recall            1.0000
    node_precision         0.9970
    node_precision_raw     0.7005
    node_f1                0.9983
    branch_score           0.9922
    gateway_recall         0.9922
    loop_recall            1.0000
    structural             0.9968

`node_recall` is 1.0 because the emitter defines a function per node by construction - that is a
self-consistency check, and it passing is the floor, not the achievement. The interesting number
is the **0.2965 gap between raw and allowlisted precision**: `run`, `accepts`, `TRANSITIONS`,
`STEPS`, `START`, `END` are real definitions corresponding to no node, and they are a large share
of a small program's definitions. Both are reported. **The `SCAFFOLD` allowlist is tuned to this
repo's emitters**, so on an unseen generator's style the raw number is the safe one and the
allowlisted number is optimistic - the single largest known weakness in this module.

## Detection rates on deliberately-corrupted code (`corruption_study`, 300 Python pairs)

Each corruption is applied to a *correct* target and the metric is asked whether it noticed. A
detection is the watched sub-metric falling below its value on the same clean pair.

    corruption      what it does                                  applicable  detected  mean drop
    drop_branch     removes one outgoing edge from the table          247       100.0%    0.1215
    drop_node       deletes one node's def and its table row          300       100.0%    0.0876
    invent_node     adds a def + table row for a node not in the IR   300       100.0%    0.0296
    rename_table    renames the TRANSITIONS dict                      300       100.0%    0.3327
    drop_loop       deletes the driver `for` loop                     161       100.0%    0.3333
    shuffle_labels  permutes branch labels, structure untouched       112         0.0%    0.0000

**The last row is the point.** `shuffle_labels` rewires which branch goes where without changing
any count, and structural fidelity is **blind to it, by design**: it measures shape, not
semantics. Anyone quoting 0.9968 as "the code is right" is quoting a number that cannot see a
fully-miswired program. 12.3.3's functional correctness is the metric that catches that; this one
is not a substitute for it and must never be reported as one.

`drop_branch` is applicable to 247 of 300 because the other 53 diagrams have no node with two
drawn outgoing arrows at all; `drop_loop` to 161 because the rest are acyclic and the emitter's
driver loop is then the only loop, whose removal `loop_recall` correctly declines to judge
(`nan`, not 0).

## What was rejected

**Tree-edit distance between the AST and the IR graph.** The natural-sounding metric, and it is
undefined: an AST is an ordered tree, an IR is an unordered cyclic graph, and any embedding of
one in the other is itself a correspondence rule - the same choice, hidden inside a distance so
it can no longer be reported. Named sub-metrics beat one opaque number.

**Fuzzy anchor matching** at 0.8 normalised similarity. Measured on the same 993 targets: node
recall could not improve (already 1.0) and raw precision fell **0.7005 -> 0.6161**, because short
slugs (`end`, `no`, `ok`, `yes`) fuzzy-match each other and each others' scaffolding. Exact
anchors only.

**Requiring one `While` per loop in tabular code.** Scored the correct reference 0.0 on every
cyclic flowchart. Replaced by the style disjunction above.

**Executing the code to observe control flow.** Answers a better question, and belongs to
12.3.2/12.3.3; it also cannot score a program that does not run, which is exactly the population
this metric most needs to be able to describe.

**Model-output numbers do not exist yet.** 12.2.x has not produced a fine-tuned model, so every
number above is over reference targets and over corruptions of them. That is what demonstrates
the metric detects what it claims; it is not a claim about any model.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

TARGETS = ROOT / "data" / "processed" / "targets"
INDEX = TARGETS / "index.json"
RUNS = ROOT / "experiments" / "eval"
OUT = RUNS / "structural.json"

#: Roles the reference emitters deliberately never emit, so their absence is a decision rather
#: than a miss (the `container`/`unknown` skip in `src/ir/targets.py`).
NON_EMITTED_ROLES = frozenset({"container", "unknown"})

#: Roles that are a branch point regardless of out-degree. Out-degree >= 2 also qualifies, so
#: this only adds decisions drawn with a single arrow actually reaching them.
GATEWAY_ROLES = frozenset({"decision", "fork"})

#: Names a generated program needs that correspond to no node. Precision is reported both with
#: and without this allowlist because the list is tuned to *this* repo's emitters: on a
#: different generator's style the raw number is the honest one.
SCAFFOLD = frozenset(
    {
        "TRANSITIONS",
        "STEPS",
        "START",
        "END",
        "ACCEPTING",
        "STATES",
        "ALPHABET",
        "run",
        "main",
        "accepts",
        "choose",
        "state",
        "current",
        "step",
        "options",
        "word",
        "symbol",
        "nxt",
        "max_steps",
        "annotations",
    }
)

#: Roles that become a table in the SQL branch of this row ("do generated tables match the IR?").
ENTITY_ROLES = frozenset({"entity", "component"})

_SLUG_KEYWORDS = frozenset(
    "False None True and as assert async await break class continue def del elif else except "
    "finally for from global if import in is lambda nonlocal not or pass raise return try "
    "while with yield".split()
)


# ------------------------------------------------------------------------------------------
# Anchors: the correspondence rule's left-hand side
# ------------------------------------------------------------------------------------------


def slug(text: str, fallback: str = "step") -> str:
    """A readable Python identifier from arbitrary handwriting.

    Deliberately a copy of `src.ir.targets.slug`, used only when that module is not importable,
    so this metric scores code from *any* generator rather than depending on the one that
    happens to live in this repo.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", (text or "").strip().lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)[:40].strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    if cleaned in _SLUG_KEYWORDS:
        cleaned += "_"
    return cleaned


def _slug() -> Any:
    try:  # the repo's own slug when present; src/codegen is another agent's, never a blocker.
        from src.ir.targets import slug as repo_slug

        return repo_slug
    except Exception:  # pragma: no cover - only on a partially-checked-out tree
        return slug


def anchors(node: dict) -> set[str]:
    """The strings that, appearing at a definition site, mean "this node".

    The node id, and the slug of its text (or of its role, which is what the emitters name an
    unlabelled box). Exact match only - see the module docstring on why fuzzy was rejected.
    """
    fn = _slug()
    role = str(node.get("semantic_role") or "unknown")
    out = {str(node["id"])}
    text = (node.get("text") or "").strip()
    out.add(fn(text or role, role.replace("-", "_")))
    return {a for a in out if a}


def node_anchor_map(diagram: dict) -> dict[str, set[str]]:
    """`{node id: anchors}` for every node, including the non-emitted roles."""
    return {str(n["id"]): anchors(n) for n in diagram.get("nodes", [])}


# ------------------------------------------------------------------------------------------
# Code symbols: the correspondence rule's right-hand side
# ------------------------------------------------------------------------------------------


@dataclass
class CodeSymbols:
    """Everything the AST offers as a possible realisation of a node, kept by site."""

    definitions: set[str] = field(default_factory=set)
    table_keys: set[str] = field(default_factory=set)
    mentions: set[str] = field(default_factory=set)
    #: `[(names mentioned anywhere in an if/elif chain's tests, branch count)]`.
    if_chains: list[tuple[set[str], int]] = field(default_factory=list)
    #: `{dict-literal string key: arity of its value}` - a tabular branch point.
    dict_arity: dict[str, int] = field(default_factory=dict)
    loop_count: int = 0
    parse_error: str = ""

    @property
    def definition_sites(self) -> set[str]:
        return self.definitions | self.table_keys


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """`id()` of every string constant that is a docstring, so it can be excluded by identity."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _chain_arity(node: ast.If) -> tuple[set[str], int]:
    """Branch count of a whole if/elif/else chain, and every name its tests mention."""
    names: set[str] = set()
    branches = 0
    current: ast.stmt | None = node
    while isinstance(current, ast.If):
        branches += 1
        for sub in ast.walk(current.test):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.add(sub.attr)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                names.add(sub.value)
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
        else:
            if current.orelse:
                branches += 1
            current = None
    return names, branches


def code_symbols(code: str, language: str = "python") -> CodeSymbols:
    """Parse `code` and collect every candidate realisation site.

    Code that does not parse yields an empty `CodeSymbols` with `parse_error` set, which scores
    0 everywhere - correctly: 12.3.1 already reports syntactic validity, and a program that does
    not parse has no structure to be faithful with.
    """
    sym = CodeSymbols()
    if language != "python":
        sym.parse_error = f"no AST available for language {language!r}"
        return sym
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        sym.parse_error = f"{type(exc).__name__}: {exc}"
        return sym

    docstrings = _docstring_nodes(tree)
    nested_if: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and len(node.orelse) == 1
            and isinstance(node.orelse[0], ast.If)
        ):
            nested_if.add(id(node.orelse[0]))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            sym.definitions.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Name):
                        sym.definitions.add(sub.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            sym.definitions.add(node.target.id)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    sym.table_keys.add(key.value)
                    if isinstance(value, ast.List | ast.Tuple | ast.Set):
                        sym.dict_arity[key.value] = len(value.elts)
                    elif isinstance(value, ast.Dict):
                        sym.dict_arity[key.value] = sum(
                            len(v.elts) if isinstance(v, ast.List | ast.Tuple | ast.Set) else 1
                            for v in value.values
                        )
        elif isinstance(node, ast.While | ast.For | ast.AsyncFor):
            sym.loop_count += 1
        elif isinstance(node, ast.If) and id(node) not in nested_if:
            names, branches = _chain_arity(node)
            if branches >= 2:
                sym.if_chains.append((names, branches))

        if isinstance(node, ast.Name):
            sym.mentions.add(node.id)
        elif isinstance(node, ast.Attribute):
            sym.mentions.add(node.attr)
        elif isinstance(node, ast.arg):
            sym.mentions.add(node.arg)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            sym.mentions.add(node.value)
    sym.mentions |= sym.definitions
    return sym


def detect_style(diagram: dict, code: str, language: str = "python") -> str:
    """`"structured"`, `"tabular"` or `"none"` - which representation the code chose.

    Decided by which site realises more of the graph's nodes, not by looking for a magic variable
    name, so a generator that calls its table something else is still recognised.
    """
    sym = code_symbols(code, language)
    if sym.parse_error:
        return "none"
    amap = node_anchor_map(diagram)
    by_def = sum(1 for a in amap.values() if a & sym.definitions)
    by_key = sum(1 for a in amap.values() if a & sym.table_keys)
    if by_key > by_def:
        return "tabular"
    if by_def or by_key:
        return "structured"
    return "none"


# ------------------------------------------------------------------------------------------
# The sub-metrics
# ------------------------------------------------------------------------------------------


def _expected_nodes(diagram: dict) -> list[dict]:
    return [
        n
        for n in diagram.get("nodes", [])
        if str(n.get("semantic_role") or "unknown") not in NON_EMITTED_ROLES
    ]


def _out_degree(diagram: dict) -> Counter:
    deg: Counter = Counter()
    for edge in diagram.get("edges", []):
        if edge.get("src") is not None and edge.get("dst") is not None:
            deg[str(edge["src"])] += 1
    return deg


def gateways(diagram: dict) -> list[tuple[str, int]]:
    """`[(node id, out-degree)]` for every branch point: out-degree >= 2, or a branching role."""
    deg = _out_degree(diagram)
    out = []
    for node in _expected_nodes(diagram):
        nid = str(node["id"])
        role = str(node.get("semantic_role") or "unknown")
        degree = deg.get(nid, 0)
        if degree >= 2 or (role in GATEWAY_ROLES and degree >= 1):
            out.append((nid, degree))
    return out


def ir_loops(diagram: dict) -> tuple[int, int]:
    """`(reducible loops, total loops)` from `src.assemble.loops` - Tarjan, reused not rewritten.

    Imported lazily and defensively: a tree without the assemble package still scores the other
    two sub-metrics, with `loop_recall` reported as undefined rather than as zero.
    """
    try:
        from src.assemble.loops import loops as find_loops
        from src.ir.model import Diagram
    except Exception:  # pragma: no cover - partially-checked-out tree
        return (0, 0)
    try:
        found = find_loops(Diagram.from_dict(diagram))
    except Exception:
        return (0, 0)
    return (sum(1 for lp in found if lp.reducible), len(found))


def node_correspondence(diagram: dict, sym: CodeSymbols) -> dict[str, float]:
    """Recall/precision/F1 of IR nodes realised as definitions or table keys."""
    expected = _expected_nodes(diagram)
    sites = sym.definition_sites
    hit = [n for n in expected if anchors(n) & sites]
    recall = len(hit) / len(expected) if expected else 1.0

    amap = node_anchor_map(diagram)
    every_anchor: set[str] = set().union(*amap.values()) if amap else set()
    attributable = {s for s in sites if s in every_anchor}
    raw_denominator = len(sites)
    allow_denominator = len({s for s in sites if s not in SCAFFOLD})
    precision_raw = len(attributable) / raw_denominator if raw_denominator else 1.0
    precision = len(attributable) / allow_denominator if allow_denominator else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "node_recall": recall,
        "node_precision": precision,
        "node_precision_raw": precision_raw,
        "node_f1": f1,
        "nodes_expected": float(len(expected)),
        "nodes_realised": float(len(hit)),
    }


def branch_correspondence(diagram: dict, sym: CodeSymbols) -> dict[str, float]:
    """Per gateway: the arity the code expresses vs the out-degree drawn.

    Either representation counts, and the better of the two wins:
      structured - an `if`/`elif` chain whose tests mention one of the gateway's anchors;
      tabular    - a dict-literal entry keyed by one of its anchors, arity = len(value).
    Scored `min/max` so an over-branching generator loses as much as an under-branching one; a
    gateway the code never branches on at all scores 0.
    """
    gws = gateways(diagram)
    if not gws:
        return {"branch_score": float("nan"), "gateway_recall": float("nan"), "gateways": 0.0}
    amap = node_anchor_map(diagram)
    scores: list[float] = []
    found = 0
    for nid, degree in gws:
        anc = amap.get(nid, set())
        arity = 0
        for names, branches in sym.if_chains:
            if names & anc:
                arity = max(arity, branches)
        for key, value_arity in sym.dict_arity.items():
            if key in anc:
                arity = max(arity, value_arity)
        if arity:
            found += 1
        want = max(degree, 1)
        scores.append(min(arity, want) / max(arity, want) if arity else 0.0)
    return {
        "branch_score": sum(scores) / len(scores),
        "gateway_recall": found / len(gws),
        "gateways": float(len(gws)),
    }


def loop_correspondence(diagram: dict, sym: CodeSymbols, style: str) -> dict[str, float]:
    """Cycles the IR marks vs loops the code writes, judged per representation."""
    reducible, total = ir_loops(diagram)
    if total == 0:
        return {
            "loop_recall": float("nan"),
            "loops_expected": 0.0,
            "loops_reducible": 0.0,
            "loops_in_code": float(sym.loop_count),
        }
    if style == "tabular":
        # One driver walks every cycle in the table; demanding n loops demands another program.
        recall = 1.0 if sym.loop_count >= 1 else 0.0
    else:
        want = max(reducible, 1)
        recall = min(sym.loop_count, want) / want
    return {
        "loop_recall": recall,
        "loops_expected": float(total),
        "loops_reducible": float(reducible),
        "loops_in_code": float(sym.loop_count),
    }


def structural_fidelity(diagram: dict, code: str, language: str = "python") -> dict[str, float]:
    """The headline metric plus every component that produced it.

    `structural` is the unweighted mean of the sub-metrics *defined* for this diagram: a graph
    with no gateway contributes no branch term rather than a free 1.0. `nan` marks an undefined
    component so a corpus mean can skip it instead of averaging in a fiction.
    """
    sym = code_symbols(code, language)
    style = detect_style(diagram, code, language)
    out: dict[str, float] = {"parsed": 0.0 if sym.parse_error else 1.0}
    out["style_tabular"] = 1.0 if style == "tabular" else 0.0
    out.update(node_correspondence(diagram, sym))
    out.update(branch_correspondence(diagram, sym))
    out.update(loop_correspondence(diagram, sym, style))
    live = [p for p in (out["node_f1"], out["branch_score"], out["loop_recall"]) if p == p]
    out["structural"] = sum(live) / len(live) if live else 0.0
    return out


def structural_detail(diagram: dict, code: str, language: str = "python") -> dict[str, Any]:
    """`structural_fidelity` plus the strings behind it - which nodes missed, which invented."""
    sym = code_symbols(code, language)
    amap = node_anchor_map(diagram)
    sites = sym.definition_sites
    every: set[str] = set().union(*amap.values()) if amap else set()
    return {
        "scores": structural_fidelity(diagram, code, language),
        "style": detect_style(diagram, code, language),
        "parse_error": sym.parse_error,
        "missing_nodes": sorted(
            str(n["id"]) for n in _expected_nodes(diagram) if not anchors(n) & sites
        ),
        "unattributable_definitions": sorted(
            s for s in sites if s not in every and s not in SCAFFOLD
        ),
        "gateways": [nid for nid, _ in gateways(diagram)],
    }


# ------------------------------------------------------------------------------------------
# The SQL branch: "do generated tables match the IR?"
# ------------------------------------------------------------------------------------------

_CREATE_TABLE = re.compile(r"create\s+table\s+(?:if\s+not\s+exists\s+)?[`\"\[]?(\w+)", re.I)
_REFERENCES = re.compile(r"references\s+[`\"\[]?(\w+)", re.I)


def sql_table_fidelity(diagram: dict, sql: str) -> dict[str, float]:
    """Tables and foreign keys in DDL vs entities and relationships in the IR.

    **Regex, not a parser.** There is no SQL AST in this environment and adding a dependency for
    a two-pattern job was rejected; the consequence is that a `CREATE TABLE` inside a string
    literal or a block comment counts, and a quoted identifier containing a space does not. Both
    are honest limitations of a line-level rule rather than accidents. No ER corpus has reached
    target-generation yet, so this arm is **untested against real DDL** - unlike the Python arm,
    it carries no measured numbers.
    """
    fn = _slug()
    entities = [
        n
        for n in diagram.get("nodes", [])
        if str(n.get("semantic_role") or "unknown") in ENTITY_ROLES
    ]
    want = {fn(n.get("text") or "", "table").lower() for n in entities}
    got = {m.lower() for m in _CREATE_TABLE.findall(sql)}
    fks = {m.lower() for m in _REFERENCES.findall(sql)}
    edges = sum(
        1 for e in diagram.get("edges", []) if e.get("src") is not None and e.get("dst") is not None
    )
    return {
        "table_recall": len(want & got) / len(want) if want else float("nan"),
        "table_precision": len(want & got) / len(got) if got else float("nan"),
        "tables_expected": float(len(want)),
        "tables_found": float(len(got)),
        "fk_targets_resolved": len(fks & got) / len(fks) if fks else float("nan"),
        "fk_count": float(len(fks)),
        "edges_expected": float(edges),
    }


# ------------------------------------------------------------------------------------------
# Corruptions: code whose damage we know, so the metric can be asked whether it noticed
# ------------------------------------------------------------------------------------------


def corrupt(code: str, kind: str, rng: random.Random) -> str | None:
    """Damage `code` in one named way, or return None when it offers no such site.

    These are the ground truth for "does the metric detect what it claims": each corruption
    changes exactly one structural fact, and a metric that does not move on it is blind to that
    fact - which `shuffle_labels` demonstrates, by design.
    """
    lines = code.splitlines()
    if kind == "drop_node":
        idx = [i for i, ln in enumerate(lines) if ln.startswith("def ") and "(state" in ln]
        idx = idx or [i for i, ln in enumerate(lines) if ln.startswith("def ")]
        if not idx:
            return None
        start = rng.choice(idx)
        end = start
        while end + 1 < len(lines) and (
            not lines[end + 1] or lines[end + 1].startswith((" ", ")"))
        ):
            end += 1
        name = lines[start][4:].split("(")[0]
        kept = lines[:start] + lines[end + 1 :]
        return "\n".join(ln for ln in kept if f": {name}," not in ln) + "\n"
    if kind == "drop_branch":
        idx = [i for i, ln in enumerate(lines) if ln.strip().startswith("'") and "), (" in ln]
        if not idx:
            return None
        i = rng.choice(idx)
        lines[i] = lines[i][: lines[i].rindex("), (")] + ")],"
        return "\n".join(lines) + "\n"
    if kind == "invent_node":
        anchor = next((i for i, ln in enumerate(lines) if ln.startswith("TRANSITIONS")), None)
        if anchor is None:
            return None
        block = [
            "def phantom_step(state: dict) -> dict:",
            "    'process: phantom'",
            "    return state",
            "",
            "",
        ]
        lines = lines[:anchor] + block + lines[anchor:]
        for i, ln in enumerate(lines):
            if ln.startswith("STEPS = {"):
                lines.insert(i + 1, "    'phantom_node': phantom_step,")
                break
        else:
            return None
        return "\n".join(lines) + "\n"
    if kind == "rename_table":
        if "TRANSITIONS" not in code:
            return None
        return code.replace("TRANSITIONS", "EDGE_MAP")
    if kind == "drop_loop":
        idx = [i for i, ln in enumerate(lines) if ln.strip().startswith(("for ", "while "))]
        if not idx:
            return None
        i = rng.choice(idx)
        indent = len(lines[i]) - len(lines[i].lstrip())
        end = i
        while end + 1 < len(lines) and (
            not lines[end + 1].strip()
            or len(lines[end + 1]) - len(lines[end + 1].lstrip()) > indent
        ):
            end += 1
        return "\n".join(lines[:i] + lines[end + 1 :]) + "\n"
    if kind == "shuffle_labels":
        labels = re.findall(r"\('([^']*)', '", code)
        if len(set(labels)) < 2:
            return None
        pool = labels[:]
        rng.shuffle(pool)
        it = iter(pool)
        return re.sub(r"\('([^']*)', '", lambda _m: f"('{next(it)}', '", code)
    raise ValueError(f"unknown corruption {kind!r}")


CORRUPTIONS = (
    "drop_branch",
    "drop_node",
    "invent_node",
    "rename_table",
    "drop_loop",
    "shuffle_labels",
)

#: Which sub-metric each corruption is expected to move. Named here so the study cannot quietly
#: be rescored against whichever number happened to react.
WATCHED = {
    "drop_branch": "branch_score",
    "drop_node": "node_recall",
    "invent_node": "node_precision_raw",
    "rename_table": "structural",
    "drop_loop": "loop_recall",
    "shuffle_labels": "structural",
}


# ------------------------------------------------------------------------------------------
# Corpus sweep
# ------------------------------------------------------------------------------------------


def reference_pairs(limit: int | None = None, language: str = "python") -> list[tuple[dict, str]]:
    """`(ir dict, code)` for every reference target of `language`, or `[]` if none are built."""
    if not INDEX.is_file():
        return []
    rows = [r for r in json.loads(INDEX.read_text(encoding="utf-8")) if r["language"] == language]
    out: list[tuple[dict, str]] = []
    for row in rows[:limit] if limit else rows:
        ir_path, code_path = ROOT / row["ir"], ROOT / row["target"]
        if ir_path.is_file() and code_path.is_file():
            out.append(
                (
                    json.loads(ir_path.read_text(encoding="utf-8")),
                    code_path.read_text(encoding="utf-8"),
                )
            )
    return out


def _mean(values: list[float]) -> float:
    live = [v for v in values if v == v]
    return sum(live) / len(live) if live else float("nan")


def corpus_study(limit: int | None = None) -> dict[str, Any]:
    """Score every reference Python target against its own IR. The floor, not the achievement."""
    rows = [structural_fidelity(d, c) for d, c in reference_pairs(limit)]
    keys = [
        "parsed",
        "style_tabular",
        "node_recall",
        "node_precision",
        "node_precision_raw",
        "node_f1",
        "branch_score",
        "gateway_recall",
        "loop_recall",
        "structural",
    ]
    return {
        "pairs": len(rows),
        "means": {k: _mean([r.get(k, float("nan")) for r in rows]) for k in keys},
    }


def corruption_study(limit: int = 300, seed: int = 0) -> dict[str, Any]:
    """For each corruption: how often the metric noticed, and by how much."""
    rng = random.Random(seed)
    pairs = reference_pairs(limit)
    out: dict[str, Any] = {}
    for kind in CORRUPTIONS:
        key = WATCHED[kind]
        detected, applicable = 0, 0
        drops: list[float] = []
        for diagram, code in pairs:
            damaged = corrupt(code, kind, rng)
            if damaged is None:
                continue
            before = structural_fidelity(diagram, code).get(key, float("nan"))
            after = structural_fidelity(diagram, damaged).get(key, float("nan"))
            if before != before or after != after:
                continue
            applicable += 1
            if after < before - 1e-9:
                detected += 1
                drops.append(before - after)
        out[kind] = {
            "watched": key,
            "applicable": applicable,
            "detected": detected,
            "detection_rate": detected / applicable if applicable else float("nan"),
            "mean_drop_when_detected": _mean(drops) if drops else 0.0,
        }
    return out


def run(limit: int | None = None, corrupt_limit: int = 300) -> dict[str, Any]:
    return {
        "corpus": corpus_study(limit),
        "corruptions": corruption_study(corrupt_limit),
        "note": "reference targets only; model-output numbers await the 12.2 fine-tune",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corrupt-limit", type=int, default=300)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    report = run(args.limit, args.corrupt_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

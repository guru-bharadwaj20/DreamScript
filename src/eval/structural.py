"""Phase 12.3.4 - structural fidelity: do the generated branches / states / tables match the IR?

    python -m src.eval.structural --study          # reference sweep + corruption detection

    from src.eval.structural import fidelity, hallucination
    fidelity(record, code)       # {"structural": 0..1, sub-metrics..., "style": ...}
    hallucination(record, code)  # {"dropped": [...], "invented": [...], rates}   (12.3.8 helper)

`record` is a 12.1.1 pair (only `diagram_type`, `language` and `ir_text` are read) and `code` is a
candidate program - a reference target or a model output. **Everything expected is derived from
`ir_text`, the text the model was shown**, never from the IR file's `attrs`, so the metric cannot
reward a model for reproducing information it could not see.

## The correspondence rule, per language

A structure metric is undefined until the rule deciding which code construct *is* which IR
element is named. Each rule below is the convention the 12.1.6 targets follow, because that is
the convention a fine-tuned model is trained to reproduce; each is exact-match on 12.1.6's own
identifier function (`targets._ident` / `_sql_ident`) - a fuzzy rule lets `process` match nine
boxes on one page, which is the false-positive mode 12.3.8 has to count.

    flowchart -> Python (AST)
        node_f1       distinct called names vs node anchors (`slug(text)`, `read_/write_` + slug
                      for io); start/end/fork/join/event/container are comments or `return`, and
                      are not anchors
        branch_score  if/while tests that call something vs IR nodes with >= 2 distinct
                      successors, scored min/max so over- and under-branching both lose
        loop_score    `while` loops vs DFS back-edge headers (structured style only)
        edge_f1       `state == "a"` ... `state = "b"` transitions vs IR edges (dispatch style
                      only - the structured form has no explicit edges to compare)
    state_machine -> Python class (AST of class-level literals)
        state_f1, transition_f1 (src, symbol, dst), initial (0/1), accepting_f1
    er_diagram -> SQL (executed on in-memory SQLite, schema read back with PRAGMAs)
        table_f1, column_f1 (table.column from `attribute` nodes), relationship_f1 (unordered
        table pairs: a foreign key, or two FKs from one junction table)
    wireframe -> React (tag scan of the JSX)
        widget_f1 (button / img / input multiset), text_f1 (label and button strings multiset)
    circuit -> SPICE (card parse)
        part_f1 (ref, value), connection_f1 (ref, +net, -net) when the IR has `wire` nets

`structural` is the unweighted mean of the sub-metrics that are *defined* (`nan` otherwise - a
graph with no branch point contributes no free 1.0). Code that does not parse scores 0.

## What it cannot see, measured rather than assumed

Structural fidelity measures shape. `corruption_study` includes controls that change meaning
without changing shape - swapping the `yes`/`no` branches of an `if`, or swapping the targets of
two state transitions that share a symbol set - and reports how often the metric notices. It is a
complement to 12.3.3's functional tests, not a substitute; see the 12.3.4 row for the numbers.

## Replaced

The inherited draft scored the Phase 2.2.6 `src/ir/targets.py` references (tabular Python and
HTML), which are not the 12.1.x training targets, had no SQL, React or SPICE arm beyond a regex,
and its corruption table could not be reproduced against the 12.1 pairs; its numbers are not
carried forward.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from typing import Any

from src.codegen import serialise
from src.codegen.targets import _ident, _sql_ident

NAN = float("nan")
_COMMENT_ROLES = frozenset({"start", "end", "fork", "join", "event", "container"})
_BUILTINS = frozenset(dir(builtins))


def _f1(expected, observed) -> float:
    """F1 over multisets (Counters) or sets. Both empty -> nan (undefined, not perfect)."""
    exp = expected if isinstance(expected, Counter) else Counter(set(expected))
    obs = observed if isinstance(observed, Counter) else Counter(set(observed))
    if not exp and not obs:
        return NAN
    hit = sum((exp & obs).values())
    if hit == 0:
        return 0.0
    p, r = hit / sum(obs.values()), hit / sum(exp.values())
    return 2 * p * r / (p + r)


def _ratio(expected: int, observed: int) -> float:
    if expected == 0 and observed == 0:
        return NAN
    return min(expected, observed) / max(expected, observed)


def _mean(values) -> float:
    live = [v for v in values if v == v]
    return sum(live) / len(live) if live else NAN


def _ir(record: dict) -> dict:
    return serialise.parse(record["ir_text"])


# ------------------------------------------------------------------------------------------------
# flowchart
# ------------------------------------------------------------------------------------------------


def _flow_anchor(node: dict) -> set[str]:
    text = node.get("text") or ""
    role = node.get("semantic_role") or ""
    if role in _COMMENT_ROLES:
        return set()
    anchors = {_ident(text, "step")}
    if role == "decision":
        # a decision with one successor is emitted as a statement, so both spellings count
        anchors.add(_ident(text.rstrip("?"), "condition"))
    if role == "io":
        slug = _ident(text, "value")
        anchors |= {f"read_{slug}", f"write_{slug}"}
        verb, _, rest = text.partition(" ")
        if verb in ("read", "write") and rest:
            anchors.add(f"{verb}_{_ident(rest, 'value')}")
    return anchors


def _successors(ir: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {n["id"]: set() for n in ir["nodes"]}
    for edge in ir["edges"]:
        if edge["src"] in out and edge["dst"] in out:
            out[edge["src"]].add(edge["dst"])
    return out


def _loop_headers(ir: dict) -> set[str]:
    from src.parse.sequences import traversal

    doc = {"nodes": [dict(n, bbox=None) for n in ir["nodes"]], "edges": ir["edges"]}
    return {dst for _src, dst in traversal(doc)[1]}


def _flowchart(record: dict, code: str) -> dict:
    ir = _ir(record)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"parsed": 0.0, "detail": f"syntax: {exc.msg}"}
    anchors = [a for n in ir["nodes"] if (a := _flow_anchor(n))]
    calls, branch_tests, loops, transitions = set(), 0, 0, set()
    dispatch = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in _BUILTINS
        ):
            calls.add(node.func.id)
        if isinstance(node, ast.If | ast.While):
            if any(isinstance(sub, ast.Call) for sub in ast.walk(node.test)):
                branch_tests += 1
            if isinstance(node, ast.While) and not (
                isinstance(node.test, ast.Constant) and node.test.value is True and _is_driver(node)
            ):
                loops += 1
            src = _state_compare(node.test)
            if src is not None:
                dispatch = True
                for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    dst = _state_assign(sub)
                    if dst is not None:
                        transitions.add((src, dst))
    all_anchors = set().union(*anchors) if anchors else set()
    realised = [a for a in anchors if a & calls]
    precision_hits = {c for c in calls if c in all_anchors}
    recall = len(realised) / len(anchors) if anchors else NAN
    precision = len(precision_hits) / len(calls) if calls else NAN
    if recall != recall and precision != precision:
        node_f1 = NAN
    elif not recall or not precision or recall != recall or precision != precision:
        node_f1 = 0.0
    else:
        node_f1 = 2 * recall * precision / (recall + precision)
    succ = _successors(ir)
    gateways = sum(1 for targets in succ.values() if len(targets) >= 2)
    edges = {(e["src"], e["dst"]) for e in ir["edges"]}
    out = {
        "parsed": 1.0,
        "style": "dispatch" if dispatch else "structured",
        "node_f1": node_f1,
        "branch_score": _ratio(gateways, branch_tests),
        "loop_score": NAN if dispatch else _ratio(len(_loop_headers(ir)), loops),
        "edge_f1": _f1(edges, transitions) if dispatch else NAN,
    }
    return out


def _is_driver(node: ast.While) -> bool:
    return any(_state_compare(sub.test) is not None for sub in node.body if isinstance(sub, ast.If))


def _state_compare(test: ast.AST) -> str | None:
    if (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "state"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and isinstance(test.comparators[0].value, str)
    ):
        return test.comparators[0].value
    return None


def _state_assign(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "state"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        return node.value.value
    return None


# ------------------------------------------------------------------------------------------------
# state machine
# ------------------------------------------------------------------------------------------------


def _state_names(ir: dict) -> dict[str, str]:
    names, taken = {}, set()
    for node in ir["nodes"]:
        name = _ident(node.get("text") or node["id"], "q")
        while name in taken:
            name = f"{name}_b"
        taken.add(name)
        names[node["id"]] = name
    return names


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _state_machine(record: dict, code: str) -> dict:
    ir = _ir(record)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"parsed": 0.0, "detail": f"syntax: {exc.msg}"}
    names = _state_names(ir)
    want_states = set(names.values())
    want_trans = {
        (names[e["src"]], _ident(e["label"] or "epsilon", "sym"), names[e["dst"]])
        for e in ir["edges"]
        if e["src"] in names and e["dst"] in names
    }
    want_initial = {names[n["id"]] for n in ir["nodes"] if n["semantic_role"] == "initial-state"}
    want_accept = {names[n["id"]] for n in ir["nodes"] if n["semantic_role"] == "final-state"}

    values: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = _literal(node.value)
            if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "set":
                values[target.id] = set()
    states = set(values.get("STATES") or [])
    got_trans = set()
    table = values.get("TABLE")
    if isinstance(table, dict):  # the table `step` executes is the behaviour; prefer it
        for src, row in table.items():
            if isinstance(row, dict):
                got_trans |= {(src, sym, dst) for sym, dst in row.items()}
    else:
        for item in values.get("TRANSITIONS") or []:
            if isinstance(item, dict) and {"trigger", "source", "dest"} <= set(item):
                got_trans.add((item["source"], item["trigger"], item["dest"]))
    initial = values.get("INITIAL")
    accepting = values.get("ACCEPTING") or set()
    return {
        "parsed": 1.0,
        "style": "class",
        "state_f1": _f1(want_states, states),
        "transition_f1": _f1(want_trans, got_trans),
        "initial": NAN if not want_initial else float(initial in want_initial),
        "accepting_f1": _f1(
            want_accept, set(accepting) if isinstance(accepting, set | list) else set()
        ),
    }


# ------------------------------------------------------------------------------------------------
# ER -> SQL
# ------------------------------------------------------------------------------------------------


def _er_expected(ir: dict) -> tuple[set, set, set]:
    nodes = {n["id"]: n for n in ir["nodes"]}
    entities = [n for n in ir["nodes"] if n["semantic_role"] == "entity"]
    table_of, used = {}, set()
    for node in entities:
        name = _sql_ident(node["text"] or node["id"], "entity")
        while name in used:
            name = f"{name}_x"
        used.add(name)
        table_of[node["id"]] = name
    columns = set()
    for edge in ir["edges"]:
        src, dst = nodes.get(edge["src"]), nodes.get(edge["dst"])
        if src and dst and edge["src"] in table_of and dst["semantic_role"] == "attribute":
            columns.add((table_of[edge["src"]], _sql_ident(dst["text"].split(":")[0], "col")))
    relations = set()
    for node in ir["nodes"]:
        if node["semantic_role"] != "relationship":
            continue
        left = [e["src"] for e in ir["edges"] if e["dst"] == node["id"] and e["src"] in table_of]
        right = [e["dst"] for e in ir["edges"] if e["src"] == node["id"] and e["dst"] in table_of]
        if left and right:
            relations.add(frozenset((table_of[left[0]], table_of[right[0]])))
    return set(table_of.values()), columns, relations


def _er(record: dict, code: str) -> dict:
    from src.eval.sql import split_statements

    ir = _ir(record)
    con = sqlite3.connect(":memory:")
    try:
        for statement in split_statements(code):
            try:
                con.execute(statement)
            except sqlite3.Error:
                continue
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables:
            return {"parsed": 0.0, "detail": "no table was created"}
        columns, fks, relations = set(), {}, set()
        for table in tables:
            for row in con.execute(f'PRAGMA table_info("{table}")'):
                columns.add((table, row[1]))
            fks[table] = [row[2] for row in con.execute(f'PRAGMA foreign_key_list("{table}")')]
        for table, parents in fks.items():
            pk = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")') if r[5]]
            if len(parents) >= 2 and len(pk) >= 2:
                relations.add(frozenset(parents[:2]))
            else:
                relations |= {frozenset((table, p)) for p in parents}
    finally:
        con.close()
    want_tables, want_columns, want_relations = _er_expected(ir)
    junctions = {t for t, parents in fks.items() if len(parents) >= 2}
    entity_tables = tables - (junctions - want_tables)
    columns = {c for c in columns if c[0] in entity_tables and not c[1].endswith("_id")}
    columns = {c for c in columns if "_fk_" not in c[1]}
    return {
        "parsed": 1.0,
        "style": "ddl",
        "table_f1": _f1(want_tables, entity_tables),
        "column_f1": _f1(want_columns, columns),
        "relationship_f1": _f1(want_relations, relations),
    }


# ------------------------------------------------------------------------------------------------
# wireframe -> React
# ------------------------------------------------------------------------------------------------

_WIDGET_OF_ROLE = {"ui-button": "button", "ui-image": "img", "ui-input": "input"}
_TAG = re.compile(r"<\s*([A-Za-z][\w.]*)")
_STRING_CHILD = re.compile(r'\{\s*"((?:[^"\\]|\\.)*)"\s*\}')


def _wireframe(record: dict, code: str) -> dict:
    ir = _ir(record)
    from src.codegen.quality import _balanced_brackets, _mask_literals

    if not _balanced_brackets(_mask_literals(code))[0]:
        return {"parsed": 0.0, "detail": "unbalanced brackets"}
    want_widgets = Counter(
        _WIDGET_OF_ROLE[n["semantic_role"]]
        for n in ir["nodes"]
        if n["semantic_role"] in _WIDGET_OF_ROLE
    )
    want_text = Counter(
        n["text"]
        for n in ir["nodes"]
        if n["semantic_role"] in ("ui-label", "ui-button") and n["text"]
    )
    tags = Counter(t.lower() for t in _TAG.findall(_mask_literals(code)))
    got_widgets = Counter({k: tags[k] for k in ("button", "img", "input") if tags[k]})
    got_text = Counter()
    for match in _STRING_CHILD.finditer(code):
        try:
            got_text[json.loads(f'"{match.group(1)}"')] += 1
        except json.JSONDecodeError:
            continue
    return {
        "parsed": 1.0,
        "style": "jsx",
        "widget_f1": _f1(want_widgets, got_widgets),
        "text_f1": _f1(want_text, got_text),
    }


# ------------------------------------------------------------------------------------------------
# circuit -> SPICE
# ------------------------------------------------------------------------------------------------


def _circuit(record: dict, code: str) -> dict:
    ir = _ir(record)
    nodes = {n["id"]: n for n in ir["nodes"]}
    parts, connections = Counter(), set()
    nets: dict[str, dict[str, str]] = {}
    for edge in ir["edges"]:
        dst = nodes.get(edge["dst"])
        if dst and dst["semantic_role"] == "wire" and edge["label"] in ("+", "-"):
            nets.setdefault(edge["src"], {})[edge["label"]] = dst["text"]
    for node in ir["nodes"]:
        if node["semantic_role"] == "wire":
            continue
        words = (node["text"] or "").split()
        if not words:
            continue
        ref = words[0].upper()
        value = words[1] if len(words) > 1 else ""
        parts[(ref, value)] += 1
        if node["id"] in nets and len(nets[node["id"]]) == 2:
            connections.add((ref, nets[node["id"]]["+"], nets[node["id"]]["-"]))
    got_parts, got_conn = Counter(), set()
    cards = 0
    for line in code.splitlines():
        tokens = line.split()
        if not tokens or tokens[0][0] in "*." or not tokens[0][0].isalpha():
            continue
        cards += 1
        ref = tokens[0].upper()
        rest = [t for t in tokens[3:] if t.upper() != "DC"]
        got_parts[(ref, rest[0] if rest else "")] += 1
        if len(tokens) >= 3:
            got_conn.add((ref, tokens[1], tokens[2]))
    if cards == 0:
        return {"parsed": 0.0, "detail": "no device cards"}
    return {
        "parsed": 1.0,
        "style": "netlist",
        "part_f1": _f1(parts, got_parts),
        "connection_f1": _f1(connections, got_conn) if connections else NAN,
    }


_ARMS = {
    "flowchart": _flowchart,
    "state_machine": _state_machine,
    "er_diagram": _er,
    "wireframe": _wireframe,
    "circuit": _circuit,
}
SUBMETRICS = {
    "flowchart": ("node_f1", "branch_score", "loop_score", "edge_f1"),
    "state_machine": ("state_f1", "transition_f1", "initial", "accepting_f1"),
    "er_diagram": ("table_f1", "column_f1", "relationship_f1"),
    "wireframe": ("widget_f1", "text_f1"),
    "circuit": ("part_f1", "connection_f1"),
}


def fidelity(record: dict, code: str) -> dict:
    """Sub-metrics plus `structural`, the mean of the defined ones (0 when the code has no form)."""
    arm = _ARMS.get(record["diagram_type"])
    if arm is None:
        return {"parsed": NAN, "structural": NAN, "detail": "no arm for this diagram type"}
    out = arm(record, code)
    if not out.get("parsed"):
        out["structural"] = 0.0
        return out
    out["structural"] = _mean(out.get(k, NAN) for k in SUBMETRICS[record["diagram_type"]])
    return out


# ------------------------------------------------------------------------------------------------
# 12.3.8 helper: which IR elements were dropped, which code elements were invented
# ------------------------------------------------------------------------------------------------


def elements(record: dict, code: str) -> tuple[Counter, Counter]:
    """(expected, observed) node-level elements for the hallucination audit, per diagram type."""
    ir, kind = _ir(record), record["diagram_type"]
    if kind == "flowchart":
        anchors = [a for n in ir["nodes"] if (a := _flow_anchor(n))]
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return Counter(min(a) for a in anchors), Counter()
        calls = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id not in _BUILTINS
        }
        expected = Counter({(sorted(a & calls) or [min(a)])[0] for a in anchors})
        return expected, Counter(calls)
    if kind == "state_machine":
        want = Counter(set(_state_names(ir).values()))
        try:
            values = {
                n.targets[0].id: _literal(n.value)
                for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            }
        except SyntaxError:
            return want, Counter()
        return want, Counter(set(values.get("STATES") or []))
    if kind == "er_diagram":
        tables, columns, _rel = _er_expected(ir)
        want = Counter(set(tables) | {f"{t}.{c}" for t, c in columns})
        scored = _er(record, code)
        if not scored.get("parsed"):
            return want, Counter()
        con = sqlite3.connect(":memory:")
        from src.eval.sql import split_statements

        for statement in split_statements(code):
            with contextlib.suppress(sqlite3.Error):
                con.execute(statement)
        got = set()
        for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            fk_count = len(list(con.execute(f'PRAGMA foreign_key_list("{table}")')))
            if fk_count >= 2 and table not in tables:
                continue
            got.add(table)
            for row in con.execute(f'PRAGMA table_info("{table}")'):
                if not row[1].endswith("_id") and "_fk_" not in row[1]:
                    got.add(f"{table}.{row[1]}")
        con.close()
        return want, Counter(got)
    if kind == "wireframe":
        want = Counter(
            f"text:{n['text']}"
            for n in ir["nodes"]
            if n["semantic_role"] in ("ui-label", "ui-button") and n["text"]
        )
        want += Counter(
            f"widget:{_WIDGET_OF_ROLE[n['semantic_role']]}"
            for n in ir["nodes"]
            if n["semantic_role"] in _WIDGET_OF_ROLE
        )
        from src.codegen.quality import _mask_literals

        tags = Counter(t.lower() for t in _TAG.findall(_mask_literals(code)))
        got = Counter({f"widget:{k}": tags[k] for k in ("button", "img", "input") if tags[k]})
        for match in _STRING_CHILD.finditer(code):
            try:
                got[f"text:{json.loads(chr(34) + match.group(1) + chr(34))}"] += 1
            except json.JSONDecodeError:
                continue
        return want, got
    if kind == "circuit":
        want = Counter(
            (n["text"] or "").split()[0].upper()
            for n in ir["nodes"]
            if n["semantic_role"] != "wire" and (n["text"] or "").split()
        )
        got = Counter(
            line.split()[0].upper()
            for line in code.splitlines()
            if line.split() and line.split()[0][0].isalpha()
        )
        return want, got
    return Counter(), Counter()


def hallucination(record: dict, code: str) -> dict:
    """Dropped (in the IR, not in the code) and invented (in the code, not in the IR) elements."""
    expected, observed = elements(record, code)
    dropped = expected - observed
    invented = observed - expected
    return {
        "expected": sum(expected.values()),
        "observed": sum(observed.values()),
        "dropped": sorted(map(str, dropped.elements())),
        "invented": sorted(map(str, invented.elements())),
        "dropped_rate": sum(dropped.values()) / max(1, sum(expected.values())),
        "invented_rate": sum(invented.values()) / max(1, sum(observed.values())),
    }


# ------------------------------------------------------------------------------------------------
# validation study: references, then code whose damage is known
# ------------------------------------------------------------------------------------------------


def corrupt(record: dict, code: str, kind: str, rng: random.Random) -> str | None:
    """One named defect, or None when the code offers no site for it."""
    lines = code.splitlines()
    t = record["diagram_type"]

    def drop(pred) -> str | None:
        idx = [i for i, ln in enumerate(lines) if pred(ln)]
        if not idx:
            return None
        i = rng.choice(idx)
        return "\n".join(lines[:i] + lines[i + 1 :]) + "\n"

    if t == "flowchart":
        if kind == "drop_node":
            return drop(lambda ln: re.match(r"\s+ctx = \w+\(ctx\)$", ln) is not None)
        if kind == "invent_node":
            idx = [i for i, ln in enumerate(lines) if re.match(r"\s+ctx = \w+\(ctx\)$", ln)]
            if not idx:
                return None
            i = rng.choice(idx)
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            return (
                "\n".join(lines[: i + 1] + [f"{indent}ctx = phantom_step(ctx)"] + lines[i + 1 :])
                + "\n"
            )
        if kind == "drop_branch":
            for i, ln in enumerate(lines):
                if re.match(r"\s+if \w+\(ctx\):$", ln):
                    return (
                        "\n".join(
                            lines[:i]
                            + [ln.replace("if ", "if True or ", 1).replace("(ctx):", "_x:")]
                            + lines[i + 1 :]
                        )
                        + "\n"
                    )
            return None
        if kind == "drop_loop":
            for i, ln in enumerate(lines):
                if re.match(r"\s+while \w+\(ctx\):$", ln):
                    return (
                        "\n".join(lines[:i] + [ln.replace("while", "if", 1)] + lines[i + 1 :])
                        + "\n"
                    )
            return None
        if kind == "swap_branches":  # control: meaning changes, shape does not
            for i, ln in enumerate(lines):
                m = re.match(r"(\s+)if (\w+)\(ctx\):$", ln)
                if m:
                    return (
                        "\n".join(
                            lines[:i] + [f"{m.group(1)}if not {m.group(2)}(ctx):"] + lines[i + 1 :]
                        )
                        + "\n"
                    )
            return None
    if t == "state_machine":
        if kind == "drop_node":
            return drop(lambda ln: re.match(r'\s{8}"[^"]+",$', ln) is not None)
        if kind == "invent_node":
            return code.replace("    STATES = [\n", '    STATES = [\n        "phantom",\n', 1)
        if kind == "drop_branch":  # one symbol out of the executed TABLE
            idx = [i for i, ln in enumerate(lines) if re.match(r'\s{8}"[^"]+": \{".+\},$', ln)]
            if not idx:
                return None
            i = rng.choice(idx)
            lines[i] = re.sub(r'\{"[^"]*": "[^"]*"(, )?', "{", lines[i], count=1)
            return "\n".join(lines) + "\n"
        if kind == "swap_branches":  # swap the destinations of two executed TABLE entries
            pattern = re.compile(r'("[^"]*": )("[^"]*")')
            sites = [
                (i, m.start(2), m.group(2))
                for i, ln in enumerate(lines)
                if re.match(r'\s{8}"[^"]+": \{".+\},$', ln)
                for m in pattern.finditer(ln.split(": ", 1)[1])
            ]
            if len(sites) < 2:
                return None
            (ia, _sa, da), (ib, _sb, db) = rng.sample(sites, 2)
            if da == db or ia == ib:
                return None
            head_a, row_a = lines[ia].split(": ", 1)
            head_b, row_b = lines[ib].split(": ", 1)
            lines[ia] = f"{head_a}: {row_a.replace(da, db, 1)}"
            lines[ib] = f"{head_b}: {row_b.replace(db, da, 1)}"
            return "\n".join(lines) + "\n"
    if t == "er_diagram":
        if kind == "drop_node":
            return drop(
                lambda ln: re.match(r"\s{4}\w+ (TEXT|REAL|INTEGER|NUMERIC|BLOB),?$", ln) is not None
                and "PRIMARY" not in ln
            )
        if kind == "invent_node":
            return code + "\nCREATE TABLE phantom (\n    phantom_id INTEGER PRIMARY KEY\n);\n"
        if kind == "drop_branch":
            return drop(lambda ln: ln.startswith("ALTER TABLE"))
    if t == "wireframe":
        if kind == "drop_node":
            return drop(lambda ln: re.match(r'\s+\{".*"\}$', ln) is not None)
        if kind == "invent_node":
            return code.replace(
                '<div className="min-h-screen bg-white p-6">',
                '<div className="min-h-screen bg-white p-6">\n      <button className="x">{"Phantom"}</button>',
                1,
            )
    if t == "circuit":
        if kind == "drop_node":
            return drop(lambda ln: re.match(r"[RCL]\d+ ", ln) is not None)
        if kind == "invent_node":
            return code.replace(".op", "R99 1 0 1k\n.op", 1)
        if kind == "drop_branch":
            idx = [i for i, ln in enumerate(lines) if re.match(r"[RCL]\d+ \S+ \S+ ", ln)]
            if not idx:
                return None
            i = rng.choice(idx)
            parts = lines[i].split()
            parts[2] = "99"
            lines[i] = " ".join(parts)
            return "\n".join(lines) + "\n"
    return None


CORRUPTIONS = ("drop_node", "invent_node", "drop_branch", "drop_loop", "swap_branches")


def study(per_type: int = 300, seed: int = 0) -> dict:
    """Reference scores over real + synthetic pairs, then detection rates per corruption."""
    from src.codegen.pairs import load_pairs

    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for record in load_pairs():
        by_type.setdefault(record["diagram_type"], []).append(record)
    report: dict = {"references": {}, "corruptions": {}}
    for kind, records in sorted(by_type.items()):
        per_source: dict[str, list[dict]] = {}
        for record in records:
            per_source.setdefault(record["source"], []).append(
                fidelity(record, record["target_code"])
            )
        report["references"][kind] = {
            source: {
                "pairs": len(rows),
                **{
                    k: round(_mean(r.get(k, NAN) for r in rows), 4)
                    for k in ("structural", *SUBMETRICS[kind])
                },
                "perfect": sum(r["structural"] >= 0.9999 for r in rows),
                "styles": dict(Counter(r.get("style") for r in rows)),
            }
            for source, rows in sorted(per_source.items())
        }
        sample = rng.sample(records, min(per_type, len(records)))
        for corruption in CORRUPTIONS:
            applied = detected = 0
            drops = []
            hall = 0
            for record in sample:
                damaged = corrupt(record, record["target_code"], corruption, rng)
                if damaged is None or damaged == record["target_code"]:
                    continue
                before = fidelity(record, record["target_code"])["structural"]
                after = fidelity(record, damaged)["structural"]
                applied += 1
                if after < before - 1e-9:
                    detected += 1
                    drops.append(before - after)
                if corruption in ("drop_node", "invent_node"):
                    h = hallucination(record, damaged)
                    base = hallucination(record, record["target_code"])
                    key = "dropped" if corruption == "drop_node" else "invented"
                    hall += len(h[key]) > len(base[key])
            if applied:
                report["corruptions"].setdefault(kind, {})[corruption] = {
                    "applied": applied,
                    "detected": detected,
                    "rate": round(detected / applied, 4),
                    "mean_drop": round(_mean(drops), 4) if drops else 0.0,
                    **(
                        {"hallucination_audit_detected": hall}
                        if corruption in ("drop_node", "invent_node")
                        else {}
                    ),
                }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.3.4 structural fidelity")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--per-type", type=int, default=300)
    args = parser.parse_args(argv)
    if args.study:
        json.dump(study(args.per_type), sys.stdout, indent=2, default=lambda x: None)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Phase 12.1.7 - the filter that makes "100% of targets compile" a postcondition, not a hope.

    python -m src.codegen.quality --self-test
    python -m src.codegen.quality --pairs data/processed/codegen/pairs/train.jsonl

A training pair whose target does not compile teaches the model to emit code that does not
compile. The only way a rate like that is ever 100% is if it is the postcondition of a filter:
`filter_pairs` partitions, and `assert_all_compile` re-runs every check on the survivors from
scratch and raises on a single miss. `src.codegen.pairs` calls both before any shard is written.

## What each language is checked with - the strongest checker available here, not a proxy

    python   `ast.parse` **and** `compile()`. Measured: `return 1` at module level and
             `def f(a, a)` both pass `ast.parse` and are rejected only by `compile()`.
    sql      12.3.7's `src.eval.sql.check`: statement-by-statement execution on in-memory
             SQLite with `PRAGMA foreign_keys=ON`, then every FK's parent table / column /
             uniqueness resolved statically, invented type names rejected, empty schema
             rejected. The inherited draft used `executescript` alone, which 12.3.7 measured
             accepting `REFERENCES nosuch(id)` and `CREATE TABLE t(x BANANA)`; both are now
             rejections in the self-test.
    react    the structural parse below as a fast first gate, then **a real build and render in
             node** (`src.eval.react`: esbuild JSX transform, then `react-dom/server`
             `renderToString` in a `vm` sandbox with a timeout). The inherited draft stopped at
             the structural parse and said so; with node 22 and a pinned esbuild/react install
             that limit no longer applies, and the render stage catches what the parse cannot -
             an undefined identifier, an import of a module that is not there, an infinite loop.
    html     `html.parser` with a strict tag stack - Sketch2Code's source pages (12.1.5) are
             HTML before conversion.
    spice    the structural rules below, then **ngspice 47 in batch mode** (`src.eval.spice`):
             the deck must parse *and* its `.op` operating point must solve. A structurally clean
             deck can be physically unsolvable - two ideal inductors in parallel, or a node that
             only capacitors touch, is a singular DC matrix - and the structural check cannot
             see it.

A checker whose tool is missing (`react.unavailable`, `spice.unavailable`) **rejects**: an
environment without node must not be able to produce a shard that claims to render.

Structural SPICE rules, enforced before the simulator: every non-comment, non-directive line is a
device card with a known device letter and at least that device's node count plus a value;
node names are alphanumeric; ground `0` appears; the deck ends with `.end`; and every node
appears on at least two cards.

## Rejected as approaches

- **Trusting the emitter.** 12.1.6 reports 1.0000 per emitter; that is a property of those
  emitters on those inputs. Every record is re-checked regardless of provenance.
- **Reporting a compile rate instead of filtering.** A 99.4% in a report does not stop the 0.6%
  entering the fine-tune.
- **`exec()`-ing Python targets here.** Compiling proves syntax; executing generated code is the
  sandbox's job (11.2.9, used by 12.3.2's helper in `src.eval.codecheck`).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

#: Languages this filter knows how to check. Anything else is rejected, loudly.
LANGUAGES = ("python", "sql", "react", "html", "spice")

#: HTML elements that never take a closing tag.
VOID_ELEMENTS = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)

#: SPICE device letter -> minimum number of nodes on the card.
SPICE_DEVICE_NODES = {
    "R": 2,
    "L": 2,
    "C": 2,
    "V": 2,
    "I": 2,
    "E": 4,
    "G": 4,
    "F": 2,
    "H": 2,
    "D": 2,
    "Q": 3,
    "J": 3,
    "M": 4,
    "X": 1,
}

_NODE_RE = re.compile(r"^[A-Za-z0-9_]+$")


class QualityError(AssertionError):
    """Raised when a record that survived the filter fails its own check."""


# ----------------------------------------------------------------------------------------
# per-language static checks: each returns (ok, kind, detail)
# ----------------------------------------------------------------------------------------


def check_python(code: str) -> tuple[bool, str, str]:
    """Parse *and* compile. ast.parse accepts things the compiler refuses."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, "python_syntax", f"{exc.msg} (line {exc.lineno})"
    try:
        compile(code, "<target>", "exec")
    except (SyntaxError, ValueError) as exc:
        return False, "python_compile", str(exc)
    return True, "ok", "parsed and compiled"


def check_sql(code: str) -> tuple[bool, str, str]:
    """12.3.7's executing, FK-resolving checker."""
    from src.eval import sql

    result = sql.check(code)
    return bool(result["ok"]), "ok" if result["ok"] else result["kind"], result["detail"]


def _mask_literals(code: str) -> str:
    """Blank out string literals and JS comments so structure can be counted. Length-preserving."""
    out = list(code)
    i, n = 0, len(code)
    quote: str | None = None
    while i < n:
        char = code[i]
        if quote:
            if char == "\\" and i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if char == quote:
                quote = None
            else:
                out[i] = " "
            i += 1
            continue
        if char in "\"'`":
            quote = char
            i += 1
            continue
        if char == "/" and i + 1 < n and code[i + 1] == "/":
            while i < n and code[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if char == "/" and i + 1 < n and code[i + 1] == "*":
            end = code.find("*/", i + 2)
            end = n if end < 0 else end + 2
            for j in range(i, end):
                if code[j] != "\n":
                    out[j] = " "
            i = end
            continue
        i += 1
    return "".join(out)


_TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][\w.:-]*)([^<>]*?)(/?)\s*>", re.S)


def _jsx_tag_balance(masked: str) -> tuple[bool, str]:
    stack: list[str] = []
    for match in _TAG_RE.finditer(masked):
        closing, name, _attrs, self_closing = match.groups()
        if name.lower() in VOID_ELEMENTS or self_closing:
            if closing:
                return False, f"<{name}> closed but is void/self-closing"
            continue
        if closing:
            if not stack:
                return False, f"</{name}> with nothing open"
            if stack[-1] != name:
                return False, f"</{name}> closes <{stack[-1]}>"
            stack.pop()
        else:
            stack.append(name)
    if stack:
        return False, f"unclosed <{stack[-1]}>"
    return True, "balanced"


def _balanced_brackets(masked: str) -> tuple[bool, str]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for char in masked:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False, f"unbalanced '{char}'"
            stack.pop()
    if stack:
        return False, f"unclosed '{stack[-1]}'"
    return True, "balanced"


_RETURN_JSX_RE = re.compile(r"\breturn\s*\(", re.S)


def check_react(code: str) -> tuple[bool, str, str]:
    """Structural JSX gate. The real build + render runs in `check_records`."""
    if not code.strip():
        return False, "empty", "no code"
    masked = _mask_literals(code)
    ok, detail = _balanced_brackets(masked)
    if not ok:
        return False, "react_brackets", detail
    ok, detail = _jsx_tag_balance(masked)
    if not ok:
        return False, "react_tags", detail
    if not re.search(r"\b(function|const|class)\s+[A-Z]\w*", code):
        return False, "react_no_component", "no capitalised component definition"
    if "export" not in masked:
        return False, "react_no_export", "component is never exported"
    if not _RETURN_JSX_RE.search(masked) and "=>" not in masked:
        return False, "react_no_return", "component returns nothing"
    roots = _return_root_counts(masked)
    if any(count != 1 for count in roots):
        return False, "react_multi_root", f"a return yields {roots} root element(s), expected 1"
    return True, "ok", "balanced JSX, single root, exported"


def _return_root_counts(masked: str) -> list[int]:
    """How many top-level elements each `return (...)` block contains."""
    counts: list[int] = []
    for match in _RETURN_JSX_RE.finditer(masked):
        start = match.end() - 1
        depth, end = 0, None
        for i in range(start, len(masked)):
            if masked[i] == "(":
                depth += 1
            elif masked[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            continue
        body = masked[start + 1 : end]
        roots, level = 0, 0
        for tag in _TAG_RE.finditer(body):
            closing, name, _attrs, self_closing = tag.groups()
            void = name.lower() in VOID_ELEMENTS or bool(self_closing)
            if closing:
                level -= 1
            elif void:
                if level == 0:
                    roots += 1
            else:
                if level == 0:
                    roots += 1
                level += 1
        counts.append(roots)
    return counts


class _StrictHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.error: str | None = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag not in VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_ELEMENTS:
            return
        if not self.stack:
            self.error = self.error or f"</{tag}> with nothing open"
            return
        if self.stack[-1] != tag:
            self.error = self.error or f"</{tag}> closes <{self.stack[-1]}>"
            return
        self.stack.pop()


def check_html(code: str) -> tuple[bool, str, str]:
    if not code.strip():
        return False, "empty", "no code"
    parser = _StrictHTML()
    parser.feed(code)
    parser.close()
    if parser.error:
        return False, "html_tags", parser.error
    if parser.stack:
        return False, "html_tags", f"unclosed <{parser.stack[-1]}>"
    if not _TAG_RE.search(code):
        return False, "html_no_tags", "no elements"
    return True, "ok", "tags balanced"


def check_spice(code: str) -> tuple[bool, str, str]:
    """The structural SPICE rules (module docstring). ngspice runs in `check_records`."""
    lines = [line.strip() for line in code.splitlines()]
    lines = [line for line in lines if line and not line.startswith("*")]
    if not lines:
        return False, "empty", "no netlist lines"
    if not lines[-1].lower().startswith(".end"):
        return False, "spice_no_end", "netlist does not end with .end"
    node_uses: Counter[str] = Counter()
    devices = 0
    for line in lines:
        if line.startswith("."):
            continue
        tokens = line.split()
        letter = tokens[0][0].upper()
        if letter not in SPICE_DEVICE_NODES:
            return False, "spice_unknown_device", f"unknown device card {tokens[0]!r}"
        if len(tokens[0]) < 2:
            return False, "spice_unnamed_device", f"device {tokens[0]!r} has no name"
        want = SPICE_DEVICE_NODES[letter]
        if len(tokens) < want + 2:
            return (
                False,
                "spice_arity",
                f"{tokens[0]} needs {want} node(s) and a value, got {len(tokens) - 1} token(s)",
            )
        for node in tokens[1 : 1 + want]:
            if not _NODE_RE.match(node):
                return False, "spice_node_name", f"{tokens[0]}: bad node {node!r}"
            node_uses[node] += 1
        devices += 1
    if devices == 0:
        return False, "spice_no_devices", "directives only, no device cards"
    if "0" not in node_uses:
        return False, "spice_no_ground", "no ground node 0"
    dangling = sorted(node for node, count in node_uses.items() if count < 2)
    if dangling:
        return False, "spice_dangling", f"node(s) touched once: {', '.join(dangling)}"
    return True, "ok", f"{devices} device card(s), {len(node_uses)} node(s)"


CHECKS: dict[str, Callable[[str], tuple[bool, str, str]]] = {
    "python": check_python,
    "sql": check_sql,
    "react": check_react,
    "html": check_html,
    "spice": check_spice,
}


def check_code(code: str, language: str) -> dict:
    """Static check plus the real tool for one target. See `check_many_code` for batches."""
    return check_many_code([(code, language)])[0]


def check_many_code(items: list[tuple[str, str]]) -> list[dict]:
    """`[{ok, kind, detail, language}]` for `(code, language)` items, tool stages batched.

    Static checks first; React survivors then go through one node batch and SPICE survivors
    through the ngspice pool, so 2,500 components cost a few node processes, not 2,500.
    """
    from src.eval import react, spice

    results: list[dict] = []
    for code, language in items:
        language = str(language).lower()
        checker = CHECKS.get(language)
        if not isinstance(code, str) or not code.strip():
            results.append({"ok": False, "kind": "empty", "detail": "blank", "language": language})
        elif checker is None:
            results.append(
                {
                    "ok": False,
                    "kind": "unsupported_language",
                    "detail": language,
                    "language": language,
                }
            )
        else:
            ok, kind, detail = checker(code)
            results.append({"ok": ok, "kind": kind, "detail": detail, "language": language})

    for language, tool in (("react", react.check_many), ("spice", spice.check_many)):
        todo = [i for i, r in enumerate(results) if r["ok"] and r["language"] == language]
        if not todo:
            continue
        for index, verdict in zip(todo, tool([items[i][0] for i in todo]), strict=True):
            if not verdict["ok"]:
                results[index].update(ok=False, kind=verdict["kind"], detail=verdict["detail"])
            else:
                results[index]["detail"] += f"; {verdict['detail']}"
    return results


# ----------------------------------------------------------------------------------------
# the record-level filter
# ----------------------------------------------------------------------------------------


def check_records(records: list[dict]) -> list[dict]:
    """`{ok, kind, detail, language, diagram_id}` per pair record: schema, then code."""
    from src.codegen import schema

    out: list[dict] = [{} for _ in records]
    todo: list[int] = []
    for i, record in enumerate(records):
        problems = schema.validate(record)
        if problems:
            out[i] = {
                "ok": False,
                "kind": "schema",
                "detail": "; ".join(problems),
                "language": record.get("language") if isinstance(record, dict) else None,
            }
        else:
            todo.append(i)
    verdicts = check_many_code([(records[i]["target_code"], records[i]["language"]) for i in todo])
    for i, verdict in zip(todo, verdicts, strict=True):
        out[i] = verdict
    for i, record in enumerate(records):
        out[i]["diagram_id"] = record.get("diagram_id") if isinstance(record, dict) else None
    return out


def check(record: dict) -> dict:
    """Is this one training pair fit to fine-tune on?"""
    return check_records([record])[0]


def filter_pairs(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Partition pairs into (kept, rejected). Rejections carry the record under `"record"`."""
    records = list(records)
    kept: list[dict] = []
    rejected: list[dict] = []
    for record, result in zip(records, check_records(records), strict=True):
        if result["ok"]:
            kept.append(record)
        else:
            rejected.append({**result, "record": record})
    return kept, rejected


def assert_all_compile(kept: Iterable[dict]) -> int:
    """12.1.7's postcondition, re-checked from scratch. Raises `QualityError` on any miss."""
    kept = list(kept)
    failures = [result for result in check_records(kept) if not result["ok"]]
    if failures:
        first = failures[0]
        raise QualityError(
            f"{len(failures)} kept pair(s) fail their own check; "
            f"first: {first['diagram_id']} [{first['kind']}] {first['detail']}"
        )
    return len(kept)


def rates(results: Iterable[dict]) -> dict:
    """Per-language kept/rejected counts and rejection kinds, from `check_records` output."""
    results = list(results)
    per_language: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    for result in results:
        language = str(result.get("language") or "unknown")
        per_language[language]["total"] += 1
        per_language[language]["kept" if result["ok"] else "rejected"] += 1
        if not result["ok"]:
            reasons[result["kind"]] += 1
    summary = {
        language: {
            "total": c["total"],
            "kept": c["kept"],
            "rejected": c["rejected"],
            "pass_rate": round(c["kept"] / c["total"], 4) if c["total"] else 0.0,
        }
        for language, c in sorted(per_language.items())
    }
    kept = sum(entry["kept"] for entry in summary.values())
    return {
        "total": len(results),
        "kept": kept,
        "rejected": len(results) - kept,
        "pass_rate": round(kept / len(results), 4) if results else 0.0,
        "by_language": summary,
        "reject_kinds": dict(reasons.most_common()),
    }


# ----------------------------------------------------------------------------------------
# the self-test corpus: valid and deliberately-invalid targets per language
# ----------------------------------------------------------------------------------------

_VALID: dict[str, list[str]] = {
    "python": [
        "def step_start():\n    return 'received'\n",
        "class Machine:\n    def __init__(self):\n        self.state = 'q0'\n",
        "def flow(x):\n    while x:\n        x -= 1\n    return x\n",
    ],
    "sql": [
        "CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT NOT NULL);",
        "CREATE TABLE a (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER REFERENCES a(id));",
        # The draft's version had no primary key; 12.3.7 rejects that as a cardinality error.
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x REAL);\nCREATE INDEX t_x ON t (x);",
    ],
    "react": [
        'export default function Page() {\n  return (\n    <div className="p-4">\n'
        '      <h1 className="text-xl">Title</h1>\n    </div>\n  );\n}\n',
        'const Card = () => {\n  return (\n    <section className="rounded border">\n'
        '      <img src="a.png" alt="a" />\n    </section>\n  );\n};\nexport default Card;\n',
        "export default function Nav() {\n  return (\n"
        '    <nav className="flex gap-2">{["a", "b"].map((x) => (\n'
        "      <a key={x}>{x}</a>\n    ))}</nav>\n  );\n}\n",
    ],
    "html": [
        '<div class="p-4"><h1>Title</h1></div>',
        '<section><img src="a.png" alt="a"><p>text</p></section>',
        "<ul><li>a</li><li>b</li></ul>",
    ],
    "spice": [
        "* divider\nV1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n",
        "* rc\nV1 1 0 DC 1\nR1 1 2 1k\nC1 2 0 1u\n.op\n.end\n",
        "* diode\nV1 a 0 5\nR1 a b 220\nD1 b 0 DMOD\n.model DMOD D\n.op\n.end\n",
    ],
}

#: (code, why, the stage expected to catch it: "static" or "tool")
_INVALID: dict[str, list[tuple[str, str, str]]] = {
    "python": [
        ("def broken(:\n    pass\n", "syntax error in the signature", "static"),
        ("return 1\n", "ast-clean, compile() rejects return outside a function", "static"),
        ("def f(a, a):\n    pass\n", "ast-clean, compile() rejects duplicate parameters", "static"),
    ],
    "sql": [
        ("CREATE TABEL t (id INT);", "misspelled keyword", "static"),
        ("CREATE TABLE t (a INT, a INT);", "duplicate column fails at execution", "static"),
        ("SELECT 1;", "runs but creates no table", "static"),
        (
            "CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES nosuch(id));",
            "executescript accepts an FK to a missing table",
            "static",
        ),
        ("CREATE TABLE t (x BANANA);", "executescript accepts an invented type", "static"),
    ],
    "react": [
        (
            'export default function P() {\n  return (\n    <div className="p">\n  );\n}\n',
            "unclosed <div>",
            "static",
        ),
        (
            "function P() {\n  return (\n    <div>{x</div>\n  );\n}\nexport default P;\n",
            "unbalanced expression brace",
            "static",
        ),
        (
            "export default function P() {\n  return (\n    <div>{items.map((i) => <b>{i}</b>)}</div>\n"
            "  );\n}\n",
            "structurally clean, `items` is undefined at render",
            "tool",
        ),
        (
            "import Chart from 'chart-lib';\nexport default function P() {\n  return (\n"
            "    <div><Chart /></div>\n  );\n}\n",
            "structurally clean, imports a module that does not exist",
            "tool",
        ),
        (
            "export default function P() {\n  const n = 1 +;\n  return (\n    <div />\n  );\n}\n",
            "structurally clean, invalid JavaScript expression",
            "tool",
        ),
    ],
    "html": [
        ("<div><p>text</div>", "</div> closes <p>", "static"),
        ("<ul><li>a</li>", "unclosed <ul>", "static"),
        ("just words", "no elements at all", "static"),
    ],
    "spice": [
        ("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n", "no .end", "static"),
        ("* x\nZ1 a b 1k\nR1 a 0 1k\n.end\n", "unknown device letter Z", "static"),
        ("* x\nV1 in 0 5\nR1 in out 1k\n.end\n", "node 'out' is dangling", "static"),
        (
            "* x\nV1 1 0 DC 5\nR1 1 2 1k\nL1 2 0 1m\nL2 2 0 1m\n.op\n.end\n",
            "structurally clean, parallel ideal inductors: singular DC matrix",
            "tool",
        ),
        (
            "* x\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 3 1n\nC2 3 0 1n\n.op\n.end\n",
            "structurally clean, node 3 has no DC path",
            "tool",
        ),
        (
            "* x\nV1 1 0 DC 5\nR1 1 2 1k\nD1 2 0 NOMODEL\n.op\n.end\n",
            "structurally clean, diode model never defined",
            "tool",
        ),
    ],
}


def self_test() -> dict:
    """Run every valid and invalid case through `check_many_code`; report the confusion."""
    items: list[tuple[str, str]] = []
    expected: list[tuple[bool, str, str]] = []
    for language, codes in _VALID.items():
        for code in codes:
            items.append((code, language))
            expected.append((True, language, ""))
    for language, cases in _INVALID.items():
        for code, _why, stage in cases:
            items.append((code, language))
            expected.append((False, language, stage))
    results = check_many_code(items)
    table: dict[str, Counter] = defaultdict(Counter)
    errors = []
    for (code, _), (want, language, stage), result in zip(items, expected, results, strict=True):
        static_ok = CHECKS[language](code)[0]
        row = table[language]
        row["valid" if want else "invalid"] += 1
        if want and result["ok"]:
            row["kept"] += 1
        elif not want and not result["ok"]:
            row["rejected"] += 1
            row["caught_only_by_tool"] += int(static_ok)
        else:
            errors.append({"language": language, "code": code, "result": result})
        if not want and stage == "tool" and not static_ok:
            errors.append({"language": language, "code": code, "note": "static check caught it"})
    return {"by_language": {k: dict(v) for k, v in table.items()}, "errors": errors}


def load_pairs(path: Path) -> list[dict]:
    """Read a JSONL pair file. Blank lines are skipped; a bad line is an error."""
    from src.codegen import schema

    return list(schema.read_jsonl(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.7 target quality filter")
    parser.add_argument("--pairs", type=Path, help="JSONL of training pairs to filter")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.pairs:
        records = load_pairs(args.pairs)
        results = check_records(records)
        print(json.dumps(rates(results), indent=2))
        return 0 if all(r["ok"] for r in results) else 1
    report = self_test()
    print(json.dumps(report, indent=2))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())

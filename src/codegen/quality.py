"""Phase 12.1.7 - the filter that makes "100% of targets compile" true rather than hoped for.

    python -m src.codegen.quality --self-test
    python -m src.codegen.quality --pairs data/processed/codegen/pairs.jsonl

A training pair whose target does not compile teaches the model to emit code that does not
compile. The row asks for "100% of targets compile", and the only way a percentage like that is
ever 100 is if it is a *postcondition of a filter*, not a property someone measured once and
hoped would hold. So this module is built the other way round from a report: `filter_pairs`
partitions, and `assert_all_compile` re-runs every check on the survivors and raises if a single
one fails. The tests assert that postcondition on a corpus that is deliberately half-invalid.

## What each language is checked with, and why that check and not a weaker one

    python   ast.parse *and* compile(). ast.parse alone accepts code the compiler rejects -
             `return` outside a function and duplicate parameter names both parse and both fail
             to compile. Measured on the self-test corpus: 2 of the 3 invalid Python cases
             are ast-clean and caught only by compile().
    sql      executed on a fresh in-memory sqlite3, not regex-matched. A DDL script that parses
             is not the same as one that builds: `REFERENCES nosuchtable` and a duplicate column
             name are both syntactically fine and both fail at execution. At least one CREATE
             TABLE must survive, so an empty-but-valid script cannot pass as an ER target.
    react    structural parse in Python. This is the one compromise in the module and it is
             documented rather than hidden: node v22.20.0 is installed but has **no JSX parser
             offline** - `require('@babel/parser')` fails and `npm ls -g` holds only corepack
             and npm. Shelling out per pair would also cost ~40ms of process start against
             ~0.1ms in-process. So JSX is checked by balance and structure (tags matched with
             the HTML void-element set honoured, self-closing tags, expression braces, exactly
             one root element in each `return`, a component that is defined and exported) with
             strings and comments masked first so a `<` inside a className cannot unbalance it.
             This catches unclosed tags, stray braces and missing exports; it does **not** catch
             an undefined identifier. Stated as a limit, not sold as a compiler.
    html      html.parser with a strict tag stack - the verbatim Sketch2Code targets are HTML,
             not JSX, and get the same balance discipline.
    spice     documented structural check. There is no netlist simulator in this environment and
             adding ngspice for a syntax check would be a heavy dependency for a light job, so
             the rules are spelled out here and enforced literally: every non-comment,
             non-directive line is a device card whose first character is a known device letter;
             the card carries at least that device's node count (R/L/C/V/I/E/G 2, D 2, Q 3,
             J 3, M 4, X >=1) plus a value or model token; node names are alphanumeric; ground
             node `0` appears at least once; the netlist ends with `.end`; and every node
             appears on at least two cards, because a node touched once is a dangling wire and
             a netlist of dangling wires is not a circuit.

## What it measured

No pair generator had written `data/processed/codegen/pairs.jsonl` when this module was built
(12.1.1-12.1.6 are still open), so the numbers below are from `self_test_corpus()` - 30
hand-written targets, 15 valid and 15 invalid, three of each per language - and they are the
filter's behaviour, not the corpus's:

    language   valid  kept   invalid  rejected
    python         3     3         3         3
    sql            3     3         3         3
    react          3     3         3         3
    html           3     3         3         3
    spice          3     3         3         3

    kept 15/30, rejected 15/30, false accepts 0, false rejects 0

Run with `--pairs` once real pairs exist and the same table is printed over them.

## What was rejected as an approach

- **Trusting the emitter.** `src/ir/targets.py` already compiles what it emits and reports
  993/993. That is a property of one emitter on one day; a filter that assumes it would pass
  through anything a *different* generator (12.1.4's synthetic pairs, 12.1.5's converted React)
  produces. Every record is re-checked here regardless of provenance.
- **Reporting a compile rate instead of filtering.** A 99.4% number in a report does not stop
  the 0.6% from entering the fine-tune.
- **`exec()`-ing Python targets.** Compiling proves syntax; executing arbitrary generated code
  in the filter is a sandbox problem, and the emitters that want execution semantics already do
  it in `src/ir/targets.py` where the inputs are known.
- **Regexing SQL.** Tried first and dropped: it accepted `CREATE TABLE t (a INT, a INT)`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

#: Languages this filter knows how to check. Anything else is rejected, loudly.
LANGUAGES = ("python", "sql", "react", "html", "spice")

#: Required keys of a training pair (the Phase 12.1.1 record contract).
REQUIRED_KEYS = (
    "diagram_id",
    "diagram_type",
    "ir_text",
    "traversal",
    "target_code",
    "language",
    "source",
    "split",
)

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
# per-language checks: each returns (ok, kind, detail)
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
    """Execute the DDL on a throwaway in-memory database. Nothing is written to disk."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(code)
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        if not tables:
            return False, "sql_no_tables", "script ran but created no table"
        connection.execute("PRAGMA foreign_key_check")
        return True, "ok", f"executed, {len(tables)} table(s)"
    except sqlite3.Error as exc:
        return False, "sql_execute", str(exc)
    finally:
        connection.close()


def _mask_literals(code: str) -> str:
    """Blank out string literals, JS comments and JSX text so structure can be counted.

    Everything replaced keeps its length, so reported offsets stay meaningful.
    """
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
    """Structural JSX check. See the module docstring for exactly what this does not catch."""
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
    try:
        parser.feed(code)
        parser.close()
    except Exception as exc:  # pragma: no cover - html.parser is lenient by design
        return False, "html_parse", str(exc)
    if parser.error:
        return False, "html_tags", parser.error
    if parser.stack:
        return False, "html_tags", f"unclosed <{parser.stack[-1]}>"
    if not _TAG_RE.search(code):
        return False, "html_no_tags", "no elements"
    return True, "ok", "tags balanced"


def check_spice(code: str) -> tuple[bool, str, str]:
    """The documented structural check. Rules are in the module docstring."""
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
                (f"{tokens[0]} needs {want} node(s) and a value, got {len(tokens) - 1} token(s)"),
            )
        nodes = tokens[1 : 1 + want]
        for node in nodes:
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


# ----------------------------------------------------------------------------------------
# the record-level filter
# ----------------------------------------------------------------------------------------


def _schema_complaint(record: Any) -> str | None:
    """Structural validation, deferring to 12.1.1's schema module when it exists."""
    try:  # another agent owns src/codegen/schema.py; never block on it
        from src.codegen import schema as _schema  # type: ignore

        validate = getattr(_schema, "validate", None)
        if callable(validate):
            problem = validate(record)
            if problem:
                return str(problem)
    except Exception:
        pass
    if not isinstance(record, dict):
        return f"record is {type(record).__name__}, not a dict"
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        return f"missing key(s): {', '.join(missing)}"
    if not isinstance(record.get("target_code"), str):
        return "target_code is not a string"
    if not isinstance(record.get("traversal"), list | tuple):
        return "traversal is not a list"
    return None


def check(record: Any) -> dict:
    """Is this training pair fit to fine-tune on?

    Returns `{ok, kind, detail, language, diagram_id}`. `kind` is `"ok"` on success and
    otherwise names the failure precisely enough to aggregate on - `python_compile`,
    `sql_execute`, `react_tags`, `spice_dangling`, `schema`, `unsupported_language`, `empty`.
    """
    diagram_id = record.get("diagram_id") if isinstance(record, dict) else None
    language = record.get("language") if isinstance(record, dict) else None
    result = {"ok": False, "kind": "schema", "detail": "", "language": language}
    result["diagram_id"] = diagram_id

    complaint = _schema_complaint(record)
    if complaint:
        result["detail"] = complaint
        return result

    language = str(record["language"]).lower()
    result["language"] = language
    code = record["target_code"]
    if not code.strip():
        result.update(kind="empty", detail="target_code is blank")
        return result
    checker = CHECKS.get(language)
    if checker is None:
        result.update(
            kind="unsupported_language",
            detail=f"{language!r} is not one of {', '.join(LANGUAGES)}",
        )
        return result

    ok, kind, detail = checker(code)
    result.update(ok=ok, kind=kind, detail=detail)
    return result


def filter_pairs(records: Iterable[Any]) -> tuple[list[Any], list[dict]]:
    """Partition pairs into (kept, rejected).

    `kept` are the records themselves, unmodified. `rejected` are the check results with the
    offending record attached under `"record"`, so a rejection can be explained without
    re-running anything.
    """
    kept: list[Any] = []
    rejected: list[dict] = []
    for record in records:
        result = check(record)
        if result["ok"]:
            kept.append(record)
        else:
            rejected.append({**result, "record": record})
    return kept, rejected


def assert_all_compile(kept: Iterable[Any]) -> int:
    """The postcondition of 12.1.7, re-checked from scratch. Raises `QualityError` on any miss.

    Deliberately not a report: this is the assertion that turns "100% of targets compile" from
    a measurement into a guarantee, and it is called by the tests and by `main`.
    """
    kept = list(kept)
    failures = [result for result in map(check, kept) if not result["ok"]]
    if failures:
        first = failures[0]
        raise QualityError(
            f"{len(failures)} kept pair(s) fail their own check; "
            f"first: {first['diagram_id']} [{first['kind']}] {first['detail']}"
        )
    return len(kept)


def rates(records: Iterable[Any]) -> dict:
    """Per-language pass/reject counts, plus the rejection reasons that produced them."""
    records = list(records)
    per_language: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    for record in records:
        result = check(record)
        language = str(result.get("language") or "unknown")
        per_language[language]["total"] += 1
        per_language[language]["kept" if result["ok"] else "rejected"] += 1
        if not result["ok"]:
            reasons[result["kind"]] += 1
    summary = {}
    for language, counts in sorted(per_language.items()):
        total = counts["total"]
        summary[language] = {
            "total": total,
            "kept": counts["kept"],
            "rejected": counts["rejected"],
            "pass_rate": round(counts["kept"] / total, 4) if total else 0.0,
        }
    total = len(records)
    kept = sum(entry["kept"] for entry in summary.values())
    return {
        "total": total,
        "kept": kept,
        "rejected": total - kept,
        "pass_rate": round(kept / total, 4) if total else 0.0,
        "by_language": summary,
        "reject_kinds": dict(reasons.most_common()),
    }


# ----------------------------------------------------------------------------------------
# the self-test corpus: valid and deliberately-invalid targets, three of each per language
# ----------------------------------------------------------------------------------------

_VALID: dict[str, list[str]] = {
    "python": [
        "def step_start():\n    return 'received'\n",
        "class Machine:\n    def __init__(self):\n        self.state = 'q0'\n",
        "def flow(x):\n    while x:\n        x -= 1\n    return x\n",
    ],
    "sql": [
        "CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT NOT NULL);",
        (
            "CREATE TABLE a (id INTEGER PRIMARY KEY);\n"
            "CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER REFERENCES a(id));"
        ),
        "CREATE TABLE t (x REAL);\nCREATE INDEX t_x ON t (x);",
    ],
    "react": [
        (
            "export default function Page() {\n"
            "  return (\n"
            '    <div className="p-4">\n'
            '      <h1 className="text-xl">Title</h1>\n'
            "    </div>\n"
            "  );\n"
            "}\n"
        ),
        (
            "const Card = () => {\n"
            "  return (\n"
            '    <section className="rounded border">\n'
            '      <img src="a.png" alt="a" />\n'
            "    </section>\n"
            "  );\n"
            "};\n"
            "export default Card;\n"
        ),
        (
            "export function Nav() {\n"
            "  return (\n"
            '    <nav className="flex gap-2">{["a", "b"].map((x) => (\n'
            "      <a key={x}>{x}</a>\n"
            "    ))}</nav>\n"
            "  );\n"
            "}\n"
        ),
    ],
    "html": [
        '<div class="p-4"><h1>Title</h1></div>',
        '<section><img src="a.png" alt="a"><p>text</p></section>',
        "<ul><li>a</li><li>b</li></ul>",
    ],
    "spice": [
        "* divider\nV1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.end\n",
        "* rc\nV1 1 0 DC 1\nR1 1 2 1k\nC1 2 0 1u\n.tran 1u 1m\n.end\n",
        "* diode\nV1 a 0 5\nR1 a b 220\nD1 b 0 DMOD\n.model DMOD D\n.end\n",
    ],
}

_INVALID: dict[str, list[tuple[str, str]]] = {
    "python": [
        ("def broken(:\n    pass\n", "syntax error in the signature"),
        ("return 1\n", "ast-clean, compile() rejects return outside a function"),
        ("def f(a, a):\n    pass\n", "ast-clean, compile() rejects duplicate parameters"),
    ],
    "sql": [
        ("CREATE TABEL t (id INT);", "misspelled keyword"),
        ("CREATE TABLE t (a INT, a INT);", "parses, fails at execution: duplicate column"),
        ("SELECT 1;", "runs but creates no table, so it is not a DDL target"),
    ],
    "react": [
        (
            'export default function P() {\n  return (\n    <div className="p">\n  );\n}\n',
            "unclosed <div>",
        ),
        (
            "function P() {\n  return (\n    <div>{x</div>\n  );\n}\nexport default P;\n",
            "unbalanced expression brace",
        ),
        (
            "export default function P() {\n  return (\n    <a />\n    <b />\n  );\n}\n",
            "two root elements in one return",
        ),
    ],
    "html": [
        ("<div><p>text</div>", "</div> closes <p>"),
        ("<ul><li>a</li>", "unclosed <ul>"),
        ("just words", "no elements at all"),
    ],
    "spice": [
        ("V1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n", "no .end"),
        ("* x\nZ1 a b 1k\nR1 a 0 1k\n.end\n", "unknown device letter Z"),
        ("* x\nV1 in 0 5\nR1 in out 1k\n.end\n", "node 'out' is dangling"),
    ],
}


def self_test_corpus() -> tuple[list[dict], list[dict]]:
    """(valid, invalid) pair records covering all five languages. Used by `--self-test`."""

    def record(language: str, index: int, code: str, tag: str) -> dict:
        return {
            "diagram_id": f"selftest/{language}/{tag}{index}",
            "diagram_type": {
                "python": "flowchart",
                "sql": "er_diagram",
                "react": "wireframe",
                "html": "wireframe",
                "spice": "circuit",
            }[language],
            "ir_text": f"node n0 [{language}]",
            "traversal": ["n0"],
            "target_code": code,
            "language": language,
            "source": "selftest",
            "split": "train",
        }

    valid = [
        record(language, i, code, "ok")
        for language, codes in _VALID.items()
        for i, code in enumerate(codes)
    ]
    invalid = []
    for language, cases in _INVALID.items():
        for i, (code, why) in enumerate(cases):
            entry = record(language, i, code, "bad")
            entry["expected_failure"] = why
            invalid.append(entry)
    return valid, invalid


def load_pairs(path: Path) -> list[dict]:
    """Read a JSONL pair file. Blank lines are skipped; a bad line is an error, not a warning."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return records


def _print_table(summary: dict) -> None:
    print(f"{'language':<10}{'total':>8}{'kept':>8}{'rejected':>10}{'pass':>8}")
    for language, entry in summary["by_language"].items():
        print(
            f"{language:<10}{entry['total']:>8}{entry['kept']:>8}"
            f"{entry['rejected']:>10}{entry['pass_rate']:>8.4f}"
        )
    print(
        f"{'TOTAL':<10}{summary['total']:>8}{summary['kept']:>8}"
        f"{summary['rejected']:>10}{summary['pass_rate']:>8.4f}"
    )
    if summary["reject_kinds"]:
        print("reject kinds: " + json.dumps(summary["reject_kinds"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.7 target quality filter")
    parser.add_argument("--pairs", type=Path, help="JSONL of training pairs to filter")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in valid/invalid corpus (the default when --pairs is absent)",
    )
    args = parser.parse_args(argv)

    if args.pairs:
        records = load_pairs(args.pairs)
        kept, rejected = filter_pairs(records)
        assert_all_compile(kept)
        _print_table(rates(records))
        print(f"kept {len(kept)} / {len(records)}; rejected {len(rejected)}")
        return 0

    valid, invalid = self_test_corpus()
    kept, rejected = filter_pairs(valid + invalid)
    assert_all_compile(kept)
    _print_table(rates(valid + invalid))
    false_rejects = [entry for entry in rejected if entry["record"] in valid]
    kept_ids = {r["diagram_id"] for r in kept}
    false_accepts = [r for r in invalid if r["diagram_id"] in kept_ids]
    print(f"false accepts: {len(false_accepts)}   false rejects: {len(false_rejects)}")
    for entry in rejected:
        print(f"  reject {entry['diagram_id']:<28} [{entry['kind']}] {entry['detail']}")
    return 0 if not (false_accepts or false_rejects) else 1


if __name__ == "__main__":
    sys.exit(main())

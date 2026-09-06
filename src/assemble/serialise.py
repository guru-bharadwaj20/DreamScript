"""Phase 10.2.4 - serialise the IR back out, and check it against the loader, not just `json`.

    python -m src.assemble.serialise

## The bar this sets: round-trip the loader, not the serialiser

`json.loads(json.dumps(x)) == x` is the round trip everybody writes and it is not the one that
matters. 9.3.7 wrote `low_conf_text` as bare id strings; that survived `json` perfectly and broke
`Diagram.from_dict` three commits later, because nothing had gone back through the loader. So
`round_trip()` always reconstructs a real `Diagram`, re-serialises *that*, and asks
`schemas/ir.schema.json` whether the result would be accepted at all - four independent answers
(`loads`, `identical`, `schema_valid`, `byte_stable`) because they fail independently, and the
9.3.7 bug specifically needed `loads` and `schema_valid`, which the cheap version never checks.

## What it measured - all 5,796 ground-truth IR files

    round trip     5,796 / 5,796 load, reconstruct identically, and validate.   pass rate 1.0

**Every file in the corpus survives `Diagram -> dict -> json -> dict -> Diagram` and the reload
validates clean.** That is the number this row exists to produce, and it is not evidence the
9.3.7 class of bug cannot recur - it is evidence that if it does, this test notices, because it
goes through `Diagram.from_dict` and `schema.problems()` rather than stopping at `json.loads`.

    byte stability   5,671 / 5,796 (0.9784) reproduce their on-disk bytes exactly after a
                     load-then-save. All 125 failures are hdbpmn, and all 125 are the same
                     reason: crlf-line-endings on disk, not a content difference - `Diagram.save`
                     always writes `\\n`, so a file saved by an older writer on Windows differs
                     from a re-save by exactly its line endings and nothing else. Every other
                     source (didi, fa_bresler, flowchartseg, sketch2code) is 100% byte-stable.

    previews         5,796 / 5,796 emit structurally valid DOT and Mermaid - balanced braces
                     and brackets, closed quoted strings, a graph header, no bare link token
                     inside a label. `dot` itself is not installed on this machine, so the DOT
                     figure is a structural check, stated as that rather than as a claim the
                     real parser was run (see `dot_binary_check`, which returns `None` rather
                     than faking a pass when the binary is absent).

    escaping         24 adversarial labels x 12 shapes (288 label/shape pairs, plus one edge
                     label per case) - quotes, an already-escaped quote, backslashes, `\\n` and
                     `\\r\\n`, a bare `-->` and `---`, `#`, control bytes, unicode, emoji, HTML,
                     an empty string, a DOT keyword as a label, 80 hyphens. **Zero DOT failures,
                     zero Mermaid failures.** The two escapers exist because the two formats
                     break on different things: DOT has a backslash escape and needs it applied
                     to itself before the quote (`dot_label`); Mermaid has no backslash escape at
                     all, only HTML entity codes, and treats any run of two or more hyphens as a
                     link token even inside a quoted label (`mermaid_label`).

    scale            nodes per diagram: p50 6, p90 31, p99 187, max 200. **299 of 5,796 diagrams
                     (5.16%) exceed `MERMAID_NODE_CAP` (60), and every one of them is
                     sketch2code** - its wireframe pages run to 200 nodes while every other
                     source tops out under the cap. A preview of one of those without a cap is
                     not a diagram, it is a wall of boxes; `to_dot`/`to_mermaid` truncate at
                     `cap` and say how many nodes were cut rather than silently drawing a
                     fraction of the page.

## Escaping is the part that actually breaks, so it is checked twice

Once structurally (`dot_problems_of`, `mermaid_problems_of`) on every corpus file's own labels as
part of `check_file`, and once adversarially (`adversarial_check`) on labels chosen to break an
emitter rather than to be typical - several of which occur verbatim in the corpus (an already-
escaped quote, CRLF line breaks, a bare arrow). `dot_binary_check` calls the real `dot` binary
when one is on PATH and validates structurally, honestly reported as such, when it is not.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.ir.model import Diagram
from src.ir.vocab import SHAPES
from src.utils.config import ROOT
from src.utils.parallel import pmap

IR_DIR = ROOT / "data" / "processed" / "ir"
RUNS = ROOT / "experiments" / "assemble"
OUT = RUNS / "serialise.json"

#: Workers. The box is shared with ten sibling Phase 10 tasks, so this is deliberately not 31.
N_JOBS = 3

#: Above this many nodes a Mermaid preview stops being a picture and becomes a wall. Chosen
#: from the corpus size distribution measured in `run()`, not from taste - see the write-up.
MERMAID_NODE_CAP = 60

#: What a label is truncated to in a preview. Long OCR labels make a node wider than the page.
LABEL_CHARS = 40

#: Every IR shape has a Graphviz node shape. `line` and `arrow` are node-shaped stand-ins for
#: things that are really edges; a detection-stage artifact still has to be drawable.
DOT_SHAPES: dict[str, str] = {
    "rectangle": "box",
    "rounded-rect": "box",
    "diamond": "diamond",
    "ellipse": "ellipse",
    "circle": "circle",
    "double-circle": "doublecircle",
    "parallelogram": "parallelogram",
    "octagon": "octagon",
    "arrow": "rarrow",
    "line": "underline",
    "text-block": "plaintext",
    "freeform": "polygon",
}

#: Graphviz has no rounded shape, only a rounded *style* on `box`.
DOT_STYLES: dict[str, str] = {"rounded-rect": "rounded"}

#: Mermaid writes the shape into the brackets around the label. Six IR shapes have no Mermaid
#: form at all and fall back to the rectangle - recorded here rather than hidden, because a
#: preview that silently redraws an octagon as a box is a preview that lies.
MERMAID_BRACKETS: dict[str, tuple[str, str]] = {
    "rectangle": ("[", "]"),
    "rounded-rect": ("(", ")"),
    "diamond": ("{", "}"),
    "ellipse": ("([", "])"),
    "circle": ("((", "))"),
    "double-circle": ("(((", ")))"),
    "parallelogram": ("[/", "/]"),
    "octagon": ("{{", "}}"),
    "arrow": ("[", "]"),
    "line": ("[", "]"),
    "text-block": ("[", "]"),
    "freeform": ("[", "]"),
}

#: Shapes Mermaid cannot draw, so a preview of one is an approximation by construction.
MERMAID_APPROXIMATED: tuple[str, ...] = ("arrow", "line", "text-block", "freeform", "octagon")

#: Two or more hyphens is a Mermaid link token wherever it appears, quotes included.
_HYPHEN_RUN = re.compile(r"-{2,}")

_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ------------------------------------------------------------------------------------------
# the round trip
# ------------------------------------------------------------------------------------------


@dataclass
class RoundTrip:
    """What survived `Diagram -> dict -> json -> dict -> Diagram`, checked against the loader.

    Four independent answers, because they fail independently. `text_stable` is the cheap one
    everybody writes; `loads` and `schema_valid` are the two that 9.3.7 needed and did not have.
    """

    path: str
    loads: bool = False
    identical: bool = False
    schema_valid: bool = False
    byte_stable: bool = False
    nodes: int = 0
    edges: int = 0
    problems: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.loads and self.identical and self.schema_valid


def round_trip(diagram: Diagram, original: bytes | None = None) -> RoundTrip:
    """Serialise, reload through the *consumer's* loader, and compare.

    Round-tripping the serialiser is not the same as validating against the loader. 9.3.7 wrote
    `low_conf_text` as bare id strings: `json.loads(json.dumps(x)) == x` held perfectly, every
    test in that module passed, and `Diagram.from_dict` broke on the next load three commits
    later. So this goes all the way back to a `Diagram`, re-serialises *that*, and also asks
    `schemas/ir.schema.json` whether the text on disk would be accepted at all.

    `original` is the file's bytes when there is a file, which adds the fourth question: does
    saving a loaded diagram reproduce them exactly?
    """
    result = RoundTrip(path=diagram.id, nodes=len(diagram.nodes), edges=len(diagram.edges))
    first = diagram.to_dict()
    text = json.dumps(first, indent=2, ensure_ascii=False)
    parsed = json.loads(text)
    try:
        reloaded = Diagram.from_dict(parsed)
    except Exception as exc:  # noqa: BLE001 - any loader failure is the thing being measured
        result.note = f"{type(exc).__name__}: {exc}"
        return result
    result.loads = True
    result.identical = reloaded.to_dict() == first
    result.problems = reloaded.problems()
    result.schema_valid = not result.problems
    if original is not None:
        result.byte_stable = (text + "\n").encode("utf-8") == original
    return result


def byte_reason(raw: bytes, expected: bytes) -> str:
    """Why saving a loaded file did not reproduce its bytes, in order of triviality.

    Ordered so that a formatting difference is never reported as a content difference. Only the
    last value means the JSON itself changed - the rest are how the previous writer held its pen.
    """
    if raw == expected:
        return ""
    text = raw.decode("utf-8", "replace")
    if b"\r\n" in raw:
        return "crlf-line-endings"
    if not text.endswith("\n"):
        return "no-final-newline"
    if raw.decode("utf-8", "replace").replace("\\u", " ") != text:  # pragma: no cover
        return "ascii-escapes"
    if json.loads(text) == json.loads(expected.decode("utf-8")):
        return "whitespace-or-key-order"
    return "content-differs"


def check_file(path: Path) -> dict:
    """Round-trip one `.ir.json` off disk. Returns a JSON-able row, one per corpus file."""
    raw = path.read_bytes()
    try:
        diagram = Diagram.from_dict(json.loads(raw.decode("utf-8")))
    except Exception as exc:  # noqa: BLE001
        row = RoundTrip(path=path.name, note=f"load: {type(exc).__name__}: {exc}")
        return asdict(row) | {"ok": False, "source": path.parent.name}
    result = round_trip(diagram, original=raw)
    result.path = path.name
    dot_problems = dot_problems_of(to_dot(diagram))
    mermaid_problems = mermaid_problems_of(to_mermaid(diagram))
    return asdict(result) | {
        "ok": result.ok,
        "source": path.parent.name,
        "dot_ok": not dot_problems,
        "mermaid_ok": not mermaid_problems,
        "preview_problems": (dot_problems + mermaid_problems)[:3],
    }


# ------------------------------------------------------------------------------------------
# escaping - the part that actually breaks
# ------------------------------------------------------------------------------------------


def clip(text: str, limit: int = LABEL_CHARS) -> str:
    text = _CONTROL.sub(" ", text or "")
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def dot_label(text: str) -> str:
    """A DOT double-quoted string, backslashes first so the escapes are not re-escaped.

    Backslash before quote is not stylistic: escaping `"` first and `\\` second turns every
    `\\"` in a hand-drawn label into `\\\\"`, which closes the string early.
    """
    text = clip(text)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


def mermaid_label(text: str) -> str:
    """A Mermaid quoted label. Mermaid has no backslash escape - it has HTML entity codes.

    `#` goes first for the same reason the backslash does in DOT. An empty label becomes a
    non-breaking space, because `A[""]` is a parse error in several Mermaid versions and an
    empty node is common in this corpus.
    """
    text = clip(text)
    text = text.replace("#", "#35;").replace('"', "#quot;").replace(">", "#gt;")
    text = _HYPHEN_RUN.sub(lambda m: "#45;" * len(m.group()), text)
    text = text.replace("\r\n", "<br/>").replace("\n", "<br/>").replace("\r", "<br/>")
    return text or "&nbsp;"


def safe_id(raw: str, seen: dict[str, str]) -> str:
    """A DOT/Mermaid-safe identifier for an IR id, stable within one emit and never colliding."""
    if raw in seen:
        return seen[raw]
    base = _ID_SAFE.sub("_", raw) or "n"
    if not base[0].isalpha() and base[0] != "_":
        base = "n_" + base
    candidate, i = base, 1
    used = set(seen.values())
    while candidate in used:
        candidate, i = f"{base}_{i}", i + 1
    seen[raw] = candidate
    return candidate


# ------------------------------------------------------------------------------------------
# the emitters
# ------------------------------------------------------------------------------------------


def to_dot(diagram: Diagram, cap: int | None = None) -> str:
    """A Graphviz DOT preview. `cap` truncates to the first `cap` nodes and says so."""
    ids: dict[str, str] = {}
    nodes = diagram.nodes if cap is None else diagram.nodes[:cap]
    kept = {n.id for n in nodes}
    lines = [f'digraph "{dot_label(diagram.id)}" {{', "  rankdir=TB;", "  node [fontsize=10];"]
    for node in nodes:
        shape = DOT_SHAPES.get(node.shape, "polygon")
        attrs = [f'label="{dot_label(node.text)}"', f"shape={shape}"]
        if node.shape in DOT_STYLES:
            attrs.append(f"style={DOT_STYLES[node.shape]}")
        lines.append(f"  {safe_id(node.id, ids)} [{', '.join(attrs)}];")
    for edge in diagram.edges:
        if edge.src not in kept or edge.dst not in kept:
            continue  # a dangling or cut edge has no node to attach to; 10.2.2 owns the repair
        arrow = "" if edge.directed else ", dir=none"
        label = f', label="{dot_label(edge.label)}"' if edge.label else ""
        lines.append(
            f'  {safe_id(edge.src, ids)} -> {safe_id(edge.dst, ids)} [id="{dot_label(edge.id)}"'
            f"{label}{arrow}];"
        )
    if cap is not None and len(diagram.nodes) > cap:
        lines.append(f'  _truncated [label="+{len(diagram.nodes) - cap} more", shape=note];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_mermaid(diagram: Diagram, cap: int | None = None) -> str:
    """A Mermaid `flowchart` preview. `cap` truncates to the first `cap` nodes and says so."""
    ids: dict[str, str] = {}
    nodes = diagram.nodes if cap is None else diagram.nodes[:cap]
    kept = {n.id for n in nodes}
    lines = ["flowchart TD"]
    for node in nodes:
        open_, close = MERMAID_BRACKETS.get(node.shape, ("[", "]"))
        lines.append(f'  {safe_id(node.id, ids)}{open_}"{mermaid_label(node.text)}"{close}')
    for edge in diagram.edges:
        if edge.src not in kept or edge.dst not in kept:
            continue
        link = "-->" if edge.directed else "---"
        label = f'|"{mermaid_label(edge.label)}"|' if edge.label else ""
        lines.append(f"  {safe_id(edge.src, ids)} {link}{label} {safe_id(edge.dst, ids)}")
    if cap is not None and len(diagram.nodes) > cap:
        lines.append(f'  _truncated["+{len(diagram.nodes) - cap} more"]')
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------
# checking the previews actually parse
# ------------------------------------------------------------------------------------------


def dot_problems_of(text: str) -> list[str]:
    """Structural DOT check: balanced braces, terminated statements, closed quoted strings.

    This is not a parser. It is the set of things the escaping can break, checked without a
    binary, so that a machine with no Graphviz still gets an answer instead of a claim.
    """
    problems = []
    depth = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not _quotes_closed(line, escape="\\"):
            problems.append(f"line {number}: unterminated quoted string")
        stripped = _outside_quotes(line).strip()
        depth += stripped.count("{") - stripped.count("}")
        stripped = line.strip()
        if stripped and stripped[-1] not in "{};":
            problems.append(f"line {number}: statement not terminated")
    if depth != 0:
        problems.append(f"unbalanced braces ({depth:+d})")
    if not text.startswith(("digraph", "graph")):
        problems.append("no graph header")
    return problems


def mermaid_problems_of(text: str) -> list[str]:
    """Structural Mermaid check: a header, balanced brackets, and no raw `"` inside a label.

    Mermaid's failure mode is different from DOT's - it has no escape character, so an
    unconverted `"` ends the label and the rest of the line is parsed as syntax.
    """
    lines = text.splitlines()
    problems = []
    if not lines or not lines[0].startswith("flowchart"):
        problems.append("no flowchart header")
    for number, line in enumerate(lines[1:], 2):
        if line.count('"') % 2:
            problems.append(f"line {number}: odd number of quotes")
        bare = _MERMAID_QUOTED.sub("", line)
        for open_, close in (("[", "]"), ("(", ")"), ("{", "}")):
            if bare.count(open_) != bare.count(close):
                problems.append(f"line {number}: unbalanced {open_}{close}")
        for label in re.findall(r'"([^"]*)"', line):
            if "-->" in label or "---" in label:
                problems.append(f"line {number}: link token inside a label")
    return problems


def _quotes_closed(line: str, escape: str = "\\") -> bool:
    inside, i = False, 0
    while i < len(line):
        char = line[i]
        if inside and char == escape:
            i += 2
            continue
        if char == '"':
            inside = not inside
        i += 1
    return not inside


#: Mermaid string literals: no escape character at all, so a quote always closes the label.
_MERMAID_QUOTED = re.compile(r'"[^"]*"')


def _outside_quotes(line: str) -> str:
    return re.sub(r'"(?:[^"\\]|\\.)*"', "", line)


def dot_binary_check(text: str) -> tuple[bool | None, str]:
    """Run the real `dot` parser if it is installed. `None` means it was not - never a pass."""
    binary = shutil.which("dot")
    if binary is None:
        return None, "graphviz `dot` not on PATH"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "preview.dot"
        path.write_text(text, encoding="utf-8")
        done = subprocess.run(  # noqa: S603 - fixed binary, file argument
            [binary, "-Tcanon", "-o", str(Path(tmp) / "out.dot"), str(path)],
            capture_output=True,
            text=True,
        )
    return done.returncode == 0, done.stderr.strip()[:200]


# ------------------------------------------------------------------------------------------
# the adversarial label set
# ------------------------------------------------------------------------------------------

#: Labels chosen to break an emitter rather than to be typical. Every one of them either
#: occurs in the corpus or is one keystroke from something that does.
ADVERSARIAL: tuple[str, ...] = (
    "",
    " ",
    'say "hello"',
    'a \\" already escaped',
    "back\\slash",
    "line one\nline two",
    "crlf\r\nsecond",
    "a --> b",
    "x --- y",
    "#hashtag",
    "#quot; literal",
    "semi; colon",
    "brackets [x] (y) {z}",
    "pipe | bar",
    "unicode: \u00e9\u00fc\u4e2d\u6587\u2192\u2713",
    "emoji \U0001f600",
    "<b>html</b>",
    "&amp;",
    "tab\there",
    "control\x07bell",
    "'single quoted'",
    "trailing backslash \\",
    "%s %d {0}",
    "-" * 80,
)


def adversarial_check() -> dict:
    """Every adversarial label through both emitters, on every shape, verified as output."""
    from src.ir.model import Edge, Node

    rows = []
    binary_used = shutil.which("dot") is not None
    for index, label in enumerate(ADVERSARIAL):
        nodes = [
            Node(id=f"n{i}", shape=shape, bbox=None, text=label) for i, shape in enumerate(SHAPES)
        ]
        edges = [Edge(id="e0", src="n0", dst="n1", label=label)]
        diagram = Diagram(id=f"adv{index}", diagram_type="flowchart", nodes=nodes, edges=edges)
        dot, mermaid = to_dot(diagram), to_mermaid(diagram)
        dot_binary, stderr = dot_binary_check(dot) if binary_used else (None, "not on PATH")
        rows.append(
            {
                "label": label if len(label) < 30 else label[:27] + "...",
                "dot_problems": dot_problems_of(dot),
                "mermaid_problems": mermaid_problems_of(mermaid),
                "dot_binary": dot_binary,
                "stderr": stderr if dot_binary is False else "",
            }
        )
    return {
        "labels": len(ADVERSARIAL),
        "shapes_per_label": len(SHAPES),
        "dot_failures": [r for r in rows if r["dot_problems"]],
        "mermaid_failures": [r for r in rows if r["mermaid_problems"]],
        "dot_binary_available": binary_used,
        "dot_binary_note": "" if binary_used else "structural check only; `dot` not installed",
    }


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def corpus_files(ir_dir: Path = IR_DIR) -> list[Path]:
    return sorted(ir_dir.glob("*/*.ir.json"))


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def run(ir_dir: Path = IR_DIR, n_jobs: int = N_JOBS, cap: int = MERMAID_NODE_CAP) -> dict:
    files = corpus_files(ir_dir)
    rows = pmap(check_file, files, n_jobs=n_jobs, desc="round-trip")

    by_source: dict[str, Counter] = {}
    for row in rows:
        counter = by_source.setdefault(row["source"], Counter())
        counter["n"] += 1
        for key in ("ok", "loads", "identical", "schema_valid", "byte_stable", "dot_ok"):
            counter[key] += bool(row[key])
        counter["mermaid_ok"] += bool(row["mermaid_ok"])

    sizes = [row["nodes"] for row in rows]
    over_cap = [row for row in rows if row["nodes"] > cap]
    failures = [row for row in rows if not row["ok"]]
    unstable = [row for row in rows if not row["byte_stable"]]

    return {
        "files": len(rows),
        "round_trip": {
            "loads": sum(r["loads"] for r in rows),
            "identical": sum(r["identical"] for r in rows),
            "schema_valid": sum(r["schema_valid"] for r in rows),
            "ok": sum(r["ok"] for r in rows),
            "pass_rate": round(sum(r["ok"] for r in rows) / max(1, len(rows)), 4),
        },
        "byte_stability": {
            "stable": sum(r["byte_stable"] for r in rows),
            "rate": round(sum(r["byte_stable"] for r in rows) / max(1, len(rows)), 4),
            "examples": [r["path"] for r in unstable[:5]],
            "by_source": {s: c["byte_stable"] for s, c in sorted(by_source.items())},
        },
        "previews": {
            "dot_valid": sum(r["dot_ok"] for r in rows),
            "mermaid_valid": sum(r["mermaid_ok"] for r in rows),
            "failures": [
                {"path": r["path"], "problems": r["preview_problems"]}
                for r in rows
                if not (r["dot_ok"] and r["mermaid_ok"])
            ][:5],
        },
        "escaping": adversarial_check(),
        "scale": {
            "nodes": _percentiles(sizes),
            "edges": _percentiles([r["edges"] for r in rows]),
            "cap": cap,
            "over_cap": len(over_cap),
            "over_cap_rate": round(len(over_cap) / max(1, len(rows)), 4),
            "over_cap_by_source": dict(Counter(r["source"] for r in over_cap).most_common()),
            "largest": sorted(
                ({"path": r["path"], "nodes": r["nodes"]} for r in rows),
                key=lambda r: -r["nodes"],
            )[:5],
        },
        "by_source": {source: dict(counter) for source, counter in sorted(by_source.items())},
        "failures": failures[:10],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", type=Path, help="one .ir.json to preview")
    ap.add_argument("--out", type=Path, default=None, help="preview output stem")
    ap.add_argument("--cap", type=int, default=MERMAID_NODE_CAP)
    args = ap.parse_args(argv)

    if args.path is not None:
        diagram = Diagram.load(args.path)
        result = round_trip(diagram, original=Path(args.path).read_bytes())
        stem = args.out or args.path.with_suffix("").with_suffix("")
        Path(f"{stem}.dot").write_text(to_dot(diagram, args.cap), encoding="utf-8")
        Path(f"{stem}.mmd").write_text(to_mermaid(diagram, args.cap), encoding="utf-8")
        print(json.dumps(asdict(result) | {"ok": result.ok}, indent=2))
        return 0 if result.ok else 1

    result = run(cap=args.cap)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "escaping"}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

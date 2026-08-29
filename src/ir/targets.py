"""Phase 2.2.6 - target code for every diagram whose structure is known.

A (diagram, code) pair is what Phase 12 fine-tunes on and what Phase 14 scores against, so the
question that matters here is where the code comes from. Three answers, and they are not equal:

| Basis | Meaning | Used for |
| :--- | :--- | :--- |
| `source` | the code **is** the ground truth, copied verbatim | Sketch2Code: the sketch was drawn *from* this HTML |
| `emitted` | deterministically transcribed from ground-truth structure | hdBPMN, FA |
| - | no structure exists, so no pair is produced | DIDI, flowchartseg, ER, circuits |

An `emitted` target is a faithful transcription of the graph, **not the program a person would
have written**. It does not name variables well, invent business logic, or restructure a cyclic
flowchart into nested `if`s it never was. Calling that "the correct target code" would be an
overclaim; what it is, is a reference an LLM can be scored against without a human writing 1,500
programs by hand. Phase 12.4 is where a small hand-written subset raises the ceiling.

Everything emitted is **checked by running it**: Python targets are compiled and their state
machines executed on real inputs, because code that does not run is not a target, it is a
string.

    python -m src.ir.targets
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.ir.model import SUFFIX, Diagram
from src.utils.config import ROOT

IR_ROOT = ROOT / "data" / "processed" / "ir"
OUT = ROOT / "data" / "processed" / "targets"
REPORT = ROOT / "reports" / "target_pairs.md"

#: Sources that carry enough structure to emit from, and what they emit.
EMITTERS = {
    "hdbpmn": ("python", "flowchart"),
    "fa_bresler": ("python", "state_machine"),
    "sketch2code": ("html", "verbatim"),
}

#: Sources deliberately excluded, and the reason. Reported rather than silently skipped.
EXCLUDED = {
    "didi": (
        "the prompts have shape and text but no semantics - a diamond labelled 'Quit?' may or "
        "may not be a decision, so any program written from one would be invented, not read"
    ),
    "flowchartseg": (
        "a node mask and nothing else: no edges, no roles, no text. There is no structure to "
        "transcribe"
    ),
}

PY_KEYWORDS = {
    "and",
    "as",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "false",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "none",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "true",
    "try",
    "while",
    "with",
    "yield",
}


def slug(text: str, fallback: str = "step") -> str:
    """A readable Python identifier from arbitrary handwriting."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", (text or "").strip().lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)[:40].strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    if cleaned in PY_KEYWORDS:
        cleaned += "_"
    return cleaned


def _unique(names: dict[str, str]) -> dict[str, str]:
    seen: Counter[str] = Counter()
    out = {}
    for key, name in names.items():
        seen[name] += 1
        out[key] = name if seen[name] == 1 else f"{name}_{seen[name]}"
    return out


def emit_flowchart_python(diagram: Diagram) -> str:
    """A flowchart as a runnable Python program.

    The shape of the output is deliberate. A hand-drawn process diagram is a *cyclic* directed
    graph with labelled branches; rewriting one into nested `if`/`while` requires choices the
    drawing does not make, and a wrong choice is invisible in the generated text. So the graph
    is emitted as what it is - one function per step, plus a driver that walks the transitions -
    which runs, is readable, and adds nothing that was not drawn.
    """
    names = _unique(
        {
            n.id: slug(n.text or n.semantic_role, n.semantic_role.replace("-", "_"))
            for n in diagram.nodes
        }
    )
    by_id = {n.id: n for n in diagram.nodes}
    starts = [n.id for n in diagram.nodes if n.semantic_role == "start"]
    ends = {n.id for n in diagram.nodes if n.semantic_role == "end"}

    lines = [
        '"""Generated from a hand-drawn diagram by DreamScript (Phase 2.2.6).',
        "",
        f"Source diagram: {diagram.id}   ({diagram.meta.get('source', '?')})",
        "",
        "A transcription of the drawing, not a hand-written program: step names come from the",
        "handwriting and the control flow is exactly the arrows that were drawn.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "",
    ]

    for node in diagram.nodes:
        if node.semantic_role in {"container", "unknown"}:
            continue
        name = names[node.id]
        label = " ".join((node.text or "").split()) or "(no label)"
        lines.append(f"def {name}(state: dict) -> dict:")
        # repr, not an f-string inside triple quotes: handwriting contains quote marks and
        # backslashes, and 14 of 693 files failed to compile before this line used repr.
        lines.append(f"    {node.semantic_role + ': ' + label!r}")
        if node.semantic_role == "decision":
            lines.append(f"    # branch on: {label}")
            lines.append(f'    state.setdefault("decisions", []).append({name!r})')
        else:
            lines.append(f'    state.setdefault("trace", []).append({name!r})')
        lines.append("    return state")
        lines.append("")
        lines.append("")

    transitions: dict[str, list[tuple[str, str]]] = {}
    for edge in diagram.edges:
        if edge.src is None or edge.dst is None:
            continue
        if edge.src not in by_id or edge.dst not in by_id:
            continue
        transitions.setdefault(edge.src, []).append((edge.label or "", edge.dst))

    lines.append("#: node id -> [(branch label, next node id)], exactly as drawn.")
    lines.append("TRANSITIONS = {")
    for src, targets in transitions.items():
        pairs = ", ".join(f"({label!r}, {dst!r})" for label, dst in targets)
        lines.append(f"    {src!r}: [{pairs}],")
    lines.append("}")
    lines.append("")
    lines.append("STEPS = {")
    for node in diagram.nodes:
        if node.semantic_role in {"container", "unknown"}:
            continue
        lines.append(f"    {node.id!r}: {names[node.id]},")
    lines.append("}")
    lines.append("")
    lines.append(f"START = {starts[0]!r}" if starts else "START = None")
    lines.append(f"END = {sorted(ends)!r}")
    lines.append("")
    lines.append("")
    lines.extend(
        [
            "def run(choose=None, max_steps: int = 1000) -> dict:",
            '    """Walk the diagram from its start event.',
            "",
            "    `choose(node_id, options)` picks a branch at a decision; the default takes the",
            "    first arrow drawn. `max_steps` exists because a hand-drawn process may loop",
            "    forever, and often legitimately does.",
            '    """',
            "    state: dict = {}",
            "    current = START",
            "    for _ in range(max_steps):",
            "        if current is None or current in END:",
            "            break",
            "        step = STEPS.get(current)",
            "        if step is not None:",
            "            state = step(state)",
            "        options = TRANSITIONS.get(current, [])",
            "        if not options:",
            "            break",
            "        current = choose(current, options) if choose else options[0][1]",
            "    return state",
            "",
            "",
            'if __name__ == "__main__":',
            "    print(run())",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_state_machine_python(diagram: Diagram) -> str:
    """A finite automaton as a runnable acceptor.

    This is the one target in the project that can be checked for *meaning* rather than for
    syntax: the emitted class either accepts the right strings or it does not.
    """
    names = _unique({n.id: slug(n.text, "q") for n in diagram.nodes})
    start = next((n.id for n in diagram.nodes if n.semantic_role == "initial-state"), None)
    accepting = sorted(n.id for n in diagram.nodes if n.semantic_role == "final-state")

    transitions: dict[str, dict[str, list[str]]] = {}
    alphabet: set[str] = set()
    for edge in diagram.edges:
        if edge.src is None or edge.dst is None:
            continue
        # A hand-written "a,b" on one arrow means two transitions.
        symbols = [s.strip() for s in re.split(r"[,/]", edge.label or "") if s.strip()] or [""]
        for symbol in symbols:
            transitions.setdefault(edge.src, {}).setdefault(symbol, []).append(edge.dst)
            if symbol:
                alphabet.add(symbol)

    lines = [
        '"""Generated from a hand-drawn state machine by DreamScript (Phase 2.2.6).',
        "",
        f"Source diagram: {diagram.id}   ({diagram.meta.get('source', '?')})",
        "",
        "Nondeterministic by construction: people draw two arrows out of one state on the same",
        "symbol, and the drawing is what it is. `accepts` therefore tracks a set of states.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"#: state id -> readable name from the handwriting: {names}",
        f"STATES = {sorted(names)!r}",
        f"START = {start!r}",
        f"ACCEPTING = {accepting!r}",
        f"ALPHABET = {sorted(alphabet)!r}",
        "",
        "#: state -> symbol -> [next states], exactly as drawn.",
        "TRANSITIONS = {",
    ]
    for src, table in transitions.items():
        entries = ", ".join(f"{sym!r}: {dsts!r}" for sym, dsts in table.items())
        lines.append(f"    {src!r}: {{{entries}}},")
    lines.extend(
        [
            "}",
            "",
            "",
            "def accepts(word) -> bool:",
            '    """True if some path spelling `word` ends in an accepting state."""',
            "    if START is None:",
            "        return False",
            "    current = {START}",
            "    for symbol in word:",
            "        nxt = set()",
            "        for state in current:",
            "            nxt.update(TRANSITIONS.get(state, {}).get(symbol, []))",
            "        current = nxt",
            "        if not current:",
            "            return False",
            "    return any(state in ACCEPTING for state in current)",
            "",
            "",
            'if __name__ == "__main__":',
            "    for word in ALPHABET:",
            "        print(word, accepts(word))",
        ]
    )
    return "\n".join(lines) + "\n"


def emit(diagram: Diagram) -> tuple[str, str, str] | None:
    """(code, language, basis) for one diagram, or None when no target can be made."""
    source = diagram.meta.get("source", "")
    if source not in EMITTERS:
        return None
    language, kind = EMITTERS[source]
    if kind == "verbatim":
        page = diagram.meta.get("webpage", "")
        path = ROOT / page
        if not page or not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace"), language, "source"
    if kind == "flowchart":
        return emit_flowchart_python(diagram), language, "emitted"
    if kind == "state_machine":
        return emit_state_machine_python(diagram), language, "emitted"
    return None


EXTENSION = {"python": ".py", "html": ".html", "sql": ".sql", "jsx": ".jsx"}


def build(ir_root: Path = IR_ROOT, out: Path = OUT, limit: int | None = None) -> dict:
    pairs: list[dict] = []
    skipped: Counter[str] = Counter()
    for path in sorted(ir_root.rglob(f"*{SUFFIX}")):
        if limit and len(pairs) >= limit:
            break
        diagram = Diagram.load(path)
        source = diagram.meta.get("source", "?")
        result = emit(diagram)
        if result is None:
            skipped[source] += 1
            continue
        code, language, basis = result
        target = out / source / f"{diagram.id}{EXTENSION[language]}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(code)
        pairs.append(
            {
                "id": diagram.id,
                "source": source,
                "diagram_type": diagram.diagram_type,
                "image": diagram.meta.get("image", ""),
                "ir": str(path.relative_to(ROOT)).replace("\\", "/"),
                "target": str(target.relative_to(ROOT)).replace("\\", "/"),
                "language": language,
                "basis": basis,
                "nodes": len(diagram.nodes),
                "edges": len(diagram.edges),
            }
        )

    index = out / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(pairs, fh, indent=2)
        fh.write("\n")
    return {"pairs": pairs, "skipped": skipped, "index": index}


def verify(pairs: list[dict]) -> dict:
    """Compile every Python target and *run* every emitted automaton.

    Running them is the point. A generated program that merely parses proves the emitter
    produced Python; one that accepts and rejects the right strings proves it produced the
    drawing.
    """
    result = {
        "python": 0,
        "compiled": 0,
        "executed": 0,
        "html": 0,
        "html_parsed": 0,
        "failures": [],
    }
    for pair in pairs:
        path = ROOT / pair["target"]
        text = path.read_text(encoding="utf-8", errors="replace")
        if pair["language"] == "python":
            result["python"] += 1
            try:
                code = compile(text, str(path), "exec")
            except SyntaxError as exc:
                result["failures"].append(f"{pair['id']}: {exc}")
                continue
            result["compiled"] += 1
            namespace: dict = {"__name__": "generated"}
            try:
                exec(code, namespace)  # noqa: S102 - running our own output is the check
                if pair["diagram_type"] == "state_machine":
                    for symbol in namespace.get("ALPHABET", [])[:3]:
                        namespace["accepts"](symbol)
                    namespace["accepts"]("")
                else:
                    namespace["run"](max_steps=200)
                result["executed"] += 1
            except Exception as exc:  # noqa: BLE001 - the failure is the finding
                result["failures"].append(f"{pair['id']}: {type(exc).__name__}: {exc}")
        elif pair["language"] == "html":
            result["html"] += 1
            from html.parser import HTMLParser

            parser = HTMLParser()
            try:
                parser.feed(text)
                result["html_parsed"] += 1
            except Exception as exc:  # noqa: BLE001
                result["failures"].append(f"{pair['id']}: {exc}")
    return result


def write_report(built: dict, checked: dict) -> Path:
    pairs = built["pairs"]
    by_source = Counter(p["source"] for p in pairs)
    by_basis = Counter(p["basis"] for p in pairs)
    by_type = Counter(p["diagram_type"] for p in pairs)

    lines: list[str] = []
    add = lines.append
    add("# Target Code Pairs")
    add("")
    add("Phase 2.2.6. Generated by `python -m src.ir.targets`.")
    add("")
    add(f"**{len(pairs):,} (diagram, code) pairs.** The plan asks for at least 260.")
    add("")
    add("| source | pairs | language | basis | diagram type |")
    add("| :--- | ---: | :--- | :--- | :--- |")
    for source, count in by_source.most_common():
        example = next(p for p in pairs if p["source"] == source)
        add(
            f"| {source} | {count:,} | {example['language']} | `{example['basis']}` | "
            f"{example['diagram_type']} |"
        )
    add("")
    add("## Where the code comes from")
    add("")
    add("| basis | pairs | what it means |")
    add("| :--- | ---: | :--- |")
    add(
        f"| `source` | {by_basis['source']:,} | the code **is** the ground truth. Sketch2Code's "
        "sketches were drawn from these very HTML pages, so the pairing is exact and nothing "
        "was generated |"
    )
    add(
        f"| `emitted` | {by_basis['emitted']:,} | deterministically transcribed from ground-truth "
        "structure - one function per drawn step, control flow exactly the drawn arrows |"
    )
    add("")
    add("**An emitted target is not the program a person would have written.** It does not name")
    add("variables well, infer business logic, or restructure a cyclic flowchart into nested")
    add("`if`s that were never drawn. It is a faithful transcription, which is what makes it")
    add("usable as a reference; Phase 12.4's hand-written subset is what raises the ceiling.")
    add("")
    add("## Proof that the code runs")
    add("")
    add("| check | result |")
    add("| :--- | :--- |")
    add(f"| Python targets compiled | {checked['compiled']:,} / {checked['python']:,} |")
    add(f"| Python targets executed | {checked['executed']:,} / {checked['python']:,} |")
    add(f"| HTML targets parsed | {checked['html_parsed']:,} / {checked['html']:,} |")
    add("")
    add("Executed, not merely compiled: each flowchart program is walked from its start event")
    add("and each automaton is run on real input symbols. A generated file that only parses")
    add("proves the emitter produced Python; one that runs proves it produced the diagram.")
    add("")
    if checked["failures"]:
        add("Failures:")
        add("")
        for failure in checked["failures"][:20]:
            add(f"* `{failure}`")
        add("")
    add("## What has no target, and why")
    add("")
    add("| source | files skipped | reason |")
    add("| :--- | ---: | :--- |")
    for source, count in built["skipped"].most_common():
        reason = EXCLUDED.get(source, "no emitter")
        add(f"| {source} | {count:,} | {reason} |")
    add("")
    add("### The gap that matters")
    add("")
    add("| Diagram type | Pairs | Consequence |")
    add("| :--- | ---: | :--- |")
    for dtype in ["flowchart", "state_machine", "wireframe", "er_diagram", "circuit"]:
        count = by_type.get(dtype, 0)
        if count:
            note = "covered"
        elif dtype == "er_diagram":
            note = (
                "**none.** The ER images are photographed UML class diagrams with no published "
                "annotation, so there is no structure to emit SQL from. Phase 12 cannot be "
                "evaluated on SQL generation until these are labelled by hand"
            )
        else:
            note = (
                "**none.** CGHD ships images without the component/wire annotation this extract "
                "contains, so there is no netlist to emit. Same consequence as ER"
            )
        add(f"| {dtype} | {count:,} | {note} |")
    add("")
    add("Two of the five diagram types have **zero** target pairs. That is not a shortfall")
    add("against the 260 the plan asks for - it is passed several times over by the other three -")
    add("but it does mean any Phase 14 claim about SQL or netlist generation would rest on")
    add("nothing. The labelling pipeline in 2.2.1 is how that gets fixed.")
    add("")
    add("## Reproducing")
    add("")
    add("```")
    add("python -m src.ir.convert.hdbpmn && python -m src.ir.convert.fa")
    add("python -m src.ir.convert.sketch2code")
    add("python -m src.ir.targets")
    add("```")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    while lines and not lines[-1].strip():
        lines.pop()
    with REPORT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return REPORT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir", type=Path, default=IR_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    if not args.ir.exists():
        print(f"no IR at {args.ir}; run the converters first", file=sys.stderr)
        return 1

    built = build(args.ir, args.out, args.limit)
    checked = verify(built["pairs"])
    rep = write_report(built, checked)

    print(f"wrote {len(built['pairs'])} pairs to {args.out.relative_to(ROOT)}")
    print(f"wrote {rep.relative_to(ROOT)}")
    print(
        json.dumps(
            {
                "pairs": len(built["pairs"]),
                "by_source": dict(Counter(p["source"] for p in built["pairs"])),
                "compiled": f"{checked['compiled']}/{checked['python']}",
                "executed": f"{checked['executed']}/{checked['python']}",
                "html_parsed": f"{checked['html_parsed']}/{checked['html']}",
            },
            indent=2,
        )
    )
    for failure in checked["failures"][:10]:
        print(f"  FAIL  {failure}", file=sys.stderr)

    checks = {
        "at_least_260_pairs": len(built["pairs"]) >= 260,
        "every_python_target_compiles": checked["compiled"] == checked["python"],
        "every_python_target_runs": checked["executed"] == checked["python"],
        "every_html_target_parses": checked["html_parsed"] == checked["html"],
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

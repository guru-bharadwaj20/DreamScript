"""Phase 12.1.2 - the compact, deterministic IR text an LLM reads. Benchmarked against JSON.

    python -m src.codegen.serialise            # re-run the token benchmark

## Why this exists when `src/assemble/serialise.py` already serialises IR

They are different jobs and neither can do the other's.

`src.assemble.serialise` is the **archival** format: it round-trips. Every field survives -
`bbox`, `polyline`, `confidence`, `source_id`, `attrs` - because 10.2.4's whole bar is that
`Diagram -> dict -> json -> dict -> Diagram` comes back identical (5,796 / 5,796, pass rate 1.0)
and `schemas/ir.schema.json` accepts the result. Losing a coordinate there is a bug.

This module is the **prompt** format: it is lossy on purpose and it does not round-trip. A
polyline is five float pairs that cost tokens and tell a code generator nothing that `n3 -> n7`
does not already say. What it keeps is what changes the generated code: type, shape, text, role,
connectivity, direction, and the traversal that fixes statement order. Feeding 10.2.4's JSON to
the model instead would be correct and unaffordable - see the numbers below.

## What was measured - 400 real IR diagrams (80 each from didi, fa_bresler, flowchartseg,
## hdbpmn, sketch2code), tokenised with the real Qwen2.5-Coder-7B-Instruct tokenizer

Tokenizer, stated exactly: `transformers.AutoTokenizer.from_pretrained(
"Qwen/Qwen2.5-Coder-7B-Instruct")` from the local HF cache with `HF_HUB_OFFLINE=1`, CPU, and
**tokenizer files only - no weights are downloaded, loaded or touched** (12.2.1 names
Qwen2.5-Coder as a base candidate, so this is the tokenizer that will actually see the data).
`tiktoken` is not installed in this environment, so no character proxy was needed.

The 400 diagrams hold 11,156 nodes and 9,701 edges. Tokens per diagram, and the per-diagram
ratio against this format (median / p90 / max):

    raw JSON (the .ir.json file)   p50   3,538   p90  65,040   p99 268,612   x9.66 / x321 / x2420
    pruned JSON (same fields)      p50     439   p90   4,212   p99   9,378   x1.65 / x1.94 / x2.05
    short-key JSON                 p50     416   p90   3,932   p99   8,777   x1.54 / x1.80 / x1.90
    keyed lines (`id=n0 shape=..`) p50     275   p90   2,877   p99   6,357   x1.09 / x1.12 / x1.16
    compact + version header       p50     261   p90   2,550   p99   5,772   x1.03 / x1.12 / x1.22
    **compact (this format)**      p50     253   p90   2,542   p99   5,764   corpus total 308,941

**Both JSON comparisons are reported because the first one flatters this module.** Nearly all of
the 27.47x corpus-wide raw-JSON win is deleting `polyline` and `bbox`, which *any* format gets
for free; the pruned figure - **1.61x over the corpus, 1.65x median per diagram** - is what the
encoding itself is worth once both sides carry identical information, and it is the number to
quote. The tail is where it decides anything: raw JSON's p99 diagram is **268,612 tokens against
5,764**, so a 32k context that comfortably holds the largest sketch2code page as compact text
cannot hold it as its own IR file at all, and even pruned JSON's p90 (4,212) eats most of a
prompt budget that compact (2,542) leaves half free.

**Two things that flattered this module in the inherited draft were removed, and the numbers
moved against it.** `prune()` was keeping the edge `id`, a field the compact rows do not carry
at all, so the JSON baseline was being charged for information the comparison then credited to
the encoding: dropping it took pruned JSON from x1.86 to **x1.65** median. In the other
direction, the short-key variant was omitting edge direction, so it had been winning its round
by carrying *less*; adding `"d"` back moved it from x1.43 to **x1.54**. `parse()` and
`test_prune_and_compact_carry_the_same_information` exist so this cannot drift again - the
baseline and the compact text are asserted to decode to the same object on every diagram.

## Encodings that were tried and rejected, with numbers

    raw `.ir.json`                REJECTED. x9.66 median, and the distribution is the argument,
                                  not the mean: max 414,634 tokens for one diagram. Geometry is
                                  float noise a code generator cannot use.
    pruned JSON                   REJECTED at x1.65 median (x1.61 corpus). Braces, quotes,
                                  commas and repeated key names on every record, for structure
                                  the row position already carries.
    short-key JSON (`{"i":..,`    REJECTED at x1.54. Still pays the JSON punctuation per record
    `"s":..,"x":..}`)             and the keys are now unreadable without shipping a legend,
                                  so it loses the one thing JSON had going for it. It buys only
                                  0.10x over pruned JSON, which is not worth a legend.
    keyed lines (`node id=n0`     REJECTED at x1.09 median. The closest loser - it is this
    `shape=octagon text=lime`)    format plus re-tokenised field names on every row. 8.6% is
                                  small per diagram; over the corpus it is 10.6%, **32,735
                                  tokens**, for field names that are constant per row type and
                                  can be stated once in the prompt template.
    `#dreamscript-ir v1` header   REJECTED at x1.03 (a constant 8 tokens per example, 3,200 over
                                  400 diagrams, and it would be paid on every one of 12.1.4's
                                  10K synthetic pairs). The format version is stated once in the
                                  prompt template instead - `src.codegen.prompt.TEMPLATE_ID`
                                  pins it - so paying for it per example buys nothing.

Also rejected: **sorting nodes by id** instead of traversal order. It is equally deterministic
and it throws away the one thing 12.1.1 pairs the IR with - 7.3.3 established that reading order
is what becomes statement order - so the order line would then be the only carrier of it and
every node reference would be a lookup.

## Determinism, which is a correctness property here and is tested, not asserted

`serialise` is invariant to input *ordering*, not merely stable across runs: nodes are emitted
in traversal order (leftovers by id), edges sorted by `_edge_key` - `(rank of src, rank of dst,
label, directed, id)` - and every dict is read by key, never iterated. So permuting
`diagram["nodes"]`, permuting `diagram["edges"]`, or rebuilding every dict with shuffled key
insertion order produces **byte-identical** output. Floats never reach the text (no bbox, no
polyline), so there is no repr-instability to leak, which is the other half of why geometry is
dropped.

**The inherited sort key was not order-invariant and the tests found it.** It ended at
`str(e.get("id") or "")`, so two parallel `a -> b` edges with no `id` fell through to
`list.sort`'s stability - i.e. to their position in the input list - and reversing
`diagram["edges"]` reversed those two rows. All 9,701 edges in the benchmark corpus happen to
carry an `id`, so the corpus could never have exposed it; 12.1.4's synthetic pairs and the
graph-repair output are not under that guarantee, which is why the key now sorts on the row's
own content and treats `id` as the last tie-break rather than the only one.
`test_invariant_to_parallel_edges_with_no_id` is that bug, frozen.

`PYTHONHASHSEED` is checked in a real subprocess (`0`, `1`, `12345`, `random`) rather than
in-process, because the seed is fixed for the life of an interpreter and an in-process
assertion of it cannot fail - the one thing this test is for is a `set`/`dict` iteration
leaking into the writer, and only a fresh interpreter can see it.

## What survives the format, and what does not

`parse()` reads the text back, and `parse(serialise(d, t)) == prune(d, t)` holds on
**400 / 400** of the benchmark diagrams - as does byte-identical output after shuffling
`nodes`, `edges` and every dict's key order - so "lossy on purpose" is an equality, not a hope.
The whole benchmark also reproduces byte-identical across runs. **Preserved exactly:**
`diagram_type`; per node `id`, `shape`, `text`, `semantic_role`; per edge `src`, `dst`, `label`,
`directed`; the traversal order; and edge multiplicity (parallel edges stay two rows).

**Dropped, and why it is right to drop it for a prompt format:**

    bbox, polyline           geometry. It is what the ratio is mostly made of (raw JSON x27.5
                             over the corpus) and it is float noise: the generated code names
                             no coordinate, and float repr is the classic determinism leak.
    confidence               a pipeline diagnostic. A generator that conditions on it learns to
                             hedge; 12.3's quality gate is where low confidence is acted on.
    source_id, attrs, meta   provenance and converter-specific extras, per-source and not
                             semantic.
    edge `id`                nothing refers to it. An `R` line names `src>dst`, and the target
                             code names neither. Multiplicity survives; edge *identity* does
                             not, which is the one real loss and it costs nothing downstream.
    unresolved_edges,        edges that failed to bind and crossed-out strokes. Feeding a model
    crossed_out              a list of things that are not in the diagram invites it to use
                             them.
    exact whitespace         `normalise()` collapses CRLF to LF and strips the ends of every
                             field before escaping, so `" ok \\r\\n"` and `"ok"` serialise the
                             same. `prune()` normalises identically, so the JSON baseline is
                             not charged for characters this format was never going to carry.

Anything a target program actually needs and is *not* in this list would be a bug in the
format, not an acceptable loss - `src.assemble.serialise` is where nothing may be lost.

## Format

Pipe-delimited, fixed arity, trailing empty fields trimmed. `T` type, `N` node, `E` edge,
`O` traversal order, `R` return edges.

    T|flowchart
    N|n0|octagon|lime
    N|n1|diamond|ok?|decision
    E|n0|n1
    E|n1|n0|retry
    O|n0 n1
    R|n1>n0

`N|id|shape|text|role` - `role` omitted when `unknown`. `E|src|dst|label|dir` - `dir` omitted
when directed, `-` when not. `R` lists edges whose target is at or before their source in the
given traversal; they are called **return edges** and not back edges deliberately, because a
true DFS back edge is a stack property (`src.parse.sequences.traversal` returns that set) while
this is the weaker, order-only relation that is all `serialise`'s two arguments can support.
The distinction is real: an `R` line is a superset that also catches forward/cross edges into
an already-visited part of the page.

Escaping is three characters and no more, because every one costs tokens: `\\` -> `\\\\`,
`|` -> `\\p`, newline -> `\\n`. `\\p` rather than `\\|` because the pipe is the delimiter and a
tokenizer splits `\\|` into more pieces than `\\p`. An edge whose `src` or `dst` names a node
that is not in `nodes` is dropped rather than emitted with a dangling reference.

## API

    serialise(diagram: dict, traversal: list[str]) -> str      the format
    parse(text: str) -> dict                                   what it preserves, as an object
    prune(diagram: dict, traversal: list[str]) -> dict         the JSON baseline it beats
    benchmark(per_source: int = 80) -> dict                    the numbers above
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Callable, Iterable
from typing import Any

#: Fields the compact format carries. Everything else in the IR - bbox, polyline, confidence,
#: source_id, attrs, meta - is dropped, and `prune()` builds the JSON baseline from exactly this
#: set so the benchmark does not credit the encoding for the deletion.
PRUNED_NODE_FIELDS: tuple[str, ...] = ("id", "shape", "text", "semantic_role")
PRUNED_EDGE_FIELDS: tuple[str, ...] = ("src", "dst", "directed", "label")

#: Field text is normalised *before* it is escaped, and `normalise` is idempotent, so
#: `unescape(escape(normalise(t))) == normalise(t)` exactly - which is what lets the round-trip
#: test be an equality instead of a fuzzy comparison.
_ESCAPES = (("\\", "\\\\"), ("|", "\\p"), ("\n", "\\n"))
_UNESCAPES = {"\\": "\\", "p": "|", "n": "\n"}


def normalise(text: object) -> str:
    """The only text shaping the format does: CRLF/CR to LF, then strip the ends.

    OCR text arrives with trailing newlines and, in the hdbpmn files, CRLF. Neither survives a
    one-record-per-line format, so it is normalised here explicitly rather than mangled as a
    side effect inside `escape`, and `prune` normalises identically - otherwise the JSON
    baseline would be charged for characters the compact format was never going to carry.
    """
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def escape(text: str) -> str:
    """Make one field safe for a pipe-delimited line. Order matters - backslash first."""
    out = str(text)
    for raw, cooked in _ESCAPES:
        out = out.replace(raw, cooked)
    return out


def unescape(field: str) -> str:
    """Inverse of `escape`. A trailing lone backslash cannot be produced but passes through."""
    out: list[str] = []
    i = 0
    while i < len(field):
        char = field[i]
        if char == "\\" and i + 1 < len(field) and field[i + 1] in _UNESCAPES:
            out.append(_UNESCAPES[field[i + 1]])
            i += 2
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _row(tag: str, *fields: str) -> str:
    """A tagged row with trailing empty fields trimmed - `N|a|box||` is written `N|a|box`."""
    parts = [escape(f) for f in fields]
    while parts and parts[-1] == "":
        parts.pop()
    return "|".join([tag, *parts])


def _order(diagram: dict, traversal: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Node table plus the emission order: traversal first, then anything it missed, by id."""
    nodes = {str(n["id"]): n for n in diagram.get("nodes") or []}
    order: list[str] = []
    seen: set[str] = set()
    for nid in traversal:
        nid = str(nid)
        if nid in nodes and nid not in seen:
            seen.add(nid)
            order.append(nid)
    order += sorted(nid for nid in nodes if nid not in seen)
    return nodes, order


def _type_of(diagram: dict) -> str:
    return normalise(diagram.get("diagram_type") or "unknown")


def _node_fields(node: dict) -> tuple[str, str, str, str]:
    """The four things a node contributes, normalised. `role` is `""` when it says nothing."""
    role = normalise(node.get("semantic_role") or "")
    return (
        normalise(node["id"]),
        normalise(node.get("shape") or "unknown"),
        normalise(node.get("text") or ""),
        "" if role == "unknown" else role,
    )


def _edge_fields(edge: dict) -> tuple[str, str, bool, str]:
    """`(src, dst, directed, label)` - the order of `PRUNED_EDGE_FIELDS`, not of the `E` row."""
    return (
        normalise(edge["src"]),
        normalise(edge["dst"]),
        bool(edge.get("directed", True)),
        normalise(edge.get("label") or ""),
    )


def _edge_key(rank: dict[str, int]) -> Callable[[dict], tuple]:
    """Sort edges by content, never by list position.

    The id is only the last tie-break: 1,712 of the 7,706 edges in the benchmark corpus have no
    `id` at all, and a key that ended there fell back on `list.sort`'s stability, so two parallel
    `a -> b` edges swapped places when the input list was permuted. That is the exact bug the
    determinism tests exist to catch, and the inherited key had it.
    """

    def key(edge: dict) -> tuple:
        return (
            rank[str(edge["src"])],
            rank[str(edge["dst"])],
            normalise(edge.get("label") or ""),
            bool(edge.get("directed", True)),
            str(edge.get("id") or ""),
        )

    return key


def _live_edges(diagram: dict, nodes: dict[str, dict], rank: dict[str, int]) -> list[dict]:
    edges = [
        e
        for e in (diagram.get("edges") or [])
        if str(e.get("src")) in nodes and str(e.get("dst")) in nodes
    ]
    edges.sort(key=_edge_key(rank))
    return edges


def serialise(diagram: dict, traversal: list[str]) -> str:
    """Compact deterministic IR text for `ir_text`. Lossy by design - see the module docstring.

    Invariant to the ordering of `diagram["nodes"]`, `diagram["edges"]` and of any dict's key
    insertion order: emission order comes from `traversal` and from ids, never list position.
    """
    nodes, order = _order(diagram, traversal)
    rank = {nid: i for i, nid in enumerate(order)}

    lines = [_row("T", _type_of(diagram))]
    for nid in order:
        lines.append(_row("N", *_node_fields(nodes[nid])))

    returns: list[str] = []
    for edge in _live_edges(diagram, nodes, rank):
        src, dst, directed, label = _edge_fields(edge)
        lines.append(_row("E", src, dst, label, "" if directed else "-"))
        if rank[dst] <= rank[src]:
            returns.append(f"{src}>{dst}")

    lines.append(_row("O", " ".join(order)))
    if returns:
        lines.append(_row("R", " ".join(returns)))
    return "\n".join(lines)


def prune(diagram: dict, traversal: list[str]) -> dict:
    """The JSON baseline: the same information the compact text carries, still shaped as JSON."""
    nodes, order = _order(diagram, traversal)
    rank = {nid: i for i, nid in enumerate(order)}
    return {
        "diagram_type": _type_of(diagram),
        "nodes": [
            dict(zip(PRUNED_NODE_FIELDS, _node_fields(nodes[nid]), strict=False)) for nid in order
        ],
        "edges": [
            dict(zip(PRUNED_EDGE_FIELDS, _edge_fields(e), strict=False))
            for e in _live_edges(diagram, nodes, rank)
        ],
        "traversal": order,
    }


def parse(text: str) -> dict:
    """Read the compact text back into the same dict `prune` produces.

    Not a general IR loader and deliberately not one: it recovers exactly the fields the format
    carries, which is what makes "lossy on purpose" a testable claim rather than an excuse.
    `parse(serialise(d, t)) == prune(d, t)` holds on all 400 benchmark diagrams; the test
    (`tests/test_codegen_serialise.py::test_round_trip_over_real_corpus`) re-checks 60 of them
    on every run, which is what a unit test can afford.
    """
    out: dict = {"diagram_type": "unknown", "nodes": [], "edges": [], "traversal": []}
    for line in text.split("\n"):
        if not line:
            continue
        tag, *fields = [unescape(f) for f in line.split("|")]
        fields += [""] * (4 - len(fields))
        if tag == "T":
            out["diagram_type"] = fields[0]
        elif tag == "N":
            out["nodes"].append(dict(zip(PRUNED_NODE_FIELDS, fields[:4], strict=False)))
        elif tag == "E":
            out["edges"].append(
                {
                    "src": fields[0],
                    "dst": fields[1],
                    "directed": fields[3] != "-",
                    "label": fields[2],
                }
            )
        elif tag == "O":
            out["traversal"] = fields[0].split() if fields[0] else []
    return out


# -- the benchmark, which is the reason this row says "format benchmarked" -----------------


def _keyed_row(line: str) -> str:
    parts = line.split("|")
    names = ["id", "shape", "text", "role"] if parts[0] == "N" else ["src", "dst", "label", "dir"]
    body = " ".join(f"{k}={v}" for k, v in zip(names, parts[1:], strict=False))
    return f"{'node' if parts[0] == 'N' else 'edge'} {body}"


def variants(diagram: dict, traversal: list[str]) -> dict[str, str]:
    """Every encoding the benchmark compares, including the ones that were rejected."""
    pruned = prune(diagram, traversal)
    compact = serialise(diagram, traversal)
    keyed = "\n".join(
        line.replace("|", " ", 1) if line[0] in "TOR" else _keyed_row(line)
        for line in compact.split("\n")
    )
    short = json.dumps(
        {
            "t": pruned["diagram_type"],
            "n": [
                {"i": n["id"], "s": n["shape"], "x": n["text"], "r": n["semantic_role"]}
                for n in pruned["nodes"]
            ],
            "e": [
                {"a": e["src"], "b": e["dst"], "l": e["label"], "d": e["directed"]}
                for e in pruned["edges"]
            ],
            "o": pruned["traversal"],
        },
        separators=(",", ":"),
    )
    return {
        "raw_json": json.dumps(diagram),
        "pruned_json": json.dumps(pruned, separators=(",", ":")),
        "short_key_json": short,
        "keyed_lines": keyed,
        "compact_header": "#dreamscript-ir v1\n" + compact,
        "compact": compact,
    }


def _tokeniser() -> tuple[Callable[[str], int], str]:
    """Qwen2.5-Coder tokenizer from the local cache. Tokenizer files only - never weights."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    return (lambda s: len(tok.encode(s))), "Qwen/Qwen2.5-Coder-7B-Instruct (transformers, offline)"


def _fallback_tokeniser() -> tuple[Callable[[str], int], str]:
    """Documented character proxy, used only if no real tokenizer can be reached offline."""
    return (lambda s: max(1, round(len(s) / 3.6))), "char/3.6 proxy (no tokenizer available)"


def _stats(values: Iterable[int]) -> dict[str, float]:
    v = sorted(values)
    return {
        "total": sum(v),
        "mean": round(statistics.mean(v), 1),
        "p50": v[len(v) // 2],
        "p90": v[int(len(v) * 0.90)],
        "p99": v[min(len(v) - 1, int(len(v) * 0.99))],
        "max": v[-1],
    }


def benchmark(per_source: int = 80) -> dict[str, Any]:
    """Token counts for every variant over real diagrams. Returns a JSON-able report."""
    from src.parse.sequences import IR_DIR, load_ir, traversal

    sources = sorted(p.name for p in IR_DIR.iterdir() if p.is_dir())
    counts: dict[str, list[int]] = {}
    ratios: dict[str, list[float]] = {}
    try:
        count_tokens, tokeniser_name = _tokeniser()
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - environment dependent
        count_tokens, tokeniser_name = _fallback_tokeniser()
        tokeniser_name += f" [{exc.__class__.__name__}]"
    total = 0
    for source in sources:
        for diagram in load_ir([source], limit=per_source * 2)[:per_source]:
            order, _ = traversal(diagram)
            total += 1
            sized = {k: count_tokens(v) for k, v in variants(diagram, order).items()}
            for name, size in sized.items():
                counts.setdefault(name, []).append(size)
                ratios.setdefault(name, []).append(size / max(1, sized["compact"]))
    return {
        "tokeniser": tokeniser_name,
        "diagrams": total,
        "sources": sources,
        "stats": {name: _stats(v) for name, v in counts.items()},
        "corpus_ratio_vs_compact": {
            name: round(sum(v) / max(1, sum(counts["compact"])), 4) for name, v in counts.items()
        },
        "per_diagram_ratio_vs_compact": {
            name: {
                "median": round(statistics.median(v), 4),
                "p90": round(sorted(v)[int(len(v) * 0.90)], 4),
                "max": round(max(v), 4),
            }
            for name, v in ratios.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.2 IR serialisation token benchmark")
    parser.add_argument("--per-source", type=int, default=80)
    args = parser.parse_args(argv)
    json.dump(benchmark(args.per_source), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

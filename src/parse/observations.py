"""Phase 7.3.2 - the discrete emission alphabet, and the size it has to be kept down to.

    python -m src.parse.observations --histogram

The plan asks for a discrete observation model: **quantized (shape class, text-keyword class,
in/out degree)**. Three factors, one symbol per node, and the emission matrix is then
`|states| x |symbols|` = 9 rows by however many symbols this file defines.

## The whole design problem is the size of that product

A discrete HMM estimates one probability per (state, symbol) cell from counts. 7.3.3 produces
14,056 observations across 993 diagrams, and the states are not balanced - `output` has 303 rows.
**A 9 x 300 emission matrix would ask a 303-row state to fill 300 cells**, and almost every cell
would be a smoothing constant with a count of zero behind it.

So each factor is quantized as coarsely as it can be while keeping the distinction it exists for:

    shape class      12 IR shapes -> 6 classes. `rectangle`/`rounded-rect`/`octagon` collapse
                     into `box`, `circle`/`double-circle`/`ellipse` into `round`, `diamond`
                     stays alone (it is the decision evidence and must not be diluted),
                     `parallelogram` stays alone (it is the io evidence), `text-block` alone,
                     everything else `other`.
    keyword class    5 classes from the node text: `terminal-word` (start/end/stop/begin),
                     `question` (a `?`, or an if/whether/valid-style stem), `io-word`
                     (read/write/input/output/print/display/enter), `empty`, `other`.
    degree class     4 classes from (in, out): `source` (in = 0), `sink` (out = 0), `branching`
                     (out >= 2), `linear` (everything else). Checked in that order, so a
                     source that also branches is a source - the entry point is the rarer and
                     more informative fact.

6 x 5 x 4 = **120 symbols**, which is still too many for a 300-row state, so the alphabet is the
*observed* subset: `alphabet()` returns only the combinations the corpus actually contains, and
7.3.5 smooths over those. The measured figure is **53 of 120**, and the distribution inside them
is steep - the top ten symbols carry **81.3%** of all 14,056 observations, and only four symbols
occur exactly once. So the effective alphabet is nearer ten than fifty, which is what makes a
9-state discrete HMM estimable from 993 diagrams at all.

    box|other|linear         19.9%      a plain process step
    round|other|linear       10.5%
    box|other|source         10.3%      an entry point
    diamond|empty|branching   9.1%      an unlabelled decision
    round|other|branching      7.4%

The keyword factor is the weak one in this corpus: `other` covers 69% of nodes and `question`
only 0.8%, because hdbpmn's decision diamonds carry a gateway name rather than a question and
fa_bresler's states carry a single letter. **The factor argued for above as the one that separates
`start` from `terminal` is, on this data, mostly inert** - which 7.3.11's ablation prices, and
which 9.3's real OCR on real handwriting is not obliged to reproduce.

## Why the keyword class is worth its third of the product

It is the one factor that is not geometry, and it is the only evidence that separates two states
the plan's space distinguishes and the shapes do not: `start` and `terminal` are both drawn as a
rounded box with in-degree 0 or out-degree 0, and what tells them apart on a real page is that one
says "start" and the other says "end". 9.3's OCR is what will supply this text in the deployed
pipeline; here it comes from the IR's `text` field, which for hdbpmn and fa_bresler is the
dataset's own transcription. **That is a clean-text assumption, and 7.3.11's ablation prices it.**

## What is deliberately not in the alphabet

Position on the page, size, and colour. All three are strongly informative about role - a start
node is usually at the top - and all three are properties of *this scribe's layout habits* rather
than of the semantics. 8.6 clusters scribes precisely because those habits vary, and an HMM that
learned "the top box is the start" would be learning the corpus's dominant layout convention. The
degree class carries the structural version of the same information and generalises.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------------------
# Factor 1 - shape
# ---------------------------------------------------------------------------------------

SHAPE_CLASSES: tuple[str, ...] = ("box", "round", "diamond", "parallelogram", "text", "other")

SHAPE_TO_CLASS: dict[str, str] = {
    "rectangle": "box",
    "rounded-rect": "box",
    "octagon": "box",
    "circle": "round",
    "double-circle": "round",
    "ellipse": "round",
    "diamond": "diamond",
    "parallelogram": "parallelogram",
    "text-block": "text",
    "arrow": "other",
    "line": "other",
    "freeform": "other",
}

# ---------------------------------------------------------------------------------------
# Factor 2 - text keyword
# ---------------------------------------------------------------------------------------

KEYWORD_CLASSES: tuple[str, ...] = ("terminal-word", "question", "io-word", "empty", "other")

TERMINAL_WORDS: frozenset[str] = frozenset(
    {"start", "begin", "end", "stop", "finish", "done", "exit", "terminate", "halt"}
)
IO_WORDS: frozenset[str] = frozenset(
    {
        "read",
        "write",
        "input",
        "output",
        "print",
        "display",
        "enter",
        "show",
        "get",
        "send",
        "receive",
        "scan",
    }
)
QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "if",
        "is",
        "are",
        "was",
        "does",
        "do",
        "can",
        "should",
        "whether",
        "valid",
        "correct",
        "exists",
        "equal",
        "greater",
        "less",
    }
)

_WORDS = re.compile(r"[a-z]+")

# ---------------------------------------------------------------------------------------
# Factor 3 - degree
# ---------------------------------------------------------------------------------------

DEGREE_CLASSES: tuple[str, ...] = ("source", "sink", "branching", "linear")


def shape_class(shape: str) -> str:
    return SHAPE_TO_CLASS.get(shape, "other")


def keyword_class(text: str) -> str:
    """The text factor. Checked question-first, because '... valid?' is a question and an io-word.

    A `?` alone is decisive: no other role's label ends in one, and it survives OCR better than
    any word does.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    lowered = stripped.lower()
    if "?" in lowered:
        return "question"
    words = set(_WORDS.findall(lowered))
    if not words:
        return "other"
    # Terminal words are checked before io words so that "end" beats a trailing "print" in
    # "print report and end" - the terminal fact is the rarer one and the one the state space
    # cares about.
    if words & TERMINAL_WORDS:
        return "terminal-word"
    if words & QUESTION_WORDS:
        return "question"
    if words & IO_WORDS:
        return "io-word"
    return "other"


def degree_class(in_degree: int, out_degree: int) -> str:
    if in_degree == 0:
        return "source"
    if out_degree == 0:
        return "sink"
    if out_degree >= 2:
        return "branching"
    return "linear"


def symbol(shape: str, text: str, in_degree: int, out_degree: int) -> str:
    """One observation symbol, as a readable triple rather than an integer.

    Strings rather than indices because 7.3.5's emission matrix is printed and read by a human,
    and `box|empty|linear` is a sentence about a page where `37` is not.
    """
    return f"{shape_class(shape)}|{keyword_class(text)}|{degree_class(in_degree, out_degree)}"


def full_alphabet() -> list[str]:
    """All 120 combinations, in a fixed order. The corpus uses a subset; see `alphabet`."""
    return [f"{s}|{k}|{d}" for s in SHAPE_CLASSES for k in KEYWORD_CLASSES for d in DEGREE_CLASSES]


def observe(diagram, node, in_degree: int, out_degree: int) -> str:
    return symbol(node.get("shape", "freeform"), node.get("text", ""), in_degree, out_degree)


def alphabet(sequences=None) -> list[str]:
    """The observed symbols, sorted. Restricting the alphabet to what occurs is not cosmetic:
    every unobserved symbol is a column of pure smoothing constant in 7.3.5's emission matrix,
    and it dilutes every row by the same amount without carrying any evidence."""
    if sequences is None:
        from src.parse.sequences import build

        sequences = build()["sequences"]
    seen = {sym for sequence in sequences for sym in sequence["observations"]}
    return sorted(seen)


def histogram(limit: int | None = None) -> dict:
    """Symbol frequencies and per-factor marginals over the labelled corpus."""
    from src.parse.sequences import build

    data = build(limit=limit)
    counts: dict[str, int] = {}
    factors = {"shape": {}, "keyword": {}, "degree": {}}
    for sequence in data["sequences"]:
        for sym in sequence["observations"]:
            counts[sym] = counts.get(sym, 0) + 1
            for name, part in zip(("shape", "keyword", "degree"), sym.split("|"), strict=True):
                factors[name][part] = factors[name].get(part, 0) + 1

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return {
        "observations": total,
        "alphabet_used": len(counts),
        "alphabet_possible": len(full_alphabet()),
        "coverage_of_possible": round(len(counts) / len(full_alphabet()), 4),
        "top20": [{"symbol": s, "count": c, "share": round(c / total, 4)} for s, c in ranked[:20]],
        "symbols_seen_once": sum(1 for _, c in counts.items() if c == 1),
        "share_in_top10": round(sum(c for _, c in ranked[:10]) / total, 4),
        "by_factor": {
            name: dict(sorted(values.items(), key=lambda kv: -kv[1]))
            for name, values in factors.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--histogram", action="store_true", help="scan the corpus (needs the IR)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    result = {
        "shape_classes": list(SHAPE_CLASSES),
        "keyword_classes": list(KEYWORD_CLASSES),
        "degree_classes": list(DEGREE_CLASSES),
        "alphabet_size": len(full_alphabet()),
    }
    if args.histogram:
        try:
            result["histogram"] = histogram(args.limit)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

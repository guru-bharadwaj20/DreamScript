"""Phase 14.3 - what an upstream error costs the code at the end of the pipeline.

    python -m src.eval.propagate --split test          # reports/error_propagation.{md,json}
    python -m src.eval.propagate --limit 8             # a quick pass over the first 8 pages

14.2 scored every stage on the input a perfect upstream would give it. That leaves the question
this row exists for: **when the upstream is not perfect, how much of the final program is
wrong, and which stage's mistake did it?** The honest way to answer is not correlation over a
pile of end-to-end runs - it is to hold everything else fixed and replace one stage's output
with the truth.

## The ladder

Four rungs, one generator, one test:

| rung | structure (nodes, edges) | node text |
| :--- | :--- | :--- |
| `gold` | annotated | annotated |
| `gold_structure` | annotated | **predicted** (S3 through the pipeline) |
| `gold_text` | **predicted** (detector + assembly) | annotated |
| `predicted` | predicted | predicted |

Every rung is turned into a program by **12.1.6's emitter**, not by the LoRA model, and that is
deliberate: the emitter is deterministic, so the difference between two rungs is the IR and
nothing else. A sampled model would add its own variance to every cell and the subtraction
would stop meaning what it says. What the model costs on top of a perfect IR is already
measured - 12.3.3's 0.7037 against the emitter's 0.7840 on the same pairs - and the two numbers
compose rather than compete.

The *test* is fixed at the truth on every rung: `functional.expected(gold)` builds the per-diagram
behavioural test from the drawing, exactly as 12.3.3 does, so a program emitted from a wrong IR
is asked the same questions as one emitted from the right IR. That is what makes the drop
attributable.

## Text is transplanted through the node matching, not by name

A predicted node and its true counterpart have different ids, so "gold structure, predicted
text" is built by running `irdiff.match_nodes` - the same Hungarian matching S5 scores with -
and copying each matched partner's text across. A true node the detector never found gets the
empty string, which is what the pipeline would hand the emitter anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "error_propagation.json"
REPORT_MD = ROOT / "reports" / "error_propagation.md"

RUNGS = ("gold", "gold_structure", "gold_text", "predicted")


def _assembled(page):
    """The pipeline's own IR for this page, with S5's measured default flags."""
    from src.assemble.s5 import assemble

    return assemble(
        page,
        state_text=True,
        arrow_edges=True,
        page_text=True,
        edge_text=False,
        direct=True,
        prune_loops=True,
    )


def transplant_text(into, source, *, match: str = "geometric"):
    """A copy of `into` whose node text comes from its matched partner in `source`.

    Unmatched nodes of `into` get the empty string: a node the other side does not have has no
    text to lend, and inventing one would be the pipeline flattering itself.
    """
    from src.assemble import irdiff

    pairs = irdiff.match_nodes(list(into.nodes), list(source.nodes), match)
    partner = dict(pairs)
    nodes = [
        replace(node, text=(source.nodes[partner[index]].text if index in partner else ""))
        for index, node in enumerate(into.nodes)
    ]
    return replace(into, nodes=nodes)


def emit(diagram, source: str) -> str | None:
    """12.1.6's emitter on one IR, or None if it refuses this graph."""
    from src.codegen.pairs import make_pair

    try:
        return make_pair(diagram.to_dict(), source)["target_code"]
    except Exception:  # an emitter that cannot answer is a rung result, not a crash
        return None


def score_page(page, sandbox_timeout: float = 20.0) -> dict[str, Any]:
    """The four rungs for one page, each scored against the truth's own behavioural test."""
    from src.assemble.corpus import truth
    from src.llm import functional

    gold = truth(page)
    predicted = _assembled(page)
    gold_dict = gold.to_dict()
    row: dict[str, Any] = {
        "page": page.name,
        "source": page.source,
        "diagram_type": gold_dict.get("diagram_type"),
        "rungs": {},
    }

    variants = {
        "gold": gold,
        "gold_structure": transplant_text(gold, predicted),
        "gold_text": transplant_text(predicted, gold),
        "predicted": predicted,
    }
    reference_code = emit(gold, page.source)
    if reference_code is None:
        row["skipped"] = "the emitter cannot answer the true IR of this page"
        return row
    expectation = functional.expected(gold_dict, reference_code)
    row["trivial_test"] = bool(functional.is_trivial(expectation))

    for name, diagram in variants.items():
        code = emit(diagram, page.source)
        if code is None:
            row["rungs"][name] = {"emitted": False, "functional": False, "reason": "no_emission"}
            continue
        signature = functional.signature(code, gold_dict, timeout_s=sandbox_timeout)
        passed, reason = functional.compare(signature, expectation)
        row["rungs"][name] = {
            "emitted": True,
            "functional": bool(passed),
            "reason": reason,
            "characters": len(code),
        }
    return row


def _rate(rows: list[dict], rung: str) -> float | None:
    scored = [row for row in rows if rung in row.get("rungs", {})]
    if not scored:
        return None
    return round(sum(row["rungs"][rung]["functional"] for row in scored) / len(scored), 4)


def summarise(rows: list[dict]) -> dict[str, Any]:
    """Pass rates per rung, the attributable drops, and the reasons behind them."""
    scored = [row for row in rows if row.get("rungs")]
    rates = {rung: _rate(scored, rung) for rung in RUNGS}
    reasons: dict[str, dict[str, int]] = {}
    for rung in RUNGS:
        counts: dict[str, int] = {}
        for row in scored:
            entry = row.get("rungs", {}).get(rung)
            if entry and not entry["functional"]:
                counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
        reasons[rung] = dict(sorted(counts.items(), key=lambda item: -item[1]))

    def drop(rung: str) -> float | None:
        if rates["gold"] is None or rates[rung] is None:
            return None
        return round(rates["gold"] - rates[rung], 4)

    by_source: dict[str, dict] = {}
    for source in sorted({row["source"] for row in scored}):
        subset = [row for row in scored if row["source"] == source]
        by_source[source] = {
            "pages": len(subset),
            **{rung: _rate(subset, rung) for rung in RUNGS},
        }
    return {
        "pages": len(scored),
        "trivial_tests": sum(1 for row in scored if row.get("trivial_test")),
        "pass_rate": rates,
        "attributable_drop": {
            "text_only": drop("gold_structure"),
            "structure_only": drop("gold_text"),
            "both": drop("predicted"),
        },
        "failure_reasons": reasons,
        "by_source": by_source,
    }


def run(split: str = "test", limit: int | None = None, source: str | None = None) -> dict:
    from src.assemble.corpus import pages

    selected = pages((split,), source)
    if limit:
        selected = selected[:limit]
    started = time.time()
    rows = []
    for index, page in enumerate(selected, 1):
        rows.append(score_page(page))
        if index % 10 == 0:
            print(f"  {index}/{len(selected)} pages", file=sys.stderr, flush=True)
    result = {
        "split": split,
        "generator": "12.1.6 emitter (deterministic; the LoRA model is 12.3.3's separate number)",
        "test": "src.llm.functional.expected on the annotated IR, identical on every rung",
        "seconds": round(time.time() - started, 1),
        **summarise(rows),
        "rows": rows,
    }
    return result


def render(result: dict) -> str:
    rates = result["pass_rate"]
    drops = result["attributable_drop"]

    def pct(value: float | None) -> str:
        return "*missing*" if value is None else f"{value:.4f}"

    lines = [
        "# Phase 14.3 - error propagation: what an upstream mistake costs the final program",
        "",
        f"Generated by `python -m src.eval.propagate --split {result['split']}` over"
        f" {result['pages']} pages in {result['seconds']} s. One generator (12.1.6's emitter,"
        " deterministic), one test (the annotated IR's own behavioural test, identical on every"
        " rung), four inputs.",
        "",
        "| rung | structure | node text | functional pass@1 | attributable drop |",
        "| :--- | :--- | :--- | ---: | ---: |",
        f"| `gold` | annotated | annotated | {pct(rates['gold'])} | - |",
        f"| `gold_structure` | annotated | predicted |"
        f" {pct(rates['gold_structure'])} | {pct(drops['text_only'])} |",
        f"| `gold_text` | predicted | annotated |"
        f" {pct(rates['gold_text'])} | {pct(drops['structure_only'])} |",
        f"| `predicted` | predicted | predicted |"
        f" {pct(rates['predicted'])} | {pct(drops['both'])} |",
        "",
        "## Per source",
        "",
        "| source | pages | " + " | ".join(f"`{rung}`" for rung in RUNGS) + " |",
        "| :--- | ---: |" + " ---: |" * len(RUNGS),
    ]
    for source, entry in result["by_source"].items():
        cells = " | ".join(pct(entry[rung]) for rung in RUNGS)
        lines.append(f"| {source} | {entry['pages']} | {cells} |")
    lines += ["", "## Why each rung fails", ""]
    for rung in RUNGS:
        counts = result["failure_reasons"].get(rung) or {}
        body = ", ".join(f"`{name}` {count}" for name, count in counts.items()) or "nothing"
        lines.append(f"* **{rung}**: {body}")
    lines += [
        "",
        f"{result['trivial_tests']} of {result['pages']} pages carry a test that cannot fail"
        " anything non-empty (`functional.is_trivial`), and they are counted in every rung"
        " alike, so they inflate all four equally and none of the differences.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--source", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = run(args.split, args.limit, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

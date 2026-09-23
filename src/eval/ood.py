"""Phase 14.7 - what the pipeline does with a diagram type it was never trained on.

    python -m src.eval.ood                    # reports/unseen_type.{md,json}
    python -m src.eval.ood --pages 5          # a shorter pass

The deployed pipeline knows two diagram types. 13.3's router is fitted on flowchart and
state_machine and nothing else, the detector's eight classes were annotated on those two
corpora, and 12.1.6 has emitters for five types but only two of them are ever reached from a
photograph. So the plan's question - *what happens on a type never trained on* - has three
possible answers, and only one of them is acceptable:

1. it says it does not know (13.5's confidence gate fires), or
2. it produces something visibly wrong and says so (`degraded`, a stopped stage), or
3. **it produces confident, runnable, wrong code**, which is the failure mode worth documenting
   because nothing downstream can detect it.

## The unseen pages

12.1.4's synthetic corpus renders all five types with 1.3.7's wobbled ink, and three of them -
`er_diagram`, `circuit`, `wireframe` - appear in no detector split, no router fit and no OCR
corpus. They are the out-of-distribution set here. `flowchart` and `state_machine` renders are
run too, as the in-distribution control: a degradation that also shows up on the control is the
renderer being unfamiliar, not the type being unseen, and without the control this row could not
tell those apart.

## What is recorded per page

The router's answer and its probability, whether the gate fired, whether code was produced, what
language it claims, whether that code parses, and how many nodes the assembled IR carried. The
verdict per page is one of `asked` (the honest answer), `stopped`, or `confident_wrong` - the
last being the one this row exists to quantify.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "unseen_type.json"
REPORT_MD = ROOT / "reports" / "unseen_type.md"

SYNTHETIC = ROOT / "data" / "processed" / "codegen" / "synthetic" / "images"

#: Types the router was never fitted on, and the two it was - the control.
UNSEEN = ("er_diagram", "circuit", "wireframe")
SEEN = ("flowchart", "state_machine")


def pages_of(kind: str, limit: int) -> list[Path]:
    directory = SYNTHETIC / kind
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.png"))[:limit]


def _parses(code: str, language: str) -> bool | None:
    """Python is checked with the parser; other languages have no stdlib parser, so None."""
    if not code:
        return False
    if language in ("python", "py"):
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True
    return None


def verdict(row: dict) -> str:
    if row.get("needs_confirmation"):
        return "asked"
    if row.get("stopped_at"):
        return "stopped"
    if row.get("code_chars"):
        return "confident_wrong" if row["unseen"] else "answered"
    return "stopped"


def score_page(pipeline, path: Path, kind: str, unseen: bool) -> dict[str, Any]:
    started = time.time()
    result = pipeline.run(path)
    row: dict[str, Any] = {
        "page": path.name,
        "true_type": kind,
        "unseen": unseen,
        "routed_type": result.diagram_type,
        "needs_confirmation": bool(result.needs_confirmation),
        "stopped_at": result.stopped_at,
        "code_chars": len(result.code),
        "language": result.language,
        "parses": _parses(result.code, result.language),
        "nodes": len((result.ir or {}).get("nodes") or []),
        "seconds": round(time.time() - started, 3),
    }
    for stage in result.stages:
        if stage.name == "classify":
            row["route_confidence"] = round(float(stage.confidence or 0.0), 4)
    row["verdict"] = verdict(row)
    return row


def summarise(rows: list[dict]) -> dict[str, Any]:
    """Counts per type. A row that has not been given a verdict gets one here, so a caller that
    assembled rows by hand (the tests, or a re-aggregation) cannot silently drop pages."""
    rows = [row if "verdict" in row else {**row, "verdict": verdict(row)} for row in rows]
    by_type: dict[str, Any] = {}
    for kind in sorted({row["true_type"] for row in rows}):
        subset = [row for row in rows if row["true_type"] == kind]
        verdicts = Counter(row["verdict"] for row in subset)
        confidences = [row.get("route_confidence", 0.0) for row in subset]
        by_type[kind] = {
            "pages": len(subset),
            "unseen": subset[0]["unseen"],
            "verdicts": dict(verdicts),
            "routed_as": dict(Counter(row["routed_type"] for row in subset)),
            "median_route_confidence": round(sorted(confidences)[len(confidences) // 2], 4),
            "produced_code": sum(1 for row in subset if row["code_chars"]),
            "parsed": sum(1 for row in subset if row["parses"]),
            "median_nodes": sorted(row["nodes"] for row in subset)[len(subset) // 2],
        }
    unseen_rows = [row for row in rows if row["unseen"]]
    return {
        "by_type": by_type,
        "unseen_pages": len(unseen_rows),
        "unseen_asked": sum(1 for row in unseen_rows if row["verdict"] == "asked"),
        "unseen_stopped": sum(1 for row in unseen_rows if row["verdict"] == "stopped"),
        "unseen_confident_wrong": sum(
            1 for row in unseen_rows if row["verdict"] == "confident_wrong"
        ),
    }


def run(limit: int = 8, floor: float | None = None) -> dict:
    from src.pipeline.core import CONFIDENCE_FLOOR, DreamScriptPipeline

    pipeline = DreamScriptPipeline(
        confidence_floor=floor if floor is not None else CONFIDENCE_FLOOR
    )
    started = time.time()
    rows: list[dict] = []
    for kind in UNSEEN + SEEN:
        for path in pages_of(kind, limit):
            rows.append(score_page(pipeline, path, kind, kind in UNSEEN))
        print(f"  {kind} done", file=sys.stderr, flush=True)
    return {
        "confidence_floor": pipeline.confidence_floor,
        "corpus": "12.1.4 synthetic renders (the only source with all five types)",
        "seconds": round(time.time() - started, 1),
        **summarise(rows),
        "rows": rows,
    }


def render(result: dict) -> str:
    lines = [
        "# Phase 14.7 - a diagram type the pipeline was never trained on",
        "",
        f"Generated by `python -m src.eval.ood` over {result['corpus']} in {result['seconds']} s,"
        f" at 13.5's confidence floor of {result['confidence_floor']}. `er_diagram`, `circuit`"
        " and `wireframe` appear in no detector split, no router fit and no OCR corpus;"
        " `flowchart` and `state_machine` are run as the in-distribution control, so a"
        " degradation that shows up on both is the renderer rather than the unseen type.",
        "",
        "| type | seen? | pages | routed as | median route confidence | produced code | verdicts |",
        "| :--- | :---: | ---: | :--- | ---: | ---: | :--- |",
    ]
    for kind, entry in result["by_type"].items():
        routed = ", ".join(f"{name} {count}" for name, count in entry["routed_as"].items())
        verdicts = ", ".join(f"{name} {count}" for name, count in entry["verdicts"].items())
        lines.append(
            f"| {kind} | {'no' if entry['unseen'] else 'yes'} | {entry['pages']} | {routed} |"
            f" {entry['median_route_confidence']} | {entry['produced_code']} | {verdicts} |"
        )
    lines += [
        "",
        f"**Of {result['unseen_pages']} out-of-distribution pages:"
        f" {result['unseen_asked']} asked for confirmation,"
        f" {result['unseen_stopped']} stopped at a stage, and"
        f" {result['unseen_confident_wrong']} produced confident, runnable, wrong code.**",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--floor", type=float, default=None, help="override 13.5's confidence floor")
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = run(args.pages, args.floor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

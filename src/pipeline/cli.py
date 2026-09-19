"""Phase 13.9 - the batch entry point.

    python -m src.pipeline run page.png
    python -m src.pipeline run photos/ --out runs/monday
    python -m src.pipeline run photos/ --out runs/monday --confidence-floor 0.8

One page or a directory of them, in, and for each one a source file, a JSON record of every
stage, and one `summary.json`/`report.md` over the batch.

## Why this is serial

The detector and the language model are each a single resident model on one card, and the
pipeline's own `StageCache` is what makes a re-run cheap. Running pages in parallel would
contend for the card, multiply the resident weights and race on the cache, so the batch is a
loop. The measurement this supports - 13.7's budget - is *per page* anyway.

## What a failure looks like

A page that stops early is still written out, with `stopped_at` and the stage table that got
that far, and it counts in the summary. The process exits non-zero only when **no** page
produced code, which is the case where something is wrong with the run rather than with a page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.pipeline.contracts import Result

#: What each generated language is written as. The emitter's own assignment (12.1.6).
SUFFIX = {"python": ".py", "sql": ".sql", "react": ".jsx", "spice": ".cir"}

#: Page extensions the detector will accept.
IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def pages_under(target: Path) -> list[Path]:
    """Every image at `target`, whether it is one file or a directory of them."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise SystemExit(f"no such file or directory: {target}")
    found = [p for p in sorted(target.rglob("*")) if p.suffix.lower() in IMAGES]
    if not found:
        raise SystemExit(f"no images under {target} (looked for {', '.join(IMAGES)})")
    return found


def write_result(result: Result, out: Path) -> None:
    """One page's outputs: the code in its own language, and the full record beside it."""
    stem = Path(result.source).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    if result.code:
        suffix = SUFFIX.get(result.language, ".txt")
        (out / f"{stem}{suffix}").write_text(result.code, encoding="utf-8")


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    value = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return round(value, 3)


def summarise(results: Sequence[Result], budget: float) -> dict[str, Any]:
    """The batch's own numbers - what got through, how long, and where the rest stopped."""
    done = [r for r in results if r.ok]

    stopped: dict[str, int] = {}
    for result in results:
        if not result.ok:
            where = result.stopped_at or "unknown"
            stopped[where] = stopped.get(where, 0) + 1

    types: dict[str, int] = {}
    for result in done:
        types[result.diagram_type] = types.get(result.diagram_type, 0) + 1

    # Cached stages are excluded from the per-stage medians: a 0.0s cache hit is a property of
    # the second run, not of what the stage costs.
    per_stage: dict[str, list[float]] = {}
    for result in results:
        for row in result.timing_table():
            if not row["cached"]:
                per_stage.setdefault(row["stage"], []).append(row["seconds"])

    seconds = sorted(result.seconds for result in results)
    return {
        "pages": len(results),
        "produced_code": len(done),
        "rate": round(len(done) / max(1, len(results)), 4),
        "needs_confirmation": sum(1 for r in results if r.needs_confirmation),
        "degraded": sum(1 for r in done if any(s.degraded for s in r.stages)),
        "stopped_at": dict(sorted(stopped.items())),
        "diagram_types": dict(sorted(types.items())),
        "seconds": {
            "median": _median(seconds),
            "p90": round(seconds[int(0.9 * (len(seconds) - 1))], 3) if seconds else 0.0,
            "max": round(seconds[-1], 3) if seconds else 0.0,
        },
        "within_budget": {
            "budget_s": budget,
            "pages": sum(1 for r in results if r.seconds < budget),
        },
        "median_seconds_per_stage": {k: _median(v) for k, v in sorted(per_stage.items())},
    }


def report_markdown(summary: dict[str, Any], results: Sequence[Result]) -> str:
    """A human-readable page of the same numbers, for `reports/`."""
    budget = summary["within_budget"]
    lines = [
        "# Phase 13.9 - batch run",
        "",
        f"{summary['pages']} pages, {summary['produced_code']} produced code "
        f"({summary['rate']:.1%}), median {summary['seconds']['median']}s, "
        f"p90 {summary['seconds']['p90']}s against a {budget['budget_s']}s budget "
        f"({budget['pages']}/{summary['pages']} inside it).",
        "",
        "| stage | median s (uncached) |",
        "| :--- | :--- |",
    ]
    for stage, value in summary["median_seconds_per_stage"].items():
        lines.append(f"| {stage} | {value} |")

    if summary["stopped_at"]:
        lines += ["", "| stopped at | pages |", "| :--- | :--- |"]
        for stage, count in summary["stopped_at"].items():
            lines.append(f"| {stage} | {count} |")

    lines += ["", "| page | type | code | s | stopped |", "| :--- | :--- | :--- | :--- | :--- |"]
    for result in results:
        produced = "yes" if result.ok else "no"
        lines.append(
            f"| {Path(result.source).name} | {result.diagram_type} | {produced} "
            f"| {result.seconds} | {result.stopped_at or '-'} |"
        )
    return "\n".join(lines) + "\n"


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Phase 13: a diagram photograph to code, one page or a directory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    runner = sub.add_parser("run", help="run the pipeline over a page or a directory")
    runner.add_argument("target", type=Path, help="an image, or a directory of them")
    runner.add_argument("--out", type=Path, default=Path("runs/latest"), help="where to write")
    runner.add_argument(
        "--confidence-floor",
        type=float,
        default=None,
        help="routing probability below which the pipeline asks rather than guesses (13.5)",
    )
    runner.add_argument("--read-text", action="store_true", help="run node-label OCR (slow)")
    runner.add_argument("--no-generate", action="store_true", help="stop before code generation")
    runner.add_argument("--no-cache", action="store_true", help="ignore the stage cache")
    runner.add_argument("--limit", type=int, default=0, help="only the first N pages")
    runner.add_argument("--quiet", action="store_true", help="no per-page line")

    args = parser.parse_args(argv)

    from src.pipeline.cache import StageCache
    from src.pipeline.core import CONFIDENCE_FLOOR, LATENCY_BUDGET_S, DreamScriptPipeline

    targets = pages_under(args.target)
    if args.limit:
        targets = targets[: args.limit]

    floor = args.confidence_floor if args.confidence_floor is not None else CONFIDENCE_FLOOR
    pipeline = DreamScriptPipeline(
        cache=StageCache(enabled=not args.no_cache),
        confidence_floor=floor,
        generate=not args.no_generate,
        read_text=args.read_text,
    )

    results: list[Result] = []
    for path in targets:
        result = pipeline.run(path)
        write_result(result, args.out)
        results.append(result)
        if not args.quiet:
            state = "ok" if result.ok else f"stopped at {result.stopped_at}"
            print(f"{path.name:40s} {result.diagram_type:15s} {state:24s} {result.seconds:6.2f}s")

    summary = summarise(results, LATENCY_BUDGET_S)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(report_markdown(summary, results), encoding="utf-8")

    print(
        f"\n{summary['produced_code']}/{summary['pages']} produced code, "
        f"median {summary['seconds']['median']}s, p90 {summary['seconds']['p90']}s "
        f"-> {args.out}"
    )
    # Non-zero only when the whole batch produced nothing: a single bad page is not a bad run.
    return 0 if summary["produced_code"] else 1


if __name__ == "__main__":
    sys.exit(run())

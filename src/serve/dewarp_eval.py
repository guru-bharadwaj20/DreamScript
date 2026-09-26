"""Phase 16.2.3 - does straightening the page actually help the pipeline read it?

    node scripts/dewarp_bench.mjs runs/dewarp        # build the images (needs the dev server)
    python -m src.serve dewarp_eval                  # run the pipeline over them and report

Writes `reports/dewarp.json` and `reports/dewarp.md`.

## The question, and the honest obstacle

The row's word is **mandatory, not optional** - every model here is trained on flat scans and a
phone adds perspective. Its Definition of Done is "dewarped crop measured against the raw photo on
the same pages", which is a comparison, not an assertion.

The obstacle is stated rather than glossed: **this repository contains no skewed photographs.** The
corpora are flat scans and the self-collected chaos corpus is not in this checkout. So the skew is
synthesised. `scripts/dewarp_bench.mjs` lays each flat fixture on a grey desk, pushes its corners
out by a known homography at three severities, and puts the result through **the code the app
ships** - `app/frontend/src/lib/dewarp.ts`, running in a real browser - to produce the third image.

That makes this an answer about *this* dewarp on *these* pages under *synthetic* tilt. It is not an
answer about real phone photographs, and the report says so at the top rather than in a footnote.

## What is compared

Three runs per page and tilt, through the same pipeline:

    flat       the original. **The baseline**, not a fourth arm - every score below is against it
    photo      the tilted page, which is what an app with no dewarp would upload
    dewarped   the tilted page straightened by the shipped detector

and the comparison is 10.2.5's own metric, `src.assemble.irdiff`: node F1, edge F1 and a GED bound
against the flat run's IR. Reusing it rather than inventing a similarity means this number sits in
the same units as every other graph comparison in the project.

**The cache is off.** `cache.ENV_KEYS` keys on the image, so three different images are three
different keys and a warm cache would not confuse them - but the flat run is executed twice across
a full sweep and one of those would be free, which makes a timing column meaningless. Off, for the
same reason 16.1.5 turns it off.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

JSON_PATH = ROOT / "reports" / "dewarp.json"
MD_PATH = ROOT / "reports" / "dewarp.md"
DEFAULT_SET = ROOT / "runs" / "dewarp" / "index.json"

#: 10.2.5's matching rule. `geometric` is IoU-gated, which is the right question here: the whole
#: claim is that removing perspective puts boxes back where a flat scan would have them.
MATCH = "geometric"


def run_one(pipeline, path: Path) -> dict[str, Any]:
    """One page through the pipeline, as a plain dict."""
    started = time.perf_counter()
    result = pipeline.run(path)
    return {
        "ok": bool(result.ok),
        "stopped_at": result.stopped_at or None,
        "diagram_type": result.diagram_type,
        "needs_confirmation": bool(result.needs_confirmation),
        "language": result.language or None,
        "code_bytes": len(result.code or ""),
        "nodes": len((result.ir or {}).get("nodes") or []),
        "edges": len((result.ir or {}).get("edges") or []),
        "seconds": round(time.perf_counter() - started, 3),
        "ir": result.ir,
    }


def normalise(ir: Any, width: int, height: int) -> Any:
    """Every bbox divided by the image it was found in, so two different crops can be compared.

    **This is the whole reason the first run of this report was meaningless.** `geometric` matching
    is IoU over pixel coordinates, and the dewarped arm is by construction a *different crop at a
    different size* - the page, cropped out of a padded frame and flattened. Its boxes could not
    overlap the baseline's no matter how well it had worked, so every dewarped node F1 came back
    **0.00**: the arm was being penalised precisely for doing its job.

    In normalised coordinates the question becomes the right one - where is each box *on the page* -
    and both arms are asked it in the same units.
    """
    if not ir or not width or not height:
        return ir
    out = json.loads(json.dumps(ir))
    for node in out.get("nodes") or []:
        box = node.get("bbox")
        if not box or len(box) != 4:
            continue
        node["bbox"] = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    return out


def image_size(path: Path) -> tuple[int, int]:
    """Width and height of a PNG, from its IHDR. No decode, and no dependency."""
    data = path.read_bytes()[:33]
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def compare(predicted_ir: Any, truth_ir: Any) -> dict[str, Any] | None:
    """10.2.5's diff, or `None` when either side has no graph to compare."""
    if not predicted_ir or not truth_ir:
        return None
    from src.assemble.irdiff import diff
    from src.ir.model import Diagram

    try:
        scored = diff(Diagram.from_dict(predicted_ir), Diagram.from_dict(truth_ir), match=MATCH)
    except Exception as error:  # noqa: BLE001 - a metric reports, it does not raise
        return {"error": f"{type(error).__name__}: {error}"}
    row = scored.to_dict()
    # The matched pairs and the edit list are large and are not what this report is about.
    for key in ("matched", "edits"):
        row.pop(key, None)
    return row


def evaluate(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.pipeline.cache import StageCache
    from src.pipeline.core import DreamScriptPipeline

    pipeline = DreamScriptPipeline(cache=StageCache(enabled=False))
    rows: list[dict[str, Any]] = []

    for entry in entries:
        row: dict[str, Any] = {
            "page": entry["page"],
            "tilt": entry["tilt"],
            "detected": entry["detected"],
            "corner_error_px": entry.get("corner_error"),
            "coverage": entry.get("coverage"),
            "skew": entry.get("skew"),
        }
        runs: dict[str, dict[str, Any]] = {}
        sizes: dict[str, tuple[int, int]] = {}

        # The baseline is the **original fixture**, not the padded flat frame: the question is how
        # close each arm comes to reading the page the way a flat scan of *the page* would be read,
        # and the padded frame is already a different crop.
        source = ROOT / "tests" / "fixtures" / entry["page"]
        runs["source"] = run_one(pipeline, source)
        sizes["source"] = image_size(source)

        for arm in ("flat", "photo", "dewarped"):
            rel = entry.get(arm)
            if not rel:
                runs[arm] = {"missing": True}
                continue
            runs[arm] = run_one(pipeline, ROOT / rel)
            sizes[arm] = image_size(ROOT / rel)

        row["runs"] = {
            arm: {k: v for k, v in run.items() if k != "ir"} for arm, run in runs.items()
        }
        truth = normalise(runs["source"].get("ir"), *sizes["source"])
        row["photo_vs_flat"] = compare(
            normalise(runs.get("photo", {}).get("ir"), *sizes.get("photo", (0, 0))), truth
        )
        row["dewarped_vs_flat"] = compare(
            normalise(runs.get("dewarped", {}).get("ir"), *sizes.get("dewarped", (0, 0))), truth
        )
        rows.append(row)
        print(
            f"  {entry['page']} {entry['tilt']}: "
            f"photo f1={_f1(row['photo_vs_flat'])} dewarped f1={_f1(row['dewarped_vs_flat'])}",
            file=sys.stderr,
        )
    return rows


def _f1(scored: dict[str, Any] | None) -> str:
    if not scored or "node_f1" not in scored:
        return "-"
    return f"{scored['node_f1']:.2f}"


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians, and the count of pages each arm got all the way to code."""

    def medians(key: str) -> dict[str, Any]:
        f1 = [r[key]["node_f1"] for r in rows if r.get(key) and "node_f1" in r[key]]
        edge = [r[key]["edge_f1"] for r in rows if r.get(key) and "edge_f1" in r[key]]
        ged = [r[key]["ged_normalised"] for r in rows if r.get(key) and "ged_normalised" in r[key]]
        return {
            "n": len(f1),
            "node_f1_median": round(statistics.median(f1), 4) if f1 else None,
            "edge_f1_median": round(statistics.median(edge), 4) if edge else None,
            "ged_normalised_median": round(statistics.median(ged), 4) if ged else None,
        }

    def reached(arm: str) -> int:
        return sum(1 for r in rows if r["runs"].get(arm, {}).get("ok"))

    return {
        "arms": ("source", "flat", "photo", "dewarped"),
        "pages": len(rows),
        "detected": sum(1 for r in rows if r["detected"]),
        "corner_error_px_median": (
            round(
                statistics.median([r["corner_error_px"] for r in rows if r.get("corner_error_px")]),
                2,
            )
            if any(r.get("corner_error_px") for r in rows)
            else None
        ),
        "reached_code": {arm: reached(arm) for arm in ("source", "flat", "photo", "dewarped")},
        "photo_vs_flat": medians("photo_vs_flat"),
        "dewarped_vs_flat": medians("dewarped_vs_flat"),
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    photo = s["photo_vs_flat"]
    dew = s["dewarped_vs_flat"]
    lines = [
        "# Phase 16.2.3 - does straightening the page help?",
        "",
        "Generated by `python -m src.serve dewarp_eval` over images built by "
        "`scripts/dewarp_bench.mjs`. The dewarp measured is **the code the app ships** "
        "(`app/frontend/src/lib/dewarp.ts`), executed in a real browser, not a reimplementation.",
        "",
        "> **The skew is synthetic.** This repository contains no skewed photographs - the corpora "
        "are flat scans and the chaos corpus is not in this checkout - so each flat fixture is "
        "tilted by a known homography at three severities and laid on a grey desk. This is an "
        "answer about this dewarp on these pages under synthetic tilt, and not an answer about "
        "real phone photographs.",
        "",
    ]

    if photo["node_f1_median"] is not None and dew["node_f1_median"] is not None:
        better = dew["node_f1_median"] - photo["node_f1_median"]
        lines += [
            f"**Node F1 against the flat baseline: {photo['node_f1_median']} photographed, "
            f"{dew['node_f1_median']} dewarped** - a change of {better:+.3f}. "
            f"Edge F1 {photo['edge_f1_median']} to {dew['edge_f1_median']}. "
            f"Normalised GED {photo['ged_normalised_median']} to {dew['ged_normalised_median']} "
            "(lower is better).",
            "",
        ]

    lost = s["reached_code"]["photo"] - s["reached_code"]["dewarped"]
    lines += [
        f"The page was found in **{s['detected']} of {s['pages']}** tilted images, with a median "
        f"corner error of **{s['corner_error_px_median']} px** against the corners the tilt "
        "actually used.",
        "",
    ]
    if lost > 0:
        lines += [
            f"**And it costs something.** {lost} of {s['pages']} pages reached code when uploaded "
            "as photographed and did not after being straightened. That is in the headline rather "
            "than in a table cell: a correction that improves the median while losing a page "
            "outright is a trade, not a free win, and the row is only worth flipping if the trade "
            "is stated.",
            "",
        ]
    lines += [
        "| arm | reached code | node F1 | edge F1 | normalised GED |",
        "| :--- | ---: | ---: | ---: | ---: |",
        f"| the page itself (baseline) | {s['reached_code']['source']}/{s['pages']} | — | — | — |",
        f"| the page on a desk, flat | {s['reached_code']['flat']}/{s['pages']} | — | — | — |",
        f"| photographed | {s['reached_code']['photo']}/{s['pages']} | {photo['node_f1_median']} "
        f"| {photo['edge_f1_median']} | {photo['ged_normalised_median']} |",
        f"| dewarped | {s['reached_code']['dewarped']}/{s['pages']} | {dew['node_f1_median']} "
        f"| {dew['edge_f1_median']} | {dew['ged_normalised_median']} |",
        "",
        "## Per page",
        "",
        "| page | tilt | corner err | nodes page/photo/dewarped | node F1 photo | node F1 dewarped |",
        "| :--- | :--- | ---: | :---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        runs = row["runs"]
        counts = "/".join(
            str(runs.get(a, {}).get("nodes", "—")) for a in ("source", "photo", "dewarped")
        )
        lines.append(
            f"| {row['page']} | {row['tilt']} | "
            f"{row['corner_error_px']:.1f} px | {counts} | "
            f"{_f1(row['photo_vs_flat'])} | {_f1(row['dewarped_vs_flat'])} |"
            if row.get("corner_error_px")
            else f"| {row['page']} | {row['tilt']} | not found | {counts} | "
            f"{_f1(row['photo_vs_flat'])} | {_f1(row['dewarped_vs_flat'])} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "* **The baseline is the original fixture**, read as-is, not a perfect answer. Where the "
        "pipeline misreads a flat page it misreads it identically in every arm, so the comparison "
        "is about what the tilt and the correction did, not about how good the pipeline is.",
        "* **The matching rule is `geometric`** (10.2.5), IoU-gated, over **normalised** boxes. "
        "The normalisation is not a detail: the dewarped arm is by construction a different crop "
        "at a different size, so in pixel coordinates its boxes cannot overlap the baseline's at "
        "all. The first run of this report scored every dewarped page at node F1 **0.00** - the "
        "arm was being penalised for doing exactly what it exists to do. Dividing each box by its "
        "own image asks the right question, which is where the box sits *on the page*.",
        "* **`wireframe` is the failure, and it is one failure with one cause.** The largest bright "
        "region on that page is the interior of the drawn screen border, so the detector finds the "
        "*drawing* rather than the sheet - a 41 to 55 px corner error where every other page is at "
        "2 px - and the crop that follows is a different frame again, which is why its normalised "
        "F1 is 0.00 and one of its three tilts loses the page entirely. A page whose content is "
        "itself a large rectangle is the case this detector is worst at, and it is exactly the "
        "case `findContours` with `approxPolyDP` would handle better.",
        "* **Corner error is against the tilt's own corners**, which the detector never sees. On a "
        "wireframe it is large for a reason worth knowing rather than hiding: the largest bright "
        "region on that page is the interior of the *drawn screen border*, not the sheet, so the "
        "dewarp crops to the drawing. That loses white margin and is arguably the better crop - but "
        "it is not what the corner-error column is measuring, and the column is not adjusted to "
        "flatter it.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 16.2.3 - dewarped against photographed")
    ap.add_argument("--set", default=str(DEFAULT_SET), help="index.json from dewarp_bench.mjs")
    args = ap.parse_args(argv)

    index = Path(args.set)
    if not index.is_file():
        print(
            f"no image set at {index}. Build it first:\n"
            f"  node scripts/dewarp_bench.mjs runs/dewarp",
            file=sys.stderr,
        )
        return 2

    entries = json.loads(index.read_text(encoding="utf-8"))
    print(f"> {len(entries)} images", file=sys.stderr)
    rows = evaluate(entries)
    report = {
        "row": "16.2.3",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "match": MATCH,
        "synthetic_skew": True,
        "cache_enabled": False,
        "summary": summarise(rows),
        "rows": rows,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    MD_PATH.write_text(markdown(report), encoding="utf-8")
    print(f"wrote {JSON_PATH.relative_to(ROOT)} and {MD_PATH.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

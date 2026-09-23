"""Phase 14.6 - leave-one-scribe-out: does the pipeline recognise diagrams or handwriting?

    python -m src.eval.crossscribe                  # reports/cross_scribe.{md,json}
    python -m src.eval.crossscribe --skip-ocr       # no GPU: classification and the read artefacts

Grouped cross-validation already holds writers out, but it reports one number over all of them.
This row asks the question a grouped average cannot answer: **how far apart are the writers?** A
stage at 0.98 whose worst writer is at 0.40 is a different engineering situation from one whose
writers all sit at 0.98, and the mean is identical.

Four stages are scored per writer, and each says which split its writers come from:

    classify     leave-one-scribe-out over the handcrafted table: the model is refitted with one
                 writer's pages entirely absent and then scored on exactly those pages
    ocr          S3's val crops, grouped by the writer who drew them (23 evaluation writers)
    assemble     S5's held-out test pages, grouped by writer, median GED each
    end_to_end   14.3's `predicted` rung, grouped by writer - the functional pass rate of the
                 program the pipeline would actually emit

## The OCR numbers are re-scored, not read

`experiments/ocr/s3_val.json` holds the per-crop predictions of whichever S3 run wrote it last,
which is not necessarily the incumbent checkpoint - at the time this row was built it held the
retrained-on-reselection run (CER 0.2073) while the incumbent is 0.2006. Per-writer numbers
derived from the wrong run would be quietly wrong, so this module re-runs the incumbent
checkpoint over the val crops and **refuses to publish the breakdown unless its own aggregate
reproduces `reports/s3_label_ocr.json`** to four decimal places.

## What leave-one-scribe-out can and cannot ask here

Only hdbpmn pages carry a writer id. sketch2code ships none, so its 731 rows are training
material in every fold and are never a held-out writer; the report says how many rows that is
rather than implying the whole corpus was covered. This is the same limitation 5.2.2 recorded,
and it has not changed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "cross_scribe.json"
REPORT_MD = ROOT / "reports" / "cross_scribe.md"

S3_REPORT = ROOT / "reports" / "s3_label_ocr.json"
S5_REPORT = ROOT / "reports" / "s5_test_large.json"
PROPAGATION = ROOT / "reports" / "error_propagation.json"


def _spread(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "p10": round(ordered[max(0, int(0.10 * (len(ordered) - 1)))], 4),
        "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
        "spread_p90_p10": round(
            ordered[int(0.90 * (len(ordered) - 1))]
            - ordered[max(0, int(0.10 * (len(ordered) - 1)))],
            4,
        ),
    }


def _writer_of(page_name: str) -> str:
    """`fa_bresler__writer015_fa_001` and `hdbpmn__ex00_writer0065` both name their writer."""
    tail = page_name.split("__")[-1]
    for chunk in tail.replace("_", " ").split():
        if chunk.startswith("writer"):
            return chunk
    return tail


def classification_loso(model: str = "logreg", limit: int | None = None) -> dict:
    """Refit with one writer's pages absent, score exactly those pages, for every known writer."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score

    from src.classify.cv import MODELS
    from src.classify.data import load

    data = load("real")
    known = np.array([str(group).startswith("scribe:") for group in data.groups])
    writers = sorted({str(group) for group in data.groups[known]})
    if limit:
        writers = writers[:limit]

    per_writer = []
    skipped = []
    started = time.time()
    for writer in writers:
        held = np.array([str(group) == writer for group in data.groups])
        if held.sum() == 0 or len(set(data.y[~held])) < 2:
            # Removing this writer would leave the training side with one class, which does not
            # measure generalisation - it measures a degenerate fit. Skipped and counted.
            skipped.append(writer.replace("scribe:", ""))
            continue
        estimator = clone(MODELS[model]())
        estimator.fit(data.X[~held], data.y[~held])
        predicted = estimator.predict(data.X[held])
        per_writer.append(
            {
                "writer": writer.replace("scribe:", ""),
                "pages": int(held.sum()),
                "accuracy": round(float(accuracy_score(data.y[held], predicted)), 4),
            }
        )
    accuracies = [row["accuracy"] for row in per_writer]
    return {
        "model": model,
        "protocol": "leave-one-scribe-out; the writer's every page is absent from the fit",
        "rows_with_a_known_writer": int(known.sum()),
        "rows_without_one": int((~known).sum()),
        "seconds": round(time.time() - started, 1),
        "writers_scored": len(per_writer),
        "writers_skipped": skipped,
        "spread": _spread(accuracies),
        "worst": sorted(per_writer, key=lambda row: row["accuracy"])[:5],
        "per_writer": per_writer,
    }


def incumbent_predictions(rescore: bool = True):
    """(truths, predictions, frame, source) for S3's val crops.

    Shared with 14.10, which needs the same crops and the same provenance question: a per-crop
    file on disk belongs to whichever run wrote it last, so a caller that wants the incumbent
    has to ask for the rescore and then check the aggregate itself.
    """
    from src.ocr.s3 import split

    files, truths, frame = split("val")
    if rescore:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        from src.ocr.s3 import CHECKPOINT, predict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proc = TrOCRProcessor.from_pretrained(CHECKPOINT)
        model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT).to(device)
        predictions = predict(model, proc, files, device)
        source = f"rescored from {CHECKPOINT.name}"
    else:
        stored = json.loads((ROOT / "experiments" / "ocr" / "s3_val.json").read_text("utf-8"))
        lookup = dict(zip(stored["files"], stored["predictions"], strict=True))
        predictions = [lookup.get(name, "") for name in frame["file"]]
        source = "experiments/ocr/s3_val.json (stored predictions)"
    return list(truths), list(predictions), frame, source


def ocr_per_writer(rescore: bool = True) -> dict:
    """S3's val crops grouped by writer, from a rescore that must reproduce the published CER."""
    from src.ocr.metrics import score

    published = json.loads(S3_REPORT.read_text(encoding="utf-8")) if S3_REPORT.is_file() else {}
    truths, predictions, frame, source = incumbent_predictions(rescore)
    aggregate = score(list(truths), list(predictions))
    entry: dict[str, Any] = {
        "source": source,
        "crops": len(truths),
        "cer": round(float(aggregate["cer"]), 4),
        "published_cer": published.get("cer"),
        "reproduces_published": bool(
            published.get("cer") is not None
            and abs(round(float(aggregate["cer"]), 4) - float(published["cer"])) < 5e-4
        ),
    }
    if not entry["reproduces_published"]:
        entry["note"] = (
            "this rescore does not reproduce the published S3 aggregate, so the per-writer"
            " breakdown is withheld rather than attributed to the incumbent checkpoint"
        )
        return entry

    buckets: dict[str, list[int]] = defaultdict(list)
    for index, writer in enumerate(frame["scribe"]):
        buckets[str(writer)].append(index)
    per_writer = []
    for writer, indices in sorted(buckets.items()):
        subset = score([truths[i] for i in indices], [predictions[i] for i in indices])
        per_writer.append(
            {
                "writer": writer,
                "crops": len(indices),
                "cer": round(float(subset["cer"]), 4),
                "exact_match": round(float(subset["exact_match"]), 4),
            }
        )
    entry["spread"] = _spread([row["cer"] for row in per_writer])
    entry["worst"] = sorted(per_writer, key=lambda row: -row["cer"])[:5]
    entry["per_writer"] = per_writer
    return entry


def _per_writer_from_pages(rows: list[dict], value, name: str) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, str] = {}
    for row in rows:
        writer = _writer_of(row["page"])
        buckets[writer].append(value(row))
        sources[writer] = row.get("source", "")
    per_writer = [
        {
            "writer": writer,
            "pages": len(values),
            "source": sources[writer],
            name: round(statistics.median(values), 4),
        }
        for writer, values in sorted(buckets.items())
    ]
    return {
        "writers": len(per_writer),
        "spread": _spread([row[name] for row in per_writer]),
        "per_writer": per_writer,
    }


def assembly_per_writer() -> dict:
    if not S5_REPORT.is_file():
        return {"note": f"{S5_REPORT.name} is absent"}
    payload = json.loads(S5_REPORT.read_text(encoding="utf-8"))
    entry = _per_writer_from_pages(payload["pages"], lambda row: float(row["ged"]), "median_ged")
    entry["source"] = "reports/s5_test_large.json"
    entry["worst"] = sorted(entry["per_writer"], key=lambda row: -row["median_ged"])[:5]
    return entry


def end_to_end_per_writer(rung: str = "predicted") -> dict:
    if not PROPAGATION.is_file():
        return {"note": f"{PROPAGATION.name} is absent; run python -m src.eval.propagate"}
    payload = json.loads(PROPAGATION.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if rung in row.get("rungs", {})]
    buckets: dict[str, list[int]] = defaultdict(list)
    sources: dict[str, str] = {}
    for row in rows:
        writer = _writer_of(row["page"])
        buckets[writer].append(int(row["rungs"][rung]["functional"]))
        sources[writer] = row.get("source", "")
    per_writer = [
        {
            "writer": writer,
            "pages": len(values),
            "source": sources[writer],
            "functional": round(sum(values) / len(values), 4),
        }
        for writer, values in sorted(buckets.items())
    ]
    return {
        "rung": rung,
        "source": "reports/error_propagation.json",
        "writers": len(per_writer),
        "spread": _spread([row["functional"] for row in per_writer]),
        "worst": sorted(per_writer, key=lambda row: row["functional"])[:5],
        "per_writer": per_writer,
    }


def overlap(ocr: dict, assembly: dict) -> dict:
    """Is a hard writer hard everywhere? Only answerable where the same writer is in both."""
    if "per_writer" not in ocr or "per_writer" not in assembly:
        return {"note": "one of the two stages has no per-writer breakdown"}
    left = {row["writer"]: row["cer"] for row in ocr["per_writer"]}
    right = {row["writer"]: row["median_ged"] for row in assembly["per_writer"]}
    shared = sorted(set(left) & set(right))
    if len(shared) < 3:
        return {
            "shared_writers": len(shared),
            "note": "S3 is scored on the validation writers and S5 on the test writers, so the"
            " two breakdowns barely overlap - the question cannot be answered on this corpus"
            " without rescoring one of them on the other's split, and is left open rather than"
            " answered from three points",
        }
    from statistics import correlation

    return {
        "shared_writers": len(shared),
        "pearson_cer_vs_ged": round(
            correlation([left[w] for w in shared], [right[w] for w in shared]), 4
        ),
    }


def collect(rescore_ocr: bool = True, model: str = "logreg", limit: int | None = None) -> dict:
    classify = classification_loso(model, limit)
    ocr = ocr_per_writer(rescore_ocr) if rescore_ocr is not None else {}
    assembly = assembly_per_writer()
    return {
        "classify": classify,
        "ocr": ocr,
        "assemble": assembly,
        "end_to_end": end_to_end_per_writer(),
        "cross_stage": overlap(ocr, assembly),
    }


def _table(entry: dict, column: str, title: str) -> list[str]:
    if "spread" not in entry:
        return [f"### {title}", "", entry.get("note", "*not measured*"), ""]
    spread = entry["spread"]
    lines = [
        f"### {title}",
        "",
        f"* source: `{entry.get('source', 're-measured here')}`",
        f"* writers: {spread['n']}; median {spread['median']}, min {spread['min']},"
        f" max {spread['max']}, p10-p90 spread {spread['spread_p90_p10']}",
        "",
        f"| writer | pages/crops | {column} |",
        "| :--- | ---: | ---: |",
    ]
    for row in entry.get("worst", []):
        count = row.get("pages") or row.get("crops")
        lines.append(f"| {row['writer']} | {count} | {row[column]} |")
    lines.append("")
    return lines


def render(result: dict) -> str:
    lines = [
        "# Phase 14.6 - cross-scribe generalisation",
        "",
        "Generated by `python -m src.eval.crossscribe`. Each stage is scored per writer, and what"
        " matters in each section is the spread rather than the mean - a stage whose worst writer"
        " is far below its median is a different problem from one whose writers agree.",
        "",
        "## Per stage, worst five writers",
        "",
    ]
    lines += _table(result["classify"], "accuracy", "classification (leave-one-scribe-out)")
    lines += _table(result["ocr"], "cer", "label OCR (S3 val crops, higher CER is worse)")
    lines += _table(result["assemble"], "median_ged", "assembly (S5 test pages, higher is worse)")
    lines += _table(result["end_to_end"], "functional", "end to end (14.3's predicted rung)")
    lines += ["## Is a hard writer hard everywhere?", "", json.dumps(result["cross_stage"]), ""]
    classify = result["classify"]
    if "spread" in classify:
        lines += [
            f"Classification holds each of {classify['spread']['n']} writers out entirely;"
            f" {classify['rows_without_one']} rows carry no writer id and are training material in"
            " every fold, so the leave-one-out claim covers"
            f" {classify['rows_with_a_known_writer']} rows rather than the whole corpus.",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-ocr", action="store_true", help="no GPU; read the other stages")
    ap.add_argument("--stored-ocr", action="store_true", help="use the stored predictions instead")
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--limit", type=int, default=None, help="only the first N writers")
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    rescore = None if args.skip_ocr else not args.stored_ocr
    result = collect(rescore, args.model, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {stage: entry.get("spread", entry.get("note")) for stage, entry in result.items()},
            indent=2,
        )
    )
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

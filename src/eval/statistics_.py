"""Phase 14.10 - confidence intervals and seed variance for every headline number.

    python -m src.eval.statistics_              # reports/statistics.md + .json
    python -m src.eval.statistics_ --resamples 2000

Every headline in this project is a point estimate over a finite sample, and several of the
decisions made on those points were smaller than the interval around them. This row attaches the
interval, from the per-item artefacts the criteria already wrote, and separates the two kinds of
uncertainty rather than pooling them into one bar:

**Sampling uncertainty** - would another 162 pages, or another 2,835 crops, have given this
number? Answered by a bootstrap over the *items*: pages for S5 and for the end-to-end pass rate,
crops for S3. 10,000 resamples by default, percentile interval, seeded so the interval is
reproducible.

**Seed uncertainty** - would another random seed have given this number? Answered only where the
criterion actually ran repeats: S1 (3 seeded repeats) and S4 (10 seeds) record every repeat, so
the spread is read from them rather than estimated. Where a criterion ran once, the report says
`single run` instead of inventing a spread, because a bootstrap over items cannot see seed
variance and printing one bar for both would be a category error.

## Why the bootstrap is over items and not over anything else

A CER is a ratio of edit distance to characters, and a pass rate is a mean over pages. Both are
statistics of a sample of items drawn from a population of drawings, so resampling items with
replacement is the interval that answers "how stable is this number". Resampling *characters*
would ignore that a crop's errors are correlated within the crop, which is exactly the
dependence that makes a per-character interval far too narrow.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from src.utils.config import ROOT

REPORT_JSON = ROOT / "reports" / "statistics.json"
REPORT_MD = ROOT / "reports" / "statistics.md"

SEED = 20260922
RESAMPLES = 10000


def bootstrap(
    items: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    resamples: int = RESAMPLES,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap over items. Returns the point estimate and the interval."""
    if not items:
        return {}
    rng = random.Random(seed)
    n = len(items)
    draws = []
    for _ in range(resamples):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        draws.append(statistic(sample))
    draws.sort()
    low = draws[int((alpha / 2) * (resamples - 1))]
    high = draws[int((1 - alpha / 2) * (resamples - 1))]
    point = statistic(items)
    return {
        "point": round(point, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "half_width": round((high - low) / 2, 4),
        "items": n,
        "resamples": resamples,
        "seed": seed,
    }


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def s5_interval(resamples: int) -> dict:
    payload = _read(ROOT / "reports" / "s5_test_large.json")
    if not payload:
        return {"note": "reports/s5_test_large.json is absent"}
    geds = [float(row["ged"]) for row in payload["pages"]]
    passes = [1.0 if value <= 3 else 0.0 for value in geds]
    return {
        "metric": "median graph edit distance (test)",
        "target": payload.get("target_median_ged"),
        "median_ged": bootstrap(geds, statistics.median, resamples),
        "pass_share": bootstrap(passes, statistics.fmean, resamples),
        "seed_variance": "single run - S5 has no seeded repeats",
        "source": "reports/s5_test_large.json",
    }


def s3_interval(resamples: int, rescore: bool = False) -> dict:
    """CER resampled over crops: distance and characters travel together, as they must.

    With ``rescore`` the incumbent checkpoint is re-run (14.6's shared path) so the interval
    belongs to the published number; without it, the per-crop file on disk is used and the
    report says whose run that file is.
    """
    payload = _read(ROOT / "reports" / "s3_label_ocr.json")
    if not payload:
        return {"note": "reports/s3_label_ocr.json is absent"}
    if not rescore and not (ROOT / "experiments" / "ocr" / "s3_val.json").is_file():
        return {"metric": "label OCR CER", "point": payload.get("cer"), "note": "no per-crop file"}

    from src.eval.crossscribe import incumbent_predictions
    from src.ocr.metrics import score

    truths, predictions, frame, _source = incumbent_predictions(rescore)
    pairs = list(zip(truths, predictions, strict=True))
    aggregate = score([t for t, _ in pairs], [p for _, p in pairs])
    reproduces = abs(round(float(aggregate["cer"]), 4) - float(payload["cer"])) < 5e-4

    def cer(sample):
        return float(score([t for t, _ in sample], [p for _, p in sample])["cer"])

    entry = {
        "metric": "label OCR CER (val)",
        "target": payload.get("target_cer"),
        "published": payload.get("cer"),
        "per_crop_file_reproduces_published": reproduces,
        "seed_variance": "single run - S3 reports one training run per arm",
        "source": _source + " + reports/s3_label_ocr.json",
    }
    if not reproduces:
        entry["note"] = (
            "the per-crop predictions on disk are not the incumbent run, so the interval below"
            " belongs to that other run and is labelled with its own point estimate"
        )
    # The bootstrap is expensive here (a full CER over 2,835 pairs per resample), so it runs at a
    # reduced count unless the caller asks for more - and the count is printed either way.
    entry["cer"] = bootstrap(pairs, cer, min(resamples, 400))
    return entry


def end_to_end_interval(resamples: int) -> dict:
    payload = _read(ROOT / "reports" / "error_propagation.json")
    if not payload:
        return {"note": "reports/error_propagation.json is absent"}
    out: dict[str, Any] = {
        "metric": "functional pass@1 by rung (test)",
        "source": "reports/error_propagation.json",
        "seed_variance": "single run - the emitter is deterministic, so there is no seed",
    }
    for rung in ("gold", "gold_structure", "gold_text", "predicted"):
        values = [
            1.0 if row["rungs"][rung]["functional"] else 0.0
            for row in payload["rows"]
            if rung in row.get("rungs", {})
        ]
        out[rung] = bootstrap(values, statistics.fmean, resamples)
    return out


def _seeded(path: Path, key: str, label: str) -> dict:
    payload = _read(path)
    if not payload:
        return {"note": f"{path.name} is absent"}
    values = [float(repeat[key]) for repeat in payload.get("repeats", [])]
    entry = {
        "metric": label,
        "point": payload.get(key),
        "target": payload.get("target_accuracy") or payload.get("target_macro_f1"),
        # `relative_to` raises for a path outside the tree (a temporary one in the tests), and a
        # provenance string is not worth an exception: fall back to the name.
        "source": (
            str(path.relative_to(ROOT)).replace("\\", "/")
            if path.is_relative_to(ROOT)
            else path.name
        ),
    }
    if not values:
        entry["seed_variance"] = "the report carries no per-repeat values"
        return entry
    entry["seeds"] = len(values)
    entry["seed_mean"] = round(statistics.fmean(values), 4)
    entry["seed_std"] = round(statistics.pstdev(values), 4)
    entry["seed_min"] = round(min(values), 4)
    entry["seed_max"] = round(max(values), 4)
    entry["seed_interval"] = [entry["seed_min"], entry["seed_max"]]
    return entry


def collect(resamples: int = RESAMPLES, rescore_ocr: bool = False) -> dict:
    return {
        "protocol": {
            "sampling": f"percentile bootstrap over items, {resamples} resamples, seed {SEED}",
            "seeds": "read from the criterion's own repeats; never estimated where absent",
        },
        "s1_classification": _seeded(
            ROOT / "reports" / "s1_heldout_scribes.json", "accuracy", "diagram-type accuracy"
        ),
        "s4_roles": _seeded(
            ROOT / "reports" / "s4_role_labelling.json", "macro_f1", "role macro F1"
        ),
        "s3_ocr": s3_interval(resamples, rescore_ocr),
        "s5_assembly": s5_interval(resamples),
        "end_to_end": end_to_end_interval(resamples),
    }


def _ci(entry: dict | None) -> str:
    if not entry:
        return "*missing*"
    if "ci95" in entry:
        return f"{entry['point']} [{entry['ci95'][0]}, {entry['ci95'][1]}]"
    return str(entry.get("point", "*missing*"))


def render(result: dict) -> str:
    lines = [
        "# Phase 14.10 - intervals, and which kind of uncertainty each one is",
        "",
        f"Generated by `python -m src.eval.statistics_`. Sampling uncertainty:"
        f" {result['protocol']['sampling']}, resampling **items** (pages, crops) rather than"
        " characters, because a crop's errors are correlated within the crop. Seed uncertainty:"
        f" {result['protocol']['seeds']}.",
        "",
        "## Headline numbers with a 95% interval",
        "",
        "| criterion | metric | estimate [95% CI] | target | seed spread |",
        "| :--- | :--- | :--- | ---: | :--- |",
    ]

    def seed_text(entry: dict) -> str:
        if "seed_std" in entry:
            return (
                f"{entry['seeds']} seeds, std {entry['seed_std']},"
                f" range [{entry['seed_min']}, {entry['seed_max']}]"
            )
        return entry.get("seed_variance", "-")

    s1, s4 = result["s1_classification"], result["s4_roles"]
    s3, s5, e2e = result["s3_ocr"], result["s5_assembly"], result["end_to_end"]
    lines += [
        f"| S1 | {s1.get('metric')} | {s1.get('point')} | {s1.get('target')} | {seed_text(s1)} |",
        f"| S4 | {s4.get('metric')} | {s4.get('point')} | {s4.get('target')} | {seed_text(s4)} |",
        f"| S3 | {s3.get('metric')} | {_ci(s3.get('cer'))} | {s3.get('target')} |"
        f" {s3.get('seed_variance', '-')} |",
        f"| S5 | {s5.get('metric')} | {_ci(s5.get('median_ged'))} | {s5.get('target')} |"
        f" {s5.get('seed_variance', '-')} |",
        f"| 14.3 | end-to-end pass@1 | {_ci(e2e.get('predicted'))} | - |"
        f" {e2e.get('seed_variance', '-')} |",
        "",
        "## The error-propagation ladder, with intervals",
        "",
        "| rung | pass@1 [95% CI] | half-width |",
        "| :--- | :--- | ---: |",
    ]
    for rung in ("gold", "gold_structure", "gold_text", "predicted"):
        entry = e2e.get(rung) or {}
        lines.append(f"| `{rung}` | {_ci(entry)} | {entry.get('half_width', '-')} |")
    gold, predicted = e2e.get("gold") or {}, e2e.get("predicted") or {}
    if gold and predicted:
        separated = gold["ci95"][0] > predicted["ci95"][1]
        lines += [
            "",
            f"The `gold` and `predicted` intervals"
            f" {'do not overlap' if separated else 'overlap'}, so 14.3's headline drop"
            f" {'survives' if separated else 'does not survive'} the sampling uncertainty of this"
            " test split.",
            "",
        ]
    s3_cer = s3.get("cer") or {}
    if s3_cer.get("resamples"):
        lines += [
            f"The S3 interval uses **{s3_cer['resamples']} resamples** rather than the count"
            " named above: each draw re-scores a full CER over"
            f" {s3_cer['items']} crops, so the count is traded against the run time and printed"
            " rather than left implicit.",
            "",
        ]
    if s3.get("per_crop_file_reproduces_published") is False:
        lines += [
            f"**The S3 interval is labelled rather than trusted**: {s3['note']}."
            f" Its own point estimate is {s3['cer']['point']} against the published"
            f" {s3['published']}.",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--rescore-ocr", action="store_true", help="re-run the incumbent S3 checkpoint")
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = collect(args.resamples, args.rescore_ocr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)[:1800])
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

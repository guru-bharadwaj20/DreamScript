"""Phase 9.3.6 - CER, WER, and the per-style-cluster breakdown, in one place.

    python -m src.ocr.metrics                    # score every model that has written predictions
    python -m src.ocr.metrics --report

This is 9.3's counterpart to `src/detect/metrics.py`: **one evaluator, so that a number from
9.3.2 and a number from 9.3.3 are the same number.** Every model in 9.3 writes predictions to
`experiments/ocr/<name>.json` in one shape, and this module is the only thing that turns them
into a score. Nothing else in Phase 9.3 is allowed to compute a CER of its own, because two
implementations of "character error rate" that differ on whitespace or case will differ by more
than the models do.

## The definition, stated because it is not unique

    CER = sum over crops of edit_distance(prediction, truth) / sum over crops of len(truth)

**Corpus-level, not the mean of per-crop rates.** The mean-of-rates form weights a two-character
`q0` the same as a twenty-eight-character BPMN task label, and this corpus contains thousands of
each, so the two definitions genuinely disagree here. The corpus form is what IAM, ICDAR and
every OCR leaderboard report, and it is what the plan's <= 0.15 target has to be read against.

**Case and whitespace are normalised, and that is a decision with a cost.** Truths are lowercased
and internal whitespace is collapsed. Nothing downstream of 9.3 distinguishes `Submit` from
`submit` - Phase 11 builds a graph and Phase 12 emits code from identifiers - so scoring case
would be scoring something the pipeline discards. `--raw` reports the unnormalised number beside
it so the size of the concession is visible rather than assumed.

A per-crop rate is capped at 1.0 for the *mean-of-rates* column only. An uncapped mean is
unbounded - a model that emits forty characters for `q0` scores 20.0 on that crop - and one such
crop moves a mean over thousands. The corpus rate is left uncapped, because there the
denominator is the whole corpus and no single crop can run away with it.

## What it measured

FILLED_IN_BELOW
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from src.utils.config import ROOT

RUNS = ROOT / "experiments" / "ocr"
REPORT = ROOT / "reports" / "ocr.md"

#: The plan's row 9.3.6 target.
TARGET_CER = 0.15

_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, collapse internal whitespace, strip. The form every score below is computed on."""
    return _WHITESPACE.sub(" ", str(text)).strip().lower()


def edit_distance(a, b) -> int:
    """Levenshtein over any two sequences, one row at a time.

    Written out rather than imported: `editdistance` is not in this environment's lock file, and
    a twenty-line dynamic program with a test against known pairs is a smaller dependency than a
    wheel. O(len(a)) memory, which matters only because the IAM pretraining set has lines of a
    hundred characters and this is called a few hundred thousand times.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = np.arange(len(b) + 1, dtype=np.int32)
    current = np.empty_like(previous)
    for i, ch in enumerate(a, start=1):
        current[0] = i
        for j, other in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + (ch != other),  # substitution
            )
        previous, current = current, previous
    return int(previous[len(b)])


def score(truths, predictions, *, raw: bool = False) -> dict:
    """The one scoring function. `truths` and `predictions` are parallel sequences of strings."""
    truths = list(truths)
    predictions = list(predictions)
    if len(truths) != len(predictions):
        raise ValueError(f"{len(truths)} truths against {len(predictions)} predictions")
    if not truths:
        return {"crops": 0, "cer": float("nan"), "wer": float("nan"), "exact_match": float("nan")}

    prepare = (lambda s: str(s)) if raw else normalise
    char_errors = char_total = word_errors = word_total = 0
    per_crop = []
    exact = 0
    for truth, prediction in zip(truths, predictions, strict=True):
        t, p = prepare(truth), prepare(prediction)
        distance = edit_distance(t, p)
        char_errors += distance
        char_total += len(t)
        per_crop.append(min(1.0, distance / len(t)) if t else float(p != ""))
        tw, pw = t.split(), p.split()
        word_errors += edit_distance(tw, pw)
        word_total += len(tw)
        exact += t == p

    return {
        "crops": len(truths),
        "cer": round(char_errors / char_total, 4) if char_total else float("nan"),
        "wer": round(word_errors / word_total, 4) if word_total else float("nan"),
        "exact_match": round(exact / len(truths), 4),
        # Reported beside the corpus rate so the difference between the two definitions is
        # visible in every table rather than being a footnote about methodology.
        "mean_crop_cer": round(float(np.mean(per_crop)), 4),
        "characters": int(char_total),
        "meets_target": bool(char_total and char_errors / char_total <= TARGET_CER),
    }


def breakdown(frame, predictions, by: str) -> list[dict]:
    """The same score computed inside each group of `frame[by]`, largest group first."""
    rows = []
    frame = frame.reset_index(drop=True)
    series = frame[by].astype(str)
    for value in series.value_counts().index:
        mask = (series == value).to_numpy()
        rows.append(
            {
                by: value,
                **score(
                    frame.loc[mask, "text"].tolist(),
                    [p for p, keep in zip(predictions, mask, strict=True) if keep],
                ),
            }
        )
    return rows


def style_clusters() -> dict[str, int]:
    """`scribe -> 8.6 style cluster`, or an empty map when 8.6 has not been run.

    Read from 8.6 rather than recomputed. A per-cluster CER computed against a *re-derived*
    partition would not be the partition 8.7 priced, and the whole value of this breakdown is
    that it is comparable to that measurement.
    """
    try:
        from src.ocr.styled import style_groups

        return {str(k): int(v) for k, v in style_groups().items()}
    except Exception:
        return {}


def load_predictions(name: str, runs: Path = RUNS) -> dict:
    path = runs / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def models(runs: Path = RUNS) -> list[str]:
    return sorted(
        p.stem
        for p in runs.glob("*.json")
        if "predictions" in json.loads(p.read_text(encoding="utf-8-sig"))
    )


def run(runs: Path = RUNS) -> dict:
    """Score every model that has written predictions, overall and per style cluster."""

    from src.ocr.textcrops import load_index

    frame = load_index()
    clusters = style_clusters()
    results: dict[str, dict] = {}
    for name in models(runs):
        payload = load_predictions(name, runs)
        files = payload["files"]
        subset = frame.set_index("file").loc[files].reset_index()
        subset["style"] = subset["scribe"].map(clusters).fillna(-1).astype(int)
        predicted = payload["predictions"]
        results[name] = {
            "overall": score(subset["text"].tolist(), predicted),
            "by_style": breakdown(subset, predicted, "style"),
            "by_source": breakdown(subset, predicted, "source"),
            "by_kind": breakdown(subset, predicted, "kind"),
            "by_provenance": breakdown(subset, predicted, "provenance"),
        }

    best = min(results, key=lambda k: results[k]["overall"]["cer"]) if results else None
    return {
        "target_cer": TARGET_CER,
        "models": results,
        "best": best,
        "best_cer": results[best]["overall"]["cer"] if best else None,
        "meets_target": bool(best and results[best]["overall"]["meets_target"]),
        "style_clusters": len(set(clusters.values())) if clusters else 0,
    }


def write_report(result: dict, path: Path = REPORT) -> Path:
    lines = ["# Phase 9.3.6 - OCR metrics", "", f"Target CER <= {result['target_cer']}.", ""]
    lines += [
        "| model | crops | CER | WER | exact | mean-crop CER |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in sorted(result["models"].items(), key=lambda kv: kv[1]["overall"]["cer"]):
        o = entry["overall"]
        lines.append(
            f"| `{name}` | {o['crops']} | {o['cer']:.4f} | {o['wer']:.4f} | "
            f"{o['exact_match']:.4f} | {o['mean_crop_cer']:.4f} |"
        )
    for name, entry in result["models"].items():
        lines += [
            "",
            f"## `{name}` by style cluster",
            "",
            "| cluster | crops | CER | WER |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for row in entry["by_style"]:
            lines.append(
                f"| {row['style']} | {row['crops']} | {row['cer']:.4f} | {row['wer']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", type=Path, default=RUNS / "metrics.json")
    args = ap.parse_args(argv)

    result = run(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.report:
        print(write_report(result))
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2))
    for name, entry in result["models"].items():
        print(name, json.dumps(entry["overall"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

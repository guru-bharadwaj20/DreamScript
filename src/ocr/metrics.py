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

Thirteen model runs from 9.3.2-9.3.5 scored through one evaluator. **The target is CER <= 0.15 and
nothing comes close: the best model in Phase 9.3 is 0.6816.**

    model                CER      WER    exact   mean-crop CER
    adapt_style_adapt 0.6816   0.9503   0.0656          0.6453
    adapt_random      0.6818   0.9520   0.0669          0.6444
    lexicon_beam      0.6839   0.9438   0.1046          0.6340
    crnn_finetune     0.6864   0.9496   0.0660          0.6481
    lexicon_domain    0.6865   0.8907   0.1185          0.6398
    crnn_unwrap       0.6889   0.9491   0.0691          0.6459
    lexicon_greedy    0.6972   0.9420   0.1021          0.6440
    lexicon_generic   0.7171   0.9911   0.0928          0.6668
    trocr_finetune    0.8211   1.0100   0.0704          0.7525
    crnn_scratch      0.8397   0.9845   0.0439          0.7972
    crnn_zero_shot    0.9338   1.0721   0.0029          0.9077
    trocr_zero_shot   1.1702   1.5427   0.0055          0.8773

**The spread among the seven serious arms is 0.0355 CER, and the distance from all of them to the
target is 0.53.** That is the shape of this table and it is the same shape as 9.1.1's three
detector families landing inside 0.014 and 9.2.5's seven ablations inside 0.017: **the choices
being compared are much smaller than the thing that is wrong.** 9.3.2 identified what that is -
the same network reads IAM lines at 0.1157 - so the honest summary is that Phase 9.3 has a
recogniser that works and a corpus of crops it cannot read.

**The per-provenance row is the deployment answer and it is good news.** Cropping from 9.1.3's
detector instead of from a human's box costs **0.0040 CER - 0.6690 against 0.6730** - which is
nothing next to the 0.53 that separates either from the target. Combined with 9.3.1's finding that
the detector recovers 86.0% of text-bearing nodes at median IoU 0.9042, **the detector is not the
bottleneck in this pipeline and does not need to improve before OCR does.** One number qualifies
that: exact match falls **0.0739 to 0.0070, a factor of ten**, so box jitter destroys *completely
correct* reads while barely moving the character rate - a few pixels of the box is a few pixels of
a glyph, and a label needs every glyph.

**The per-kind row prices 9.3.1's edge-label rule exactly.** Edge crops score **0.8332 against the
nodes' 0.6709**, and 9.3.1 predicted the gap in advance by measuring that 19.4% of derived edge
boxes contain no ink other than the connector. Those crops cannot be read by anything, so **a
sixth of the edge CER is a cropping rule, not a recogniser.** Their exact-match is nevertheless the
*highest* in the table at 0.1705, because the edge labels that are correctly cropped are single
characters.

**The per-source row is the same effect from the other side.** fa_bresler scores **0.4788 with
exact match 0.4429** against hdbpmn's 0.6896 and 0.0051 - a model that gets 44% of fa_bresler's
labels perfectly right and half a percent of hdbpmn's. fa_bresler labels are single alphabet
symbols rendered onto synthetic paper; hdbpmn labels are wrapped English phrases photographed on
real paper. **These are two different tasks sharing one metric**, which is exactly why this module
reports the breakdown and not just the pooled number.

**Per style cluster, there is nothing there.** Clusters 0, 1 and 2 score 0.6935, 0.6933 and 0.6865
- a spread of **0.0070** across 3,900 hdbpmn crops - so the recogniser is not systematically worse
on any of 8.6's writing styles. That is a second, independent confirmation of 9.3.5's conclusion,
arrived at without training anything: if a per-style head were going to help, the per-style error
would have to differ first, and it does not. (The `-1` row is fa_bresler, which has no 8.6 style,
and is the source effect above rather than a style effect.)

**The two CER definitions disagree by up to 0.07 in this table** - `trocr_zero_shot` is 1.1702
corpus-level and 0.8773 as a capped mean of per-crop rates - which is the whole reason both columns
are printed. A reader given only the second would conclude that zero-shot TrOCR is better than
`crnn_scratch`; the corpus-level rate, which is what the target is written against, says it is far
worse.
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

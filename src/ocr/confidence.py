"""Phase 9.3.7 - a confidence on every read, calibrated, and written into the IR.

    python -m src.ocr.confidence
    python -m src.ocr.confidence --annotate     # fill `low_conf_text` on the hdbpmn IR files

9.3.2's recogniser returns a string. Phase 16's UI has to decide which of those strings to put in
front of a person to correct, and it can only do that if the string comes with a number saying
how much to trust it. The IR has carried a `low_conf_text` list since 2.1 and it is empty in all
5,796 files. This fills it.

## The confidence, and why it is the CTC posterior rather than the softmax maximum

A CTC decode is a *path*, and its probability is a product over columns. Three quantities are
computed from that path and compared, because the obvious one is not obviously right:

    min_char     the least confident emitting column. One bad character makes the label wrong,
                 so the minimum is what a decision about the whole string should arguably use.
    mean_char    the average over emitting columns. Forgiving of a single bad glyph.
    path_logp    the mean per-column log-probability of the whole path, blanks included.

They are scored against each other rather than one being asserted, because they disagree in a way
that matters: `min_char` is dominated by the worst character in a 28-character BPMN label and so
falls with length, while `mean_char` does not.

## The three controls, and why the third one is the real test

A confidence that ranks reads by how likely they are to be wrong is useful. A number that
correlates with error *for a reason that has nothing to do with the model* is not, and there are
two such numbers sitting in plain sight:

    random       shuffled confidences. AUROC 0.5 by construction; the floor.
    length       the length of the predicted string. **Long labels are harder**, so this alone
                 predicts error, and any model confidence that does not beat it has added nothing.
    ink          the crop's ink share. A faint or cluttered crop is harder to read.

`length` is the one to watch. It costs nothing, needs no model, and on a corpus that mixes `q0`
with `evaluate the Application` it will not be close to chance. A model confidence that beats it
is carrying information about *this image*; one that does not is a length detector.

## Calibration

AUROC says the ranking is useful; it says nothing about whether 0.9 means 90%. Reliability is
reported as expected calibration error over ten equal-count bins against **exact match**, since
that is the event the UI acts on - a label is either right or it needs a person - and a bin's
observed CER is reported beside it so a partially-correct read is visible too.

## What it measured

2,811 validation crops from 9.3.2's fine-tuned CRNN. **89.79% of reads are not exactly correct**,
which is what a CER of 0.69 looks like per crop and which makes this a well-populated ranking
problem rather than a rare-event one.

    score          AUROC (predicts an error)
    path_logp        0.9497
    min_char         0.9345
    length           0.8880   <- the control that costs nothing
    mean_char        0.8716
    ink              0.5889
    random           0.4849

**The confidence is real: `path_logp` beats the length baseline by 0.0617 and chance by 0.4648.**
That first margin is the one that matters. Label length alone reaches **0.8880** on this corpus,
because `q0` is easy and a 28-character BPMN phrase is not, and a confidence that had merely
learned "long labels are wrong" would have landed there. It does not; the model's own posterior
carries **0.06 AUROC of information about this particular image** beyond what the string's length
already says.

**The same ordering holds on the number the UI actually acts on.** Flagging the least confident
20% of reads captures **34.44% of all character error - a lift of 1.72** - against **27.23% and
1.36** for the length baseline. So a reviewer working through a fifth of the page sees a third of
the errors, and a third more of them than sorting by length would have shown.

**`mean_char` is the worst of the three model scores and that is the informative part.** Averaging
over emitting columns forgives the single bad glyph that makes a label wrong, so it ranks a
28-character read with one hopeless character above a 3-character read that is merely uncertain -
exactly backwards for a decision about the whole string. `min_char`, which is the pessimist, beats
it by 0.0629, and `path_logp`, which includes the blanks and so also carries *how confidently the
model placed nothing*, beats it by 0.0781. The obvious choice was the wrong one.

**Calibration is a disaster and it is separate from the ranking.** ECE **0.8121**: the bins from
confidence 0.80 up to 0.92 - **1,687 crops, 60% of the corpus** - contain **not one exactly
correct read between them**, and even the top bin at confidence 0.9992 is right 58.36% of the
time. The model is confident about paths, not about labels, and nothing in CTC training asks those
two to agree. **So the number must be used as a rank and never as a probability**, and Phase 16
must not show it to a user as "97% sure". A threshold chosen at a review budget - which is what
`budget_capture` returns and what `annotate` writes - is the correct way to consume it, because a
budget only needs the ordering.

**467 nodes across 125 hdbpmn pages are now flagged in `low_conf_text`**, at the threshold that
flags a fifth of reads, each as `{"ref", "confidence", "note"}` per 2.1's schema.

**Getting that into the IR took two bugs, and both are the same mistake.** The first `annotate`
wrote to `<page>.json` where the files are `<page>.ir.json`, and silently annotated **zero** pages
while still reporting 467 flags - so `pages_not_found` is now returned beside `pages_annotated`,
because a writer that finds nothing to write to must say so rather than report success. The second
wrote the entries as **bare node-id strings** instead of the schema's objects. That round-trips
through `json` perfectly, passes every test in this file, and breaks `Diagram.from_dict` on the
next load - which is where it was caught, by `tests/test_preprocess_page.py`, three commits later
and nowhere near here. **The IR is a schema, not a dictionary that happens to be JSON**, and the
lesson worth carrying is that a module writing into shared state should be validated against the
consumer's loader and not against its own round trip. `annotate` now also drops any `ref` that
does not name a node on that page, which is the invariant `src/ir/qa.py` checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS, normalise
from src.utils.config import ROOT

IR_DIR = ROOT / "data" / "processed" / "ir"

#: Share of reads the UI is willing to put in front of a person. The operating point every
#: threshold below is chosen at, so that "how much error does flagging 20% catch" has one answer.
REVIEW_BUDGET = 0.20

BINS = 10
SCORES = ("min_char", "mean_char", "path_logp")


# ------------------------------------------------------------------------------------------
# reading a confidence off a CTC posterior
# ------------------------------------------------------------------------------------------


def path_scores(logprobs: np.ndarray) -> dict[str, float]:
    """The three candidate confidences for one crop's posterior, all in [0, 1].

    Emitting columns are the best-path columns that are neither blank nor a repeat of the
    previous column - exactly the columns that contribute a character - because a string's
    confidence should not be diluted by the blanks between its letters, and on these crops the
    blanks are most of the columns.
    """
    probabilities = np.exp(logprobs)
    best = logprobs.argmax(axis=-1)
    top = probabilities.max(axis=-1)
    emitting = [
        i for i, index in enumerate(best) if index != 0 and (i == 0 or best[i - 1] != index)
    ]
    if not emitting:
        return {
            "min_char": 0.0,
            "mean_char": 0.0,
            "path_logp": float(np.exp(logprobs.max(-1).mean())),
        }
    values = top[emitting]
    return {
        "min_char": float(values.min()),
        "mean_char": float(values.mean()),
        "path_logp": float(np.exp(logprobs.max(axis=-1).mean())),
    }


# ------------------------------------------------------------------------------------------
# scoring a confidence
# ------------------------------------------------------------------------------------------


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUROC, ties averaged. `positive` marks the reads that are wrong.

    Confidences are negated by the caller when higher means *more* confident, so that this
    function always answers "does a high score predict an error".
    """
    scores, positive = np.asarray(scores, float), np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration(confidence: np.ndarray, correct: np.ndarray, bins: int = BINS) -> dict:
    """Equal-count reliability bins, plus expected calibration error against exact match."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, bool)
    order = np.argsort(confidence)
    chunks = np.array_split(order, bins)
    rows, error = [], 0.0
    for chunk in chunks:
        if not len(chunk):
            continue
        mean_confidence = float(confidence[chunk].mean())
        observed = float(correct[chunk].mean())
        rows.append(
            {
                "n": int(len(chunk)),
                "confidence": round(mean_confidence, 4),
                "exact_match": round(observed, 4),
            }
        )
        error += len(chunk) * abs(mean_confidence - observed)
    return {"bins": rows, "ece": round(error / max(1, len(confidence)), 4)}


def budget_capture(confidence: np.ndarray, distances: np.ndarray, budget: float) -> dict:
    """Flag the least confident `budget` share: how much of the total character error is caught?

    This is the number the UI cares about and AUROC does not directly give. Reviewing 20% of
    reads is only worth doing if those 20% contain much more than 20% of the errors, and the
    ratio to `budget` is what says whether the ranking is buying anything.
    """
    confidence, distances = np.asarray(confidence, float), np.asarray(distances, float)
    k = max(1, int(round(budget * len(confidence))))
    flagged = np.argsort(confidence)[:k]
    total = distances.sum()
    captured = distances[flagged].sum()
    return {
        "budget": budget,
        "flagged": int(k),
        "error_captured": round(float(captured / total), 4) if total else float("nan"),
        "lift": round(float((captured / total) / budget), 3) if total else float("nan"),
        "threshold": round(float(np.sort(confidence)[k - 1]), 4),
    }


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def run(model_name: str = "finetune", budget: float = REVIEW_BUDGET) -> dict:
    from src.ocr.crnn import diagram_split, greedy_decode, load
    from src.ocr.lexicon import posteriors
    from src.ocr.metrics import edit_distance

    model, chars, device = load(model_name)
    files, texts, frame = diagram_split("val")
    columns = posteriors(model, files, device)

    predictions = [greedy_decode(c, chars) for c in columns]
    measures = {name: np.array([path_scores(c)[name] for c in columns]) for name in SCORES}

    truths = [normalise(t) for t in texts]
    distances = np.array(
        [edit_distance(t, p) for t, p in zip(truths, predictions, strict=True)], dtype=float
    )
    wrong = distances > 0
    correct = ~wrong

    import cv2

    from src.ocr.textcrops import OUT as CROPS

    ink = np.array(
        [
            float((cv2.imread(f, cv2.IMREAD_GRAYSCALE) < 128).mean()) if Path(f).is_file() else 0.0
            for f in files
        ]
    )
    lengths = np.array([len(p) for p in predictions], dtype=float)
    rng = np.random.default_rng(42)

    #: Every candidate and every control, scored the same way: AUROC of "predicts an error".
    candidates = {
        **{name: -measures[name] for name in SCORES},
        "length": lengths,
        "ink": -ink,
        "random": rng.permutation(-measures["mean_char"]),
    }
    ranking = {name: round(auroc(values, wrong), 4) for name, values in candidates.items()}

    best = max(SCORES, key=lambda n: ranking[n])
    result = {
        "model": model_name,
        "crops": len(files),
        "error_rate": round(float(wrong.mean()), 4),
        "auroc": ranking,
        "best_model_score": best,
        # The two comparisons that decide whether this is a confidence or a length detector.
        "beats_length_by": round(ranking[best] - ranking["length"], 4),
        "beats_random_by": round(ranking[best] - ranking["random"], 4),
        "calibration": {name: calibration(measures[name], correct) for name in SCORES},
        "capture": {name: budget_capture(measures[name], distances, budget) for name in SCORES},
        "length_capture": budget_capture(-lengths, distances, budget),
        "crops_root": str(CROPS),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "confidence_scores.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "files": frame["file"].tolist(),
                "confidence": measures[best].round(5).tolist(),
                "score": best,
                "threshold": result["capture"][best]["threshold"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return result


# ------------------------------------------------------------------------------------------
# writing it into the IR
# ------------------------------------------------------------------------------------------


def annotate(
    source: str = "hdbpmn",
    threshold: float | None = None,
    scores_path: Path = RUNS / "confidence_scores.json",
    ir_dir: Path = IR_DIR,
    write: bool = True,
) -> dict:
    """Fill each diagram's `low_conf_text` with the reads that fell below `threshold`.

    2.1 froze the entry shape as `{"ref", "confidence"}` with an optional `alternatives` and
    `note`, and `src/ir/qa.py` checks that every `ref` names a real element. The first version of
    this function wrote bare node-id strings, which round-trips through `json` perfectly well and
    then breaks `Diagram.from_dict` on the next load - caught by `test_preprocess_page.py`, not
    here, three commits later. The IR is a schema and not a dictionary that happens to be JSON.
    """
    payload = json.loads(scores_path.read_text(encoding="utf-8-sig"))
    threshold = payload["threshold"] if threshold is None else threshold
    flagged: dict[str, list[dict]] = {}
    for name, value in zip(payload["files"], payload["confidence"], strict=True):
        if value >= threshold:
            continue
        parts = name.replace(".png", "").split("__")
        if len(parts) < 3 or not parts[2].startswith("n_"):
            continue
        flagged.setdefault(parts[1], []).append(
            {
                "ref": parts[2][2:],
                "confidence": round(float(value), 4),
                "note": f"9.3.7 {payload['score']} below {float(threshold):.4f}",
            }
        )

    touched = missing = 0
    for page, ids in flagged.items():
        path = ir_dir / source / f"{page}.ir.json"
        if not path.is_file():
            missing += 1
            continue
        diagram = json.loads(path.read_text(encoding="utf-8"))
        known = {n["id"] for n in diagram.get("nodes", [])}
        entries = {e["ref"]: e for e in ids if e["ref"] in known}
        diagram["low_conf_text"] = [entries[k] for k in sorted(entries)]
        if write:
            path.write_text(json.dumps(diagram, indent=2) + "\n", encoding="utf-8")
        touched += 1
    return {
        "threshold": round(float(threshold), 4),
        "pages_annotated": touched,
        # Non-zero here means the crop filename no longer maps onto an IR filename, which is
        # exactly how the first run silently annotated nothing at all.
        "pages_not_found": missing,
        "nodes_flagged": int(sum(len(v) for v in flagged.values())),
        "entry_shape": ["ref", "confidence", "note"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="finetune")
    ap.add_argument("--budget", type=float, default=REVIEW_BUDGET)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", type=Path, default=RUNS / "confidence.json")
    args = ap.parse_args(argv)

    result = run(args.model, args.budget)
    if args.annotate:
        result["ir"] = annotate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "calibration"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

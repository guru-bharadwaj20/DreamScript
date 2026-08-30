"""Phase 5.2.7 - are the probabilities honest? Reliability diagrams and Brier scores.

    python -m src.classify.calibration          # writes reports/figures/p5_calibration.png

A model that says "0.9 flowchart" should be right nine times in ten. Nothing so far in Phase 5
has checked that: 5.2.3 scored the argmax, 5.2.4 scored the ranking, and both are indifferent to
whether the numbers mean anything as probabilities. They have to mean something here, because
Phase 13 hands this confidence to a user and Phase 10 uses it to decide when to ask.

    reliability curve   predicted probability against observed frequency, in ten bins
    Brier score         mean squared error of the probability vector; lower is better
    ECE                 expected calibration error - the average gap between the two axes,
                        weighted by how many predictions fall in each bin

Both corrections the plan asks about are applied and measured: **Platt scaling** (a sigmoid
fitted to the scores) and **isotonic regression** (a monotone step function, more flexible and
hungrier for data). Each is fitted inside the training folds only, through
`CalibratedClassifierCV`, so the calibration never sees the rows it is evaluated on.

## What it measured

    model    variant        Brier    ECE     accuracy   mean confidence   over-confidence
    logreg   uncalibrated   0.0986  0.0278    0.9410        0.9641            +0.023
    logreg   platt          0.1159  0.0638    0.9291        0.8750            -0.054
    logreg   isotonic       0.1038  0.0353    0.9381        0.9041            -0.034

    knn      uncalibrated   0.1448  0.0724    0.9276        1.0000            +0.072
    knn      platt          0.1184  0.0683    0.9343        0.8667            -0.068
    knn      isotonic       0.1171  0.0580    0.9336        0.8763            -0.057

    tree     uncalibrated   0.1892  0.0881    0.9022        0.9900            +0.088
    tree     platt          0.1429  0.0989    0.9239        0.8256            -0.098
    tree     isotonic       0.1360  0.0831    0.9261        0.8431            -0.083

**Logistic regression is already calibrated and both corrections make it worse.** ECE 0.0278
uncalibrated against 0.0353 isotonic and 0.0638 Platt; the Brier score moves the same way. That
is the expected result for a model fitted by maximum likelihood on a proper scoring rule - its
probabilities *are* the fit - and it is worth recording because "calibrate everything" is the
reflex, and here it would cost accuracy (0.9410 -> 0.9291) as well as calibration.

**The tree and kNN are badly over-confident and both corrections help them.** The tree averages
0.990 confidence at 0.902 accuracy - a nine-point gap - because a leaf that holds seven pure
training rows reports 1.00 whatever it actually knows. Isotonic regression cuts its Brier from
0.1892 to 0.1360 and *raises* its accuracy to 0.9261, which is the rare case where calibrating
improves the argmax too: re-ranking the classes by corrected probability changes some decisions,
and on this corpus it changes them for the better.

kNN's uncalibrated mean confidence is **exactly 1.0000**, which is not over-confidence so much
as the absence of any confidence at all: 5.1.2 selected k = 1, so every probability vector is
one-hot, the reliability curve has a single point, and there is nothing to be miscalibrated. Its
ECE of 0.0724 is just its error rate. Calibration gives it a usable probability for the first
time, which matters more for Phase 13 than the Brier improvement suggests.

## What Phase 5 recommends downstream

Use **logistic regression uncalibrated** where a probability is needed - it is the best
calibrated model here and the correction is not worth its cost. Use **isotonic** if the tree or
kNN is ever the model in play, and never Platt for the tree, which is the one combination that
made ECE worse (0.0881 -> 0.0989) while improving Brier. The two metrics disagreeing there is
not a contradiction: Platt sharpens the average squared error while shifting the whole curve
below the diagonal, trading over-confidence for under-confidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize

from src.classify.cv import MODELS, REAL_MODELS, SEEDS, out_of_fold, splitter
from src.classify.data import Dataset, load
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p5_calibration.png"

N_BINS = 10


def one_hot(y_true: np.ndarray, classes) -> np.ndarray:
    """One column per class, always. `label_binarize` collapses the two-class case to one."""
    truth = label_binarize(y_true, classes=list(classes))
    if truth.shape[1] == 1:
        truth = np.hstack([1 - truth, truth])
    return truth


def brier(y_true: np.ndarray, y_proba: np.ndarray, classes) -> float:
    """Multiclass Brier score: mean squared distance from the one-hot truth."""
    return float(np.mean(np.sum((y_proba - one_hot(y_true, classes)) ** 2, axis=1)))


def reliability(confidence: np.ndarray, correct: np.ndarray, bins: int = N_BINS) -> dict:
    """Observed accuracy against predicted confidence, plus the expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    rows, gap, total = [], 0.0, len(confidence)
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        mean_confidence = float(confidence[mask].mean())
        observed = float(correct[mask].mean())
        rows.append(
            {
                "bin": b,
                "confidence": round(mean_confidence, 4),
                "accuracy": round(observed, 4),
                "count": int(mask.sum()),
            }
        )
        gap += mask.sum() / total * abs(mean_confidence - observed)
    return {"bins": rows, "ece": round(float(gap), 4)}


def calibrated_out_of_fold(
    model: str, dataset: Dataset, method: str, *, seed: int = SEEDS[0], n_splits: int = 5
):
    """Out-of-fold predictions for a calibrated wrapper around one of the registry's models.

    The wrapper refits its own internal 3-fold split *inside* each training fold, so the sigmoid
    or the isotonic map is estimated on data the outer fold is allowed to see and never on the
    rows being scored.
    """
    classes = np.array(sorted(set(dataset.y.tolist())), dtype=object)
    probabilities = np.zeros((len(dataset.y), len(classes)))
    predictions = np.empty(len(dataset.y), dtype=object)

    for train_index, test_index in splitter(dataset, "stratified", seed, n_splits):
        wrapper = CalibratedClassifierCV(clone(MODELS[model]()), method=method, cv=3)
        wrapper.fit(dataset.X[train_index], dataset.y[train_index])
        raw = wrapper.predict_proba(dataset.X[test_index])
        for position, name in enumerate(wrapper.classes_):
            probabilities[test_index, int(np.where(classes == name)[0][0])] = raw[:, position]
        predictions[test_index] = wrapper.predict(dataset.X[test_index])
    return predictions, probabilities, classes


def assess(dataset: Dataset, model: str, seed: int = SEEDS[0]) -> dict:
    """Uncalibrated, Platt-scaled and isotonic, scored the same way."""
    base = out_of_fold(model, dataset, seed=seed)
    out = {"model": model, "variants": {}}

    def score(name: str, y_pred, y_proba) -> dict:
        confidence = y_proba.max(axis=1)
        correct = (y_pred == dataset.y).astype(float)
        curve = reliability(confidence, correct)
        return {
            "variant": name,
            "brier": round(brier(dataset.y, y_proba, base.classes), 4),
            "ece": curve["ece"],
            "accuracy": round(float(correct.mean()), 4),
            "mean_confidence": round(float(confidence.mean()), 4),
            "overconfidence": round(float(confidence.mean() - correct.mean()), 4),
            "curve": curve["bins"],
        }

    out["variants"]["uncalibrated"] = score("uncalibrated", base.y_pred, base.y_proba)
    for method, label in (("sigmoid", "platt"), ("isotonic", "isotonic")):
        predictions, probabilities, _ = calibrated_out_of_fold(model, dataset, method, seed=seed)
        out["variants"][label] = score(label, predictions, probabilities)
    return out


def figure(results: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"uncalibrated": "#a63d54", "platt": "#3b6ea5", "isotonic": "#4b8b3b"}
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.4), squeeze=False)
    for ax, result in zip(axes[0], results, strict=True):
        ax.plot([0, 1], [0, 1], ls=":", lw=0.9, color="#999999")
        for name, row in result["variants"].items():
            curve = row["curve"]
            ax.plot(
                [point["confidence"] for point in curve],
                [point["accuracy"] for point in curve],
                marker="o",
                ms=3.5,
                lw=1.4,
                color=styles.get(name, "#777777"),
                label=f"{name} (Brier {row['brier']:.3f}, ECE {row['ece']:.3f})",
            )
        ax.set_title(result["model"], fontsize=10)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("observed accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(corpus: str = "real", models=REAL_MODELS, write: bool = True) -> dict:
    dataset = load(corpus)
    results = [assess(dataset, model) for model in models]
    summary = {
        "corpus": corpus,
        "models": {
            row["model"]: {
                name: {k: v for k, v in variant.items() if k != "curve"}
                for name, variant in row["variants"].items()
            }
            for row in results
        },
    }
    if write:
        summary["figure"] = str(figure(results).relative_to(ROOT))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=list(REAL_MODELS))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, tuple(args.models), not args.no_figure), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

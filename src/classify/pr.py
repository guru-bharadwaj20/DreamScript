"""Phase 5.2.5 - precision-recall curves, and what the minority classes would cost to recover.

    python -m src.classify.pr          # writes reports/figures/p5_pr.png

ROC flatters a rare class. With 40 circuits among 1,340 pages, a model can label a hundred
flowcharts as circuits and barely move the false-positive rate, because the denominator is the
1,300 pages that are not circuits. Precision-recall has no such denominator: **every false
positive shows up immediately as lost precision**, which is why the plan asks for these curves
specifically for the minority type.

The baseline is different too, and it is drawn on every panel. A random model's PR curve sits at
the class's prevalence - 0.030 for circuits, 0.448 for flowcharts - so an average precision of
0.30 is a disaster for one class and a triumph for another. There is no shared "chance line".

## The question this task exists to answer

5.2.4 found that logistic regression ranks circuits at 0.788 AUC while predicting only 23% of
them, which means the information is there and the argmax is throwing it away. The right
follow-up is not "is the model good at circuits" but **"what would it cost to actually predict
them"** - and a PR curve answers exactly that: pick a recall, read off the precision you would
have to accept.

## What it measured

Average precision per class, with each class's own prevalence as its baseline and the lift over
it. Logistic regression:

    class            AP     prevalence   lift
    wireframe      0.994      0.448       2.2x
    state_machine  0.990      0.037      26.5x
    flowchart      0.971      0.448       2.2x
    er_diagram     0.804      0.037      21.5x
    circuit        0.262      0.030       8.8x

    macro AP: logreg 0.804, knn 0.673, tree 0.654

**Lift is the column that makes the minority classes readable.** State machines and ER diagrams
score 26x and 21x their base rate - by that measure they are the *best* classified types on the
page, better than the two big classes, whose 2.2x is all a 45%-prevalence class can achieve.
Circuits at 8.8x are genuinely learned too. What the AP column says is that 8.8 times 0.030 is
still only 0.26.

## The price of recovering the circuits, in one row

5.2.4 established that the information is there and the argmax discards it. Here is what
harvesting it would cost, reading logistic regression's circuit curve:

    recall wanted    best precision available
        0.25              0.379
        0.50              0.212
        0.75              0.070
        0.90              0.033

**Half the circuits can be found at 21% precision** - four false alarms for every hit - and 75%
of them only at 7%, which is barely above the 3.0% you would get by calling everything a
circuit. So the honest answer to "can a threshold fix the circuit class" is: partly, and not
cheaply. A downstream stage that can afford to check a shortlist would take the 0.25/0.38 point;
an autonomous pipeline that must commit to one type should not.

The same table for kNN is flat at 0.030 - the prevalence - at every recall level. That is the
one-hot probability problem from 5.2.4 again: with k = 1 there is no threshold to move, so
there is no trade to price.

## ER diagrams behave completely differently, and that is the encouraging part

    recall wanted    best precision available (er_diagram)
        0.25              0.960
        0.50              0.903
        0.75              0.813
        0.90              0.257

**Three quarters of the ER diagrams are recoverable at 81% precision.** With 50 rows - ten more
than circuits - the same pipeline gives a class that is genuinely usable. Whatever is wrong with
circuits is specific to circuits, not a general statement about small classes, which is the same
conclusion 5.2.3 reached from the other direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from src.classify.cv import REAL_MODELS, SEEDS, out_of_fold
from src.classify.data import Dataset, load
from src.classify.roc import COLOURS
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURE = ROOT / "reports" / "figures" / "p5_pr.png"

#: Recall levels at which the achievable precision is tabulated. The point of the table is to
#: price the trade rather than to describe the curve.
RECALL_TARGETS = (0.25, 0.5, 0.75, 0.9)


def precision_at_recall(precision: np.ndarray, recall: np.ndarray, target: float) -> float:
    """The best precision available at or above a given recall. `nan` if unreachable."""
    reachable = precision[recall >= target]
    return float(reachable.max()) if len(reachable) else float("nan")


def curves(
    dataset: Dataset, model: str, strategy: str = "stratified", seed: int = SEEDS[0]
) -> dict:
    predictions = out_of_fold(model, dataset, strategy=strategy, seed=seed)
    classes = list(predictions.classes)
    truth = label_binarize(predictions.y_true, classes=classes)

    per_class = {}
    for index, name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(
            truth[:, index], predictions.y_proba[:, index]
        )
        prevalence = float(truth[:, index].mean())
        per_class[name] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "average_precision": round(
                float(average_precision_score(truth[:, index], predictions.y_proba[:, index])), 4
            ),
            "prevalence": round(prevalence, 4),
            # How many times better than guessing at the class's own base rate.
            "lift": round(
                float(average_precision_score(truth[:, index], predictions.y_proba[:, index]))
                / prevalence,
                2,
            ),
            "precision_at_recall": {
                str(target): round(precision_at_recall(precision, recall, target), 4)
                for target in RECALL_TARGETS
            },
        }

    return {
        "model": model,
        "classes": classes,
        "per_class": per_class,
        "macro_average_precision": round(
            float(np.mean([row["average_precision"] for row in per_class.values()])), 4
        ),
    }


def figure(results: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.4), squeeze=False)
    for ax, result in zip(axes[0], results, strict=True):
        for name in result["classes"]:
            row = result["per_class"][name]
            ax.step(
                row["recall"],
                row["precision"],
                where="post",
                lw=1.4,
                color=COLOURS.get(name, "#777777"),
                label=f"{name} (AP {row['average_precision']:.3f})",
            )
            ax.axhline(
                row["prevalence"], lw=0.7, ls=":", color=COLOURS.get(name, "#777777"), alpha=0.7
            )
        ax.set_title(
            f"{result['model']} — macro AP {result['macro_average_precision']:.3f}", fontsize=10
        )
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def run(
    corpus: str = "real", models=REAL_MODELS, strategy: str = "stratified", write: bool = True
) -> dict:
    dataset = load(corpus)
    results = [curves(dataset, model, strategy) for model in models]
    summary = {
        "corpus": corpus,
        "strategy": strategy,
        "models": {
            row["model"]: {
                "macro_average_precision": row["macro_average_precision"],
                "per_class": {
                    name: {
                        "average_precision": row["per_class"][name]["average_precision"],
                        "prevalence": row["per_class"][name]["prevalence"],
                        "lift": row["per_class"][name]["lift"],
                        "precision_at_recall": row["per_class"][name]["precision_at_recall"],
                    }
                    for name in row["classes"]
                },
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
    ap.add_argument("--strategy", default="stratified", choices=["stratified", "grouped"])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        print(
            json.dumps(
                run(args.corpus, tuple(args.models), args.strategy, not args.no_figure), indent=2
            )
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 5.2.4 - one-vs-rest ROC curves, and macro and micro AUC.

    python -m src.classify.roc          # writes reports/figures/p5_roc.png

Five binary problems per model - each class against the other four - drawn on one axis, plus the
two averages that summarise them:

    micro AUC   pools every (row, class) decision into one curve; dominated by the big classes
    macro AUC   averages the five per-class AUCs; every class counts once

Both are reported for the reason 5.2.3 reports macro and weighted F1: on a corpus where three
classes hold 3% of the rows each, the two numbers answer different questions and only the macro
one reflects what this project claims to support.

## AUC is threshold-free, which is why it is here

Everything else in 5.2 scores the models at the operating point their `predict` happens to
choose. AUC asks a different question - **how well does the model rank a page of this class
above pages that are not** - and it is the metric that separates "the model has no idea" from
"the model knows but its threshold is in the wrong place". 5.2.7's calibration curves are the
other half of that distinction.

The probabilities are 5.2.1's out-of-fold ones, so no row is scored by a model that saw it.

## What it measured

    model     macro AUC   micro AUC   weighted AUC
    logreg      0.9454      0.9903        0.9837
    tree        0.8798      0.9536        0.9432
    knn         0.8655      0.9548        0.9443

    per-class AUC     circuit   er_diagram   flowchart   state_machine   wireframe
    logreg             0.788      0.959       0.984         1.000          0.997
    tree               0.662      0.849       0.942         0.978          0.968
    knn                0.606      0.824       0.946         0.980          0.972

**Logistic regression ranks circuits at 0.788 AUC while finding only 23% of them** (5.2.3). That
gap is the most useful thing this task produced: the model is not blind to circuits, it ranks
them well above chance - it simply never puts enough probability on the class to win a
five-way argmax against flowchart and wireframe. A circuit page is usually the model's second
or third choice. **This is a threshold problem, not a representation problem**, and it means a
class-specific threshold or a cost-sensitive decision rule could recover a large part of the
missing recall without touching the features. 5.2.5's precision-recall curves are where the
price of that trade is read off.

## kNN's curves are staircases, and that is a property of k = 1

5.1.2 selected k = 1, so every out-of-fold probability vector kNN produces is **one-hot**: the
single nearest neighbour is either of the class or it is not. A one-hot score has exactly two
distinct values, so its ROC curve has one interior point and the AUC collapses towards
`(sensitivity + specificity) / 2`. kNN's 0.8655 macro AUC is therefore not comparable with
logistic regression's 0.9454 as a *ranking* quality - there is no ranking to measure.

That is not an argument for changing k, which was selected on macro F1 against a proper sweep.
It is an argument for reading AUC and calibration (5.2.7) as inapplicable to this particular
model, and for saying so here rather than letting a lower number look like worse performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from src.classify.cv import REAL_MODELS, SEEDS, out_of_fold
from src.classify.data import Dataset, load
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURE = ROOT / "reports" / "figures" / "p5_roc.png"

#: Fixed per class so this figure, 5.2.5's and 5.2.6's all colour a class the same way.
COLOURS = {
    "flowchart": "#3b6ea5",
    "wireframe": "#c1663d",
    "state_machine": "#4b8b3b",
    "er_diagram": "#8a5fa8",
    "circuit": "#a63d54",
}


def curves(
    dataset: Dataset, model: str, strategy: str = "stratified", seed: int = SEEDS[0]
) -> dict:
    """Per-class ROC curves plus micro and macro AUC, from out-of-fold probabilities."""
    predictions = out_of_fold(model, dataset, strategy=strategy, seed=seed)
    classes = list(predictions.classes)
    truth = label_binarize(predictions.y_true, classes=classes)
    scores = predictions.y_proba

    per_class = {}
    for index, name in enumerate(classes):
        fpr, tpr, _ = roc_curve(truth[:, index], scores[:, index])
        per_class[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(auc(fpr, tpr)), 4),
            "support": int(truth[:, index].sum()),
        }

    micro_fpr, micro_tpr, _ = roc_curve(truth.ravel(), scores.ravel())
    return {
        "model": model,
        "classes": classes,
        "per_class": per_class,
        "micro": {
            "fpr": micro_fpr.tolist(),
            "tpr": micro_tpr.tolist(),
            "auc": round(float(auc(micro_fpr, micro_tpr)), 4),
        },
        "macro_auc": round(float(roc_auc_score(truth, scores, average="macro")), 4),
        "weighted_auc": round(float(roc_auc_score(truth, scores, average="weighted")), 4),
    }


def figure(results: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.4), squeeze=False)
    for ax, result in zip(axes[0], results, strict=True):
        for name in result["classes"]:
            row = result["per_class"][name]
            ax.plot(
                row["fpr"],
                row["tpr"],
                lw=1.4,
                color=COLOURS.get(name, "#777777"),
                label=f"{name} ({row['auc']:.3f})",
            )
        ax.plot(
            result["micro"]["fpr"],
            result["micro"]["tpr"],
            lw=1.2,
            ls="--",
            color="#444444",
            label=f"micro ({result['micro']['auc']:.3f})",
        )
        ax.plot([0, 1], [0, 1], lw=0.8, ls=":", color="#999999")
        ax.set_title(f"{result['model']} — macro AUC {result['macro_auc']:.3f}", fontsize=10)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(fontsize=7, loc="lower right", frameon=False)
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
                "macro_auc": row["macro_auc"],
                "micro_auc": row["micro"]["auc"],
                "weighted_auc": row["weighted_auc"],
                "per_class_auc": {n: row["per_class"][n]["auc"] for n in row["classes"]},
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

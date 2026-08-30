"""Phase 5.2.3 - the full classification report, per class and per model.

    python -m src.classify.metrics          # writes reports/classification_report.md

Accuracy is one number for a five-class problem whose classes differ in size by a factor of
fifteen, and 5.1.4 measured what that hides: a constant predictor scores 0.448. This module
reports what the models actually do, class by class:

    precision, recall, F1, support     per class, averaged over the three CV repeats
    macro F1                           every class counted once - Phase 5's headline
    weighted F1                        every *row* counted once - what accuracy nearly is
    balanced accuracy                  mean per-class recall

Everything is computed from 5.2.1's out-of-fold predictions, so the report describes exactly the
same fits that 5.2.4's curves and 5.2.8's significance test do.

## Macro against weighted, and why both are printed

They disagree by 0.15 here and the disagreement is the finding, not a technicality. Macro F1
gives the 40-row circuit class the same weight as the 600-row flowchart class; weighted F1
gives it 40/1340 of the weight. **A model that never predicted a circuit at all would lose 0.20
macro F1 and 0.02 weighted F1.** The first number is the one that reflects what DreamScript
claims to do - five diagram types, all of them supported - so it is the one Phase 5 selects on,
and the second is printed beside it so the gap stays visible.

## What it measured

    model      macro F1   weighted F1   balanced acc   macro - weighted
    logreg      0.7917      0.9379         0.7891          -0.1461
    knn         0.7614      0.9224         0.7523          -0.1611
    tree        0.7434      0.9001         0.7596          -0.1568
    majority    0.1237      0.2770         0.2000          -0.1533

**Every model loses about 0.15 between weighted and macro F1**, including the constant
predictor, and that is the corpus rather than the models: three of five classes hold 3-4% of the
rows each. Read weighted, this problem looks 93% solved. Read macro, it is 79%.

Per class, logistic regression:

    class            precision   recall     F1     support
    wireframe          0.981     0.981    0.981     600
    state_machine      0.920     0.993    0.955      50
    flowchart          0.941     0.958    0.950     600
    er_diagram         0.814     0.780    0.796      50
    circuit            0.343     0.233    0.277      40

**Four classes are solved and one is not.** Circuits are found less than a quarter of the time
by every model - 0.233 recall for logreg, 0.208 for kNN, 0.325 for the tree - and when a model
does say "circuit" it is right about a third of the time. That single class costs the headline
about 0.14 macro F1 all by itself.

Scarcity is not the explanation, and the corpus makes that easy to check: **state machines have
50 rows to circuits' 40 and score 0.955 F1**. The difference is in the features, and 4.1.1 named
it a phase early - a circuit's components are open symbols rather than closed outlines, so
`node_count` sees 3.66 fewer nodes than the truth and most of the per-node features are
undefined on those pages. The classifier is being asked to recognise a class whose measurements
are mostly missing.

The tree is the only model with circuit recall above 0.30, and it is the worst model overall.
That is the shape of the trade-off on this corpus: the tree spends splits on the small classes
that the linear model spends on the large ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

from src.classify.cv import SEEDS, out_of_fold
from src.classify.data import Dataset, load
from src.utils.config import ROOT

REPORT = ROOT / "reports" / "classification_report.md"


def _relative(path) -> str:
    """Repo-relative when it can be; absolute when a caller redirected it elsewhere."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def report(dataset: Dataset, model: str, strategy: str = "stratified", seeds=SEEDS) -> dict:
    """Per-class precision, recall and F1, averaged over the repeats."""
    classes = sorted(set(dataset.y.tolist()))
    runs = [out_of_fold(model, dataset, strategy=strategy, seed=seed) for seed in seeds]

    stacked = {name: {"precision": [], "recall": [], "f1": []} for name in classes}
    macro, weighted, balanced = [], [], []
    for run in runs:
        precision, recall, f1, support = precision_recall_fscore_support(
            run.y_true, run.y_pred, labels=classes, zero_division=0
        )
        for index, name in enumerate(classes):
            stacked[name]["precision"].append(float(precision[index]))
            stacked[name]["recall"].append(float(recall[index]))
            stacked[name]["f1"].append(float(f1[index]))
        macro.append(float(f1_score(run.y_true, run.y_pred, average="macro", zero_division=0)))
        weighted.append(
            float(f1_score(run.y_true, run.y_pred, average="weighted", zero_division=0))
        )
        balanced.append(float(balanced_accuracy_score(run.y_true, run.y_pred)))

    support = {name: int((dataset.y == name).sum()) for name in classes}
    return {
        "model": model,
        "strategy": strategy,
        "repeats": len(runs),
        "per_class": {
            name: {
                "precision": round(float(np.mean(stacked[name]["precision"])), 4),
                "recall": round(float(np.mean(stacked[name]["recall"])), 4),
                "f1": round(float(np.mean(stacked[name]["f1"])), 4),
                "f1_std": round(float(np.std(stacked[name]["f1"])), 4),
                "support": support[name],
            }
            for name in classes
        },
        "macro_f1": round(float(np.mean(macro)), 4),
        "weighted_f1": round(float(np.mean(weighted)), 4),
        "balanced_accuracy": round(float(np.mean(balanced)), 4),
        "macro_minus_weighted": round(float(np.mean(macro) - np.mean(weighted)), 4),
    }


def to_markdown(reports: list[dict], dataset: Dataset) -> str:
    classes = sorted(set(dataset.y.tolist()))
    lines = [
        "# Phase 5.2.3 — classification report",
        "",
        f"Out-of-fold predictions from {reports[0]['repeats']} repeats of stratified 5-fold CV "
        f"over {len(dataset.y)} real photographs. Every number is an average over the repeats.",
        "",
        "## Headline",
        "",
        "| model | macro F1 | weighted F1 | balanced accuracy | macro − weighted |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in reports:
        lines.append(
            f"| `{row['model']}` | **{row['macro_f1']:.4f}** | {row['weighted_f1']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['macro_minus_weighted']:+.4f} |"
        )

    for row in reports:
        lines += [
            "",
            f"## `{row['model']}`",
            "",
            "| class | precision | recall | F1 | ± | support |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in classes:
            cell = row["per_class"][name]
            lines.append(
                f"| {name} | {cell['precision']:.3f} | {cell['recall']:.3f} | "
                f"{cell['f1']:.3f} | {cell['f1_std']:.3f} | {cell['support']} |"
            )
    lines.append("")
    return "\n".join(lines)


def run(
    corpus: str = "real",
    models=("logreg", "knn", "tree", "majority"),
    strategy: str = "stratified",
    write: bool = True,
) -> dict:
    dataset = load(corpus)
    reports = [report(dataset, model, strategy) for model in models]
    if write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(to_markdown(reports, dataset), encoding="utf-8")
    return {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "reports": reports,
        "path": _relative(REPORT) if write else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=["logreg", "knn", "tree", "majority"])
    ap.add_argument("--strategy", default="stratified", choices=["stratified", "grouped"])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.corpus, tuple(args.models), args.strategy, not args.no_write)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

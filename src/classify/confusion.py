"""Phase 5.2.6 - confusion matrices, raw and row-normalised.

    python -m src.classify.confusion          # writes reports/figures/p5_confusion.png

Two matrices per model, because they answer different questions and the raw one alone is
unreadable on a corpus this skewed:

    raw               how many pages went where. The 600-row classes dominate every cell.
    row-normalised    what fraction of each true class went where - per-class recall, spread
                      across the columns that took it.

The row-normalised matrix is the one to read. On this corpus the raw matrix's largest
off-diagonal entry can be a *smaller* error rate than a cell a tenth its size, simply because
one class has fifteen times the rows.

Everything comes from 5.2.1's out-of-fold predictions, summed over the three repeats, so each
cell counts 4,020 page-decisions rather than 1,340.

## What it measured

4,020 page-decisions per model. Logistic regression, the five worst confusions **by share of
the true class**:

    true            predicted        count   share of true class
    circuit         flowchart          67          0.558
    er_diagram      flowchart          29          0.193
    circuit         wireframe          16          0.133
    circuit         state_machine       8          0.067
    er_diagram      wireframe           4          0.027

and the same model's five worst **by raw count**:

    circuit    -> flowchart    67   (0.558 of circuits)
    flowchart  -> circuit      47   (0.026 of flowcharts)
    er_diagram -> flowchart    29   (0.193 of ER diagrams)
    circuit    -> wireframe    16   (0.133 of circuits)
    flowchart  -> wireframe    15   (0.008 of flowcharts)

**The two orderings disagree in exactly the way the docstring predicted.** `flowchart ->
circuit` is the second-largest error by count and the *twelfth* by share: 47 pages is 2.6% of
the flowchart class. Read from the raw matrix alone, a reader would spend their time on
flowcharts, which are 95.8% correct.

## Everything drains into flowchart

**More than half of all circuits are called flowcharts** (0.558 for logreg, 0.500 for kNN, 0.475
for the tree), and a fifth of ER diagrams go the same way. Flowchart is the sink for anything
the model cannot place - which is what a 45%-prevalence class does under a probabilistic argmax,
and it is also substantively right: a hand-drawn circuit *is* a set of symbols joined by lines,
and 4.1's features describe it as a page with few closed shapes and many strands, which is not
far from a sparse flowchart.

The reverse direction is where the models differ. The tree's largest error by count is
`flowchart -> circuit` at 75 pages, which is the cost of the `class_weight="balanced"` it
selected in 5.1.3: the tree is the only model that spends real flowchart accuracy trying to
find circuits, and 5.2.3 shows it is also the only one with circuit recall above 0.30. That is
the same trade priced two ways.

**No model confuses wireframes with anything.** The wireframe row is 0.98 diagonal for every
model, which is consistent with 4.1's finding that wireframes are the type the handcrafted
features describe most distinctly - they are the only class whose ink is mostly not shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.classify.cv import REAL_MODELS, SEEDS, out_of_fold
from src.classify.data import Dataset, load
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURE = ROOT / "reports" / "figures" / "p5_confusion.png"


def matrices(dataset: Dataset, model: str, strategy: str = "stratified", seeds=SEEDS) -> dict:
    """Raw counts summed over the repeats, and the row-normalised version."""
    classes = sorted(set(dataset.y.tolist()))
    total = np.zeros((len(classes), len(classes)), dtype=int)
    for seed in seeds:
        predictions = out_of_fold(model, dataset, strategy=strategy, seed=seed)
        total += confusion_matrix(predictions.y_true, predictions.y_pred, labels=classes)

    row_sums = total.sum(axis=1, keepdims=True)
    normalised = np.divide(total, row_sums, out=np.zeros(total.shape), where=row_sums > 0)

    confusions = [
        {
            "true": classes[i],
            "predicted": classes[j],
            "count": int(total[i, j]),
            "share_of_true_class": round(float(normalised[i, j]), 4),
        }
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and total[i, j] > 0
    ]
    return {
        "model": model,
        "classes": classes,
        "raw": total.tolist(),
        "normalised": normalised.round(4).tolist(),
        "worst_by_count": sorted(confusions, key=lambda row: -row["count"])[:5],
        "worst_by_share": sorted(confusions, key=lambda row: -row["share_of_true_class"])[:5],
        "decisions": int(total.sum()),
    }


def figure(results: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(results), figsize=(4.3 * len(results), 8.4), squeeze=False)
    for column, result in enumerate(results):
        classes = result["classes"]
        for row, (key, title, fmt) in enumerate(
            [("raw", "counts", "{:.0f}"), ("normalised", "row-normalised", "{:.2f}")]
        ):
            ax = axes[row][column]
            data = np.array(result[key], dtype=float)
            ax.imshow(data / data.max() if key == "raw" else data, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            ax.set_title(f"{result['model']} — {title}", fontsize=10)
            if row == 1:
                ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            limit = (data / data.max() if key == "raw" else data).max() * 0.6
            for i in range(len(classes)):
                for j in range(len(classes)):
                    shade = (data[i, j] / data.max()) if key == "raw" else data[i, j]
                    ax.text(
                        j,
                        i,
                        fmt.format(data[i, j]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if shade > limit else "#222222",
                    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def run(
    corpus: str = "real", models=REAL_MODELS, strategy: str = "stratified", write: bool = True
) -> dict:
    dataset = load(corpus)
    results = [matrices(dataset, model, strategy) for model in models]
    summary = {
        "corpus": corpus,
        "strategy": strategy,
        "models": {
            row["model"]: {
                "decisions": row["decisions"],
                "worst_by_count": row["worst_by_count"],
                "worst_by_share": row["worst_by_share"],
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

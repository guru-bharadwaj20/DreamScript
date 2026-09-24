"""Phase 5.3.1 - what the kNN decision boundary actually looks like, and what the picture costs.

    python -m src.classify.boundaries      # writes reports/figures/p5_knn_boundaries.png

The plan asks for 2-D projections showing the diagram types clustering. 4.2.7 already made the
scatter plots and reported that the types are **not** linearly separable in two dimensions; this
task adds the thing a scatter plot cannot show - where the classifier puts its boundaries, and
how the choice of `k` changes their shape.

Two projections, because they fail differently:

    PCA(2)        the two directions of greatest variance in the scaled 33-column table
    top-2 by MI   the two single most informative features, from 4.2.6's ranking

PCA keeps a little of every feature and 4.2.7 measured it holding 45% of the variance; the
top-2 pair keeps two features completely and throws away the other 31. The first is the better
summary and the second is the more readable axis, so both are drawn.

Three values of `k` per projection, because the boundary *is* the lesson of this model. At
k = 1 - the value 5.1.2 selected - every training point owns a cell and the boundary is a
Voronoi tessellation with islands in it; at k = 15 the islands are gone and the regions are
convex-ish blobs. The minority classes live in those islands, which is exactly why 5.1.2 found
k = 1 winning and every larger k erasing the 40-row circuit class.

## The honest number: what two dimensions cost

A decision-boundary plot is a picture of a **different model** from the one Phase 5 selected -
the real kNN sees 33 features. Drawing it without saying so would be the most misleading figure
in this repository, so the projection is scored: the same repeated CV, on the 2-D projection,
refitting the projection **inside each training fold** so the picture's honesty and the score's
honesty are the same question. The panel titles carry the number.

## What it measured

    projection   axes                              k=1     k=5     k=15
    PCA(2)       PC1 (35%), PC2 (11%)             0.384   0.405   0.352
    top-2 by MI  global_aspect, dir_flow_axis     0.576   0.583   0.531

    the model Phase 5 actually selected, 33 features, k=1:  0.764

**Two dimensions cost half the model.** The best projected score is 0.583 against 0.764 for the
same estimator on the full table - the picture is a classifier roughly as good as guessing the
two large classes. That is the number this task exists to publish: the figure is a useful
picture of *where the classes sit* and a bad picture of *what the model does*, and 4.2.7 already
said the same thing without a classifier attached to it. Anyone quoting a decision-boundary plot
as evidence of separability is quoting 0.58.

**PCA is the worse projection and the more honest one.** It scores 0.38 while keeping a little
of every feature; the top-2 pair scores 0.58 while keeping two features and discarding 31. The
gap says the signal in this table is spread thin - no two orthogonal directions of variance
carry it - which is the same finding as 4.2.7's 45% of variance in two components, priced.

**The most informative pair of axes is led by the camera leak.** Mutual information picks
`global_aspect` first, which 4.1.5 established is a property of the photograph and 5.1.1
measured as worth 0.0052 macro F1 to the real model. The most readable 2-D picture of this
corpus is therefore partly a picture of which dataset each page came from, and the panel is
labelled rather than quietly used.

**The circuit region disappears at k = 15, in both projections, to exactly 0.0 of the plane.**
This is 5.1.2's finding made visible: it selected k = 1 because any larger k erases the 40-row
circuit class, and here the class's decision region shrinks from 3.3% of the PCA plane at k = 1
to 0.04% at k = 5 to nothing at all at k = 15. A minority class in a kNN model survives as
islands, and smoothing is what removes islands.

One inversion worth recording: **k = 5 edges out k = 1 in both projections** (0.405 vs 0.384,
0.583 vs 0.576) while k = 1 wins on the full table. The optimal neighbourhood size is a property
of the dimension, not of the corpus - in 2-D the points are dense enough that averaging five
neighbours helps, and in 33-D they are not. The panels show a model selected for a space it is
not being drawn in, which is one more reason not to read the figure as the classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.classify.cv import N_SPLITS, SEEDS, out_of_fold, splitter
from src.classify.data import Dataset, load
from src.classify.knn import BEST_PARAMS
from src.features.scaling import feature_scaler
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURE = ROOT / "reports" / "figures" / "p5_knn_boundaries.png"

#: k = 1 is 5.1.2's selection; the other two are there to show the boundary smoothing.
K_VALUES = (1, 5, 15)

GRID_STEPS = 320

#: Colour per diagram type, shared between the filled regions and the scattered points so a
#: misplaced point is visible as a colour sitting in the wrong field.
COLOURS = {
    "circuit": "#a63d54",
    "er_diagram": "#c98a2b",
    "flowchart": "#3b6ea5",
    "state_machine": "#4b8b3b",
    "wireframe": "#6b5b95",
}


def project(dataset: Dataset, kind: str, *, fit_rows: np.ndarray | None = None):
    """A fitted 33-to-2 projection and the names of its axes.

    Returns `(transform, axis_names)`, where `transform` maps a raw feature matrix to two
    columns. `fit_rows` restricts what the projection learns from, which is how the scored path
    keeps the projection inside the training fold.
    """
    rows = np.arange(len(dataset.y)) if fit_rows is None else fit_rows
    scaler = feature_scaler().fit(dataset.X[rows])

    if kind == "pca":
        pca = PCA(n_components=2, random_state=SEEDS[0]).fit(scaler.transform(dataset.X[rows]))
        variance = pca.explained_variance_ratio_
        names = [f"PC1 ({variance[0]:.0%})", f"PC2 ({variance[1]:.0%})"]
        return (lambda X: pca.transform(scaler.transform(X))), names

    if kind == "top2":
        from src.features.importance import rank

        # The scaler emits 4.2.3's `*_missing` indicators alongside the features, so the names
        # have to come from the fitted imputer rather than from the dataset's own list.
        names = list(scaler.named_steps["impute"].get_feature_names_out(dataset.feature_names))
        ranked = rank(scaler.transform(dataset.X[rows]), dataset.y[rows], names)
        chosen = [row["feature"] for row in ranked[:2]]
        columns = [names.index(name) for name in chosen]
        return (lambda X: scaler.transform(X)[:, columns]), chosen

    raise ValueError(f"projection must be pca or top2; got {kind!r}")


def projected_score(
    dataset: Dataset, kind: str, k: int, *, seed: int = SEEDS[0], n_splits: int = N_SPLITS
) -> float:
    """Macro F1 of a kNN fitted on the 2-D projection, projection refit inside each fold.

    This is the number that makes the figure honest. Fitting the projection on the whole corpus
    and then cross-validating on top of it would leak the test rows into the axes themselves,
    and the picture would come with a score it has not earned.
    """
    predictions = np.empty(len(dataset.y), dtype=object)
    for train_index, test_index in splitter(dataset, "stratified", seed, n_splits):
        transform, _ = project(dataset, kind, fit_rows=train_index)
        model = KNeighborsClassifier(
            n_neighbors=k, metric=BEST_PARAMS["metric"], weights=BEST_PARAMS["weights"]
        )
        model.fit(transform(dataset.X[train_index]), dataset.y[train_index])
        predictions[test_index] = model.predict(transform(dataset.X[test_index]))
    return float(f1_score(dataset.y, predictions, average="macro", zero_division=0))


def boundary(embedded: np.ndarray, y: np.ndarray, k: int, steps: int = GRID_STEPS) -> dict:
    """A kNN fitted on two columns, evaluated over a mesh - the raw material of one panel."""
    classes = sorted(set(y.tolist()))
    model = KNeighborsClassifier(
        n_neighbors=min(k, len(y)), metric=BEST_PARAMS["metric"], weights=BEST_PARAMS["weights"]
    ).fit(embedded, y)

    pad = 0.05 * (embedded.max(axis=0) - embedded.min(axis=0) + 1e-9)
    low, high = embedded.min(axis=0) - pad, embedded.max(axis=0) + pad
    xs = np.linspace(low[0], high[0], steps)
    ys = np.linspace(low[1], high[1], steps)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    flat = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    labelled = model.predict(flat)
    regions = np.array([classes.index(name) for name in labelled]).reshape(mesh_x.shape)

    area = {name: round(float((regions == index).mean()), 4) for index, name in enumerate(classes)}
    return {
        "k": k,
        "classes": classes,
        "extent": [float(low[0]), float(high[0]), float(low[1]), float(high[1])],
        "regions": regions,
        "area_share": area,
        "resubstitution_accuracy": round(float((model.predict(embedded) == y).mean()), 4),
    }


def panel(dataset: Dataset, kind: str, k_values=K_VALUES) -> dict:
    """One projection: the embedding for the picture, and a cross-validated score per k."""
    transform, axis_names = project(dataset, kind)
    embedded = transform(dataset.X)
    return {
        "projection": kind,
        "axes": axis_names,
        "embedded": embedded,
        "boundaries": [
            {
                **boundary(embedded, dataset.y, k),
                "macro_f1": round(projected_score(dataset, kind, k), 4),
            }
            for k in k_values
        ],
    }


def figure(panels: list[dict], y: np.ndarray, full_scores: dict, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    classes = sorted(set(y.tolist()))
    cmap = ListedColormap([COLOURS.get(name, "#888888") for name in classes])
    rows, columns = len(panels), len(panels[0]["boundaries"])
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.1 * rows), squeeze=False)

    for row, sheet in enumerate(panels):
        embedded = sheet["embedded"]
        for column, result in enumerate(sheet["boundaries"]):
            ax = axes[row][column]
            ax.imshow(
                result["regions"],
                extent=result["extent"],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                alpha=0.28,
                interpolation="nearest",
                vmin=0,
                vmax=len(classes) - 1,
            )
            for index, name in enumerate(classes):
                mask = y == name
                ax.scatter(
                    embedded[mask, 0],
                    embedded[mask, 1],
                    s=7,
                    lw=0.2,
                    edgecolor="#ffffff",
                    color=cmap(index),
                    label=name if row == 0 and column == 0 else None,
                )
            reference = full_scores.get(result["k"])
            cost = f" (33-D {reference:.3f})" if reference is not None else ""
            ax.set_title(
                f"{sheet['projection']}, k={result['k']} — 2-D macro F1 "
                f"{result['macro_f1']:.3f}{cost}",
                fontsize=9,
            )
            ax.set_xlabel(sheet["axes"][0], fontsize=8)
            ax.set_ylabel(sheet["axes"][1], fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0 and column == 0:
                ax.legend(fontsize=7, loc="best", frameon=False, markerscale=1.6)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def run(corpus: str = "real", projections=("pca", "top2"), write: bool = True) -> dict:
    dataset = load(corpus)
    panels = [panel(dataset, kind) for kind in projections]

    # The model Phase 5 actually selected, on all 33 features, so the panels have something to
    # be compared against rather than being read as the classifier itself.
    full = out_of_fold("knn", dataset)
    full_scores = {BEST_PARAMS["n_neighbors"]: round(full.macro_f1(), 4)}

    summary = {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "full_dimension_macro_f1": full_scores[BEST_PARAMS["n_neighbors"]],
        "projections": {
            sheet["projection"]: {
                "axes": sheet["axes"],
                "by_k": {
                    str(result["k"]): {
                        "macro_f1": result["macro_f1"],
                        "resubstitution_accuracy": result["resubstitution_accuracy"],
                        "area_share": result["area_share"],
                    }
                    for result in sheet["boundaries"]
                },
            }
            for sheet in panels
        },
    }
    if write:
        summary["figure"] = str(figure(panels, dataset.y, full_scores).relative_to(ROOT))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--projections", nargs="*", default=["pca", "top2"])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, tuple(args.projections), not args.no_figure), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

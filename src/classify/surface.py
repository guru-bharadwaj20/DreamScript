"""Phase 6.3.6 - what each kernel's decision surface looks like, and what the picture costs.

    python -m src.classify.surface        # writes reports/figures/p6_kernel_surfaces.png

The plan asks for 2-D projected decision surfaces per kernel. 5.3.1 did the same exercise for
kNN and established the rule this task inherits:

> A decision-boundary plot is a picture of a **different model** from the one Phase 5 selected -
> the real kNN sees 33 features. Drawing it without saying so would be the most misleading figure
> in this repository.

It also measured the cost: the best 2-D projection scored 0.583 against 0.764 for the same
estimator on the full table. **Two dimensions cost half the model.** So the projection here is
scored the same way - the same repeated CV, on the projection, refitting the projection inside
each training fold - and every panel carries its own number.

The 161-column hybrid table makes this worse, not better, than it was for 5.3.1's 33 columns.
6.1.3 measured the embedding needing 128 PCA components to hold 91.5% of its variance; two
components hold a small fraction of that, so the surfaces below are drawn in a space that has
thrown away most of what the real models use.

## What the figure is for, given that

Not for judging which kernel is better - the scores in 6.3.1 through 6.3.3 do that on the full
table. It is for showing **the shape each kernel's hypothesis class can produce**: a linear SVM
draws straight lines whatever the data does, a degree-2 polynomial draws conics, an RBF draws
closed contours around clusters. That property is genuine at any dimension and it is the thing a
viva asks to see, so the figure shows it and the caption carries the score that says what else it
is not showing.

## What it measured

Hybrid table, PCA to 2 components refitted inside every fold, each kernel at 6.3.1-6.3.3's
settings:

    kernel     2-D macro F1     161-D macro F1     cost of the picture
    linear        0.4978           0.9628               -0.4650
    poly          0.5078           0.9690               -0.4612
    rbf           0.5075           0.9690               -0.4615

    variance retained by two components:  16.4%

**Two dimensions cost 0.46 macro F1 - a little under half the model, on all three kernels.**
5.3.1 measured the same thing for kNN on the 33-column table and found the projection costing
about half; on 161 columns the fraction is the same and the absolute loss is larger, because the
model being given up is much better. Two components hold **16.4%** of this table's variance, so
the surfaces below are drawn in a space missing five sixths of what the real models use.

Anyone quoting one of these panels as evidence that the classes are separable is quoting 0.50.

## The three kernels partition the plane almost identically

    kernel     circuit   er_diagram   flowchart   state_machine   wireframe
    linear      0.0055     0.0000      0.6456        0.0976        0.2513
    poly        0.0045     0.0000      0.6403        0.1026        0.2526
    rbf         0.0044     0.0000      0.6365        0.1026        0.2565

**The area each class occupies is the same to within half a percentage point across all three
kernels**, and their projected scores agree to within 0.010. The panels differ in the *shape* of
their boundaries - straight lines, conics, closed contours - which is what the figure exists to
show, and they agree almost exactly on which regions belong to whom. At two dimensions the kernel
is a stylistic choice; the 0.006 that separated them in 6.3.2 and 6.3.3 lives in the 159
dimensions this picture discards.

## The class that has no region at all

**`er_diagram` receives exactly 0.0% of the plane under every kernel.** All 50 of its pages are
plotted, and there is nowhere in the projection where the classifier would predict it - the
region is squeezed out entirely by flowchart, which takes 64%. `circuit` survives on 0.44-0.55%,
which at this resolution is a sliver.

This is 5.3.1's finding recurring on a different model: it reported the circuit region vanishing
to 0.0 of the plane at k = 15. Two of the three minority classes are geometrically invisible in
two dimensions, and they are the classes 5.3.4, 6.3.4 and 6.3.5 all identified as the hard ones.
The figure cannot show the part of the problem that is actually difficult, which is the most
useful thing it has to say about itself.

## What the figure is therefore good for

The shape of each hypothesis class, and nothing quantitative. A linear SVM produces straight
boundaries, the degree-2 polynomial produces curved ones, and the RBF produces closed contours
around dense regions - all three visible in the panels, all three true at 161 dimensions as well,
and none of them a claim about accuracy. Every panel title carries both numbers so the figure
cannot be quoted without its own refutation attached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classify.activations import TABLES

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p6_kernel_surfaces.png"

SEED = 42

KERNELS = ("linear", "poly", "rbf")

#: Full-table macro F1 for each kernel on the hybrid table, from 6.3.1-6.3.3, so each panel can
#: print what its own projection cost rather than only what the projection scored.
FULL_DIMENSION = {"linear": 0.9628, "poly": 0.9690, "rbf": 0.9690}


def projector(n_components: int = 2):
    """PCA to two dimensions, fitted inside whatever fold it is given.

    A projection fitted on all rows and then cross-validated is the same leak 4.2.4's scaler was
    built to avoid: the two components would have been chosen using the test fold's variance.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    return Pipeline([("prepare", feature_scaler()), ("pca", PCA(n_components, random_state=SEED))])


def projected_estimator(kernel: str, table: str = "hybrid"):
    """The scaler, PCA to 2-D, then the kernel - one estimator so CV refits all three per fold."""
    from sklearn.pipeline import Pipeline

    from src.classify.multiclass import base_estimator

    return Pipeline([("project", projector()), ("model", base_estimator(kernel, table))])


def score_projection(
    data, kernel: str, table: str = "hybrid", folds: int = 5, n_jobs: int | None = None
) -> float:
    """Macro F1 of the model the picture actually shows - the honesty number 5.3.1 demanded."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    predicted = cross_val_predict(
        projected_estimator(kernel, table),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    return round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4)


def surface(data, kernel: str, table: str = "hybrid", resolution: int = 300) -> dict:
    """The fitted 2-D model, its grid of predictions, and the projected points to draw over it."""
    fitted = projected_estimator(kernel, table).fit(data.X, data.y)
    points = fitted.named_steps["project"].transform(data.X)

    pad = 0.06 * (points.max(axis=0) - points.min(axis=0))
    x_min, y_min = points.min(axis=0) - pad
    x_max, y_max = points.max(axis=0) + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # The grid is already in projected space, so it goes straight to the classifier - pushing it
    # back through the projection would be a different transform and a different picture.
    model = fitted.named_steps["model"]
    grid = np.c_[xx.ravel(), yy.ravel()]
    classes = list(model.classes_)
    zz = np.array([classes.index(label) for label in model.predict(grid)]).reshape(xx.shape)

    areas = {name: round(float((zz == index).mean()), 4) for index, name in enumerate(classes)}
    return {
        "kernel": kernel,
        "points": points,
        "xx": xx,
        "yy": yy,
        "zz": zz,
        "classes": classes,
        "area_share": areas,
        # A class with no region in the plane is invisible to the classifier the picture shows,
        # which is 5.3.1's finding about circuits at k = 15 and worth checking per kernel.
        "classes_with_no_region": [name for name, share in areas.items() if share == 0.0],
    }


def figure(
    panels: list[dict],
    y: np.ndarray,
    scores: dict,
    full: dict = FULL_DIMENSION,
    path: Path = FIGURE,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    colours = plt.get_cmap("tab10")

    for ax, panel in zip(axes, panels, strict=True):
        classes = panel["classes"]
        ax.contourf(
            panel["xx"],
            panel["yy"],
            panel["zz"],
            levels=np.arange(-0.5, len(classes)),
            colors=[colours(i) for i in range(len(classes))],
            alpha=0.22,
        )
        for index, name in enumerate(classes):
            mask = y == name
            ax.scatter(
                panel["points"][mask, 0],
                panel["points"][mask, 1],
                s=6,
                color=colours(index),
                label=name,
                edgecolors="none",
                alpha=0.75,
            )
        projected = scores[panel["kernel"]]
        reference = full.get(panel["kernel"])
        ax.set_title(
            f"{panel['kernel']}  -  2-D {projected:.3f}"
            + (f"  |  161-D {reference:.3f}" if reference else ""),
            fontsize=9,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(fontsize=6.5, loc="best", frameon=False, markerscale=1.8)
    fig.suptitle(
        "Phase 6.3.6 - kernel decision surfaces on PCA(2). The scores say what the picture "
        "leaves out.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(
    table: str = "hybrid",
    corpus: str = "real",
    kernels=KERNELS,
    n_jobs: int | None = None,
    write: bool = True,
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    scores = {kernel: score_projection(data, kernel, table, n_jobs=n_jobs) for kernel in kernels}
    panels = [surface(data, kernel, table) for kernel in kernels]

    summary = {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "projected_macro_f1": scores,
        "full_dimension_macro_f1": {k: FULL_DIMENSION[k] for k in kernels if k in FULL_DIMENSION},
        "cost_of_two_dimensions": {
            kernel: round(FULL_DIMENSION[kernel] - scores[kernel], 4)
            for kernel in kernels
            if kernel in FULL_DIMENSION
        },
        "area_share": {panel["kernel"]: panel["area_share"] for panel in panels},
        "classes_with_no_region": {
            panel["kernel"]: panel["classes_with_no_region"] for panel in panels
        },
        "retained_variance": retained_variance(data),
    }
    if write:
        summary["figure"] = str(figure(panels, data.y, scores).relative_to(ROOT))
    return summary


def retained_variance(data, n_components: int = 2) -> float:
    """How much of the table two components actually keep - the size of what the figure discards."""
    fitted = projector(n_components).fit(data.X)
    return round(float(fitted.named_steps["pca"].explained_variance_ratio_.sum()), 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=list(TABLES))
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--kernels", nargs="*", default=list(KERNELS))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.table, args.corpus, tuple(args.kernels), args.jobs, not args.no_figure)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

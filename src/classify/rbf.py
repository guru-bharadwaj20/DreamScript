"""Phase 6.3.3 - the RBF kernel, a gamma x C grid, and the one hyperparameter Phase 6 has found.

    python -m src.classify.rbf --table all

The plan asks for a gamma and C grid and a comparison. Two tasks of context make that more
pointed than it sounds.

6.3.1 measured the linear SVM at 0.9628 on the hybrid table - the best number in Phase 6, ahead
of every network 6.2 produced. So the question here is not "is a kernel better than a linear
model", it is **whether an infinite-dimensional feature map buys anything over the one that
already won.**

6.2.3 measured weight decay ranging 0.0008 across four orders of magnitude and 6.3.1 found C
sitting on a one-decade plateau. Almost nothing in this phase has been sensitive to its
hyperparameters. `gamma` is the exception a priori: it is the inverse width of the kernel, so it
sets *how far one training point's influence reaches*, and at the extremes the model degenerates
in two opposite and diagnosable ways.

## The two degeneracies, which is why the grid is wide

**gamma too large.** Each point's Gaussian bump is narrower than the distance to its neighbours,
every training row becomes its own support vector, and the model memorises the training set and
predicts the majority class everywhere else. The signature is a training score near 1.0 with a
held-out macro F1 near the majority baseline, and a support-vector count approaching 100%.

**gamma too small.** Every kernel value approaches 1, the Gram matrix approaches a constant, and
the decision function approaches a single hyperplane - the RBF degenerates toward a linear model.

Both are measurable rather than quotable, so this task reports the **support-vector share and the
training score beside every cell**, and the diagnosis is read off them rather than asserted.

## What it measured

35 cells (7 gammas x 5 C) per table, stratified 5-fold:

    table          best cell                 macro F1   linear (6.3.1)   diff     collapsed cells
    hybrid         gamma 1e-4, C 1000         0.9690       0.9628       +0.0062      22 of 35
    embedding      gamma 1e-5, C 1000         0.9742       0.9691       +0.0051      18 of 35
    handcrafted    gamma scale, C 10          0.8310       0.7656       +0.0654      20 of 35

**gamma is the most consequential hyperparameter in Phase 6, by a wide margin.** Its range on the
hybrid table is **0.7866** - from 0.9690 down to 0.1824 - against 0.5812 for C on the same table,
0.186 for 6.3.2's coef0, and under 0.03 for everything in 6.2. After five tasks reporting that
nothing matters, two kernel parameters in a row matter enormously.

    hybrid, best score at each gamma:
    1e-5 -> 0.9658    1e-4 -> 0.9690    1e-3 -> 0.9632    scale -> 0.9205
    1e-2 -> 0.6692    1e-1 -> 0.1824     1.0 -> 0.1977

## The predicted degeneracy, with the signature it was predicted to have

The docstring above named the large-gamma failure before the grid was run: every training row
becomes its own support vector, the training score goes to 1.0, and the held-out score falls to
the majority baseline. The worst cell on the hybrid table is gamma 1.0, C 1.0:

    macro F1 0.1704    training macro F1 1.0000    support-vector share 1.000

**All three numbers are exactly the predicted signature.** 100% of rows are support vectors, the
model reproduces its training set perfectly, and it generalizes at 0.17 - which for five classes
with two holding 90% of the rows is what predicting the majority looks like under macro F1. This
is why the support-vector share and training score are reported beside every cell and why
`diagnose` names the mode: **22 of the 35 hybrid cells are degenerate**, and a grid that printed
only macro F1 would show two thirds of itself as unexplained bad numbers.

The small-gamma end behaves as predicted too, and less dramatically: at 1e-5 the kernel is nearly
constant and the model scores 0.9658 - within 0.003 of the linear SVM's 0.9628, which is exactly
what "the RBF degenerates toward a linear model" means when it is measured rather than said.

**sklearn's `gamma="scale"` is not the best value on either learned table** - 0.9205 against
0.9690 on the hybrid, a gap of 0.049 - though it is the winner on the handcrafted one. That is
the same pattern as 6.3.1's C and 6.3.2's coef0: the third default in a row that this corpus does
not agree with, and the third reason the grids are wide.

## What the kernel is worth

**The gains over the linear SVM are real but small on the tables that were already good, and
large on the one that was not.** +0.0062 on the hybrid and +0.0051 on the embedding, against
**+0.0654 on the handcrafted table** - where 0.8310 also beats every network 6.2 produced on that
table (best 0.8242) and 6.1.4's logistic regression (0.7981) by 0.033.

That split is the same one 6.3.1 found from the other side. The learned tables are close to
linearly separable, so an infinite-dimensional feature map has almost nothing left to add; the
33-column geometric table is not, and it is the one place in Phase 6 where model capacity - of any
kind, network or kernel - has bought a substantial amount. **0.9742 on the embedding table is the
best number in Phase 6.**
"""

from __future__ import annotations

import argparse
import json
import sys

from src.classify.activations import TABLES

SEED = 42

#: Six orders of magnitude, plus sklearn's adaptive default. The extremes are in the grid to be
#: diagnosed, not because they are candidates.
GAMMAS = ("scale", 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

CS = (0.1, 1.0, 10.0, 100.0, 1000.0)

#: 6.3.1's linear SVM, per table - the number an infinite-dimensional kernel has to beat.
LINEAR = {"hybrid": 0.9628, "embedding": 0.9691, "handcrafted": 0.7656}


def pipeline(gamma="scale", C: float = 1.0, **kwargs):  # noqa: N803
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from src.features.scaling import feature_scaler

    settings = {
        "kernel": "rbf",
        "gamma": gamma,
        "C": C,
        "random_state": SEED,
        **kwargs,
    }
    return Pipeline([("prepare", feature_scaler()), ("model", SVC(**settings))])


def grid(gammas=GAMMAS, cs=CS) -> list[dict]:
    return [{"gamma": gamma, "C": C} for gamma in gammas for C in cs]


def score_cell(data, config: dict, folds: int = 5, n_jobs: int | None = None) -> dict:
    """One cell, with the two diagnostics that let a bad score be explained rather than reported.

    The support-vector share and the training score come from a single fit on all rows, not from
    the cross-validation - they are descriptions of what the kernel does at this gamma, and one
    fit is enough to see a model that has made every row a support vector.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    estimator = pipeline(**config)
    predicted = cross_val_predict(
        estimator,
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    fitted = pipeline(**config).fit(data.X, data.y)
    model = fitted.named_steps["model"]
    support = int(model.n_support_.sum())

    return {
        **config,
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == data.y).mean()), 4),
        "train_macro_f1": round(
            float(f1_score(data.y, fitted.predict(data.X), average="macro", zero_division=0)), 4
        ),
        "support_vector_share": round(support / len(data.y), 4),
        "distinct_predictions": int(len(set(predicted.tolist()))),
    }


def diagnose(row: dict, n_classes: int) -> str:
    """Name the failure mode from the diagnostics, rather than leaving a low score unexplained."""
    if row["distinct_predictions"] == 1:
        return "collapsed to one class"
    if row["support_vector_share"] > 0.95 and row["train_macro_f1"] > 0.99:
        return "memorised: every row a support vector"
    if row["support_vector_share"] > 0.95:
        return "every row a support vector"
    if row["distinct_predictions"] < n_classes:
        return f"predicts only {row['distinct_predictions']} of {n_classes} classes"
    return "ok"


def run(table: str = "hybrid", corpus: str = "real", n_jobs: int | None = None) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    n_classes = len(data.classes)
    rows = [score_cell(data, config, n_jobs=n_jobs) for config in grid()]
    for row in rows:
        row["diagnosis"] = diagnose(row, n_classes)
    rows.sort(key=lambda row: -row["macro_f1"])

    best_by_gamma: dict = {}
    for row in rows:
        name = str(row["gamma"])
        if name not in best_by_gamma:
            best_by_gamma[name] = row["macro_f1"]
    best_by_c: dict = {}
    for row in rows:
        name = str(row["C"])
        if name not in best_by_c:
            best_by_c[name] = row["macro_f1"]

    linear = LINEAR.get(table)
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "cells": len(rows),
        "best": rows[0],
        "top5": rows[:5],
        "worst": rows[-1],
        "best_by_gamma": {k: best_by_gamma[k] for k in sorted(best_by_gamma, key=str)},
        "best_by_C": best_by_c,
        "gamma_range": round(max(best_by_gamma.values()) - min(best_by_gamma.values()), 4),
        "c_range": round(max(best_by_c.values()) - min(best_by_c.values()), 4),
        "collapsed_cells": sum(1 for row in rows if row["diagnosis"] != "ok"),
        "linear_svm_6_3_1": linear,
        "rbf_minus_linear": round(rows[0]["macro_f1"] - linear, 4) if linear else None,
        "all": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {name: run(name, args.corpus, args.jobs) for name in tables}
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if not args.full:
        for value in result.values():
            value.pop("all", None)
    print(json.dumps(result if len(result) > 1 else next(iter(result.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

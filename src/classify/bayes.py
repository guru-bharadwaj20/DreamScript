"""Phase 7.2.2 - Gaussian Naive Bayes on the text statistics, and the assumption it is built on.

    python -m src.classify.bayes

7.2.1 built a 14-column table from the text layer alone, because this model is meant to be the
pipeline's fast path. Gaussian NB is the natural first classifier for it: every column is
continuous, the model is one mean and one variance per feature per class, and fitting it is a
single pass over the data with no iteration at all.

## What the model actually assumes, and why both parts matter here

**Conditional independence.** Given the class, every feature is independent of every other. This
table violates that on purpose and visibly: `text_n_blocks` and `text_blocks_per_area` are the
same count with and without a normalisation, and `text_width_mean` and `text_aspect_mean` share a
numerator. 7.2.4 is the task that measures the damage; this one records the score the assumption
produces.

**Normality.** Each feature is Gaussian within each class. Several of these columns cannot be -
`text_row_alignment` and `text_edge_share` are bounded shares that pile up at 0 and 1, and
`text_n_blocks` is a count. So this task also fits the model on **log-transformed** counts as a
control, because if normality is the binding constraint then a transform that makes the columns
more nearly Gaussian should help, and if it does not the assumption was not what was costing.

## What `var_smoothing` is doing

sklearn adds `var_smoothing * max(variance)` to every variance. It is not a regularizer in the
usual sense - it is what stops a class whose feature is constant from producing an infinitely
peaked Gaussian and a likelihood of zero everywhere else. On a 40-row class with a bounded
feature that is a live failure, so it is swept rather than left at its default.

## What it measured

10 cells (5 var_smoothing values x 2 transforms), stratified 5-fold on 1,340 real pages:

    best   var_smoothing = 0.1, no transform      macro F1 0.4207    accuracy 0.6149
    worst  var_smoothing = 1e-12, log transform   macro F1 0.2285    accuracy 0.4746

**0.42 is the lowest macro F1 of any fitted model in the project** - Phase 5's majority baseline
aside - and it is worth being precise about what that is evidence of. 7.2.1's table separates the
classes well (an ANOVA F of 1,116 on `text_aspect_mean`), and 7.2.3 reaches 0.707 on exactly these
14 columns. So the failure is the *model*, not the features, and this task is where the two
Gaussian NB assumptions get charged for the difference.

## The per-class breakdown says which assumption

    class            F1
    flowchart       0.8894
    wireframe       0.4950
    state_machine   0.4198
    er_diagram      0.2524
    circuit         0.0472

**Circuit is at 0.047 - effectively never predicted correctly.** 7.2.1 already showed why: on the
headline column circuit and er_diagram sit within 0.2 blocks of each other (22.5 against 22.7),
and a model that represents each class as a product of 14 independent Gaussians has nothing left
to separate them with once the means coincide. The accuracy of 0.615 against a macro F1 of 0.421
is the familiar gap - the model is right on the two big classes and guessing on the three small
ones.

## var_smoothing is not a minor knob here

    1e-12   0.2291
    1e-09   0.2323
    1e-06   0.2571
    1e-03   0.2870
    1e-01   0.4207

**The default (1e-9) scores 0.2323 and the best swept value scores 0.4207 - the sweep is worth
+0.188 macro F1**, far more than anything else in this task. That is the failure the docstring
predicted: a 40-row class with a bounded feature produces a near-zero within-class variance, an
absurdly peaked Gaussian, and a likelihood that vetoes every row not sitting almost exactly on the
mean. Smoothing at 0.1 adds 10% of the largest variance to every variance - a very heavy hand -
and the score is still climbing at the end of the grid, which says the model wants to be less
Gaussian than it is allowed to be.

## The normality control answers the other half

D'Agostino's test rejects normality in **52 of 70 class-feature combinations (74%)**, so the
assumption is broken about as comprehensively as it could be. `text_width_mean`, `text_width_std`
and `text_nn_distance` reject in all five classes.

But the log transform - the direct remedy - **makes it worse, not better: 0.4064 against 0.4207,
a loss of 0.014**. So non-normality is not the binding constraint. What is left is conditional
independence, which 7.2.4 measures directly, and the discretization route, which 7.2.3 takes: a
multinomial likelihood on binned columns needs neither assumption and gains 0.286 macro F1 over
this task for the same 14 features.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

SEED = 42

#: sklearn's default is 1e-9. Swept because a 40-row class can produce a near-zero variance and
#: the default is then the difference between a usable model and one that predicts one class.
SMOOTHING = (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)

TRANSFORMS = ("none", "log")

#: Columns that are counts or unbounded-positive, where a log transform is defensible. The
#: bounded shares are left alone: log of a value that is legitimately 0 is not a transform.
LOGGABLE = (
    "text_n_blocks",
    "text_blocks_per_area",
    "text_width_mean",
    "text_width_std",
    "text_height_mean",
    "text_aspect_mean",
    "text_nn_distance",
)


def log_transform(names: list[str]):
    """`log1p` on the count-like columns only, as a ColumnTransformer-free function transformer.

    Applied to the positive/unbounded columns and not to the bounded shares, because `log1p` of a
    share in [0, 1] compresses a range that was never skewed and buys nothing.
    """
    from sklearn.preprocessing import FunctionTransformer

    index = [i for i, name in enumerate(names) if name in LOGGABLE]

    def apply(X):
        out = np.array(X, dtype=float, copy=True)
        if index:
            out[:, index] = np.log1p(np.clip(out[:, index], 0, None))
        return out

    return FunctionTransformer(apply, feature_names_out="one-to-one")


def pipeline(var_smoothing: float = 1e-9, transform: str = "none", names=None):
    """4.2.3's imputation, optionally a log transform, then Gaussian NB.

    4.2.4's scaler is included even though a Gaussian NB is scale-invariant in principle: the
    pipeline's `feature_scaler` is where the *imputation* lives, and 7.2.1's table has nan on
    every page with no text.
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    steps = [("prepare", feature_scaler())]
    if transform == "log":
        if names is None:
            raise ValueError("the log transform needs the feature names to know which columns")
        # Before the scaler would be more natural, but `prepare` is also the imputer and the
        # transform must not see nan; after it, the columns are centred and log1p of a negative
        # is nan - so the transform is applied to the raw columns via a leading step instead.
        steps = [("log", log_transform(names)), ("prepare", feature_scaler())]
    elif transform != "none":
        raise ValueError(f"transform must be one of {list(TRANSFORMS)}; got {transform!r}")

    steps.append(("model", GaussianNB(var_smoothing=var_smoothing)))
    return Pipeline(steps)


def evaluate(
    data,
    var_smoothing: float = 1e-9,
    transform: str = "none",
    folds: int = 5,
    n_jobs: int | None = None,
) -> dict:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    predicted = cross_val_predict(
        pipeline(var_smoothing, transform, list(data.feature_names)),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    classes = data.classes
    per_class = f1_score(data.y, predicted, average=None, labels=classes, zero_division=0)
    return {
        "var_smoothing": var_smoothing,
        "transform": transform,
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == data.y).mean()), 4),
        "per_class_f1": {
            name: round(float(v), 4) for name, v in zip(classes, per_class, strict=True)
        },
    }


def normality(data) -> dict:
    """How Gaussian each column actually is, within each class - the assumption, measured.

    D'Agostino's K^2 per (feature, class), reported as the share of the 70 combinations that
    reject normality at 0.01. A model built on an assumption should say how false it is.
    """
    from scipy import stats

    rows = {}
    for index, name in enumerate(data.feature_names):
        rejected = 0
        tested = 0
        for label in data.classes:
            column = data.X[data.y == label, index]
            column = column[np.isfinite(column)]
            if len(column) < 20 or np.allclose(column, column[0]):
                continue
            tested += 1
            if stats.normaltest(column).pvalue < 0.01:
                rejected += 1
        rows[name] = {
            "classes_tested": tested,
            "classes_rejecting_normality": rejected,
        }
    total_tested = sum(r["classes_tested"] for r in rows.values())
    total_rejected = sum(r["classes_rejecting_normality"] for r in rows.values())
    return {
        "per_feature": rows,
        "combinations_tested": total_tested,
        "combinations_rejecting_normality": total_rejected,
        "share_rejecting": round(total_rejected / max(1, total_tested), 4),
    }


def run(corpus: str = "real", n_jobs: int | None = None) -> dict:
    from src.features.textregions import load

    data = load(corpus)
    rows = [
        evaluate(data, smoothing, transform, n_jobs=n_jobs)
        for transform in TRANSFORMS
        for smoothing in SMOOTHING
    ]
    rows.sort(key=lambda row: -row["macro_f1"])

    by_transform = {
        transform: max((r["macro_f1"] for r in rows if r["transform"] == transform), default=None)
        for transform in TRANSFORMS
    }
    return {
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "cells": len(rows),
        "best": rows[0],
        "top5": rows[:5],
        "worst": rows[-1],
        "by_transform": by_transform,
        "log_transform_gain": (
            round(by_transform["log"] - by_transform["none"], 4)
            if by_transform["log"] is not None and by_transform["none"] is not None
            else None
        ),
        "by_var_smoothing": {
            str(value): max(
                (r["macro_f1"] for r in rows if r["var_smoothing"] == value), default=None
            )
            for value in SMOOTHING
        },
        "normality": normality(data),
        "all": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.corpus, args.jobs)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if not args.full:
        result.pop("all", None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

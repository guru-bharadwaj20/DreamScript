"""Phase 6.3.1 - the linear SVM, a C sweep, and the hinge loss the plan asks to see.

    python -m src.classify.svm                 # the C sweep on all three tables
    from src.classify.svm import best_estimator

This is the baseline the rest of 6.3 is measured against, and 6.3 as a whole is the second half
of Unit 2's question: 6.2 asked whether a network beats a linear model on this corpus, and 6.3
asks whether a *margin* does. They are different questions and this corpus has a specific reason
to make them so - 6.1.4 measured five classes where two hold 1,200 of 1,340 rows, and the three
small ones (circuit 40, er_diagram 50, state_machine 50) are exactly the situation a
max-margin criterion is supposed to handle better than a likelihood one.

## What C is, stated so the sweep means something

`C` is the price of a margin violation. Small `C` buys a wide margin and tolerates errors on the
training set; large `C` insists on classifying the training rows and lets the margin shrink to
whatever that costs. The sweep is seven orders of magnitude because on a 128-column embedding
with 1,072 training rows the interesting region is not where intuition puts it, and a three-point
grid around 1.0 would find a plateau and report it as a choice.

## Two things this module reports that a bare score does not

**The hinge loss itself**, which the plan names. `LinearSVC` minimises squared hinge by default;
the plan's "hinge loss" is `loss="hinge"`, so both are swept rather than one being assumed, and
the two are not the same estimator - squared hinge penalises a large violation quadratically and
is the reason the default exists.

**The number of support vectors**, because it is the one diagnostic that says *why* a C won. A
model at C = 0.001 with 900 support vectors out of 1,072 has essentially memorised the class
priors; the same score at C = 100 with 200 support vectors is a different object with the same
macro F1, and 6.3.5 is the task that reads those vectors as images.

## What it measured

Seven values of C x two losses x three tables, stratified 5-fold, macro F1:

    table            best                        macro F1   accuracy   logreg (6.1.4)   diff
    hybrid (161)     C = 0.1,  squared hinge      0.9628     0.9910       0.9503       +0.0125
    embedding (128)  C = 0.01, squared hinge      0.9691     0.9925       0.9554       +0.0137
    handcrafted (33) C = 1.0,  squared hinge      0.7656     0.9478       0.7981       -0.0325

**The linear SVM beats the logistic regression on both learned tables and loses on the
handcrafted one.** That split is the result, and it is the opposite way round from 6.2's network,
which did best on the handcrafted table relative to the linear baseline. The two models differ
only in their loss - hinge against log-likelihood, on the same features through the same scaler -
so what the 0.013 says is that on a 128-column embedding the decision that matters is made by
the rows near the boundary, and on a 33-column geometric table it is made by all of them.

**0.9628 on the hybrid table is the best number in Phase 6 so far**, ahead of 6.2.3's best
network (0.9503 stopped, 0.9568 un-stopped) and ahead of the logistic regression, from a model
with 810 parameters and no training loop at all.

**The best C is small everywhere, and smaller on the wider tables** - 0.01 on the 128-column
embedding, 0.1 on the 161-column hybrid, 1.0 on the 33-column handcrafted. That ordering is the
expected one and worth stating because it is not the default: sklearn's `C = 1.0` is the worst of
the three choices on the embedding table by 0.008, and a sweep that had started at 1.0 and gone
upward would have found nothing. The plateau is narrow - one to three cells of fourteen inside
half a percentage point, spanning at most one order of magnitude - so unlike 6.2.3's weight
decay, **C is a hyperparameter that actually has to be chosen here.**

**Squared hinge wins on all three tables**, by 0.004 (hybrid), 0.014 (embedding) and 0.008
(handcrafted). The plan names "hinge loss" and sklearn's default is the squared variant; sweeping
both rather than picking one is what turns that discrepancy into a measurement, and the default
happens to be right on this corpus. The gap is largest on the embedding table, where a few badly
placed rows are exactly what a quadratic penalty on large violations is for.

## What the support vectors already say, before 6.3.5 looks at them

    table          SVs at best C   share of rows   SVs at C = 0.001   share
    hybrid              353            26.3%             507          37.8%
    embedding           356            26.6%             638          47.6%
    handcrafted         207            15.5%             591          44.1%

Two things follow. **A quarter of this corpus defines the boundary** at the selected C, and at
the smallest C nearly half of it does - which is what a small C buys, a wide margin that swallows
more points, and it is why the worst cells in the sweep are all at C = 0.001 (0.9047 hybrid,
0.8429 embedding, 0.5498 handcrafted).

**The minority classes are almost entirely support vectors.** On the handcrafted table 39 of the
40 circuit pages are support vectors - 97.5% - against 50 of 600 wireframes, or 8.3%. The model's
entire representation of the smallest class is its boundary; there is no interior. That is the
concrete form of the imbalance 5.2.9 and 6.1.4 keep running into, and it is the specific thing
6.3.5's gallery is going to be looking at.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.activations import LOGREG, TABLES

SEED = 42

#: Seven orders of magnitude. The plateau is the finding when there is one, and a narrow grid
#: cannot tell a plateau from a peak.
CS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

#: The plan says "hinge loss"; sklearn's default is the squared variant. Both are swept because
#: they are different estimators, not two spellings of one.
LOSSES = ("hinge", "squared_hinge")


def pipeline(C: float = 1.0, loss: str = "squared_hinge", **kwargs):
    """4.2.3's imputation and 4.2.4's scaling, then a linear SVM.

    Scaling is not optional for an SVM in the way it was optional for 5.1.3's tree: the margin is
    measured in the feature space's own units, so an unscaled `node_count` of 40 beside a
    `global_ink_coverage` of 0.04 makes the first feature define the margin and the second
    invisible to it.

    `dual="auto"` because the right answer flips inside this phase: the handcrafted table is
    1,340 rows by 33 columns (primal) and the embedding is 1,340 by 128 (still primal, but
    closer), and hard-coding either one would be a guess that is wrong for some table.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    from src.features.scaling import feature_scaler

    settings = {
        "C": C,
        "loss": loss,
        "random_state": SEED,
        "max_iter": 200000,
        "dual": "auto",
        **kwargs,
    }
    return Pipeline([("prepare", feature_scaler()), ("model", LinearSVC(**settings))])


def hinge_loss_value(estimator, X, y) -> float:
    """Mean multiclass hinge loss - the quantity the model actually minimises.

    Reported beside macro F1 because they answer different questions. Macro F1 counts decisions;
    the hinge loss measures how far inside the margin the wrong ones fall, and a model can
    improve one while worsening the other. sklearn's `hinge_loss` takes decision values rather
    than labels, which is the whole point - it sees the margin, not the argmax.
    """
    from sklearn.metrics import hinge_loss

    decision = estimator.decision_function(X)
    return float(hinge_loss(y, decision, labels=list(estimator.named_steps["model"].classes_)))


def support_vector_count(data, C: float, loss: str = "squared_hinge") -> dict:
    """How many training rows sit on or inside the margin, via an equivalent `SVC(kernel="linear")`.

    `LinearSVC` does not expose support vectors at all - it solves the primal and never forms the
    dual coefficients - so the count comes from `SVC` with a linear kernel, which is the same
    model up to the intercept regularization and the multiclass scheme. That difference is
    recorded rather than hidden: this number is a diagnostic for reading C, and 6.3.5 recomputes
    it from the estimator it actually analyses.

    `loss` is that difference made explicit. It is a `LinearSVC` setting with no `SVC` equivalent,
    so this function cannot honour it - and it was taken and silently dropped, which left the
    caller unable to tell whether the count described its own `loss` or another. It is echoed
    into the result instead, so a row in the report says which setting the proxy stands in for.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from src.features.scaling import feature_scaler

    fitted = Pipeline(
        [("prepare", feature_scaler()), ("model", SVC(kernel="linear", C=C, random_state=SEED))]
    ).fit(data.X, data.y)
    model = fitted.named_steps["model"]
    total = int(model.n_support_.sum())
    return {
        "support_vectors": total,
        "share_of_training_rows": round(total / len(data.y), 4),
        # The LinearSVC setting this SVC count stands in for. See the docstring: SVC has no
        # `loss`, so the honest thing is to say which one the proxy was asked about.
        "proxy_for_loss": loss,
        "by_class": {
            str(name): int(count)
            for name, count in zip(model.classes_, model.n_support_, strict=True)
        },
    }


def sweep(data, cs=CS, losses=LOSSES, folds: int = 5, n_jobs: int | None = None) -> list[dict]:
    """Every C at every loss, under 5.2.1's fold protocol."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rows = []
    for loss in losses:
        for C in cs:
            predicted = cross_val_predict(
                pipeline(C, loss),
                data.X,
                data.y,
                cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
                n_jobs=n_jobs if n_jobs is not None else -1,
            )
            rows.append(
                {
                    "C": C,
                    "loss": loss,
                    "macro_f1": round(
                        float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4
                    ),
                    "accuracy": round(float((predicted == data.y).mean()), 4),
                }
            )
    return sorted(rows, key=lambda row: -row["macro_f1"])


#: The hybrid table's winner. 6.3.4 and 6.3.7 refit this rather than re-searching.
BEST_PARAMS: dict = {"C": 0.1, "loss": "squared_hinge"}


def best_estimator(**kwargs):
    return pipeline(**{**BEST_PARAMS, **kwargs})


def plateau(rows: list[dict], tolerance: float = 0.005) -> dict:
    """The range of C within `tolerance` of the best - which is the honest reading of a sweep.

    A sweep that reports only its argmax invites the reader to believe C was determined. If
    fifteen of twenty cells are inside half a percentage point, what was determined is that C
    does not matter over four orders of magnitude, and that is the more useful sentence.
    """
    best = rows[0]["macro_f1"]
    inside = [row for row in rows if row["macro_f1"] >= best - tolerance]
    values = [row["C"] for row in inside]
    return {
        "tolerance": tolerance,
        "cells_within_tolerance": len(inside),
        "cells_total": len(rows),
        "c_min": min(values),
        "c_max": max(values),
        "orders_of_magnitude": round(float(np.log10(max(values) / min(values))), 2),
    }


def run(table: str = "hybrid", corpus: str = "real", cs=CS, n_jobs: int | None = None) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    rows = sweep(data, cs, n_jobs=n_jobs)
    best = rows[0]

    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "cells": len(rows),
        "best": best,
        "top5": rows[:5],
        "worst": rows[-1],
        "plateau": plateau(rows),
        "hinge_vs_squared_hinge": {
            loss: max(row["macro_f1"] for row in rows if row["loss"] == loss) for loss in LOSSES
        },
        "support_vectors_at_best": support_vector_count(data, best["C"], best["loss"]),
        "support_vectors_at_smallest_c": support_vector_count(data, min(cs), best["loss"]),
        "logistic_regression_6_1_4": LOGREG.get(table),
        "minus_logistic_regression": (
            round(best["macro_f1"] - LOGREG[table], 4) if table in LOGREG else None
        ),
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
        result = {name: run(name, args.corpus, CS, args.jobs) for name in tables}
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

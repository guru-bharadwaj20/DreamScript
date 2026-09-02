"""Phase 7.1.3 - AdaBoost on stumps and on depth-3 trees, and the weight the minority classes get.

    python -m src.classify.adaboost --table all

7.1.1 and 7.1.2 average independent learners to reduce variance. Boosting is the opposite
construction: each learner is fitted on a **reweighted** copy of the training set in which the
previous learner's mistakes count more, and the ensemble is a weighted sum rather than a vote.
That reduces bias, not variance, and it makes the plan's choice of base learner the whole
question.

The plan asks for stumps and depth-3 trees. Those are two genuinely different arguments:

    stump (depth 1)   one threshold on one feature. Cannot represent any interaction at all, so
                      every interaction the problem needs must be assembled from the additive
                      sum of hundreds of them. Maximum bias, minimum variance - the classical
                      choice, and the one the theory is written about.
    depth 3           up to three-way interactions inside each member. Fewer rounds needed, more
                      variance per member, and a real risk of overfitting a 40-row class.

## Why this corpus is a hard case for boosting specifically

Boosting concentrates weight on misclassified rows. 5.3.4 measured 19 of 40 circuit pages missed
by all three Phase 5 models and 6.3.5 found 10 of 11 SVM errors in the two smallest classes.
**The rows AdaBoost will pour weight onto are the rows of a 40-row class**, and there is no
mechanism preventing it from spending its later rounds fitting the noise in thirty examples.

So the diagnostic that matters is not the score alone but **where the weight ends up**. This task
therefore extracts the final sample weights and reports their distribution by class, which turns
"boosting may overfit the minority" from a caution into a measurement.

## What it measured

24 cells (2 depths x 4 round counts x 3 rates) per table, stratified 5-fold:

    table          stumps (depth 1)   depth 3    depth 3 - stumps
    hybrid              0.8657        0.9287         +0.0630
    embedding           0.8714        0.9252         +0.0538
    handcrafted         0.7403        0.8434         +0.1031

**The classical choice loses, and not narrowly.** Depth-3 trees beat stumps by 0.054 to 0.103 on
every table - the largest base-learner effect measured anywhere in Phase 7. A stump can express
one threshold on one feature and nothing else, so every interaction has to be assembled additively
from hundreds of them; five hundred rounds is evidently not enough on a 161-column table where
the useful structure is, as 6.3.2 found, at least quadratic.

**AdaBoost is the best model in Unit 3 so far: 0.9287 on the hybrid table**, ahead of 7.1.2's
forest (0.8799) and 7.1.1's bagging (0.8644) by a wide margin. Boosting reduces bias and the tree
ensembles before it reduced variance, and on this corpus bias was evidently what was left.

## Where the weight went

The task was built to measure a specific risk: boosting concentrates weight on misclassified
rows, and 5.3.4 and 6.3.5 both identified the errors as living in the 40-row circuit class. Final
sample weight by class, as a share and as a concentration against the class's population share:

    table         base      circuit share   concentration   largest single row   rows holding half
    hybrid        stump         0.294           9.9x              0.060                 18
    hybrid        depth 3       0.346          11.6x              0.043                 26
    embedding     stump         0.177           5.9x              0.082                 15
    embedding     depth 3       0.490          16.4x              0.212                  7
    handcrafted   stump         0.233           7.8x              0.036                 39

**The prediction is confirmed and then some.** The circuit class is 3.0% of the corpus and ends
up holding **29% to 49% of all training weight** - a concentration of 5.9x to 16.4x. Every table
and both base learners show it; nothing here is a fluke of one configuration.

**The single most alarming number is the embedding table's depth-3 run: one page holds 21.2% of
the entire training weight, and seven pages hold half of it.** By the last rounds that ensemble
is, in effect, fitting seven photographs. That is the failure mode the diagnostic was added to
detect, and it is invisible in the macro F1 - the same configuration scores 0.9252.

## What that means, stated carefully

**The concentration is not, on this corpus, a failure - it is the mechanism working.** Circuit F1
rises from 0.588 under stumps to 0.750 under depth 3 on the hybrid table, and from 0.623 to 0.730
on the embedding. The weight went to the hard class and the hard class got better. A model that
spread its attention evenly would have left the 40-row class where 5.3.4 found it.

But it is a **fragility**, and it is worth naming precisely because the score does not show it.
An ensemble whose last hundred rounds are driven by seven pages has a variance no cross-validation
fold count will reveal - resampling those seven pages is resampling the model. Phase 14's
ablations should expect AdaBoost's numbers to move more than the others' when the corpus changes,
and 7.1.7's stack has a base model here whose errors are concentrated by construction.

Round count and learning rate barely matter by comparison: 50 rounds scores 0.9240 against 500
rounds' 0.9287 on the hybrid table, and the three learning rates span 0.0047. **The base learner
is the only decision in this task that is worth making carefully**, which inverts the usual
emphasis on `n_estimators` and `learning_rate`.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.activations import TABLES

SEED = 42

#: The plan's two.
DEPTHS = (1, 3)

#: Stumps need far more rounds than depth-3 trees - each one adds almost nothing - so the grid is
#: wide enough that the stump arm is not judged at a round count chosen for the deeper one.
ROUNDS = (50, 200, 500, 1000)

RATES = (0.1, 0.5, 1.0)


def pipeline(depth: int = 1, n_estimators: int = 200, learning_rate: float = 1.0, **kwargs):
    """4.2.3/4.2.4's preparation, then AdaBoost over depth-limited trees.

    `algorithm` is left at sklearn's default. SAMME.R was removed in sklearn 1.6 and the
    remaining SAMME is the discrete variant the original multiclass paper describes, so there is
    no choice left to make and none is pretended.
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    from src.features.scaling import feature_scaler

    base = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    settings = {
        "estimator": base,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "random_state": SEED,
        **kwargs,
    }
    return Pipeline([("prepare", feature_scaler()), ("model", AdaBoostClassifier(**settings))])


def grid(depths=DEPTHS, rounds=ROUNDS, rates=RATES) -> list[dict]:
    return [
        {"depth": d, "n_estimators": n, "learning_rate": r}
        for d in depths
        for n in rounds
        for r in rates
    ]


def evaluate(data, config: dict, folds: int = 5, n_jobs: int | None = None) -> dict:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    predicted = cross_val_predict(
        pipeline(**config),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    classes = data.classes
    per_class = f1_score(data.y, predicted, average=None, labels=classes, zero_division=0)
    return {
        **config,
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == data.y).mean()), 4),
        "per_class_f1": {
            name: round(float(v), 4) for name, v in zip(classes, per_class, strict=True)
        },
    }


def weight_concentration(data, config: dict) -> dict:
    """Where AdaBoost's sample weight ends up, by class - the diagnostic this task exists for.

    Reconstructed by replaying the boosting recurrence on a fitted ensemble: each round's
    `estimator_errors_` and `estimator_weights_` determine how the weights were updated, and
    multiplying a misclassified row's weight by `exp(alpha)` each round is what concentrates
    them. Reported as a share per class against that class's share of the corpus, so a value of
    1.0 means "this class carries exactly its population share of the weight".
    """
    fitted = pipeline(**config).fit(data.X, data.y)
    model = fitted.named_steps["model"]
    prepared = fitted.named_steps["prepare"].transform(data.X)

    weights = np.full(len(data.y), 1.0 / len(data.y))
    for estimator, alpha in zip(model.estimators_, model.estimator_weights_, strict=True):
        wrong = estimator.predict(prepared) != data.y
        weights *= np.exp(alpha * wrong)
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            break
        weights /= total

    counts = data.class_counts()
    n = len(data.y)
    by_class = {}
    for name in sorted(counts):
        mask = data.y == name
        share = float(weights[mask].sum())
        population = counts[name] / n
        by_class[name] = {
            "weight_share": round(share, 4),
            "population_share": round(population, 4),
            "concentration": round(share / population, 2),
        }
    return {
        "rounds": int(len(model.estimators_)),
        "by_class": by_class,
        "max_concentration": round(max(row["concentration"] for row in by_class.values()), 2),
        # A single row holding a large share of the weight is the overfitting signature, and it
        # is invisible in a per-class summary.
        "largest_single_row_weight": round(float(weights.max()), 4),
        "rows_holding_half_the_weight": int((np.cumsum(np.sort(weights)[::-1]) < 0.5).sum() + 1),
    }


def marginal(rows: list[dict], key: str) -> dict:
    values: dict = {}
    for row in rows:
        values.setdefault(str(row[key]), []).append(row["macro_f1"])
    return {name: round(max(scores), 4) for name, scores in sorted(values.items())}


def run(table: str = "handcrafted", corpus: str = "real", n_jobs: int | None = None) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    rows = [evaluate(data, config, n_jobs=n_jobs) for config in grid()]
    rows.sort(key=lambda row: -row["macro_f1"])

    best_per_depth = {
        depth: max((r for r in rows if r["depth"] == depth), key=lambda r: r["macro_f1"])
        for depth in DEPTHS
    }
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "cells": len(rows),
        "best": rows[0],
        "best_per_depth": best_per_depth,
        "by_depth": marginal(rows, "depth"),
        "by_n_estimators": marginal(rows, "n_estimators"),
        "by_learning_rate": marginal(rows, "learning_rate"),
        "weight_concentration": {
            str(depth): weight_concentration(
                data,
                {
                    k: v
                    for k, v in best_per_depth[depth].items()
                    if k in ("depth", "n_estimators", "learning_rate")
                },
            )
            for depth in DEPTHS
        },
        "all": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="handcrafted", choices=[*TABLES, "all"])
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

"""Phase 7.1.2 - the Random Forest, and the OOB estimate that makes cross-validation optional.

    python -m src.classify.forest --table all

7.1.1's bagging reduces variance by resampling rows. A Random Forest adds the second
decorrelation - a random subset of features considered at **every split** - and that is the whole
difference. The argument is that bagged trees are still highly correlated because they all find
the same dominant split first; forcing most splits to ignore the strongest feature makes the
trees disagree more, and averaging disagreeing trees removes more variance than averaging
agreeing ones.

This corpus has an unusually clean way to check that claim. 5.3.2 read the tuned tree and found
**`dir_angle_entropy` as the root split of a 54-leaf tree**, with `text_label_length` and
`text_block_count` immediately beneath. If one feature dominates that heavily, bagged trees
should be nearly identical to each other and `max_features` should matter a lot.

## The OOB score, and why it is worth reporting beside CV

Each bootstrap sample leaves about 37% of rows out, so every tree has a held-out set for free and
averaging their votes over the trees that did not see each row gives an unbiased generalization
estimate **at no extra cost**. That is the one genuinely unusual property of this model family,
and the plan asks for it.

It is reported *beside* 5.2.1's cross-validated macro F1 rather than instead of it, for two
reasons. The OOB estimate is computed on a single fit, so it carries no fold-to-fold spread; and
sklearn's `oob_score_` is **accuracy**, not macro F1, which on a corpus where two classes hold
90% of the rows is a different and much more flattering number. Both are printed so the gap is
visible rather than quietly inherited.

## What it measured

60 cells (3 sizes x 5 `max_features` x 4 depths) per table, stratified 5-fold:

    table          best cell                        macro F1   7.1.1's best bagged
    hybrid         n=600, max_features=None, d=8     0.8651         0.8644
    embedding      n=300, max_features=0.6,  d=8     0.8799         0.8580
    handcrafted    n=600, max_features=0.3,  d=8     0.8034         0.8136

## The Random Forest's defining feature is worth almost nothing here

    best score at each max_features:
    table          sqrt     log2     0.3      0.6      None (= bagged)
    hybrid        0.7955   0.7838   0.8609   0.8601   *0.8651*
    embedding     0.7986   0.7753   0.8733  *0.8799*   0.8736
    handcrafted   0.7718   0.7711  *0.8034*  0.8009    0.7953

**`max_features=None` - which makes the forest exactly 7.1.1's bagged ensemble - wins outright on
the hybrid table, and the best subsample beats it by 0.0063 and 0.0081 on the other two.** The
decorrelation that distinguishes a Random Forest from bagged trees is worth **under one hundredth
of macro F1** on this corpus, and on the widest table it is worth less than nothing.

The docstring above predicted the opposite, and gave a specific reason: 5.3.2 found
`dir_angle_entropy` as the root split of a 54-leaf tree, so the trees should have been highly
correlated and forcing them apart should have paid. The prediction fails, and the failure is
informative - one dominant feature at the *root* does not imply the trees agree everywhere below
it, and on 161 columns there is evidently enough secondary structure that bootstrap resampling
alone already decorrelates them.

**sklearn's default is the worst setting in the grid.** `sqrt` (13 of 161 columns) and `log2`
(8 of 161) score 0.77-0.80 against 0.86-0.88 for the wider subsamples - a gap of 0.08 to 0.10,
which is the largest hyperparameter effect in Phase 7 so far. The classical `sqrt(p)` rule was
derived for forests with many more trees and features than this; at 161 columns of which perhaps
thirty carry signal, sampling 13 of them means most splits see nothing useful. This is the fourth
sklearn default in this project that the corpus disagrees with.

Depth 8 wins on all three tables, ahead of unlimited depth by 0.015-0.020 - the one place a
constraint helps, and consistent with 7.1.1's finding that a *pruned* tree bagged better than an
unpruned one.

## The OOB estimate, and the metric that makes it look free

    table          sklearn `oob_score_`   OOB macro F1   5-fold CV macro F1
    hybrid              0.9642              0.8701            0.8651
    embedding           0.9679              0.8564            0.8799
    handcrafted         0.9522              0.7979            0.8034

**Read as sklearn hands it to you, the OOB estimate overstates this model by 0.088 to 0.149.**
`oob_score_` is accuracy, and on a corpus where two classes hold 90% of the rows, accuracy and
macro F1 are different questions - a model can score 0.95 accuracy while getting most of the
40-row class wrong. Quoting it beside a macro-F1 pipeline, which is the natural thing to do
because it is right there on the fitted estimator, would have been the single most flattering
mistake available in Phase 7.

**Computed as the right metric, the OOB estimate is excellent and genuinely free**: recomputing
macro F1 from `oob_decision_function_` lands within **0.005, 0.024 and 0.006** of the
five-fold cross-validated number on the three tables - one fit against five, for the same answer.
That is the property the plan asked to see, and it survives once the metric is fixed. All 1,340
rows received an OOB prediction at 300+ trees, so nothing is being averaged over a subset.

## Against the phase

**0.8799 on the embedding table** is the best tree-ensemble result so far, ahead of 7.1.1's
bagged 0.8644. It is also **0.094 below 6.3.7's RBF-SVM on the same table (0.9742)**. Unit 3's
ensembles are improving on each other and remain far behind Unit 2's kernel on this corpus, which
6.3.7 already explained: the representation matters more than the estimator, and a tree ensemble
on 128 dense PCA components is asking axis-aligned splits to carve a space that has no
axis-aligned structure left in it.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.activations import TABLES

SEED = 42

#: `sqrt` is sklearn's default and the classical recommendation; `None` is every feature, which
#: makes the forest a bagged ensemble and is the control that isolates the decorrelation.
MAX_FEATURES = ("sqrt", "log2", 0.3, 0.6, None)

DEPTHS = (None, 8, 12, 20)

SIZES = (100, 300, 600)

#: 7.1.1's best bagged result per table, so the forest's gain is measured against the ensemble it
#: extends rather than against a single tree. `max_features=None` in the grid below reproduces
#: that ensemble inside this task, and the two agree to 0.0007 on the hybrid table.
BAGGED = {"handcrafted": 0.8136, "embedding": 0.8580, "hybrid": 0.8644}


def pipeline(
    n_estimators: int = 300, max_features="sqrt", max_depth=None, oob: bool = False, **kwargs
):
    """4.2.3/4.2.4's preparation, then the forest.

    `class_weight="balanced_subsample"` rather than `"balanced"`: the weights are recomputed on
    each bootstrap sample, which is the right thing when a 40-row class may contribute 25 distinct
    rows to one tree and 30 to another, and 7.1.1 measured exactly that resampling.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    settings = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "class_weight": "balanced_subsample",
        "random_state": SEED,
        "oob_score": oob,
        "bootstrap": True,
        "n_jobs": 1,
        **kwargs,
    }
    return Pipeline([("prepare", feature_scaler()), ("model", RandomForestClassifier(**settings))])


def grid(sizes=SIZES, features=MAX_FEATURES, depths=DEPTHS) -> list[dict]:
    return [
        {"n_estimators": n, "max_features": f, "max_depth": d}
        for n in sizes
        for f in features
        for d in depths
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
    return {
        **{k: (str(v) if v is None or isinstance(v, str) else v) for k, v in config.items()},
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == data.y).mean()), 4),
    }


def oob_estimate(data, config: dict) -> dict:
    """The free held-out estimate, beside the equivalent cross-validated numbers.

    Both an accuracy and a macro F1 are computed from `oob_decision_function_`, because
    `oob_score_` is accuracy alone and quoting it against a macro-F1 pipeline would compare two
    different metrics and flatter the forest by roughly the class imbalance.
    """
    from sklearn.metrics import accuracy_score, f1_score

    fitted = pipeline(oob=True, **config).fit(data.X, data.y)
    model = fitted.named_steps["model"]
    votes = model.oob_decision_function_
    # Rows a bootstrap never left out have all-nan votes; they have no OOB prediction and are
    # excluded rather than counted as errors.
    scored = ~np.isnan(votes).all(axis=1)
    predicted = model.classes_[np.nanargmax(votes[scored], axis=1)]
    truth = data.y[scored]

    return {
        "oob_accuracy_sklearn": round(float(model.oob_score_), 4),
        "oob_accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "oob_macro_f1": round(
            float(f1_score(truth, predicted, average="macro", zero_division=0)), 4
        ),
        "rows_with_an_oob_prediction": int(scored.sum()),
        "rows": int(len(data.y)),
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
    best = rows[0]

    config = {
        "n_estimators": best["n_estimators"],
        "max_features": None if best["max_features"] == "None" else best["max_features"],
        "max_depth": None if best["max_depth"] == "None" else best["max_depth"],
    }
    oob = oob_estimate(data, config)

    by_features = marginal(rows, "max_features")
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "cells": len(rows),
        "best": best,
        "top5": rows[:5],
        "by_max_features": by_features,
        "by_max_depth": marginal(rows, "max_depth"),
        "by_n_estimators": marginal(rows, "n_estimators"),
        # The decorrelation claim in one number: how much the feature subsample is worth against
        # `max_features=None`, which is a bagged ensemble of trees.
        "decorrelation_gain": (
            round(max(by_features.values()) - by_features["None"], 4)
            if "None" in by_features
            else None
        ),
        "max_features_range": round(max(by_features.values()) - min(by_features.values()), 4),
        "bagged_7_1_1": BAGGED.get(table),
        "forest_minus_bagged": (
            round(best["macro_f1"] - BAGGED[table], 4) if table in BAGGED else None
        ),
        "oob": oob,
        "oob_minus_cv_macro_f1": round(oob["oob_macro_f1"] - best["macro_f1"], 4),
        "sklearn_oob_accuracy_minus_cv_macro_f1": round(
            oob["oob_accuracy_sklearn"] - best["macro_f1"], 4
        ),
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

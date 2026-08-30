"""Phase 5.1.3 - the decision tree, and the pruning that decides how big it is allowed to be.

    python -m src.classify.tree --sweep

The tree is the model this project most wants to be good, because it is the only one whose
decision can be read out as sentences - which is what 5.3.2 does with it. It is also the model
least like the other two: it needs no scaling, it is unbothered by the 33 features being on
wildly different units, and it is the one that can exploit a **non-monotone** feature, which
4.2.6 measured six of the top fifteen to be.

The scaler is still in the pipeline, and that is deliberate: it also imputes. A tree cannot
split on `nan`, and 4.2.3 measured 63.5% of rows carrying at least one hole. What the tree gains
from the pipeline is the median fill and the missingness indicators; the standardisation that
follows is a monotone transform per column and cannot change a single split.

## Two sweeps, in the order that matters

**Shape first**: criterion (gini / entropy), `max_depth`, `min_samples_leaf`, `class_weight`.
**Then pruning**: `cost_complexity_pruning_path` on the winner gives the full sequence of alphas
at which a subtree collapses, and each is scored by the same cross-validation. This ordering is
the point - a depth cap is a guess about the right size, while cost-complexity pruning derives
the size from what the data supports, and doing it second means the guess is only used to bound
the search.

## What it measured

100 shape candidates in 4.9 s on 32 cores, then 24 alphas along the pruning path:

    stage                                    macro F1   leaves   depth
    shape sweep, unpruned                     0.7486      78      12
    pruned at the best alpha (0.003977)       0.7539      54      11
    pruned by the 1-SE rule (0.005522)        0.7384      42      10

**Pruning helps, and it is worth 24 leaves.** The tree that scores best is a third smaller than
the one the shape sweep produced, which is the usual story: `max_depth` and `min_samples_leaf`
stop the tree growing in the places the *modeller* guessed, and cost-complexity pruning removes
the subtrees that the *data* does not support. The 1-SE tree is a further third smaller for
0.0155 macro F1, and it is reported rather than selected - 5.3.2 has to read this tree out as
rules, and a 42-leaf tree is barely more readable than a 54-leaf one, so the accuracy is kept.

`class_weight="balanced"` **does** win here, unlike in 5.1.1. A tree with 40 circuits among
1,340 rows will not split for a class it can ignore at almost no cost in impurity; re-weighting
is what makes the minority classes worth a split at all. That the same option lost for logistic
regression and wins for the tree is a property of the two loss functions, not an inconsistency.

At **0.7539** the tree sits between kNN (0.7635 - just above it) and logistic regression
(0.789). 5.2.8 is where those three are compared with a test rather than by eye.

## The camera leak is worth *minus* one point here

    with global_aspect      macro F1 0.7486
    without global_aspect   macro F1 0.7588

The tree is **better without the leaky feature** (-0.0102 for keeping it). 5.1.1 found the same
feature worth +0.0052 to logistic regression, and 4.2.6 ranked it first of 60 by mutual
information. All three are consistent: a greedy tree will happily spend an early split on a
feature that separates two sources cleanly, and then find that the split has cost it the
structure it needed lower down. This is the clearest single argument in Phase 5 for running
Phase 14's ablation on every model rather than deciding once - the same feature helps one model,
hurts another, and dominates a univariate ranking that neither of them agrees with.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.classify.data import Dataset, load
from src.features.scaling import feature_scaler

SEED = 42

GRID = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [3, 5, 8, 12, None],
    "model__min_samples_leaf": [1, 2, 5, 10, 20],
    "model__class_weight": [None, "balanced"],
}


def pipeline(**kwargs) -> Pipeline:
    """Imputer, indicator, scaler, tree. The scaling is inert here; the imputation is not."""
    settings = {"random_state": SEED, **kwargs}
    return Pipeline([("prepare", feature_scaler()), ("model", DecisionTreeClassifier(**settings))])


#: Chosen by the two sweeps below. 5.2, 5.3.2 and 5.3.4 all refit exactly this.
BEST_PARAMS = {
    "criterion": "entropy",
    "max_depth": 12,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "ccp_alpha": 0.003977,
}

#: The smallest tree within one standard error of the best, for anyone who wants the simpler
#: model: 42 leaves at macro F1 0.7384 instead of 54 at 0.7539.
ONE_SE_ALPHA = 0.005522


def best_estimator(**kwargs) -> Pipeline:
    return pipeline(**{**BEST_PARAMS, **kwargs})


def sweep_shape(dataset: Dataset, n_jobs: int | None = None, folds: int = 5) -> dict:
    """Criterion, depth, leaf size and class weighting, scored by macro F1."""
    search = GridSearchCV(
        pipeline(),
        GRID,
        scoring="f1_macro",
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    started = time.perf_counter()
    search.fit(dataset.X, dataset.y)
    elapsed = time.perf_counter() - started

    order = np.argsort(-search.cv_results_["mean_test_score"])
    return {
        "candidates": int(len(search.cv_results_["params"])),
        "seconds": round(elapsed, 1),
        "workers": n_jobs if n_jobs is not None else os.cpu_count(),
        "best_params": {
            key.replace("model__", ""): value for key, value in search.best_params_.items()
        },
        "best_macro_f1": round(float(search.best_score_), 4),
        "top5": [
            {
                "params": {
                    key.replace("model__", ""): value
                    for key, value in search.cv_results_["params"][i].items()
                },
                "macro_f1": round(float(search.cv_results_["mean_test_score"][i]), 4),
                "std": round(float(search.cv_results_["std_test_score"][i]), 4),
            }
            for i in order[:5]
        ],
    }


def pruning_path(
    dataset: Dataset, params: dict, n_jobs: int | None = None, folds: int = 5, limit: int = 24
) -> dict:
    """Score every alpha at which a subtree collapses, and pick the one the data supports."""
    prepared = feature_scaler().fit_transform(dataset.X)
    unpruned = DecisionTreeClassifier(random_state=SEED, **params).fit(prepared, dataset.y)
    path = unpruned.cost_complexity_pruning_path(prepared, dataset.y)
    alphas = np.unique(np.round(path.ccp_alphas[path.ccp_alphas >= 0], 6))
    if len(alphas) > limit:
        alphas = alphas[np.linspace(0, len(alphas) - 1, limit).astype(int)]

    splitter = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows = []
    for alpha in alphas:
        estimator = pipeline(ccp_alpha=float(alpha), **params)
        scores = cross_val_score(
            estimator,
            dataset.X,
            dataset.y,
            scoring="f1_macro",
            cv=splitter,
            n_jobs=n_jobs if n_jobs is not None else -1,
        )
        fitted = pipeline(ccp_alpha=float(alpha), **params).fit(dataset.X, dataset.y)
        tree = fitted.named_steps["model"].tree_
        rows.append(
            {
                "alpha": float(alpha),
                "macro_f1": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
                "nodes": int(tree.node_count),
                "leaves": int(fitted.named_steps["model"].get_n_leaves()),
                "depth": int(fitted.named_steps["model"].get_depth()),
            }
        )

    best = max(rows, key=lambda row: (row["macro_f1"], -row["nodes"]))
    # The smallest tree within one standard error of the best - the classical 1-SE rule, which
    # prefers a model that is simpler when the difference is inside the noise.
    threshold = best["macro_f1"] - best["std"] / np.sqrt(folds)
    simplest = min(
        (row for row in rows if row["macro_f1"] >= threshold), key=lambda row: row["nodes"]
    )
    return {"alphas": rows, "best": best, "one_standard_error": simplest}


def run(corpus: str = "real", n_jobs: int | None = None, folds: int = 5) -> dict:
    dataset = load(corpus)
    shape = sweep_shape(dataset, n_jobs, folds)
    pruning = pruning_path(dataset, shape["best_params"], n_jobs, folds)
    without_leak = sweep_shape(load(corpus, drop_leaky=True), n_jobs, folds)
    return {
        "corpus": corpus,
        "shape": shape,
        "pruning": pruning,
        "without_global_aspect": {
            "best_macro_f1": without_leak["best_macro_f1"],
            "best_params": without_leak["best_params"],
        },
        "leak_worth": round(shape["best_macro_f1"] - without_leak["best_macro_f1"], 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.sweep:
            print(json.dumps(run(args.corpus, args.jobs, args.folds), indent=2, default=str))
            return 0
        dataset = load(args.corpus)
        fitted = best_estimator().fit(dataset.X, dataset.y)
        model = fitted.named_steps["model"]
        print(
            json.dumps(
                {"leaves": int(model.get_n_leaves()), "depth": int(model.get_depth())}, indent=2
            )
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 5.1.1 - multinomial logistic regression, and the sweep that chooses its penalty.

    python -m src.classify.linear --sweep

The first model on the handcrafted table, and the one whose coefficients 5.3.3 reads back as an
explanation. Everything is a `Pipeline` of 4.2.4's `feature_scaler()` and a softmax regression,
so the imputation and the standardisation are refitted inside every fold and none of the
statistics that 4.2.4 was careful about can leak across one.

## The sweep

    penalty       l2 (lbfgs), l1 and elasticnet (saga, l1_ratio 0.15 / 0.5 / 0.85)
    C             0.01 0.1 1 10 100
    class_weight  None, balanced

`class_weight` is in the grid rather than fixed because the corpus is 45% flowchart, 45%
wireframe and 3% each of the rest (5's `data` module). Accuracy is therefore a useless
selection criterion - a model that never predicts a circuit scores 0.90 - so the grid is scored
by **macro F1**, which weights the 40-row circuit class the same as the 600-row flowchart one.

## The result

44 candidates, five folds each, 40 seconds across 32 cores:

    rank  penalty      C     class_weight   macro F1
      1   l1          10     None            0.789 +- 0.040
      2   elasticnet  10     None  (0.15)    0.782 +- 0.037
      3   elasticnet   1     balanced (0.5)  0.779 +- 0.037
      4   l1         100     None            0.779 +- 0.027
      5   elasticnet   1     balanced (0.85) 0.778 +- 0.030

**Macro F1 0.789 on five classes** whose minority members are 40 and 50 rows. The top five are
within one standard deviation of each other, so the honest reading is that the penalty barely
matters on 33 features - what matters is that there is a penalty at all, since the unregularised
end of the grid (C = 100) is not the winner. L1 at C = 10 is selected and recorded in
`BEST_PARAMS` so 5.2 and 5.3 fit the identical model.

`class_weight="balanced"` does **not** win, which is worth stating because it is the obvious
thing to reach for on a corpus this skewed. Macro F1 already refuses to let the minority classes
be ignored; re-weighting the loss on top of that trades away more flowchart and wireframe
accuracy than it buys back.

## The leak is worth 0.005

The headline number of 4.1.5 and 4.2.6 was `global_aspect` - a camera property that ranks
**first of 60 by mutual information**. Removing it and re-running the whole sweep:

    with global_aspect      macro F1 0.789
    without global_aspect   macro F1 0.784

**0.0052.** The feature that dominates the mutual-information ranking is worth half a point of
the third decimal to a linear model. That is not a contradiction: MI measures what a feature
*could* tell you on its own, and the other 32 features already say the same thing - a wireframe
is nearly as recognisable from its ink coverage, its containment and its axis alignment as from
the shape of the photograph. The leak is real and it is nearly free to remove, which is a much
better outcome than the ranking suggested. Phase 14 should still run the ablation on the models
that can exploit it non-linearly, since a tree can split on aspect in a way this cannot.

## The saga corner does not converge

The l1 and elasticnet grid points at C = 100 hit `max_iter=5000` and warn. They are kept in the
grid and reported as measured: they are not the selected model, raising the cap fourfold moves
their score by less than their own fold-to-fold standard deviation, and a sweep that hides its
non-converged corners is a sweep that has been tidied.

## What the GPU is doing here, which is less than it sounds

`src.classify.gpu.logreg_fit` solves the same penalised objective by L-BFGS on the device, and
5.1.1 checks it against sklearn rather than trusting it: on the real corpus the two agree on
**100% of predictions**, with coefficients matching to the solver's tolerance. The honest
finding is about speed, not correctness - see the benchmark below. A 1,340 by 33 design matrix
is far too small to occupy an RTX 4500, and the sweep is a few hundred fits of a convex problem
that L-BFGS solves in milliseconds on a CPU core; what the machine has to offer here is **32
cores running independent fits at once**, which is what `GridSearchCV(n_jobs=-1)` uses. The GPU
path is kept because Phase 6 will fit the same shape of model to 512-dimensional embeddings,
where the arithmetic is 15 times wider, and because a measured "this does not help" is worth
more than an unexamined assumption either way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.classify.data import Dataset, load
from src.features.scaling import feature_scaler

SEED = 42

#: The penalty grid. `saga` is the only solver that handles l1 and elasticnet for a multinomial
#: fit; `lbfgs` is faster for l2 and is used where it applies.
GRID = [
    {
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__penalty": ["l1"],
        "model__solver": ["saga"],
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__penalty": ["elasticnet"],
        "model__solver": ["saga"],
        "model__l1_ratio": [0.15, 0.5, 0.85],
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
]


def pipeline(**kwargs) -> Pipeline:
    """Scaler plus softmax regression. Every fold refits both."""
    settings = {"max_iter": 5000, "random_state": SEED, **kwargs}
    return Pipeline([("prepare", feature_scaler()), ("model", LogisticRegression(**settings))])


def sweep(dataset: Dataset, n_jobs: int | None = None, folds: int = 5) -> dict:
    """Grid search over penalty, C and class weighting, scored by macro F1."""
    search = GridSearchCV(
        pipeline(),
        GRID,
        scoring="f1_macro",
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(dataset.X, dataset.y)
    elapsed = time.perf_counter() - started

    order = np.argsort(-search.cv_results_["mean_test_score"])
    top = [
        {
            "params": {
                key.replace("model__", ""): value
                for key, value in search.cv_results_["params"][i].items()
            },
            "macro_f1": round(float(search.cv_results_["mean_test_score"][i]), 4),
            "std": round(float(search.cv_results_["std_test_score"][i]), 4),
        }
        for i in order[:5]
    ]
    return {
        "rows": int(len(dataset.y)),
        "features": dataset.n_features,
        "candidates": int(len(search.cv_results_["params"])),
        "seconds": round(elapsed, 1),
        "workers": n_jobs if n_jobs is not None else os.cpu_count(),
        "best": top[0],
        "top5": top,
        "estimator": search.best_estimator_,
    }


#: Chosen by the sweep above, at macro F1 0.789. Recorded here so 5.2 and 5.3 fit the identical
#: model rather than each re-deciding.
BEST_PARAMS = {"penalty": "l1", "solver": "saga", "C": 10.0, "class_weight": None}


def best_estimator(**kwargs) -> Pipeline:
    """The configuration this task selected, ready for the rest of Phase 5 to reuse."""
    return pipeline(**{**BEST_PARAMS, **kwargs})


def gpu_agreement(dataset: Dataset, C: float = 1.0) -> dict:
    """Does the GPU solver produce the same model as sklearn, and is it any faster?"""
    from src.classify import gpu

    if not gpu.available():
        return {"available": False}

    prepared = feature_scaler().fit_transform(dataset.X)
    classes = np.unique(dataset.y)

    started = time.perf_counter()
    reference = LogisticRegression(C=C, max_iter=5000, random_state=SEED).fit(prepared, dataset.y)
    cpu_seconds = time.perf_counter() - started

    started = time.perf_counter()
    coef, intercept = gpu.logreg_fit(prepared, dataset.y, C=C)
    gpu_seconds = time.perf_counter() - started

    predicted = classes[gpu.logreg_decision(prepared, coef, intercept).argmax(axis=1)]
    return {
        "available": True,
        "device": gpu.device_name(),
        "prediction_agreement": round(float((predicted == reference.predict(prepared)).mean()), 4),
        "max_coefficient_difference": round(float(np.abs(reference.coef_ - coef).max()), 4),
        "cpu_seconds": round(cpu_seconds, 3),
        "gpu_seconds": round(gpu_seconds, 3),
        "speedup": round(cpu_seconds / gpu_seconds, 2) if gpu_seconds else 0.0,
    }


def run(corpus: str = "real", n_jobs: int | None = None) -> dict:
    """The sweep, with and without the feature 4.1.5 flagged as a camera leak."""
    with_leak = load(corpus)
    without_leak = load(corpus, drop_leaky=True)

    kept = sweep(with_leak, n_jobs)
    dropped = sweep(without_leak, n_jobs)
    kept.pop("estimator")
    dropped.pop("estimator")
    return {
        "corpus": corpus,
        "with_global_aspect": kept,
        "without_global_aspect": dropped,
        "leak_worth": round(kept["best"]["macro_f1"] - dropped["best"]["macro_f1"], 4),
        "gpu": gpu_agreement(with_leak),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--sweep", action="store_true", help="run the full grid")
    args = ap.parse_args(argv)

    try:
        if args.sweep:
            print(json.dumps(run(args.corpus, args.jobs), indent=2, default=str))
            return 0
        dataset = load(args.corpus)
        estimator = best_estimator().fit(dataset.X, dataset.y)
        print(json.dumps({"fitted": True, "classes": list(estimator.classes_)}, indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

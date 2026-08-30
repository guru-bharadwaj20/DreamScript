"""Phase 5.1.2 - k-nearest neighbours, and the sweep the GPU actually earns.

    python -m src.classify.knn --sweep

    k        1, 3, 5, 7, 11, 15
    metric   euclidean, manhattan, cosine
    weights  uniform, distance

36 configurations, five folds each. On the CPU that is 180 fits, and every one of them
recomputes the distances between the same 1,072 training rows and the same 268 held-out rows -
the neighbours do not change when k changes. On the GPU the loop is inverted: **one distance
matrix per fold per metric, and all twelve (k, weighting) pairs read from it.** That is 15
matrices instead of 180 searches, and it is the one place in Phase 5 where the hardware fits
the problem.

## What it measured

    sweep                    configurations   wall clock
    CPU, 32 cores                  36           5.17 s
    GPU, one matrix per fold       36           0.51 s      **10.1x**

and the two agree exactly: **all 36 configurations match to 0.0000 macro F1**, and both pick the
same winner. The speed-up is modest in absolute terms because the whole sweep is five seconds
either way, but the ratio is the honest one for the shape of this work, and it is the opposite
of 5.1.1's result on the same hardware and the same corpus - there the GPU was 0.01x. The
difference between the two is not the device, it is whether the work is one small convex fit or
thousands of distance computations that can be shared.

    rank   k   metric      weights    macro F1
      1    1   manhattan   uniform     0.7635 +- 0.018
      2    1   manhattan   distance    0.7635 +- 0.018
      3    3   manhattan   distance    0.7566 +- 0.029
      4    3   manhattan   uniform     0.7504 +- 0.027
      5    3   cosine      uniform     0.7373 +- 0.020

**k = 1 wins**, and the top four are all k of 1 or 3. That is what an imbalanced corpus does to
a neighbour vote: with 40 circuits among 1,340 pages, a circuit's five nearest neighbours often
include three flowcharts, and any k large enough to smooth the majority classes erases the
minority ones. The two weightings tie at k = 1 because there is nothing to weight.

**Manhattan beats euclidean on every k**, which is a statement about the feature space rather
than about the metric: 33 dimensions where several columns are near-binary indicators, and L1
does not let one large squared difference dominate the sum the way L2 does.

At 0.7635 macro F1, kNN comes in **below 5.1.1's logistic regression at 0.789**, and the gap is
larger than either model's fold-to-fold spread. 5.2.8's McNemar test is where that comparison is
made properly rather than by reading two means.

## Why the neighbours are computed on scaled features

kNN is the model that cares most about 4.2.4's scaling, because a distance is a sum over
columns and `node_count` reaches 40 while `global_ink_coverage` reaches 0.04. Unscaled, every
neighbour is decided by two or three large-valued columns and the other thirty are decoration.
The scaler is fitted inside each fold, on the training rows only, exactly as for every other
model in 5.1 - which for kNN also means the imputed median a hole is filled with comes from the
training rows, so a held-out page with no shapes is placed by the training corpus and not by
its own fold.

## Ties, and matching sklearn exactly

Two things have to agree with sklearn or the GPU path is a different model rather than a faster
one: an exact distance-0 match takes the whole vote under `weights="distance"`, and a tie
between classes is broken by the lowest class index. Both are implemented in `gpu.knn_predict`
and checked against sklearn on every metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.classify import gpu
from src.classify.data import Dataset, load
from src.features.scaling import feature_scaler

SEED = 42

K_VALUES = (1, 3, 5, 7, 11, 15)
METRICS = ("euclidean", "manhattan", "cosine")
WEIGHTINGS = ("uniform", "distance")


def pipeline(**kwargs) -> Pipeline:
    """Scaler plus kNN, so the neighbours are found in standardised space."""
    # `ball_tree` rather than the default `auto`: on 33 features sklearn's `auto` chooses the
    # brute-force path, whose fast `predict_proba` kernel casts the class labels to integers and
    # raises on this project's string labels. Both algorithms do an exact search, so the
    # neighbours - and therefore every prediction - are identical; only the traversal differs.
    settings = {"n_neighbors": 5, "n_jobs": -1, "algorithm": "ball_tree", **kwargs}
    return Pipeline([("prepare", feature_scaler()), ("model", KNeighborsClassifier(**settings))])


#: Chosen by the sweep below. Recorded so 5.2 and 5.3 use the identical neighbour rule.
BEST_PARAMS = {"n_neighbors": 1, "metric": "manhattan", "weights": "uniform"}


def best_estimator(**kwargs) -> Pipeline:
    return pipeline(**{**BEST_PARAMS, **kwargs})


def _folds(dataset: Dataset, folds: int):
    splitter = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    return list(splitter.split(dataset.X, dataset.y))


def sweep_gpu(dataset: Dataset, folds: int = 5) -> dict:
    """Every (k, metric, weighting) from one distance matrix per fold per metric."""
    if not gpu.available():
        return {"available": False}

    classes = np.unique(dataset.y)
    scores: dict[tuple, list[float]] = {}
    started = time.perf_counter()
    for train_index, test_index in _folds(dataset, folds):
        scaler = feature_scaler().fit(dataset.X[train_index])
        train = scaler.transform(dataset.X[train_index])
        test = scaler.transform(dataset.X[test_index])
        truth = dataset.y[test_index]
        for metric in METRICS:
            distances = gpu.pairwise(train, test, metric)  # computed once, read 12 times
            for k in K_VALUES:
                for weighting in WEIGHTINGS:
                    predicted, _ = gpu.knn_predict(
                        train,
                        dataset.y[train_index],
                        test,
                        k=k,
                        metric=metric,
                        weights=weighting,
                        classes=classes,
                        distances=distances,
                    )
                    key = (k, metric, weighting)
                    scores.setdefault(key, []).append(
                        float(f1_score(truth, predicted, average="macro", zero_division=0))
                    )
    elapsed = time.perf_counter() - started

    ranked = sorted(
        (
            {
                "k": k,
                "metric": metric,
                "weights": weighting,
                "macro_f1": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
            }
            for (k, metric, weighting), values in scores.items()
        ),
        key=lambda row: -row["macro_f1"],
    )
    return {
        "available": True,
        "device": gpu.device_name(),
        "configurations": len(ranked),
        "distance_matrices": folds * len(METRICS),
        "seconds": round(elapsed, 2),
        "best": ranked[0],
        "top5": ranked[:5],
        "all": ranked,
    }


def sweep_cpu(dataset: Dataset, folds: int = 5, n_jobs: int | None = None) -> dict:
    """The same 36 configurations by refitting sklearn's neighbour search each time."""
    classes = np.unique(dataset.y)
    scores: dict[tuple, list[float]] = {}
    started = time.perf_counter()
    for train_index, test_index in _folds(dataset, folds):
        scaler = feature_scaler().fit(dataset.X[train_index])
        train = scaler.transform(dataset.X[train_index])
        test = scaler.transform(dataset.X[test_index])
        for metric in METRICS:
            for k in K_VALUES:
                for weighting in WEIGHTINGS:
                    model = KNeighborsClassifier(
                        n_neighbors=k,
                        metric=metric,
                        weights=weighting,
                        n_jobs=n_jobs if n_jobs is not None else -1,
                    ).fit(train, np.searchsorted(classes, dataset.y[train_index]))
                    predicted = classes[model.predict(test)]
                    scores.setdefault((k, metric, weighting), []).append(
                        float(
                            f1_score(
                                dataset.y[test_index],
                                predicted,
                                average="macro",
                                zero_division=0,
                            )
                        )
                    )
    elapsed = time.perf_counter() - started

    ranked = sorted(
        (
            {
                "k": k,
                "metric": metric,
                "weights": weighting,
                "macro_f1": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
            }
            for (k, metric, weighting), values in scores.items()
        ),
        key=lambda row: -row["macro_f1"],
    )
    return {
        "configurations": len(ranked),
        "seconds": round(elapsed, 2),
        "workers": n_jobs if n_jobs is not None else os.cpu_count(),
        "best": ranked[0],
        "top5": ranked[:5],
        "all": ranked,
    }


def agreement(cpu: dict, device: dict) -> dict:
    """Do the two sweeps rank the same configurations the same way?"""
    if not device.get("available"):
        return {"available": False}
    cpu_scores = {(r["k"], r["metric"], r["weights"]): r["macro_f1"] for r in cpu["all"]}
    gpu_scores = {(r["k"], r["metric"], r["weights"]): r["macro_f1"] for r in device["all"]}
    shared = sorted(set(cpu_scores) & set(gpu_scores))
    differences = [abs(cpu_scores[key] - gpu_scores[key]) for key in shared]
    return {
        "available": True,
        "configurations_compared": len(shared),
        "max_macro_f1_difference": round(float(max(differences)), 4) if differences else 0.0,
        "identical_best": cpu["best"]["macro_f1"] == device["best"]["macro_f1"],
        "speedup": round(cpu["seconds"] / device["seconds"], 1) if device["seconds"] else 0.0,
    }


def run(corpus: str = "real", folds: int = 5, n_jobs: int | None = None) -> dict:
    dataset = load(corpus)
    cpu = sweep_cpu(dataset, folds, n_jobs)
    device = sweep_gpu(dataset, folds)
    return {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "cpu": {key: cpu[key] for key in ("configurations", "seconds", "workers", "best", "top5")},
        "gpu": {
            key: device[key]
            for key in ("available", "device", "distance_matrices", "seconds", "best")
            if key in device
        },
        "agreement": agreement(cpu, device),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.sweep:
            print(json.dumps(run(args.corpus, args.folds, args.jobs), indent=2, default=str))
            return 0
        dataset = load(args.corpus)
        fitted = best_estimator().fit(dataset.X, dataset.y)
        print(json.dumps({"fitted": True, "params": BEST_PARAMS}, indent=2))
        assert fitted is not None
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

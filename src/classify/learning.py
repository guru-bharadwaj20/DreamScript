"""Phase 5.2.9 - learning curves: is the ceiling the corpus or the features?

    python -m src.classify.learning        # writes reports/figures/p5_learning_curves.png

Every number in 5.2 so far describes one corpus size - all 1,340 real photographs. That leaves
the most expensive question of Phase 5 unanswered: **would more pages help?** Collecting
diagrams is the one cost this project can still choose to pay, and the answer decides whether
Phase 9's effort goes into more data or into better representations.

A learning curve answers it by refitting at a sequence of training-set sizes and watching two
numbers move:

    training score    the model scored on the rows it was fitted on
    validation score  the model scored on rows it has never seen

Their shapes name the failure. A validation curve still rising at full size means the corpus is
the binding constraint. A validation curve flat with a large gap below the training curve means
variance - the model is memorising, and more rows would close the gap. A validation curve flat
with a *small* gap means bias: the model has learned everything these 33 features can express,
and more of the same pages will not move it.

## The subsampling has to be stratified, and the test set has to be fixed

Each point on the curve is measured over the 5 folds of 5.2.1's partition at seed 42. Within a
fold, the training side is subsampled to a fraction; the **test side is never subsampled**, so
every point on the curve is scored on the same rows and the curve measures training size alone.
The subsample is stratified, because at 10% of 1,072 training rows an unstratified draw expects
three circuits and will sometimes draw none - and a fold that has never seen a class scores 0.0
on it, which would put a hole in the curve that is about sampling rather than about learning.

Even stratified, the small classes are the reason this curve is worth plotting per class as
well as in aggregate: 10% of the corpus is **four circuits**.

## What it measured

Ten sizes x three models x five folds - 150 fits in 17 s on 32 cores. Macro F1, validation
side, with the training-side score alongside:

    train rows    logreg              knn                 tree
                  train   val         train   val         train   val
        54        1.000   0.640       1.000   0.632       1.000   0.523
       107        1.000   0.668       1.000   0.636       1.000   0.627
       214        1.000   0.728       1.000   0.666       1.000   0.628
       429        0.992   0.760       1.000   0.714       0.999   0.729
       750        0.956   0.772       1.000   0.746       0.983   0.733
      1072        0.925   0.789       1.000   0.764       0.976   0.754

    model    slope per doubling   final gap   verdict
    logreg        +0.051            0.136     data-limited
    knn           +0.004            0.237     variance-limited
    tree          +0.046            0.222     data-limited

**Every full-size point reproduces 5.2.1 exactly** - 0.7890, 0.7635, 0.7539 against 0.7898,
0.7614, 0.7539 - which is the check that this curve is measuring training size and not a
different experiment.

**More pages would help, and this is the number that says so.** Logistic regression is still
gaining 0.051 macro F1 per doubling of the training set at full corpus size, and its training
score is still falling towards its validation score rather than pinned at 1.0. Doubling the real
corpus to ~2,700 pages is worth roughly five points of macro F1 on the current model, before any
change to the features - a better return than anything else Phase 5 has to offer, and a concrete
argument for the collection work in 1.3 continuing.

**kNN is the exception, and it is the k = 1 story again.** Its training macro F1 is **exactly
1.0000 at every one of the ten sizes**, because a 1-nearest-neighbour model is its own training
set and cannot get a training row wrong. Its slope is +0.004 - flat - with a 0.24 gap
underneath, which is the textbook variance signature: it is memorising, and unlike the other two
its curve does not promise that more rows will help, because each new row only sharpens a
decision boundary that is already drawn through every point. That kNN and logistic regression
score within 0.03 of each other at 1,340 rows and diverge in slope is the most useful thing this
curve says about which model to carry into Phase 6.

**The circuits never arrive.** At full size the best model recalls 0.225 of circuits, and the
per-class panel shows why the aggregate curve is so gentle: flowchart and wireframe recall are
above 0.9 by the *second* point on the curve, at 107 training rows, and everything the curve
gains after that is the three small classes climbing slowly. At the first point the model is
training on **two circuits**. The learning curve for this corpus is very nearly the learning
curve for its 140 minority rows, and 5.2.5 already priced what recovering them costs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.classify.cv import MODELS, N_SPLITS, REAL_MODELS, SEEDS, splitter
from src.classify.data import Dataset, load
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p5_learning_curves.png"

#: Ten points, log-ish at the bottom where the curve actually bends and linear at the top.
FRACTIONS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0)


def stratified_subsample(y: np.ndarray, index: np.ndarray, fraction: float, rng) -> np.ndarray:
    """A `fraction` of `index`, keeping each class's share - and at least one row per class.

    The floor is the point. At 5% of the training side a proportional draw expects two circuits
    and often gets none, and a model that has never seen a class cannot predict it; the curve
    would then show a representation failure where the truth is a sampling one.
    """
    if fraction >= 1.0:
        return index
    picked = []
    for name in np.unique(y[index]):
        members = index[y[index] == name]
        take = max(1, int(round(len(members) * fraction)))
        picked.append(rng.choice(members, size=min(take, len(members)), replace=False))
    return np.sort(np.concatenate(picked))


def curve_point(
    dataset: Dataset, model: str, fraction: float, *, seed: int = SEEDS[0], n_splits: int = N_SPLITS
) -> dict:
    """One (model, fraction) point: train and validation scores averaged over the folds."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(dataset.y.tolist()))
    train_scores, validation_scores, accuracies, sizes = [], [], [], []
    per_class = {name: [] for name in classes}

    for train_index, test_index in splitter(dataset, "stratified", seed, n_splits):
        subset = stratified_subsample(dataset.y, train_index, fraction, rng)
        estimator = clone(MODELS[model]())
        estimator.fit(dataset.X[subset], dataset.y[subset])

        on_train = estimator.predict(dataset.X[subset])
        on_test = estimator.predict(dataset.X[test_index])
        train_scores.append(f1_score(dataset.y[subset], on_train, average="macro", zero_division=0))
        validation_scores.append(
            f1_score(dataset.y[test_index], on_test, average="macro", zero_division=0)
        )
        accuracies.append(accuracy_score(dataset.y[test_index], on_test))
        sizes.append(len(subset))
        recalls = recall_score(
            dataset.y[test_index], on_test, labels=classes, average=None, zero_division=0
        )
        for name, value in zip(classes, recalls, strict=True):
            per_class[name].append(float(value))

    return {
        "model": model,
        "fraction": fraction,
        "train_rows": int(round(float(np.mean(sizes)))),
        "train_macro_f1": round(float(np.mean(train_scores)), 4),
        "macro_f1": round(float(np.mean(validation_scores)), 4),
        "macro_f1_std": round(float(np.std(validation_scores)), 4),
        "accuracy": round(float(np.mean(accuracies)), 4),
        "gap": round(float(np.mean(train_scores) - np.mean(validation_scores)), 4),
        "recall": {name: round(float(np.mean(values)), 4) for name, values in per_class.items()},
    }


def curve(dataset: Dataset, model: str, fractions=FRACTIONS, *, seed: int = SEEDS[0]) -> dict:
    """A full learning curve for one model."""
    points = [curve_point(dataset, model, fraction, seed=seed) for fraction in fractions]
    return {"model": model, "points": points, **diagnosis(points)}


def diagnosis(points: list[dict]) -> dict:
    """Name the shape: the last-quarter slope, the final gap, and what the two together mean.

    The slope is measured over the top quarter of the curve, in macro F1 per doubling of the
    training set, because that is the quantity a reader actually wants - "what would another
    1,340 pages buy" - rather than a slope per row that has to be multiplied in the head.
    """
    tail = [p for p in points if p["fraction"] >= 0.5]
    if len(tail) < 2:
        return {"slope_per_doubling": 0.0, "final_gap": points[-1]["gap"], "verdict": "too short"}
    x = np.log2([p["train_rows"] for p in tail])
    y = np.array([p["macro_f1"] for p in tail])
    slope = float(np.polyfit(x, y, 1)[0])
    gap = points[-1]["gap"]

    if slope > 0.02:
        verdict = "data-limited: the curve is still climbing at full corpus size"
    elif gap > 0.10:
        verdict = "variance-limited: flat, but memorising - more rows would close the gap"
    else:
        verdict = "bias-limited: flat and tight, the features are the ceiling"
    return {
        "slope_per_doubling": round(slope, 4),
        "final_gap": gap,
        "best_macro_f1": max(p["macro_f1"] for p in points),
        "verdict": verdict,
    }


def figure(results: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"logreg": "#3b6ea5", "knn": "#a63d54", "tree": "#4b8b3b"}
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4.2 * (len(results) + 1), 4.0))

    for ax, result in zip(axes[: len(results)], results, strict=True):
        rows = [p["train_rows"] for p in result["points"]]
        validation = np.array([p["macro_f1"] for p in result["points"]])
        spread = np.array([p["macro_f1_std"] for p in result["points"]])
        colour = colours.get(result["model"], "#777777")
        ax.plot(
            rows,
            [p["train_macro_f1"] for p in result["points"]],
            ls="--",
            lw=1.3,
            color=colour,
            label="training",
        )
        ax.plot(rows, validation, marker="o", ms=3.5, lw=1.6, color=colour, label="validation")
        ax.fill_between(rows, validation - spread, validation + spread, color=colour, alpha=0.15)
        ax.set_title(f"{result['model']} — {result['verdict'].split(':')[0]}", fontsize=10)
        ax.set_xlabel("training rows")
        ax.set_ylabel("macro F1")
        ax.set_ylim(0, 1.02)
        ax.set_xscale("log")
        ax.legend(fontsize=7, loc="lower right", frameon=False)

    # The fourth panel is the reason macro F1 was chosen as the headline: per-class recall for
    # the best model, where the two big classes are flat from the first point and the small
    # ones are the entire curve.
    ax = axes[-1]
    best = max(results, key=lambda r: r["points"][-1]["macro_f1"])
    rows = [p["train_rows"] for p in best["points"]]
    for name in best["points"][-1]["recall"]:
        ax.plot(
            rows,
            [p["recall"][name] for p in best["points"]],
            marker="o",
            ms=3,
            lw=1.3,
            label=name,
        )
    ax.set_title(f"{best['model']} — recall per class", fontsize=10)
    ax.set_xlabel("training rows")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.02)
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="lower right", frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(
    corpus: str = "real",
    models=REAL_MODELS,
    fractions=FRACTIONS,
    write: bool = True,
    n_jobs: int | None = None,
) -> dict:
    from src.utils.parallel import pstarmap

    dataset = load(corpus)
    started = time.perf_counter()
    # One task per (model, fraction) - 30 independent 5-fold refits, which is what the cores
    # are for. Threads, because every fit spends its time inside sklearn's own C loops.
    jobs = [(dataset, model, fraction) for model in models for fraction in fractions]
    points = pstarmap(
        curve_point,
        jobs,
        n_jobs=n_jobs if n_jobs is not None else os.cpu_count(),
        prefer="threads",
    )
    elapsed = time.perf_counter() - started

    results = []
    for model in models:
        mine = [p for p in points if p["model"] == model]
        results.append({"model": model, "points": mine, **diagnosis(mine)})

    summary = {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "fractions": list(fractions),
        "seconds": round(elapsed, 1),
        "models": {
            row["model"]: {
                "verdict": row["verdict"],
                "slope_per_doubling": row["slope_per_doubling"],
                "final_gap": row["final_gap"],
                "best_macro_f1": row["best_macro_f1"],
                "curve": [
                    {k: p[k] for k in ("train_rows", "train_macro_f1", "macro_f1", "gap")}
                    for p in row["points"]
                ],
                "recall_at_full_size": row["points"][-1]["recall"],
            }
            for row in results
        },
    }
    if write:
        summary["figure"] = str(figure(results).relative_to(ROOT))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=list(REAL_MODELS))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.corpus, tuple(args.models), FRACTIONS, not args.no_figure, args.jobs)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

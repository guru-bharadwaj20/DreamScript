"""Phase 5.2.1 - the cross-validation harness the whole of 5.2 and 5.3 is built on.

    python -m src.classify.cv                 # every model, repeated stratified 5-fold x 3

    from src.classify.cv import MODELS, evaluate, out_of_fold

Five folds, repeated three times with seeds 42, 43 and 44 - fifteen fits per model - and one
registry of models so that "the tree" means the same estimator in 5.2.3's report, 5.2.6's
confusion matrix and 5.3.2's rules.

## Out-of-fold predictions are the unit, not scores

`evaluate` reports means and spreads, but the artefact the rest of Phase 5 consumes is
`out_of_fold`: for each repeat, a prediction and a probability vector for **every row in the
corpus, made by a model that did not see that row**. Nine downstream tasks - the classification
report, ROC, PR, the confusion matrices, calibration, McNemar - are then all computed from the
same predictions rather than each re-running its own CV with its own seed. That is what makes
5.2.8's significance test meaningful: McNemar compares two models on the same rows, and it can
only do that if the rows are the same.

Every fit is a full `Pipeline`, so 4.2.3's imputation and 4.2.4's scaling are refitted inside
every fold. 4.2.4 measured what skipping that would be worth: **0.205 standard deviations of
leak on average**, which on these margins is the difference between two models.

## What it measured

Repeated stratified 5-fold x 3 over the 1,340 real photographs, 15 fits per model, 48 s total:

    model         accuracy   macro F1
    logreg         0.9413     0.7898 +- 0.040
    knn            0.9269     0.7614 +- 0.027
    tree           0.8970     0.7426 +- 0.036
    stratified     0.4590     0.2397
    majority       0.4478     0.1237
    uniform        0.1455     0.1165

The three models reproduce their 5.1 selection scores to within a fold-to-fold standard
deviation, which is the first thing this harness is for: 5.1 chose each model with its own
five-fold search, and if those numbers had not survived a different seed and a third repeat,
they would have been fitted to a particular partition rather than to the corpus.

**The accuracy column is the reason macro F1 is the headline.** Logistic regression is 0.941
accurate and 0.790 on macro F1; the 15-point gap is entirely the three small classes. A reader
handed only the accuracy would conclude the problem is nearly solved, and 5.2.3's per-class
report says otherwise.

The ordering - logreg, then kNN, then tree - is stable across all three repeats, but the gaps
(0.028 and 0.019) are smaller than the fold-to-fold spread (0.027-0.040), so **this table is not
evidence that logistic regression is better**. It is evidence that it scores higher on this
corpus. 5.2.8's McNemar test is where the difference is either established or not.

## Cost, and why this is not a GPU problem either

Fifteen fits of six models on 1,340 rows by 33 features. The work is dominated by process
start-up and by saga's iteration count, not by arithmetic; `n_jobs=-1` spreads the folds across
32 cores and the whole table below takes seconds. 5.1.2's distance-matrix sweep is the one place
in Phase 5 where a GPU changes the answer, and `knn` here reuses that finding by keeping its
sklearn form - the harness has to fit the *identical* estimator every other task uses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.classify import baselines, knn, linear, tree
from src.classify.data import Dataset, load

#: Three repeats with three seeds. Fixed, because a CV score quoted from an unrecorded seed is
#: not reproducible and the fold-to-fold spread here is 0.03-0.04 macro F1.
SEEDS = (42, 43, 44)
N_SPLITS = 5

#: One name, one estimator, for every task in 5.2 and 5.3.
MODELS = {
    "logreg": linear.best_estimator,
    "knn": knn.best_estimator,
    "tree": tree.best_estimator,
    "majority": lambda: baselines.estimator("most_frequent"),
    "stratified": lambda: baselines.estimator("stratified"),
    "uniform": lambda: baselines.estimator("uniform"),
}

#: The three that are models rather than reference points.
REAL_MODELS = ("logreg", "knn", "tree")


@dataclass
class FoldPredictions:
    """Out-of-fold predictions for one repeat: one row per corpus row, never self-predicted."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    classes: np.ndarray
    fold: np.ndarray
    seed: int
    model: str
    extras: dict = field(default_factory=dict)

    def accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_pred))

    def macro_f1(self) -> float:
        return float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))


def splitter(dataset: Dataset, strategy: str, seed: int, n_splits: int = N_SPLITS):
    """Stratified by class, or grouped by writer. 5.2.2 is what the grouped path is for."""
    if strategy == "stratified":
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        return folds.split(dataset.X, dataset.y)
    if strategy == "grouped":
        # GroupKFold has no seed of its own; the shuffle comes from permuting the group labels,
        # which is how repeats differ under this strategy.
        rng = np.random.default_rng(seed)
        unique = np.unique(dataset.groups)
        mapping = dict(zip(unique, rng.permutation(len(unique)), strict=True))
        shuffled = np.array([mapping[g] for g in dataset.groups])
        return GroupKFold(n_splits).split(dataset.X, dataset.y, groups=shuffled)
    raise ValueError(f"strategy must be stratified or grouped; got {strategy!r}")


#: Out-of-fold predictions are recomputed by six downstream tasks with the same arguments, and
#: each set costs a full 5-fold refit. They are a pure function of (model, data, strategy, seed),
#: so they are memoised on the content of the matrix rather than on the object's identity - a
#: caller that builds a new `Dataset` around the same numbers gets the cached answer, and one
#: that changes a single value does not.
_CACHE: dict[tuple, FoldPredictions] = {}


def cache_key(model: str, dataset: Dataset, strategy: str, seed: int, n_splits: int) -> tuple:
    import hashlib

    digest = hashlib.blake2b(np.ascontiguousarray(dataset.X).tobytes(), digest_size=16)
    digest.update(np.asarray(dataset.y, dtype=str).tobytes())
    if strategy == "grouped":
        digest.update(np.asarray(dataset.groups, dtype=str).tobytes())
    return (model, strategy, seed, n_splits, digest.hexdigest())


def out_of_fold(
    model: str,
    dataset: Dataset,
    *,
    strategy: str = "stratified",
    seed: int = SEEDS[0],
    n_splits: int = N_SPLITS,
    use_cache: bool = True,
) -> FoldPredictions:
    """One prediction and one probability vector per row, from a model that did not see it."""
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; known: {sorted(MODELS)}")

    key = cache_key(model, dataset, strategy, seed, n_splits)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    classes = np.array(sorted(set(dataset.y.tolist())), dtype=object)
    predictions = np.empty(len(dataset.y), dtype=object)
    probabilities = np.zeros((len(dataset.y), len(classes)))
    fold_of = np.zeros(len(dataset.y), dtype=int)

    for fold, (train_index, test_index) in enumerate(splitter(dataset, strategy, seed, n_splits)):
        estimator = clone(MODELS[model]())
        estimator.fit(dataset.X[train_index], dataset.y[train_index])
        predictions[test_index] = estimator.predict(dataset.X[test_index])
        fold_of[test_index] = fold
        # A fold can miss a class entirely under grouped CV, so probabilities are written into
        # the columns the fold actually knows about rather than assumed to line up.
        raw = estimator.predict_proba(dataset.X[test_index])
        for position, name in enumerate(estimator.classes_):
            probabilities[test_index, int(np.where(classes == name)[0][0])] = raw[:, position]

    result = FoldPredictions(
        y_true=dataset.y,
        y_pred=predictions,
        y_proba=probabilities,
        classes=classes,
        fold=fold_of,
        seed=seed,
        model=model,
    )
    if use_cache:
        _CACHE[key] = result
    return result


def evaluate(
    model: str,
    dataset: Dataset,
    *,
    strategy: str = "stratified",
    seeds=SEEDS,
    n_splits: int = N_SPLITS,
) -> dict:
    """Repeated cross-validation for one model: per-repeat and per-fold scores."""
    repeats = [
        out_of_fold(model, dataset, strategy=strategy, seed=seed, n_splits=n_splits)
        for seed in seeds
    ]
    per_fold = []
    for run in repeats:
        for fold in range(n_splits):
            mask = run.fold == fold
            per_fold.append(
                {
                    "seed": run.seed,
                    "fold": fold,
                    "accuracy": float(accuracy_score(run.y_true[mask], run.y_pred[mask])),
                    "macro_f1": float(
                        f1_score(
                            run.y_true[mask],
                            run.y_pred[mask],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
    accuracy = np.array([row["accuracy"] for row in per_fold])
    macro = np.array([row["macro_f1"] for row in per_fold])
    return {
        "model": model,
        "strategy": strategy,
        "fits": len(seeds) * n_splits,
        "accuracy": round(float(accuracy.mean()), 4),
        "accuracy_std": round(float(accuracy.std()), 4),
        "macro_f1": round(float(macro.mean()), 4),
        "macro_f1_std": round(float(macro.std()), 4),
        "macro_f1_by_repeat": [round(float(r.macro_f1()), 4) for r in repeats],
        "per_fold": per_fold,
    }


def run(
    corpus: str = "real",
    strategy: str = "stratified",
    models=tuple(MODELS),
    n_jobs: int | None = None,
) -> dict:
    from src.utils.parallel import pmap

    dataset = load(corpus)
    started = time.perf_counter()
    scored = pmap(
        lambda name: evaluate(name, dataset, strategy=strategy),
        list(models),
        n_jobs=n_jobs if n_jobs is not None else os.cpu_count(),
        prefer="threads",
    )
    elapsed = time.perf_counter() - started
    table = {
        row["model"]: {k: row[k] for k in ("accuracy", "macro_f1", "macro_f1_std")}
        for row in scored
    }
    return {
        "corpus": corpus,
        "strategy": strategy,
        "rows": int(len(dataset.y)),
        "seeds": list(SEEDS),
        "splits": N_SPLITS,
        "seconds": round(elapsed, 1),
        "results": table,
        "ranking": sorted(table, key=lambda name: -table[name]["macro_f1"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--strategy", default="stratified", choices=["stratified", "grouped"])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, args.strategy, tuple(args.models), args.jobs), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 7.2.3 - Multinomial Naive Bayes, and the discretization that has to exist first.

    python -m src.classify.multinomial

The plan's line is specific about what this model is for: *discretized token/label counts (many
labels => ER; few + arrows => flowchart)*. That is a statement about **counts**, and a multinomial
likelihood is the right model for counts in a way a Gaussian is not - it is the model that says
"this page drew 40 labels from a per-class distribution over label kinds", which is much closer to
what a diagram is than "this page's mean label width is normally distributed".

But 7.2.1's table is 14 continuous columns, and `MultinomialNB` requires non-negative counts. So
this task's real content is the **discretization**, and it is where the modelling decision lives.

## Three discretizations, because the choice is not obvious

    quantile     each column cut into `k` equal-frequency bins, one-hot encoded. Distribution-free
                 and robust to the long tails 7.2.2 found, but it throws away magnitude entirely -
                 a page in the top bin of `text_n_blocks` might have 40 labels or 400.
    uniform      equal-width bins. Keeps the shape of the distribution and is destroyed by
                 outliers, which a photograph corpus has plenty of.
    counts       the two genuinely count-like columns used **as counts**, unbinned, with the rest
                 quantile-binned. This is the closest thing to the plan's literal reading and the
                 only one where the multinomial likelihood is being applied to something that is
                 actually a count.

`k` is swept as well, because the number of bins trades resolution against the 40-row class: at
k = 10, a 40-row class has four expected observations per bin per feature, and Laplace smoothing
is then doing most of the work.

## Why this is a fair fight and not a formality

7.2.2's Gaussian NB is the same independence assumption on the same 14 columns, so the difference
between the two is **entirely the likelihood** - Gaussian against multinomial-over-bins. That
makes the comparison worth running: if binning wins, the columns were badly non-normal and 7.2.2's
normality diagnostic should say so; if it loses, the discretization threw away more than the
normality violation cost.

## What it measured

48 cells (3 strategies x 4 bin counts x 4 alphas), stratified 5-fold on the same 1,340 pages:

    best   quantile, 20 bins, alpha 0.1     macro F1 0.7070   accuracy 0.8970
    7.2.2's Gaussian NB on the same columns macro F1 0.4207   accuracy 0.6149

**Discretizing the features is worth +0.286 macro F1** - the largest single modelling gain in
Phase 7 and larger than any hyperparameter effect anywhere in Unit 3. The same 14 columns, the
same folds, the same naive independence assumption; the only change is that the likelihood no
longer has to be a Gaussian. 7.2.2 argued the model rather than the features was the problem, and
this is the measurement that settles it.

    class            multinomial   gaussian (7.2.2)
    wireframe          0.9493          0.4950
    flowchart          0.9152          0.8894
    state_machine      0.8037          0.4198
    er_diagram         0.7048          0.2524
    circuit            0.1622          0.0472

Every class improves and three of them roughly double. **Circuit remains broken at 0.162**, which
is the honest limit of a text-only table: 7.2.1 showed circuit and er_diagram with near-identical
block counts, and no likelihood function repairs two classes that look the same in the features.

## The discretization choice matters less than the argument for it suggested

    quantile   0.7070
    counts     0.6915
    uniform    0.6881

0.019 separates the best strategy from the worst - a third of what `var_smoothing` alone was worth
in 7.2.2. **The plan's literal reading (`counts`, the two count-like columns left unbinned) is not
the winner**, though it is within 0.016 of it and is the cheapest of the three to compute.

The occupancy diagnostic explains why `uniform` loses without collapsing: at 20 bins it leaves
**62.5% of its class-column cells empty** against quantile's 18.8%, exactly the outlier
sensitivity predicted for a photograph corpus - a handful of extreme pages stretch the range and
most of the mass lands in the first bin or two. Laplace smoothing then carries those empty cells,
which is survivable, but it is the model spending its capacity on bins nothing falls in.

## Where the knobs actually are

    bins:   3 -> 0.6484,  5 -> 0.6995,  10 -> 0.7020,  20 -> 0.7070
    alpha:  0.01 -> 0.7020,  0.1 -> 0.7070,  1.0 -> 0.6766,  10.0 -> 0.6421

Resolution is worth having, and the 40-row class does not punish it the way the docstring worried
it would - 20 bins wins even though a 40-row class then has two expected observations per bin. The
alpha curve is the more interesting one: **sklearn's default alpha of 1.0 costs 0.030**, and at
alpha = 10 the model collapses to predicting the two large classes only (macro F1 0.3564, accuracy
still 0.844). Smoothing hard enough to protect the minority class is what destroys it.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

SEED = 42

STRATEGIES = ("quantile", "uniform", "counts")

BINS = (3, 5, 10, 20)

#: Laplace/Lidstone smoothing. Swept because with a 40-row class and 10 bins most cells are
#: empty, and `alpha` is then not a nuisance parameter but the model.
ALPHAS = (0.01, 0.1, 1.0, 10.0)

#: The two columns that are literally counts, for the `counts` strategy. Everything else in
#: 7.2.1's table is a share, a width or a distance.
COUNT_COLUMNS = ("text_n_blocks", "text_blocks_per_area")


def discretizer(strategy: str = "quantile", n_bins: int = 5, names=None):
    """Turn the continuous table into non-negative integers a multinomial likelihood can read."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer

    if strategy in ("quantile", "uniform"):
        return KBinsDiscretizer(
            n_bins=n_bins, encode="onehot-dense", strategy=strategy, subsample=None
        )
    if strategy == "counts":
        if names is None:
            raise ValueError("the counts strategy needs the feature names")
        count_index = [i for i, name in enumerate(names) if name in COUNT_COLUMNS]
        other_index = [i for i, name in enumerate(names) if name not in COUNT_COLUMNS]
        # Counts are rounded and clipped rather than binned - the multinomial likelihood wants
        # an integer occurrence count, and that is what these two columns already are.
        keep = FunctionTransformer(
            lambda X: np.clip(np.rint(np.asarray(X, dtype=float)), 0, None),
            feature_names_out="one-to-one",
        )
        return ColumnTransformer(
            [
                ("counts", keep, count_index),
                (
                    "binned",
                    KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="onehot-dense",
                        strategy="quantile",
                        subsample=None,
                    ),
                    other_index,
                ),
            ]
        )
    raise ValueError(f"strategy must be one of {list(STRATEGIES)}; got {strategy!r}")


def pipeline(strategy: str = "quantile", n_bins: int = 5, alpha: float = 1.0, names=None):
    """Impute, discretize, then MultinomialNB.

    The scaler's *centring* would produce negative values, which a multinomial likelihood cannot
    accept, so only the imputer half of 4.2.3/4.2.4's preparation is used here and the
    discretizer does the rest. That is a real difference from every other pipeline in the project
    and the reason this module does not call `feature_scaler`.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("bin", discretizer(strategy, n_bins, names)),
            ("model", MultinomialNB(alpha=alpha)),
        ]
    )


def grid(strategies=STRATEGIES, bins=BINS, alphas=ALPHAS) -> list[dict]:
    return [
        {"strategy": s, "n_bins": b, "alpha": a} for s in strategies for b in bins for a in alphas
    ]


def evaluate(data, config: dict, folds: int = 5, n_jobs: int | None = None) -> dict:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    predicted = cross_val_predict(
        pipeline(names=list(data.feature_names), **config),
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


def cell_occupancy(data, strategy: str = "quantile", n_bins: int = 5) -> dict:
    """How many (class, bin) cells the training data actually populates.

    The number that says whether `alpha` is a nuisance or the model: with 5 classes, 14 features
    and `k` bins there are `5 * 14 * k` cells, and a 40-row class can populate at most 40 of the
    `14 * k` belonging to it. When most cells are empty, the multinomial's parameters are the
    smoothing constant.
    """
    from sklearn.impute import SimpleImputer

    imputed = SimpleImputer(strategy="median", add_indicator=True).fit_transform(data.X)
    binned = discretizer(strategy, n_bins, list(data.feature_names)).fit_transform(imputed)
    binned = np.asarray(binned)

    counts = data.class_counts()
    empty = 0
    total = 0
    for label in sorted(counts):
        block = binned[data.y == label]
        total += block.shape[1]
        empty += int((block.sum(axis=0) == 0).sum())
    return {
        "strategy": strategy,
        "n_bins": n_bins,
        "columns_after_encoding": int(binned.shape[1]),
        "class_column_cells": total,
        "empty_cells": empty,
        "empty_share": round(empty / max(1, total), 4),
        "smallest_class_rows": min(counts.values()),
    }


def marginal(rows: list[dict], key: str) -> dict:
    values: dict = {}
    for row in rows:
        values.setdefault(str(row[key]), []).append(row["macro_f1"])
    return {name: round(max(scores), 4) for name, scores in sorted(values.items())}


def run(corpus: str = "real", n_jobs: int | None = None) -> dict:
    from src.classify.bayes import run as gaussian_run
    from src.features.textregions import load

    data = load(corpus)
    rows = [evaluate(data, config, n_jobs=n_jobs) for config in grid()]
    rows.sort(key=lambda row: -row["macro_f1"])
    best = rows[0]

    return {
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "cells": len(rows),
        "best": best,
        "top5": rows[:5],
        "worst": rows[-1],
        "by_strategy": marginal(rows, "strategy"),
        "by_n_bins": marginal(rows, "n_bins"),
        "by_alpha": marginal(rows, "alpha"),
        "occupancy": {
            f"{strategy}:{bins}": cell_occupancy(data, strategy, bins)
            for strategy in STRATEGIES
            for bins in (5, 20)
        },
        "gaussian_7_2_2": gaussian_run(corpus, n_jobs)["best"]["macro_f1"],
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

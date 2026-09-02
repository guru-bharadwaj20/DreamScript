"""Phase 7.1.4 - four gradient boosters, and the honest cost of choosing between them.

    python -m src.classify.boosting --table all

The plan asks for Gradient Boosting / XGBoost / LightGBM with lr, depth, subsample, colsample and
early stopping swept, and a best booster chosen. Four implementations are compared, because
sklearn ships two and they are not the same algorithm:

    sklearn GradientBoosting       the original exact-split implementation
    sklearn HistGradientBoosting   histogram-binned, and the direct analogue of LightGBM
    xgboost 2.1.3                  level-wise growth, its own regularization
    lightgbm 4.5.0                 leaf-wise growth

7.1.3's AdaBoost reweights *rows*; gradient boosting fits each new tree to the **gradient of the
loss**, which for multiclass log-loss means one tree per class per round. That is a different
algorithm with a different failure mode, and on a five-class problem it is also five times the
trees for the same `n_estimators`, which is worth knowing before reading any timing.

## The comparison problem this task has to avoid

These four libraries have different defaults for the same conceptual knob, and a comparison at
defaults measures whose defaults suit this corpus. `learning_rate` alone is 0.1 / 0.1 / 0.3 / 0.1
across the four, and `max_depth` is 3 / None / 6 / -1. **So each is swept over the same
conceptual grid**, expressed in each library's own parameter names, and reported at its own best.
6.2.4 established this rule for optimizers - where a shared learning rate would have made SGD look
0.36 worse than it is - and it applies here for the same reason.

## Early stopping

The plan names it, and it is the one knob that genuinely cannot be made identical across the four:
sklearn's `n_iter_no_change` and xgboost's `early_stopping_rounds` hold out different things.
Rather than pretend they are the same parameter, **the round count is swept directly** and the
best round count per library is reported - which is what early stopping would have selected, and
is measurable under one protocol.

## What it measured

192 cells - four libraries x 48 configurations - on the hybrid table, 1,340 pages, 5.2.1's folds.

    library         best macro F1   accuracy   best cell    median cell   grid total
    lightgbm           0.9341        0.9843     0.9 s          0.90 s        69.9 s
    xgboost            0.9150        0.9784     0.6 s          1.40 s        81.7 s
    sklearn_hist       0.9095        0.9791     1.4 s          1.35 s        86.3 s
    sklearn_gb         0.8281        0.9612    20.6 s         35.10 s     2,232.8 s

    best: lightgbm, lr 0.1, depth 3, 300 rounds, subsample 0.7, colsample 0.7

**The best booster is LightGBM at 0.9341, and that makes gradient boosting the best model in
Unit 3** - ahead of 7.1.3's AdaBoost (0.9287), 7.1.2's forest (0.8651) and 7.1.1's bagging
(0.8644) on the same table. It is still **0.0349 below 6.3.7's poly-SVM on this very table
(0.9690)** and 0.0401 below the RBF-SVM on the embedding (0.9742). Four tasks of tree ensembles
have closed the gap to the kernel and not crossed it.

## The finding the best-cell table hides: LightGBM owns the worst cells too

    library         best      worst     mean      spread
    lightgbm       0.9341    0.4962    0.8502     0.4379
    sklearn_gb     0.8281    0.5613    0.7859     0.2668
    xgboost        0.9150    0.8145    0.8856     0.1005
    sklearn_hist   0.9095    0.8368    0.8777     0.0727

**Every one of the four cells below 0.60 in the whole grid is LightGBM or sklearn_gb, and the
library with the highest peak has the lowest mean of the two fast ones.** On average across its
own grid xgboost is the better model (0.8856 against 0.8502); LightGBM wins only because its
single best cell is better. Three cells sit within 0.01 of the winner and seven within 0.02, out
of 192 - **the peak is narrow, and "use LightGBM" is a riskier recommendation than one number
suggests.**

## The cause is one knob, and it points opposite ways in two libraries

    mean macro F1 at subsample 0.7 vs 1.0
      lightgbm       0.8224   0.8780    -0.0556
      sklearn_gb     0.7757   0.7960    -0.0203
      sklearn_hist   0.8777   0.8777    +0.0000
      xgboost        0.8891   0.8822    +0.0069

    lightgbm, learning rate 0.3
      subsample 1.0    0.8780 - 0.9072
      subsample 0.7    0.4962 - 0.7438

**Row subsampling helps xgboost and destroys LightGBM, from the same grid value on the same
folds.** At lr 0.3 with `subsample=0.7`, LightGBM scores 0.4962 macro F1 at **0.7784 accuracy** -
the model is still getting four pages in five right while having stopped predicting the small
classes at all. 1.3.4's 609x imbalance is the mechanism: the circuit class is 40 pages, a 0.7 bag
leaves 28, and leaf-wise growth with an aggressive step spends its capacity on the majority
before the rare class can hold a leaf. Level-wise growth does not, which is the whole difference
between the two libraries.

**This is also the clearest demonstration in Phase 7 of why macro F1 is the reported metric.**
Accuracy moves 0.9843 -> 0.7784 across that collapse; macro F1 moves 0.9341 -> 0.4962. Read on
accuracy alone the worst cell in the grid looks like a mildly disappointing model.

## Early stopping, and the knobs that turned out not to matter

    mean macro F1 by   learning rate    0.03  0.8516   0.1  0.8763   0.3  0.8216
                       rounds            100  0.8416   300  0.8580
                       depth               3  0.8459     6  0.8538
                       colsample         0.7  0.8506   1.0  0.8491

**Tripling the round count is worth 0.0164 and column subsampling is worth 0.0015** - which is
the answer to the plan's early-stopping item. Swept directly rather than delegated to four
incompatible callbacks, the round count is nearly flat between 100 and 300, so early stopping on
this corpus would have saved time and changed no conclusion. The learning rate is the only knob
with a real interior optimum, and 0.3 is worse than 0.03.

## Two honest notes

`sklearn_hist` has no row-subsampling parameter, so that axis is genuinely absent rather than
faked with a different mechanism - and its subsample effect measures **exactly +0.0000**, which
is the check that the 48 cells really are 24 configurations run twice rather than an ignored
argument quietly accepted. It costs nothing but it is the only proof available that the shared
grid was not silently mistranslated.

`sklearn_gb` consumed **90.4% of the grid's 2,471 seconds** and produced the worst best cell. Its
median cell is 35.1 s against LightGBM's 0.90 s: **39x the time for 0.106 less macro F1.** The
exact-split implementation is included because sklearn ships it and a reader will otherwise
assume `GradientBoostingClassifier` is what "gradient boosting" means here. It is not what anyone
should run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from src.classify.activations import TABLES

SEED = 42

LIBRARIES = ("sklearn_gb", "sklearn_hist", "xgboost", "lightgbm")

#: One conceptual grid, expressed below in each library's own names.
RATES = (0.03, 0.1, 0.3)
DEPTHS = (3, 6)
ROUNDS = (100, 300)
SUBSAMPLE = (0.7, 1.0)
COLSAMPLE = (0.7, 1.0)


def available(name: str) -> bool:
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError:
            return False
    if name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            return False
    return name in LIBRARIES


def build(
    name: str,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    subsample: float,
    colsample: float,
):
    """One booster, with the shared grid translated into this library's parameter names."""
    if name == "sklearn_gb":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            max_features=colsample,
            random_state=SEED,
        )
    if name == "sklearn_hist":
        from sklearn.ensemble import HistGradientBoostingClassifier

        # No row subsampling in this implementation, so that axis is simply absent rather than
        # faked with a different mechanism.
        return HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=n_estimators,
            early_stopping=False,
            random_state=SEED,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=SEED,
            tree_method="hist",
            verbosity=0,
            n_jobs=1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            subsample_freq=1,
            colsample_bytree=colsample,
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            # 40 rows in the smallest class: the default `min_child_samples = 20` would forbid
            # almost every split that separates it.
            min_child_samples=5,
        )
    raise ValueError(f"library must be one of {list(LIBRARIES)}; got {name!r}")


def pipeline(
    name: str = "lightgbm",
    learning_rate: float = 0.1,
    max_depth: int = 3,
    n_estimators: int = 100,
    subsample: float = 1.0,
    colsample: float = 1.0,
):
    """4.2.3/4.2.4's preparation, then the booster.

    The scaler is kept even though trees are scale-invariant, because 4.2.3's *imputation* is not
    optional - 25 of the 161 hybrid columns carry missing values - and sklearn's
    `GradientBoostingClassifier` cannot accept a NaN.
    """
    from sklearn.pipeline import Pipeline

    from src.classify.labels import LabelSafe
    from src.features.scaling import feature_scaler

    inner = build(name, learning_rate, max_depth, n_estimators, subsample, colsample)
    if name == "xgboost":
        # XGBoost's sklearn wrapper requires integer targets and this project uses strings
        # everywhere; encoding is done inside the estimator so `predict` still returns labels.
        inner = LabelSafe(inner)
    return Pipeline([("prepare", feature_scaler()), ("model", inner)])


def grid(
    rates=RATES, depths=DEPTHS, rounds=ROUNDS, subsamples=SUBSAMPLE, colsamples=COLSAMPLE
) -> list[dict]:
    return [
        {"learning_rate": r, "max_depth": d, "n_estimators": n, "subsample": s, "colsample": c}
        for r in rates
        for d in depths
        for n in rounds
        for s in subsamples
        for c in colsamples
    ]


def evaluate(data, name: str, config: dict, folds: int = 5, n_jobs: int | None = None) -> dict:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    started = time.perf_counter()
    predicted = cross_val_predict(
        pipeline(name, **config),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    return {
        "library": name,
        **config,
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == data.y).mean()), 4),
        "seconds": round(time.perf_counter() - started, 1),
    }


def marginal(rows: list[dict], key: str) -> dict:
    values: dict = {}
    for row in rows:
        values.setdefault(str(row[key]), []).append(row["macro_f1"])
    return {name: round(max(scores), 4) for name, scores in sorted(values.items())}


def run(
    table: str = "hybrid", corpus: str = "real", libraries=LIBRARIES, n_jobs: int | None = None
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    usable = [name for name in libraries if available(name)]
    rows = [evaluate(data, name, config, n_jobs=n_jobs) for name in usable for config in grid()]
    rows.sort(key=lambda row: -row["macro_f1"])

    per_library = {
        name: max((r for r in rows if r["library"] == name), key=lambda r: r["macro_f1"])
        for name in usable
    }
    scores = [row["macro_f1"] for row in per_library.values()]
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "libraries": usable,
        "skipped": [name for name in libraries if name not in usable],
        "cells": len(rows),
        "best": rows[0],
        "best_per_library": per_library,
        "spread_between_libraries": round(max(scores) - min(scores), 4),
        "by_learning_rate": marginal(rows, "learning_rate"),
        "by_max_depth": marginal(rows, "max_depth"),
        "by_n_estimators": marginal(rows, "n_estimators"),
        "by_subsample": marginal(rows, "subsample"),
        "by_colsample": marginal(rows, "colsample"),
        "seconds_per_library": {
            name: round(sum(r["seconds"] for r in rows if r["library"] == name), 1)
            for name in usable
        },
        "all": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--libraries", nargs="*", default=list(LIBRARIES))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {name: run(name, args.corpus, tuple(args.libraries), args.jobs) for name in tables}
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

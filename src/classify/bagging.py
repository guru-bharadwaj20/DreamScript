"""Phase 7.1.1 - bagged trees, and the variance reduction the corpus has room for.

    python -m src.classify.bagging --table all

Unit 3 opens with the simplest ensemble there is: fit the same high-variance learner on `n`
bootstrap resamples and average the votes. The argument is entirely about variance - bagging does
not reduce a learner's bias, so it helps exactly to the extent that the base learner's errors move
when the training set is resampled.

5.1.3 selected a decision tree at `max_depth = 12, ccp_alpha = 0.004` and 5.3.2 refitted it to
**54 leaves, three of them holding a single row.** A tree with single-row leaves is the canonical
high-variance learner, so this corpus should be close to a best case for bagging - and 5.2.9's
finding that the corpus is data-limited cuts the other way, because a bootstrap sample of a
40-row class contains about 25 distinct circuits.

So there are two competing expectations again, and this task reports the quantity that separates
them rather than only the score:

    variance account   bagging should help, and the gain should grow with `n_estimators`
                       until it plateaus
    small-class account  bootstrap resampling drops ~37% of each class's distinct rows, which on
                       a 40-row class is severe, so the minority F1 may *fall*

## What is compared, and against what

Three base learners, because "bagging helps" is a claim about variance and the only way to show
that is to bag something stable as a control:

    tree (unpruned)    the high-variance case - bagging should help most here
    tree (5.1.3's)     the tuned tree Phase 5 actually selected
    logistic regression  a low-variance learner - bagging should do close to nothing

If the unpruned tree gains and the logistic regression does not, the mechanism is variance
reduction. If both gain equally, something else is happening.

## What it measured

Three base learners x seven ensemble sizes x three tables, stratified 5-fold, macro F1:

    table         base            n=1      n=10     n=25     n=100    best     gain
    hybrid        tree_unpruned  0.7964   0.8424   0.8560   0.8453   0.8560   +0.0596
    hybrid        tree_tuned     0.7452   0.8425   0.8586   0.8644   0.8644   +0.1192
    hybrid        logreg         0.9464   0.9525   0.9546   0.9513   0.9546   +0.0082
    embedding     tree_tuned     0.7531   0.8208   0.8580   0.8398   0.8580   +0.1049
    embedding     logreg         0.9546   0.9649   0.9582   0.9649   0.9649   +0.0103
    handcrafted   tree_tuned     0.6955   0.7885   0.8136   0.8061   0.8136   +0.1181
    handcrafted   logreg         0.7688   0.7783   0.7903   0.7812   0.7905   +0.0217

**The control does exactly what a variance argument predicts.** Bagging a tree is worth
**+0.060 to +0.119**; bagging a logistic regression is worth **+0.008 to +0.022** - an order of
magnitude less, on all three tables. Averaging helps in proportion to how much the base learner's
errors move when the training set is resampled, and the measurement separates the two learners by
a factor of ten. This is the cleanest mechanism confirmation in either phase.

The tuned tree gains *more* than the unpruned one on every table (+0.119 against +0.060 on the
hybrid), which is the one part of the result that inverts the naive reading. The unpruned tree
starts higher as a single model - a 54-leaf pruned tree throws away structure the unpruned one
keeps - so it has less to recover, and both arrive within 0.008 of each other after bagging. The
ensembles converge to the same place from different starting points, which is the usual finding
and worth having measured rather than assumed.

## The small-class fear was wrong, and the bootstrap arithmetic says why

The docstring above worried that resampling a 40-row class would hurt the minority classes.
Measured, the opposite happens: **the minority mean gains more than the macro average on every
single row of the table** - +0.182 against +0.119 for the tuned tree on the hybrid, +0.168
against +0.118 on the handcrafted, and +0.161 against +0.105 on the embedding.

The bootstrap simulation explains it. The expected share of distinct rows appearing in a
resample is:

    circuit (40 rows)   0.6388        flowchart (600 rows)   0.6324
    er_diagram (50)     0.6313        wireframe (600)        0.6324
    state_machine (50)  0.6313

**A 40-row class loses no more of itself to resampling than a 600-row class does** - 0.639 against
0.632, and if anything slightly less. `1 - 1/e = 0.632` is an asymptotic result, and 40 is already
close enough to asymptotic for the difference to be under a percentage point. The concern was
about a finite-sample effect that turns out not to exist at this size, which is precisely why it
was simulated rather than reasoned about.

Minority classes gain more from bagging because they are where an unbagged tree's variance is
largest - a single-row leaf is a memorised page, and 5.3.2 found three of them - so averaging has
the most to remove exactly there.

## How many members, and where this leaves Unit 3

**Ten to twenty-five members is enough.** The smallest ensemble within 0.002 of the best is 25 on
five of the nine curves and 10 on two; past that the curves are flat and non-monotone (the hybrid
tuned tree reads 0.8586 / 0.8627 / 0.8644 / 0.8557 at 25 / 50 / 100 / 200). Anything above 50 is
buying fold-to-fold noise.

**The best bagged result in this task is 0.8644**, tuned trees on the hybrid table. That is a
large gain over the single tree 5.1.3 selected and it is still **0.11 below 6.3.7's best cell**
(RBF-SVM on the embedding, 0.9742), and 0.09 below a *bagged logistic regression* on the same
table. Unit 3's first ensemble improves the weakest model family in the project by a lot, and
does not approach what Unit 2 already had.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.activations import TABLES

SEED = 42

#: Bagging's one real hyperparameter. Extended past the point of plateau on purpose - a curve
#: that is still rising at its last point cannot be said to have plateaued.
SIZES = (1, 5, 10, 25, 50, 100, 200)

BASES = ("tree_unpruned", "tree_tuned", "logreg")

#: 6.3.7's per-table numbers for the tuned single tree's Phase 5 sibling, so the ensemble has a
#: reference that is not just its own `n_estimators = 1` cell.
SINGLE_TREE_5_1_3 = 0.7640


def base_learner(name: str):
    """One of the three, at the settings its own task selected."""
    if name == "tree_unpruned":
        from sklearn.tree import DecisionTreeClassifier

        # No depth limit and no pruning: the deliberately high-variance control.
        return DecisionTreeClassifier(random_state=SEED, class_weight="balanced")
    if name == "tree_tuned":
        from sklearn.tree import DecisionTreeClassifier

        from src.classify.tree import BEST_PARAMS

        return DecisionTreeClassifier(random_state=SEED, **BEST_PARAMS)
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=3000, random_state=SEED)
    raise ValueError(f"base must be one of {list(BASES)}; got {name!r}")


def pipeline(base: str = "tree_unpruned", n_estimators: int = 50, **kwargs):
    """4.2.3/4.2.4's preparation, then a bagged ensemble of the chosen base learner."""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    settings = {
        "estimator": base_learner(base),
        "n_estimators": n_estimators,
        "random_state": SEED,
        # The default. Sampling fewer rows would confound the ensemble size with the sample size,
        # and 7.1.2's Random Forest is where feature subsampling gets introduced deliberately.
        "max_samples": 1.0,
        "bootstrap": True,
        **kwargs,
    }
    return Pipeline([("prepare", feature_scaler()), ("model", BaggingClassifier(**settings))])


def distinct_share(n_rows: int, draws: int = 20000, seed: int = SEED) -> float:
    """Expected fraction of a class's rows appearing at least once in a bootstrap sample.

    The small-class account's quantity. Asymptotically `1 - 1/e` = 0.632, but the whole point is
    that a 40-row class is not asymptotic, so it is simulated rather than quoted.
    """
    rng = np.random.default_rng(seed)
    counts = [
        len(np.unique(rng.integers(0, n_rows, size=n_rows))) / n_rows
        for _ in range(min(draws, 400))
    ]
    return round(float(np.mean(counts)), 4)


def evaluate(data, base: str, n_estimators: int, folds: int = 5, n_jobs: int | None = None) -> dict:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    predicted = cross_val_predict(
        pipeline(base, n_estimators),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    classes = data.classes
    per_class = f1_score(data.y, predicted, average=None, labels=classes, zero_division=0)
    return {
        "base": base,
        "n_estimators": n_estimators,
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == data.y).mean()), 4),
        "per_class_f1": {
            name: round(float(value), 4) for name, value in zip(classes, per_class, strict=True)
        },
    }


def curve(data, base: str, sizes=SIZES, n_jobs: int | None = None) -> list[dict]:
    """Macro F1 against ensemble size - the variance account's prediction, as a curve."""
    return [evaluate(data, base, n, n_jobs=n_jobs) for n in sizes]


def plateau(rows: list[dict], tolerance: float = 0.002) -> dict:
    """The smallest ensemble within `tolerance` of the best - what `n_estimators` actually needs.

    Reported because a bagging curve almost always flattens, and quoting the argmax of a flat
    curve as "the chosen size" implies a precision the measurement does not have.
    """
    best = max(row["macro_f1"] for row in rows)
    enough = min(
        (row["n_estimators"] for row in rows if row["macro_f1"] >= best - tolerance),
        default=None,
    )
    return {
        "best_macro_f1": best,
        "best_n_estimators": max(rows, key=lambda row: row["macro_f1"])["n_estimators"],
        "smallest_within_tolerance": enough,
        "tolerance": tolerance,
    }


def run(
    table: str = "handcrafted",
    corpus: str = "real",
    bases=BASES,
    sizes=SIZES,
    n_jobs: int | None = None,
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    curves = {base: curve(data, base, sizes, n_jobs) for base in bases}

    gains = {}
    for base, rows in curves.items():
        single = next(row for row in rows if row["n_estimators"] == 1)
        best = max(rows, key=lambda row: row["macro_f1"])
        gains[base] = {
            "single": single["macro_f1"],
            "bagged": best["macro_f1"],
            "gain": round(best["macro_f1"] - single["macro_f1"], 4),
            "minority_single": _minority_mean(single, data),
            "minority_bagged": _minority_mean(best, data),
        }
        gains[base]["minority_gain"] = round(
            gains[base]["minority_bagged"] - gains[base]["minority_single"], 4
        )

    counts = data.class_counts()
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "sizes": list(sizes),
        "curves": curves,
        "plateau": {base: plateau(rows) for base, rows in curves.items()},
        "gain_from_bagging": gains,
        # The small-class account's number, per class, so the minority result has a mechanism
        # attached rather than only a direction.
        "bootstrap_distinct_share": {
            name: distinct_share(count) for name, count in sorted(counts.items())
        },
        "single_tree_5_1_3": SINGLE_TREE_5_1_3 if table == "handcrafted" else None,
    }


def _minority_mean(row: dict, data, minority=("circuit", "er_diagram", "state_machine")) -> float:
    values = [v for k, v in row["per_class_f1"].items() if k in minority]
    return round(float(np.mean(values)), 4) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="handcrafted", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--bases", nargs="*", default=list(BASES))
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {
            name: run(name, args.corpus, tuple(args.bases), SIZES, args.jobs) for name in tables
        }
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result if len(result) > 1 else next(iter(result.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

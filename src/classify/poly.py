"""Phase 6.3.2 - the polynomial kernel, on the boundary the plan named and the one that is hard.

    python -m src.classify.poly              # both binary studies plus the multiclass grid
    python -m src.classify.poly --pair circuit flowchart

The plan asks for degree {2, 3, 4}, `coef0` and `gamma` swept, and says to **target the flowchart
vs. state-diagram boundary**. That instruction is followed and it is also checked, because 5.3.4
already looked:

> The task line reads *expect flowchart <-> state_machine*. **It does not hold.** The largest
> confusion is `circuit -> flowchart`, and the predicted pair appears at no rank at all inside
> the top ten.

5.3.4 measured zero state_machine pages missed by all three Phase 5 models, against 19 of 40
circuit pages missed by all three. So the plan's named boundary is the one place in this corpus
where there is nothing to study, and running only that study would produce a table of 1.000s and
call it a success.

Both are therefore run. **`flowchart | state_machine` is the plan's request, delivered and
reported as specified.** **`circuit | flowchart` is the same study on the pair 5.3.4 identified**,
and it is where the kernel has anything to do: 64 of the 120 pooled confusions in Phase 5 are
circuit pages called flowcharts, which is 53% of the entire circuit class.

## What a polynomial kernel is doing here

`K(x, z) = (gamma * <x, z> + coef0) ^ degree` computes an inner product in the space of all
monomials up to `degree` without ever forming it - for the 161-column hybrid table at degree 3
that space has about 700,000 dimensions. `coef0` is the part usually left at its default and it
is not cosmetic: at `coef0 = 0` the kernel is *homogeneous* and only the degree-`d` monomials
appear, so a degree-3 kernel cannot represent a quadratic boundary at all. It is swept for that
reason and not for completeness.

## What it measured

144 cells (3 degrees x 4 gammas x 3 coef0 x 4 C) per study, hybrid table, stratified 5-fold.

### The plan's pair: flowchart | state_machine, 650 rows

    best cell        degree 2, gamma scale, coef0 0, C 1     macro F1  1.0000
    linear SVM (6.3.1) on the same two classes                         1.0000
    best at every degree / every coef0 / every gamma / every C         1.0000

**Every non-degenerate cell in the grid scores exactly 1.0000, and so does the linear SVM.**
There is no boundary here to target. 5.3.4 predicted this from Phase 5's confusion matrices -
zero state_machine pages missed by all three models - and the kernel study confirms it in the
strongest available form: 600 flowcharts and 50 state machines are linearly separable in the
hybrid feature space, and no amount of kernel machinery can improve on a perfect score. The
plan's instruction is carried out and its premise is refuted by carrying it out.

(The grid's worst cell here is 0.4800, which is a degenerate kernel predicting one class, not a
hard boundary - see the coef0 result below.)

### 5.3.4's pair: circuit | flowchart, 640 rows

    best cell        degree 2, gamma scale, coef0 10, C 0.1  macro F1  0.9346
    linear SVM (6.3.1) on the same two classes                         0.8908
    kernel - linear                                                   +0.0438

**This is where the kernel earns its keep, and it is the largest model-family gain in Phase 6.**
+0.0438 on the pair that produces 64 of Phase 5's 120 pooled confusions - 53% of the entire
circuit class - against +0.0000 on the pair the plan named. A study run only where the plan
pointed would have reported a tie and learned nothing; run where 5.3.4 pointed, it finds the one
place in this corpus where a nonlinear decision surface is worth having.

## The finding: coef0 is the hyperparameter, not degree

    circuit | flowchart      coef0 0 -> 0.7487    coef0 1 -> 0.9264    coef0 10 -> 0.9346
    five-class multiclass    coef0 0 -> 0.8875    coef0 1 -> 0.9582    coef0 10 -> 0.9690

    circuit | flowchart      degree 2 -> 0.9346   degree 3 -> 0.9346   degree 4 -> 0.9264
    five-class multiclass    degree 2 -> 0.9690   degree 3 -> 0.9690   degree 4 -> 0.9690

**`coef0` is worth 0.186 on the binary study and 0.082 on the multiclass one. Degree is worth
0.008 and 0.000.** The plan names degree {2, 3, 4} as the thing to sweep and leaves coef0 as an
afterthought; the measurement inverts that completely, and **0.186 is the largest single
hyperparameter effect anywhere in Phase 6** - larger than the entire activation question (0.026),
the entire optimizer question (0.017), or the entire regularization grid (0.009).

The mechanism is the one stated above the fold rather than found afterwards: at `coef0 = 0` the
kernel is homogeneous, `(gamma * <x,z>)^d` contains only degree-`d` monomials, and a degree-3
kernel cannot represent a quadratic - or a linear - boundary at all. Raising coef0 puts the whole
polynomial back in the expansion. sklearn's default is `coef0 = 0.0`, so **the default polynomial
kernel is the degenerate one**, and a sweep over degree alone at the default would have concluded
that polynomial kernels are bad on this corpus.

Degree not mattering once coef0 is nonzero is the complement of the same fact: the useful
structure is quadratic, and degrees 3 and 4 contain the quadratic terms and simply do not need
the rest.

## Against the phase

The five-class polynomial kernel reaches **0.9690 (degree 2, gamma 1e-3, coef0 10, C 10) against
6.3.1's linear 0.9628** - a gain of 0.0062, and the best five-class number in Phase 6 so far. C
is flat across all four values, which is 6.3.1's plateau again.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.activations import TABLES

SEED = 42

#: The plan's three.
DEGREES = (2, 3, 4)

#: `scale` is sklearn's default (1 / (n_features * X.var())); the numbers bracket it. A fixed
#: gamma on a 33-column table and a 161-column one are different kernels, which is why the
#: adaptive default is in the grid rather than replaced by it.
GAMMAS = ("scale", 0.001, 0.01, 0.1)

#: Zero is the homogeneous kernel - only the top-degree monomials - and it is in the grid because
#: it is the one value that changes what the kernel can represent rather than how sharply.
COEF0S = (0.0, 1.0, 10.0)

CS = (0.1, 1.0, 10.0, 100.0)

#: The plan's target, and 5.3.4's. Both are studied; see the module docstring.
PLAN_PAIR = ("flowchart", "state_machine")
HARD_PAIR = ("circuit", "flowchart")


def pipeline(degree: int = 3, gamma="scale", coef0: float = 0.0, C: float = 1.0, **kwargs):
    """4.2.3's imputation and 4.2.4's scaling, then a polynomial-kernel SVC.

    Scaling matters more for a polynomial kernel than for any other model in this phase: the
    kernel raises an inner product to the third or fourth power, so a column ten times larger
    than its neighbours is a thousand or ten thousand times larger inside the kernel.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from src.features.scaling import feature_scaler

    settings = {
        "kernel": "poly",
        "degree": degree,
        "gamma": gamma,
        "coef0": coef0,
        "C": C,
        "random_state": SEED,
        # A degree-4 kernel with a large coef0 produces enormous kernel values and libsvm can
        # grind; the cap turns "this cell never returns" into "this cell is reported as slow".
        "max_iter": 5_000_000,
        **kwargs,
    }
    return Pipeline([("prepare", feature_scaler()), ("model", SVC(**settings))])


def binary_subset(data, pair: tuple[str, str]):
    """The two named classes only, as a Dataset - the plan's "focused binary study"."""
    from src.classify.data import Dataset

    keep = np.isin(data.y, list(pair))
    if not keep.any():
        raise KeyError(f"neither of {pair} is present in {data.corpus}")
    return Dataset(
        X=data.X[keep],
        y=data.y[keep],
        groups=data.groups[keep],
        ids=data.ids[keep],
        feature_names=list(data.feature_names),
        corpus=f"{data.corpus}:{pair[0]}|{pair[1]}",
        sources=data.sources[keep] if len(data.sources) else data.sources,
    )


def grid(degrees=DEGREES, gammas=GAMMAS, coef0s=COEF0S, cs=CS) -> list[dict]:
    return [
        {"degree": degree, "gamma": gamma, "coef0": coef0, "C": C}
        for degree in degrees
        for gamma in gammas
        for coef0 in coef0s
        for C in cs
    ]


def score_grid(data, configs, folds: int = 5, n_jobs: int | None = None) -> list[dict]:
    """Every cell under 5.2.1's fold protocol, macro F1."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    splitter = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows = []
    for config in configs:
        predicted = cross_val_predict(
            pipeline(**config),
            data.X,
            data.y,
            cv=splitter,
            n_jobs=n_jobs if n_jobs is not None else -1,
        )
        rows.append(
            {
                **config,
                "macro_f1": round(
                    float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4
                ),
                "accuracy": round(float((predicted == data.y).mean()), 4),
            }
        )
    return sorted(rows, key=lambda row: -row["macro_f1"])


def marginal(rows: list[dict], key: str) -> dict:
    """Best score at each level of one kernel parameter, holding nothing else fixed.

    The best of each level rather than the mean: a polynomial grid contains genuinely broken
    cells - degree 4 at gamma 0.1 with C 100 is a kernel that overflows into a constant
    prediction - and a mean over them measures how many broken cells the grid happens to contain
    rather than what the parameter is worth.
    """
    levels: dict = {}
    for row in rows:
        levels.setdefault(str(row[key]), []).append(row["macro_f1"])
    return {name: round(max(values), 4) for name, values in sorted(levels.items())}


def study(data, pair: tuple[str, str], configs=None, n_jobs: int | None = None) -> dict:
    """One focused binary study: the grid, the marginals, and the linear baseline it must beat."""
    from src.classify.svm import BEST_PARAMS
    from src.classify.svm import pipeline as linear_pipeline

    subset = binary_subset(data, pair)
    rows = score_grid(subset, configs or grid(), n_jobs=n_jobs)

    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    linear = cross_val_predict(
        linear_pipeline(**BEST_PARAMS),
        subset.X,
        subset.y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    linear_score = round(float(f1_score(subset.y, linear, average="macro", zero_division=0)), 4)

    return {
        "pair": list(pair),
        "rows": int(len(subset.y)),
        "class_counts": subset.class_counts(),
        "cells": len(rows),
        "best": rows[0],
        "top5": rows[:5],
        "worst": rows[-1],
        "linear_svm_6_3_1": linear_score,
        "kernel_minus_linear": round(rows[0]["macro_f1"] - linear_score, 4),
        "by_degree": marginal(rows, "degree"),
        "by_coef0": marginal(rows, "coef0"),
        "by_gamma": marginal(rows, "gamma"),
        "by_C": marginal(rows, "C"),
        "all": rows,
    }


def run(
    table: str = "hybrid",
    corpus: str = "real",
    pairs=(PLAN_PAIR, HARD_PAIR),
    n_jobs: int | None = None,
    multiclass: bool = True,
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    result = {
        "table": table,
        "corpus": corpus,
        "studies": {f"{a}|{b}": study(data, (a, b), n_jobs=n_jobs) for a, b in pairs},
    }
    if multiclass:
        # The five-class grid, so the kernel has a number comparable with 6.3.1's 0.9628 rather
        # than only with itself on two classes.
        rows = score_grid(data, grid(), n_jobs=n_jobs)
        result["multiclass"] = {
            "cells": len(rows),
            "best": rows[0],
            "top5": rows[:5],
            "by_degree": marginal(rows, "degree"),
            "by_coef0": marginal(rows, "coef0"),
            "linear_svm_6_3_1": {"hybrid": 0.9628, "embedding": 0.9691, "handcrafted": 0.7656}.get(
                table
            ),
            "all": rows,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=list(TABLES))
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--pair", nargs=2, action="append", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-multiclass", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)

    pairs = [tuple(p) for p in args.pair] if args.pair else (PLAN_PAIR, HARD_PAIR)
    try:
        result = run(args.table, args.corpus, pairs, args.jobs, not args.no_multiclass)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if not args.full:
        for value in result["studies"].values():
            value.pop("all", None)
        if "multiclass" in result:
            result["multiclass"].pop("all", None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

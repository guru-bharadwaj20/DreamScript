"""Phase 6.3.4 - one-vs-one against one-vs-rest, and the parameter that does not choose between them.

    python -m src.classify.multiclass --table all

The plan asks for an OvO / OvR comparison, chosen and justified. There is a trap in the way
sklearn spells this and it is worth stating before the numbers, because falling into it produces
a table where the two strategies are identical and the conclusion is that the choice does not
matter.

**`SVC(decision_function_shape="ovr")` does not train a one-vs-rest classifier.** `SVC` is always
one-vs-one internally - it fits `k(k-1)/2` binary problems, full stop - and
`decision_function_shape` only controls whether the `decision_function` output is *reshaped* from
10 columns to 5 by voting. The fitted model, its support vectors and its predictions are
identical either way. A comparison built on that parameter compares nothing.

A real one-vs-rest SVM requires wrapping the estimator: `OneVsRestClassifier(SVC(...))`, which
fits `k` binary problems each against the pooled remainder. That is what is compared here, with
`OneVsOneClassifier(SVC(...))` as the explicit OvO counterpart so both arms go through the same
meta-estimator machinery.

## Why the choice has teeth on this corpus specifically

The class counts are 600 / 600 / 50 / 50 / 40, and the two schemes see that imbalance completely
differently.

**OvO** trains `circuit vs er_diagram` on 40 + 50 = 90 rows - balanced, tiny, and never sees a
flowchart. Each of its 10 problems is roughly balanced, which is the standard argument in its
favour, and each is trained on a fraction of the corpus.

**OvR** trains `circuit vs everything` on 40 against 1,300 - a 1:32 imbalance, on the class that
5.3.4 already identified as the one every Phase 5 model loses. This is the standard argument
against OvR, and this corpus is close to a worst case for it.

So the prediction going in is that **OvO should win, and the gap should live almost entirely in
the minority classes.** Per-class F1 is therefore reported, not just the macro average, because
that is what makes the prediction checkable rather than merely confirmed.

## What it measured

Three kernels x two schemes x three tables, stratified 5-fold, each kernel at the
hyperparameters 6.3.1-6.3.3 selected for it:

    table          kernel    OvO      OvR     OvO - OvR    minority gap   majority gap
    hybrid         linear   0.9658   0.9581    +0.0077        +0.0126        +0.0004
    hybrid         rbf      0.9658   0.9613    +0.0045        +0.0076        +0.0000
    hybrid         poly     0.9658   0.9581    +0.0077        +0.0126        +0.0004
    embedding      linear   0.9637   0.9597    +0.0040        +0.0063        +0.0004
    embedding      rbf      0.9742   0.9686    +0.0056        +0.0088        +0.0009
    embedding      poly     0.9661   0.9652    +0.0009        +0.0015        +0.0000
    handcrafted    linear   0.7946   0.7477    +0.0469        +0.0755        +0.0039
    handcrafted    rbf      0.8227   0.7912    +0.0315        +0.0535        -0.0014
    handcrafted    poly     0.7894   0.7616    +0.0278        +0.0467        -0.0005

**One-vs-one wins all nine comparisons, and the prediction about *where* it wins is confirmed as
sharply as the ranking.** The minority gap exceeds the majority gap in every single row, and on
the handcrafted table it does so by a factor of nineteen (+0.0755 against +0.0039). The two
schemes are essentially identical on the 600-row classes - flowchart and wireframe move by 0.001
or less almost everywhere - and differ only where the class is small.

The `circuit` column is the whole effect in one number:

    handcrafted, linear kernel      circuit F1   OvO 0.217    OvR 0.093
    handcrafted, rbf kernel         circuit F1   OvO 0.387    OvR 0.214
    hybrid, rbf kernel              circuit F1   OvO 0.880    OvR 0.857

**OvO more than doubles the F1 of the 40-row class on the handcrafted table.** That is the
mechanism stated above the fold, measured: OvR asks `circuit vs everything`, a 1:32 imbalance on
the class every Phase 5 model already lost, while OvO's hardest circuit problem is 40 against 50.

The effect is largest exactly where it should be. On the handcrafted table - where 6.3.3 found
the classes are not close to separable - the scheme is worth 0.03-0.05. On the two learned tables,
where almost everything is separable anyway, it is worth 0.001-0.008. **The multiclass scheme
matters in proportion to how hard the problem is**, which is the reason to justify the choice on
this corpus rather than adopt a default.

## The trap, measured

    identical predictions        True
    rows that differ             0 of 1,340
    support vectors, "ovo" shape 363
    support vectors, "ovr" shape 363

`SVC(decision_function_shape="ovr")` and `SVC(decision_function_shape="ovo")` produce **the same
fitted model, the same 363 support vectors, and the same prediction on all 1,340 rows.** The
parameter reshapes `decision_function`'s output from 10 columns to 5 by voting and does nothing
else; `SVC` is one-vs-one always. A comparison built on that parameter - which is the natural
reading of the sklearn API - would have reported a difference of exactly zero and concluded that
the multiclass scheme is irrelevant. It is not; it is worth up to 0.047 macro F1 and up to 0.124
on the smallest class, and reaching that required `OneVsRestClassifier` around the estimator.

## What is chosen, and what it costs

**One-vs-one**, on every table and every kernel. The cost is the `k(k-1)/2 = 10` binary fits
against OvR's 5, but each OvO problem sees only two classes: at these class counts the ten OvO
problems total 5,360 training rows against OvR's 6,700, so **OvO is fitting twice as many models
on less data overall**, and the measured wall-clock difference is 1.4 s against 0.9 s at worst.
For a five-class problem the quadratic term is not yet a reason to prefer OvR; at twenty classes
it would be 190 fits against 20, and this recommendation would need revisiting.

One curiosity worth recording: on the hybrid table all three kernels score **exactly 0.9658**
under OvO. Given 6.3.1-6.3.3's finding that the learned tables are nearly linearly separable,
three differently-shaped boundaries converging on the same 1,340 decisions is what that
separability looks like from the inside.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from src.classify.activations import TABLES

SEED = 42

STRATEGIES = ("ovo", "ovr")

#: The kernels to compare the schemes under. The strategy could interact with the kernel - a
#: linear model's OvR problems are harder than an RBF's - so it is not assumed away.
KERNELS = ("linear", "rbf", "poly")


def base_estimator(kernel: str = "rbf", table: str = "hybrid"):
    """One binary SVC at the settings 6.3.1-6.3.3 selected for this table.

    Each kernel is given the hyperparameters its own task chose rather than a shared default,
    because otherwise this task would be comparing tuning effort across kernels while claiming to
    compare multiclass schemes.
    """
    from sklearn.svm import SVC

    if kernel == "linear":
        from src.classify.svm import BEST_PARAMS

        return SVC(kernel="linear", C=BEST_PARAMS["C"], random_state=SEED)
    if kernel == "rbf":
        gamma = {"hybrid": 1e-4, "embedding": 1e-5, "handcrafted": "scale"}.get(table, "scale")
        C = {"hybrid": 1000.0, "embedding": 1000.0, "handcrafted": 10.0}.get(table, 1.0)
        return SVC(kernel="rbf", gamma=gamma, C=C, random_state=SEED)
    if kernel == "poly":
        return SVC(kernel="poly", degree=2, gamma=1e-3, coef0=10.0, C=10.0, random_state=SEED)
    raise ValueError(f"kernel must be one of {list(KERNELS)}; got {kernel!r}")


def pipeline(
    strategy: str = "ovo", kernel: str = "rbf", table: str = "hybrid", n_jobs: int | None = None
):
    """Scaler, then the chosen meta-estimator around a binary SVC.

    Both arms go through a meta-estimator - `OneVsOneClassifier` rather than bare `SVC` - so the
    comparison is between the two schemes and not between a wrapper and its absence.
    """
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    inner = base_estimator(kernel, table)
    if strategy == "ovo":
        meta = OneVsOneClassifier(inner, n_jobs=n_jobs)
    elif strategy == "ovr":
        meta = OneVsRestClassifier(inner, n_jobs=n_jobs)
    else:
        raise ValueError(f"strategy must be one of {list(STRATEGIES)}; got {strategy!r}")
    return Pipeline([("prepare", feature_scaler()), ("model", meta)])


def binary_problems(strategy: str, n_classes: int) -> int:
    """How many binary fits the scheme costs - `k(k-1)/2` against `k`.

    The number the plan's "justified" has to weigh against the score: at five classes OvO fits
    ten models and OvR fits five, but OvO's are on small subsets and OvR's are on everything.
    """
    if strategy == "ovo":
        return n_classes * (n_classes - 1) // 2
    return n_classes


def evaluate(
    data, strategy: str, kernel: str, table: str, folds: int = 5, n_jobs: int | None = None
) -> dict:
    """One scheme under 5.2.1's fold protocol, with per-class F1 and wall-clock cost."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    started = time.perf_counter()
    predicted = cross_val_predict(
        pipeline(strategy, kernel, table),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    elapsed = time.perf_counter() - started

    classes = data.classes
    per_class = f1_score(data.y, predicted, average=None, labels=classes, zero_division=0)
    return {
        "strategy": strategy,
        "kernel": kernel,
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == data.y).mean()), 4),
        "per_class_f1": {
            name: round(float(value), 4) for name, value in zip(classes, per_class, strict=True)
        },
        "binary_problems": binary_problems(strategy, len(classes)),
        "seconds": round(elapsed, 1),
    }


def compare(data, table: str, kernels=KERNELS, n_jobs: int | None = None) -> list[dict]:
    return [
        evaluate(data, strategy, kernel, table, n_jobs=n_jobs)
        for kernel in kernels
        for strategy in STRATEGIES
    ]


def minority_gap(rows: list[dict], minority=("circuit", "er_diagram", "state_machine")) -> dict:
    """Where the two schemes differ, split by class size - the prediction this task made.

    A macro average over five classes hides which three moved. If OvO's advantage is the minority
    classes, this is where it shows; if the two differ uniformly, the explanation offered above
    is wrong even if the ranking is right.
    """
    result = {}
    for kernel in {row["kernel"] for row in rows}:
        by_strategy = {row["strategy"]: row for row in rows if row["kernel"] == kernel}
        if set(by_strategy) != set(STRATEGIES):
            continue
        ovo, ovr = by_strategy["ovo"], by_strategy["ovr"]
        small = [name for name in ovo["per_class_f1"] if name in minority]
        large = [name for name in ovo["per_class_f1"] if name not in minority]
        result[kernel] = {
            "macro_f1_ovo_minus_ovr": round(ovo["macro_f1"] - ovr["macro_f1"], 4),
            "minority_mean_ovo": round(float(np.mean([ovo["per_class_f1"][n] for n in small])), 4),
            "minority_mean_ovr": round(float(np.mean([ovr["per_class_f1"][n] for n in small])), 4),
            "majority_mean_ovo": round(float(np.mean([ovo["per_class_f1"][n] for n in large])), 4),
            "majority_mean_ovr": round(float(np.mean([ovr["per_class_f1"][n] for n in large])), 4),
        }
        result[kernel]["minority_gap"] = round(
            result[kernel]["minority_mean_ovo"] - result[kernel]["minority_mean_ovr"], 4
        )
        result[kernel]["majority_gap"] = round(
            result[kernel]["majority_mean_ovo"] - result[kernel]["majority_mean_ovr"], 4
        )
    return result


def decision_shape_is_not_a_strategy(data, kernel: str = "rbf", table: str = "hybrid") -> dict:
    """The trap, measured: `decision_function_shape` changes nothing about the fitted model.

    Reported as a number rather than as a warning, because "this parameter is not what you think"
    is exactly the kind of claim that should be demonstrated on the corpus in front of it.
    """
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    predictions = {}
    supports = {}
    for shape in ("ovo", "ovr"):
        estimator = base_estimator(kernel, table)
        estimator.set_params(decision_function_shape=shape)
        fitted = Pipeline([("prepare", feature_scaler()), ("model", estimator)]).fit(data.X, data.y)
        predictions[shape] = fitted.predict(data.X)
        supports[shape] = int(fitted.named_steps["model"].n_support_.sum())

    return {
        "identical_predictions": bool(np.array_equal(predictions["ovo"], predictions["ovr"])),
        "rows_that_differ": int((predictions["ovo"] != predictions["ovr"]).sum()),
        "support_vectors_ovo_shape": supports["ovo"],
        "support_vectors_ovr_shape": supports["ovr"],
        "note": "SVC is always one-vs-one; decision_function_shape only reshapes the output",
    }


def run(
    table: str = "hybrid", corpus: str = "real", kernels=KERNELS, n_jobs: int | None = None
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    rows = compare(data, table, kernels, n_jobs)
    ranked = sorted(rows, key=lambda row: -row["macro_f1"])
    wins = {
        kernel: max(
            (row for row in rows if row["kernel"] == kernel), key=lambda row: row["macro_f1"]
        )["strategy"]
        for kernel in kernels
    }
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "class_counts": data.class_counts(),
        "results": rows,
        "best": ranked[0],
        "winner_per_kernel": wins,
        "minority_analysis": minority_gap(rows),
        "decision_function_shape_check": decision_shape_is_not_a_strategy(data, "rbf", table),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--kernels", nargs="*", default=list(KERNELS))
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {name: run(name, args.corpus, tuple(args.kernels), args.jobs) for name in tables}
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result if len(result) > 1 else next(iter(result.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

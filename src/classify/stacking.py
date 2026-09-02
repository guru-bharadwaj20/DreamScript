"""Phase 7.1.7 - RF + SVM + MLP under a logistic meta-learner, and whether it beats its best base.

    python -m src.classify.stacking --table all

The plan asks for a stack of Random Forest, SVM and MLP with a logistic meta-learner, and says
the deliverable is "**meta-model beats best base, or documented as not**". That phrasing is the
right one and this task takes it literally: the interesting outcome is as likely to be the
negative, and 6.3.7 already explains why.

Stacking works when the base models make **different** mistakes. 6.3.7 measured the opposite on
the two learned tables: six models spanning a logistic regression and a 228,357-parameter network
landed inside 0.012 of each other on the hybrid table, and 6.3.6 found three kernels partitioning
a projection of it almost identically. Models that agree have nothing to combine.

So the quantity this task reports beside the score is **base-model disagreement** - the share of
rows on which the three bases predict different things, and the oracle score a perfect combiner
would reach. The oracle is the ceiling: if it is barely above the best single model, no
meta-learner of any kind can help, and that is a stronger statement than the stack's own number.

## Making the stack honest

`StackingClassifier` fits the meta-learner on **cross-validated** base predictions rather than
in-sample ones. That is not a detail: an RF's in-sample predictions are near-perfect, so a
meta-learner trained on them would learn "trust the forest" and collapse. sklearn does the right
thing by default with `cv=5`, and it is stated explicitly here rather than inherited silently.

The whole stack then sits inside 5.2.1's outer cross-validation, so the reported number involves
two nested levels of splitting and no base model ever sees a row its meta-learner is scored on.

## What it measured

Three bases under a logistic meta-learner, `cv=5` inside 5.2.1's outer five folds:

    table          forest    svm      mlp     stacked   stacked - best base
    hybrid         0.7797   0.9690   0.9595   0.9723        +0.0033
    embedding      0.7753   0.9742   0.9468   0.9725        -0.0017
    handcrafted    0.7319   0.8310   0.7777   0.8064        -0.0246

**The answer is the plan's second branch: documented as not.** The stack beats its best base on
one table out of three, by 0.0033, on 5 discordant rows out of 1,340 - McNemar gives p = 1.0. On
the other two it loses, and on the handcrafted table it loses by 0.025. Nothing here is a
meta-learner that helps.

## The disagreement diagnostic said so before the stack was fitted

    table          all three agree   oracle acc   best single acc   headroom
    hybrid             0.9612          0.9955         0.9918         0.0037
    embedding          0.9612          0.9948         0.9925         0.0022
    handcrafted        0.9246          0.9739         0.9507         0.0231

**A perfect combiner - one picking the right base whenever any base is right - would gain 0.0022
to 0.0037 on the learned tables.** That is the ceiling for *every possible* combination rule, not
just this one, and the logistic meta-learner captured essentially all of it on the hybrid table
(+0.0033 of an available +0.0037). The stack is not underperforming; there is nothing there to
win.

This is what 6.3.7 predicted. Models that land inside 0.012 of each other make the same mistakes:
the SVM and MLP disagree on **1.1% of rows** on the learned tables, and the three bases are
unanimous on 96%. Where they are unanimous they are also right 95.8% of the time, so the
disagreement set is both small and hard.

## The handcrafted table is the informative failure

It has the most disagreement (7.5% of rows, oracle headroom 0.0231) and it is the table where the
stack loses most. More diversity did not become more accuracy, because diversity from a *weak*
base is noise: the forest scores 0.7319 there, and the meta-learner has to learn when to trust a
model that is wrong a quarter of the time, from 1,340 rows across five classes. With cross-fitted
base probabilities as its only input it cannot, and the extra freedom costs more than the extra
information is worth.

## What this means for the pipeline

Phase 13's routing should not stack. The single RBF SVM of 6.3.7 costs one model's latency,
matches the stack within noise on both learned tables, and beats it outright on one. 7.1.8 prices
the difference; the accuracy case for the ensemble is already absent here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np

from src.classify.activations import TABLES

SEED = 42

BASES = ("forest", "svm", "mlp")


def base_estimator(name: str, table: str = "hybrid"):
    """One base model at the settings its own task selected."""
    if name == "forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        )
    if name == "svm":
        from src.classify.multiclass import base_estimator as kernel

        model = kernel("rbf", table)
        # The meta-learner wants probabilities; `SVC` only provides them with this flag, which
        # costs an internal Platt calibration and is the reason it is not on by default.
        model.set_params(probability=True, random_state=SEED)
        return model
    if name == "mlp":
        from src.classify.regularize import ACTIVATION, best_settings
        from src.classify.torchnet import TorchMLP

        return TorchMLP(
            hidden_layer_sizes=(512, 256),
            activation=ACTIVATION,
            batch_size=16,
            device="cpu",
            **best_settings(table),
        )
    raise ValueError(f"base must be one of {list(BASES)}; got {name!r}")


def stack(table: str = "hybrid", bases=BASES, inner_folds: int = 5):
    """The stack, with the scaler shared and the meta-learner fitted on out-of-fold predictions."""
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    estimators = [(name, base_estimator(name, table)) for name in bases]
    meta = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=3000, random_state=SEED),
        # Explicit rather than inherited: in-sample base predictions would teach the meta-learner
        # to trust whichever base memorises hardest.
        cv=inner_folds,
        stack_method="predict_proba",
        n_jobs=1,
    )
    return Pipeline([("prepare", feature_scaler()), ("model", meta)])


def base_pipeline(name: str, table: str = "hybrid"):
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    return Pipeline([("prepare", feature_scaler()), ("model", base_estimator(name, table))])


def out_of_fold(data, estimator, folds: int = 5, n_jobs: int | None = None) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    return cross_val_predict(
        estimator,
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else 1,
    )


def disagreement(predictions: dict, truth: np.ndarray) -> dict:
    """How much room a combiner has, before any combiner is fitted.

    `oracle` is the share of rows at least one base gets right - the ceiling for *any* stacking
    rule, including ones nobody has invented. If it sits close to the best single base, the
    negative result is structural rather than a failure of the meta-learner.
    """
    names = sorted(predictions)
    correct = {name: predictions[name] == truth for name in names}

    pairs = {}
    for a, b in itertools.combinations(names, 2):
        pairs[f"{a} vs {b}"] = round(float((predictions[a] != predictions[b]).mean()), 4)

    any_right = np.zeros(len(truth), dtype=bool)
    all_right = np.ones(len(truth), dtype=bool)
    for name in names:
        any_right |= correct[name]
        all_right &= correct[name]

    best_single = max(float(correct[name].mean()) for name in names)
    return {
        "pairwise_prediction_disagreement": pairs,
        "rows_where_all_three_agree": round(
            float(
                np.mean(
                    [len({predictions[name][i] for name in names}) == 1 for i in range(len(truth))]
                )
            ),
            4,
        ),
        "oracle_accuracy": round(float(any_right.mean()), 4),
        "unanimous_accuracy": round(float(all_right.mean()), 4),
        "best_single_accuracy": round(best_single, 4),
        "headroom_above_best_single": round(float(any_right.mean()) - best_single, 4),
    }


def run(
    table: str = "hybrid", corpus: str = "real", bases=BASES, n_jobs: int | None = None
) -> dict:
    from sklearn.metrics import f1_score

    from src.embed.hybrid import dataset

    data = dataset(table, corpus)

    predictions = {
        name: out_of_fold(data, base_pipeline(name, table), n_jobs=n_jobs) for name in bases
    }
    base_scores = {
        name: round(float(f1_score(data.y, pred, average="macro", zero_division=0)), 4)
        for name, pred in predictions.items()
    }
    stacked = out_of_fold(data, stack(table, bases), n_jobs=n_jobs)
    stacked_score = round(float(f1_score(data.y, stacked, average="macro", zero_division=0)), 4)

    best_base = max(base_scores, key=lambda name: base_scores[name])
    from src.classify.significance import mcnemar

    verdict = mcnemar(stacked == data.y, predictions[best_base] == data.y)

    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "base_macro_f1": base_scores,
        "best_base": best_base,
        "stacked_macro_f1": stacked_score,
        "stacked_minus_best_base": round(stacked_score - base_scores[best_base], 4),
        # The plan's "or documented as not", answered as a test rather than a sign.
        "beats_best_base": bool(stacked_score > base_scores[best_base]),
        "mcnemar_vs_best_base": {
            "stacked_right_base_wrong": verdict["b"],
            "base_right_stacked_wrong": verdict["c"],
            "discordant": verdict["discordant"],
            "p_value": round(verdict["p_value"], 6),
            "significant_at_05": bool(verdict["p_value"] < 0.05),
        },
        "disagreement": disagreement(predictions, data.y),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--bases", nargs="*", default=list(BASES))
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {name: run(name, args.corpus, tuple(args.bases), args.jobs) for name in tables}
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result if len(result) > 1 else next(iter(result.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

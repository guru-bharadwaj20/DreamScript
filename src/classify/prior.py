"""Phase 7.2.5 - Naive Bayes as a prior into the downstream classifier, integrated and ablated.

    python -m src.classify.prior

The plan asks for the NB output to be used as a prior into the downstream classifier **and** as
the HMM's initial distribution, integrated and ablated. The HMM does not exist until 7.3, so this
task does the half that can be done now and builds the interface the other half will use:
`page_prior()` returns a class distribution per page, which is exactly the vector 7.3.4's initial
state distribution needs, and 7.3 consumes it rather than recomputing one.

## Three ways to combine, because they are not the same idea

The phrase "use as a prior" hides a real choice, and all three are measured:

    features      the 5 posterior columns appended to the downstream model's feature table. The
                  downstream model learns what the prior is worth, including learning to ignore
                  it. Most flexible, and the least like a prior.
    multiply      the downstream model's posterior multiplied by the NB posterior and
                  renormalised - the literal Bayesian reading, and the one with no free
                  parameters. Also the one that double-counts, since both models saw the same
                  page.
    interpolate   a weighted geometric mean, `p_down^(1-w) * p_nb^w`, with `w` swept. Contains
                  `multiply` at w = 0.5 in log space and the downstream model alone at w = 0,
                  so it is the honest superset of the other two.

## The ablation, and the leak it has to avoid

The prior has to be computed **out-of-fold**. An NB fitted on all rows and then handed to a
downstream model inside cross-validation leaks the test fold's labels through the prior column,
and the leak is not subtle - NB's training-set posteriors are far sharper than its held-out ones.
So the prior is produced by `cross_val_predict` under the same partition the downstream model is
scored on, and the ablation is the same downstream model with the prior removed.

## What it measured

Out-of-fold NB posteriors combined with 6.3.7's RBF SVM on the hybrid table, 1,340 real pages:

    downstream alone (SVM)          0.9718
    naive bayes alone               0.2323
    NB posteriors as features       0.9627    (-0.0091)
    multiply (literal Bayes)        0.2427    (-0.7291)
    interpolate, best w = 0.0       0.9718    ( 0.0000)

**No combination mode helps, and the best interpolation weight is exactly zero** - the ablation
selects "do not use the prior at all" out of a grid that contained the prior. That is a clean
negative result, and the interpolation curve shows how sharp it is:

    w      0.0      0.1      0.25     0.5      0.75     1.0
    F1   0.9718   0.9323   0.2723   0.2427   0.2338   0.2323

**A quarter weight on the prior costs 0.70 macro F1.** The collapse between w = 0.1 and w = 0.25
is the signature of a *confident* wrong prior: NB's posteriors are near-one-hot even when they are
wrong - 7.2.2's var_smoothing sensitivity is the same peakedness measured a different way - so in
log space a small weight on a saturated distribution is not a small nudge.

`multiply`, the literal Bayesian reading with no free parameter, is the worst mode at 0.2427,
barely above NB alone. It double-counts by construction, and when one of the two models being
multiplied is 0.73 macro F1 worse than the other, double-counting means the weak model wins.

## The features mode is the informative one

Appending the 5 posterior columns to the SVM's 161 loses 0.0091. **The downstream model was given
the chance to learn what the prior is worth, including learning to ignore it, and it still ended
up worse** - the columns are 5 more dimensions for an RBF kernel to spend bandwidth on, and they
carry nothing the table did not already have. The cost is small and it is not zero, which is the
cleanest statement available of "this prior contains no new information".

## What that does *not* say, and what 7.3 inherits

This is a negative result about **combining with an already-strong classifier**, not about the
prior being uninformative in general. NB alone is 0.2323 macro F1, above the 0.20 a uniform guess
over five classes would give, so it does carry signal; there is simply no signal in it that a
0.9718 model does not already have.

7.3.4's initial state distribution is a different situation: there the alternative is a uniform
prior over roles, not a model that already scores 0.97. So `page_prior()` is kept, and it returns
out-of-fold posteriors so the HMM inherits the leak-free version. The interface is built and
exercised; the ablation on *this* consumer says do not wire it in, and Phase 13's routing should
take the SVM's output directly.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

SEED = 42

MODES = ("none", "features", "multiply", "interpolate")

#: Interpolation weights. 0 is the downstream model alone and 1 is Naive Bayes alone, so the
#: sweep contains both ablations as endpoints rather than as separate runs.
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def page_prior(corpus: str = "real", folds: int = 5, n_jobs: int | None = None):
    """Out-of-fold Naive Bayes posteriors, one row per page.

    Returned in `classes` order alongside the ids, so 7.3.4 can index it by page without
    re-deriving the class ordering. Out-of-fold by construction: see the module docstring.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from src.classify.bayes import pipeline
    from src.features.textregions import load

    data = load(corpus)
    proba = cross_val_predict(
        pipeline(names=list(data.feature_names)),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        method="predict_proba",
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    return {
        "ids": data.ids,
        "classes": np.array(sorted(set(data.y.tolist()))),
        "proba": proba,
        "y": data.y,
    }


def downstream_proba(
    table: str = "hybrid", corpus: str = "real", folds: int = 5, n_jobs: int | None = None
):
    """6.3.7's winning model's out-of-fold posteriors, under the same partition as the prior."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from src.embed.hybrid import dataset
    from src.features.scaling import feature_scaler

    data = dataset(table, corpus)
    gamma = {"hybrid": 1e-4, "embedding": 1e-5, "handcrafted": "scale"}.get(table, "scale")
    C = {"hybrid": 1000.0, "embedding": 1000.0, "handcrafted": 10.0}.get(table, 1.0)
    estimator = Pipeline(
        [
            ("prepare", feature_scaler()),
            ("model", SVC(kernel="rbf", gamma=gamma, C=C, probability=True, random_state=SEED)),
        ]
    )
    proba = cross_val_predict(
        estimator,
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        method="predict_proba",
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    return data, proba


def combine(down: np.ndarray, prior: np.ndarray, weight: float) -> np.ndarray:
    """Geometric interpolation in log space, renormalised.

    `weight = 0` is the downstream model untouched, `1` is Naive Bayes alone, and `0.5` is the
    plain product. Clipped before the log because both models can emit an exact zero and
    `log(0)` would propagate a nan through the whole row.
    """
    floor = 1e-12
    mixed = (1 - weight) * np.log(np.clip(down, floor, None)) + weight * np.log(
        np.clip(prior, floor, None)
    )
    mixed -= mixed.max(axis=1, keepdims=True)
    out = np.exp(mixed)
    return out / out.sum(axis=1, keepdims=True)


def as_features(
    table: str = "hybrid", corpus: str = "real", folds: int = 5, n_jobs: int | None = None
) -> dict:
    """The prior's five columns appended to the downstream feature table, scored the same way."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from src.embed.hybrid import dataset
    from src.features.scaling import feature_scaler

    data = dataset(table, corpus)
    prior = page_prior(corpus, folds, n_jobs)
    aligned = _align(prior, data.ids)

    gamma = {"hybrid": 1e-4, "embedding": 1e-5, "handcrafted": "scale"}.get(table, "scale")
    C = {"hybrid": 1000.0, "embedding": 1000.0, "handcrafted": 10.0}.get(table, 1.0)
    estimator = Pipeline(
        [
            ("prepare", feature_scaler()),
            ("model", SVC(kernel="rbf", gamma=gamma, C=C, random_state=SEED)),
        ]
    )
    combined = np.hstack([data.X, aligned])
    predicted = cross_val_predict(
        estimator,
        combined,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    return {
        "macro_f1": round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4),
        "columns_added": int(aligned.shape[1]),
    }


def _align(prior: dict, ids) -> np.ndarray:
    """The prior's rows in the caller's id order, raising rather than concatenating by position."""
    position = {identifier: index for index, identifier in enumerate(prior["ids"])}
    missing = [identifier for identifier in ids if identifier not in position]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(ids)} ids have no Naive Bayes prior "
            f"(first: {missing[0]!r}); rebuild the text table"
        )
    return prior["proba"][[position[identifier] for identifier in ids]]


def run(table: str = "hybrid", corpus: str = "real", n_jobs: int | None = None) -> dict:
    from sklearn.metrics import f1_score

    data, down = downstream_proba(table, corpus, n_jobs=n_jobs)
    prior = page_prior(corpus, n_jobs=n_jobs)
    aligned = _align(prior, data.ids)

    classes = np.array(sorted(set(data.y.tolist())))
    if list(classes) != list(prior["classes"]):
        raise ValueError("the prior and the downstream model disagree about the class ordering")

    def score(proba: np.ndarray) -> float:
        predicted = classes[proba.argmax(axis=1)]
        return round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4)

    curve = {str(w): score(combine(down, aligned, w)) for w in WEIGHTS}
    features = as_features(table, corpus, n_jobs=n_jobs)

    baseline = curve["0.0"]
    best_weight = max(WEIGHTS[:-1], key=lambda w: curve[str(w)])
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "downstream_alone": baseline,
        "naive_bayes_alone": curve["1.0"],
        "multiply": curve["0.5"],
        "interpolate_curve": curve,
        "best_interpolation_weight": best_weight,
        "best_interpolated": curve[str(best_weight)],
        "as_features": features,
        # The plan's "integrated and ablated", as one line per integration mode.
        "gain_over_downstream_alone": {
            "features": round(features["macro_f1"] - baseline, 4),
            "multiply": round(curve["0.5"] - baseline, 4),
            "interpolate": round(curve[str(best_weight)] - baseline, 4),
        },
        "any_mode_helps": bool(
            max(features["macro_f1"], curve["0.5"], curve[str(best_weight)]) > baseline
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=["handcrafted", "embedding", "hybrid"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.table, args.corpus, args.jobs), indent=2))
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

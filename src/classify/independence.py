"""Phase 7.2.4 - where the independence assumption breaks, and why the classifier survives it.

    python -m src.classify.independence

The plan asks for a written analysis of where conditional independence fails and why Naive Bayes
still works. Both halves are measurable on this corpus rather than quotable, and that is what this
task does.

## The assumption

Naive Bayes computes `P(class | x) ∝ P(class) * prod_j P(x_j | class)`. The product is the
assumption: given the class, the features carry no information about each other. 7.2.1's table
violates it by construction and it is worth naming the specific pairs -

    text_n_blocks / text_blocks_per_area    the same count, normalised and not
    text_width_mean / text_aspect_mean      share a numerator
    text_x_std / text_y_std                 both grow when the writing is spread out
    text_coverage / text_ink_share          both grow with how much writing there is

- so the question is not *whether* it is violated but *how much that costs*.

## Three measurements, each answering a different part of the question

**How badly is it violated?** Pearson correlation between every pair of features, computed
*within* each class and averaged - which is exactly the quantity the assumption says should be
zero. A pooled correlation would not be: two features can be uncorrelated within every class and
strongly correlated overall, and that case does not violate the assumption at all.

**What does the violation cost?** Gaussian NB against a **Quadratic Discriminant Analysis** on the
same columns. QDA is the same generative model - a Gaussian per class - with the diagonal
covariance assumption removed. So the gap between them is the price of the assumption, isolated
from every other difference, and it is the cleanest control available for this question.

**Why does it survive?** Because classification takes an argmax and the assumption damages the
*probabilities* far more than the *ranking*. Multiplying dependent features double-counts shared
evidence, which pushes the winning class's posterior toward 1 and the rest toward 0 - a monotone
distortion that usually leaves the argmax alone. That predicts **badly calibrated probabilities
alongside a usable accuracy**, and both are measured: Brier score and the share of predictions
made at over 0.99 confidence.

## What it measured

## How badly the assumption is violated

Mean absolute within-class correlation over the 14 columns, per class:

    state_machine   0.762
    wireframe       0.588
    flowchart       0.494
    circuit         0.463
    er_diagram      0.419
    ---------------------
    mean            0.545

**The quantity the model assumes is zero averages 0.545.** Three of the six pairs the docstring
named in advance are above 0.5, and the worst is the one predicted first:

    text_n_blocks / text_blocks_per_area   0.944
    text_ink_share / text_coverage         0.611
    text_n_blocks  / text_coverage         0.505

A correlation of 0.944 *within* each class is close to the degenerate case - those two columns
are the same count with and without a normalisation, and Naive Bayes multiplies both likelihoods
in as though they were separate evidence. The page is not counted once; it is counted twice.

The class ordering is worth noting: **state_machine is the most correlated class (0.762) and it is
not the worst-scoring one** in 7.2.2 (0.4198, third of five). Violation severity per class does not
predict per-class F1, which is the first sign that the violation is not what is costing the score.

## What the violation costs

QDA is the same generative model with the diagonal assumption removed:

    naive bayes            0.2323
    QDA, reg_param 0.00    0.0116
    QDA, reg_param 0.01    0.6904
    QDA, reg_param 0.10    0.6242
    QDA, reg_param 0.50    0.5762

**Modelling the covariance is worth +0.458 macro F1** - the assumption is not free, and on this
table it is the single most expensive modelling decision measured in 7.2. But the control that
makes the number meaningful is the unregularized row: **QDA at reg_param = 0 scores 0.0116**, near
total collapse, because a 40-row class in 14 dimensions has a singular covariance matrix. The full
covariance model is only usable *once it is regularized back toward the diagonal*, and the score
then falls monotonically as more regularization is applied. The best cell, 0.01, is the least
diagonal one that is numerically alive.

So the honest statement is narrower than "independence costs 0.46": independence costs 0.46
against the most weakly regularized covariance model this corpus can support, and a 40-row class
is what sets that floor.

## Why the classifier survives it anyway - and where it does not

Classification needs the *argmax* of the posterior to be right, not the posterior itself. Counting
correlated evidence twice sharpens the distribution without necessarily moving its peak, and the
calibration numbers show exactly that happening:

    mean confidence     0.9754
    accuracy            0.4791
    overconfidence      0.4963
    Brier (one-vs-rest) 0.2017

**84% of predictions come back above 0.99 confidence, and those are right 53% of the time.** The
model is wrong by half a probability unit on average. Sorted by confidence band, the ordering is
still correct - 11% accuracy below 0.9, 55% above 0.999999 - so the score is *monotone* in
confidence and useless as a probability.

That resolves the plan's question in a specific way. Naive Bayes survives the violation as a
**ranker**: its argmax is 0.479-accurate, well above chance, because doubled evidence usually
doubles in the direction the single copy pointed. It does not survive as a **probability model**,
and it is the probability that 7.2.5 tried to use - which is why the prior collapsed there at a
quarter weight, and why 7.3's initial distribution should take the ranking and re-normalise rather
than take the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

SEED = 42


def within_class_correlation(data) -> dict:
    """Mean |correlation| between feature pairs, computed inside each class.

    The pooled correlation is the wrong quantity: two features can be independent given the class
    and still correlate across the whole corpus, purely because both track the class. Only the
    within-class version is what Naive Bayes assumes away.
    """
    names = list(data.feature_names)
    # The usable columns have to be chosen once, across the whole table, rather than per class:
    # a per-class mask gives each class a different set of columns, and the matrices are then
    # not the same shape and not about the same pairs.
    finite = np.isfinite(data.X).all(axis=0)
    finite_names = [name for name, keep in zip(names, finite, strict=True) if keep]

    per_class = {}
    accumulated = []
    for label in data.classes:
        usable = data.X[data.y == label][:, finite]
        if usable.shape[0] < 3 or usable.shape[1] < 2:
            continue
        matrix = np.nan_to_num(np.corrcoef(usable, rowvar=False))
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        per_class[label] = round(float(np.abs(upper).mean()), 4)
        accumulated.append(np.abs(matrix))

    mean_matrix = np.mean(accumulated, axis=0) if accumulated else np.zeros((1, 1))

    pairs = []
    if mean_matrix.shape[0] == len(finite_names):
        for i in range(len(finite_names)):
            for j in range(i + 1, len(finite_names)):
                pairs.append((finite_names[i], finite_names[j], float(mean_matrix[i, j])))
    pairs.sort(key=lambda item: -item[2])

    return {
        "mean_absolute_within_class_correlation": (
            round(float(np.mean(list(per_class.values()))), 4) if per_class else None
        ),
        "per_class": per_class,
        "most_dependent_pairs": [
            {"a": a, "b": b, "correlation": round(value, 4)} for a, b, value in pairs[:8]
        ],
        "pairs_above_0_5": int(sum(1 for _, _, value in pairs if value > 0.5)),
        "pairs_total": len(pairs),
    }


def qda_control(data, folds: int = 5, n_jobs: int | None = None) -> dict:
    """Gaussian NB against QDA - the same model with the diagonal assumption removed.

    `reg_param` is not zero: a 40-row class in 14 dimensions gives a singular covariance matrix,
    and QDA without regularization would fail on exactly the class this project cares about.
    That regularization is itself a partial return toward the diagonal, so it is reported.
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    splitter = StratifiedKFold(folds, shuffle=True, random_state=SEED)

    def score(estimator):
        predicted = cross_val_predict(
            estimator, data.X, data.y, cv=splitter, n_jobs=n_jobs if n_jobs is not None else -1
        )
        return round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4)

    naive = Pipeline([("prepare", feature_scaler()), ("model", GaussianNB())])
    results = {"naive_bayes": score(naive)}
    for reg in (0.0, 0.01, 0.1, 0.5):
        full = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", QuadraticDiscriminantAnalysis(reg_param=reg)),
            ]
        )
        try:
            results[f"qda_reg_{reg}"] = score(full)
        except Exception:  # noqa: BLE001 - a singular class covariance is the expected failure
            results[f"qda_reg_{reg}"] = None

    usable = [v for k, v in results.items() if k.startswith("qda") and v is not None]
    return {
        "scores": results,
        "best_qda": max(usable) if usable else None,
        "cost_of_the_assumption": (
            round(max(usable) - results["naive_bayes"], 4) if usable else None
        ),
    }


def calibration(data, folds: int = 5, n_jobs: int | None = None) -> dict:
    """Is Naive Bayes a bad probability estimator while remaining a usable classifier?

    The mechanism the whole survival argument rests on. Double-counting dependent evidence pushes
    the winning posterior toward 1; a model that is right 80% of the time while claiming 0.999 is
    exactly what that predicts, and it is directly observable.
    """
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    splitter = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    estimator = Pipeline([("prepare", feature_scaler()), ("model", GaussianNB())])
    proba = cross_val_predict(
        estimator,
        data.X,
        data.y,
        cv=splitter,
        method="predict_proba",
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    classes = np.array(sorted(set(data.y.tolist())))
    predicted = classes[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    correct = predicted == data.y

    # One-vs-rest Brier, averaged over classes - the multiclass form that does not need the
    # probabilities to be binned.
    brier = float(
        np.mean(
            [
                brier_score_loss((data.y == label).astype(int), proba[:, index])
                for index, label in enumerate(classes)
            ]
        )
    )

    bands = {}
    for low, high in ((0.0, 0.9), (0.9, 0.99), (0.99, 0.999999), (0.999999, 1.01)):
        mask = (confidence >= low) & (confidence < high)
        if mask.any():
            bands[f"{low}-{high}"] = {
                "rows": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "accuracy": round(float(correct[mask].mean()), 4),
                "mean_confidence": round(float(confidence[mask].mean()), 4),
            }

    return {
        "accuracy": round(float(accuracy_score(data.y, predicted)), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "overconfidence": round(float(confidence.mean() - correct.mean()), 4),
        "brier_ovr": round(brier, 4),
        "share_above_0_99": round(float((confidence > 0.99).mean()), 4),
        "accuracy_when_above_0_99": (
            round(float(correct[confidence > 0.99].mean()), 4)
            if (confidence > 0.99).any()
            else None
        ),
        "by_confidence_band": bands,
    }


def run(corpus: str = "real", n_jobs: int | None = None) -> dict:
    from src.features.textregions import load

    data = load(corpus)
    return {
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "violation": within_class_correlation(data),
        "cost": qda_control(data, n_jobs=n_jobs),
        "calibration": calibration(data, n_jobs=n_jobs),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, args.jobs), indent=2))
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

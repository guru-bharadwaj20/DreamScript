"""Phase 5.2.8 - is the difference real? McNemar between models, and honest CV intervals.

    python -m src.classify.significance

5.2.1 measured logistic regression at 0.7898 macro F1 and kNN at 0.7614, and said in the same
breath that the gap is smaller than the fold-to-fold spread. This module settles it two ways.

## McNemar, on the rows both models saw

The test uses only the pages where the two models **disagree**: `b` is the count where the first
is right and the second wrong, `c` the reverse. Under the null the two are exchangeable, so
`b ~ Binomial(b + c, 0.5)` and the exact binomial test is used rather than the chi-square
approximation, which is unreliable when `b + c` is small - and on a 1,340-row corpus where the
models agree on nine tenths of the pages, `b + c` is small by construction.

The rows have to be the same for this to mean anything, which is exactly why 5.2.1 makes
out-of-fold predictions the shared artefact: both models are compared on the same 1,340 pages,
predicted under the same fold partition from the same seed.

## Confidence intervals with the correction that repeated CV needs

The naive interval - `mean +- t * sd / sqrt(n)` over 15 fold scores - is **too narrow**, and
badly so. The 15 scores are not independent: every pair of training sets overlaps by about 80%
of the corpus, so the folds are correlated and the usual variance estimate under-counts it. The
Nadeau-Bengio correction replaces `1/n` with `1/n + test_ratio/train_ratio`, which for 5-fold CV
inflates the standard error by a factor of about 1.7. Both intervals are reported so the
difference is visible rather than assumed.

## What it measured

McNemar over the 1,340 real photographs, on out-of-fold predictions from seed 42:

    comparison        a right, b wrong   b right, a wrong   discordant   p        significant
    logreg vs knn            51                 33              84       0.063        no
    logreg vs tree           91                 39             130       6.0e-6       yes
    knn vs tree              85                 51             136       0.0045       yes

And the intervals over the 15 fold scores of repeated 5-fold CV:

    model    mean macro F1   naive 95%          Nadeau-Bengio 95%    inflation
    logreg      0.7898       0.767 - 0.813       0.739 - 0.840         2.18x
    knn         0.7614       0.746 - 0.777       0.727 - 0.795         2.18x
    tree        0.7426       0.722 - 0.763       0.698 - 0.788         2.18x

**Both models beat the tree, and the gap between logistic regression and kNN is not
established.** At p = 0.063 the logreg-kNN comparison sits just the wrong side of 0.05 on 84
discordant pages: logreg wins 51 of them and kNN 33, which is the direction 5.2.1's ranking
predicted and not enough of a majority to call. The honest reading of Phase 5's headline is that
*logistic regression and kNN are indistinguishable on this corpus and both are better than the
tree*, not that logistic regression is the best model.

**The two methods disagree, and the disagreement is the point.** Every pair of corrected
intervals overlaps - including logreg against tree, which McNemar settles at p = 6e-6. Both are
right, because they answer different questions. Overlapping intervals ask whether two *separate*
estimates could share a mean; McNemar asks whether one model beats the other **on the same
pages**, and pairing removes the fold-to-fold variance that makes the intervals wide. A reader
who compares models by eyeballing error bars will call this corpus a three-way tie and be wrong
about two of the three comparisons. This is exactly why 5.2.1 made out-of-fold predictions the
shared artefact rather than letting each task re-run its own CV.

**The naive interval is wrong by a factor of 2.18 and always by that factor.** The inflation is
`sqrt((1/15 + 1/4) / (1/15))`, so it depends on the fold count and the number of scores and not
at all on the data - which is why it is worth stating once here rather than rediscovering per
model. Reported naively, logistic regression's 0.7898 comes with +-0.023; corrected, +-0.050.
The uncorrected interval would exclude kNN's mean and imply a difference the paired test
declines to confirm, so the correction and the pairing pull in opposite directions and both are
needed to read the result correctly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
from scipy import stats

from src.classify.cv import N_SPLITS, REAL_MODELS, SEEDS, evaluate, out_of_fold
from src.classify.data import Dataset, load


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on two boolean vectors of per-row correctness."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "discordant": 0, "p_value": 1.0, "statistic": 0.0}
    # Two-sided exact binomial test: the chi-square form is unreliable for small n, and n is
    # small here by construction - the models agree on most pages.
    p_value = float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    statistic = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {
        "b": b,
        "c": c,
        "discordant": n,
        "statistic": round(float(statistic), 4),
        "p_value": p_value,
    }


def compare_models(
    dataset: Dataset, a: str, b: str, seed: int = SEEDS[0], strategy: str = "stratified"
) -> dict:
    """McNemar for one pair, on identical out-of-fold predictions."""
    first = out_of_fold(a, dataset, strategy=strategy, seed=seed)
    second = out_of_fold(b, dataset, strategy=strategy, seed=seed)
    correct_a = first.y_pred == dataset.y
    correct_b = second.y_pred == dataset.y
    result = mcnemar(correct_a, correct_b)
    return {
        "a": a,
        "b_model": b,
        "accuracy_a": round(float(correct_a.mean()), 4),
        "accuracy_b": round(float(correct_b.mean()), 4),
        "a_right_b_wrong": result["b"],
        "b_right_a_wrong": result["c"],
        "discordant": result["discordant"],
        "p_value": round(result["p_value"], 6),
        "significant_at_05": bool(result["p_value"] < 0.05),
        "winner": a if result["b"] > result["c"] else b,
    }


def interval(scores: np.ndarray, n_splits: int = N_SPLITS, confidence: float = 0.95) -> dict:
    """Naive and Nadeau-Bengio-corrected intervals for a set of cross-validated scores."""
    scores = np.asarray(scores, float)
    n = len(scores)
    mean = float(scores.mean())
    variance = float(scores.var(ddof=1))
    critical = float(stats.t.ppf(0.5 + confidence / 2, n - 1))

    naive = critical * np.sqrt(variance / n)
    # Nadeau & Bengio: the folds share training data, so the variance of the mean is
    # (1/n + test/train) * sigma^2 rather than sigma^2/n.
    test_ratio = 1.0 / n_splits
    corrected = critical * np.sqrt(variance * (1.0 / n + test_ratio / (1.0 - test_ratio)))
    return {
        "mean": round(mean, 4),
        "scores": int(n),
        "naive_half_width": round(float(naive), 4),
        "corrected_half_width": round(float(corrected), 4),
        "naive_interval": [round(mean - naive, 4), round(mean + naive, 4)],
        "corrected_interval": [round(mean - corrected, 4), round(mean + corrected, 4)],
        "inflation": round(float(corrected / naive), 2) if naive else 0.0,
    }


def run(corpus: str = "real", models=REAL_MODELS, strategy: str = "stratified") -> dict:
    dataset = load(corpus)
    intervals = {}
    for model in models:
        scored = evaluate(model, dataset, strategy=strategy)
        intervals[model] = interval(np.array([row["macro_f1"] for row in scored["per_fold"]]))

    pairs = [
        compare_models(dataset, a, b, strategy=strategy)
        for a, b in itertools.combinations(models, 2)
    ]
    overlaps = {
        f"{a}_vs_{b}": bool(
            intervals[a]["corrected_interval"][0] <= intervals[b]["corrected_interval"][1]
            and intervals[b]["corrected_interval"][0] <= intervals[a]["corrected_interval"][1]
        )
        for a, b in itertools.combinations(models, 2)
    }
    return {
        "corpus": corpus,
        "strategy": strategy,
        "rows": int(len(dataset.y)),
        "intervals": intervals,
        "mcnemar": pairs,
        "corrected_intervals_overlap": overlaps,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=list(REAL_MODELS))
    ap.add_argument("--strategy", default="stratified", choices=["stratified", "grouped"])
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, tuple(args.models), args.strategy), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

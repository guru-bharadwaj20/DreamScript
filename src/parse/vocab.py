"""Phase 7.4.2 - a Gaussian mixture over 7.4.1's descriptors, and which covariance it can afford.

    python -m src.parse.vocab

7.4.1 built a 22-column descriptor for every labelled hdbpmn node. This task fits a Gaussian
mixture to it and answers the plan's question - full, diagonal or tied - which is the same
bias-variance question 7.3.12 asked of the continuous HMM, on the same corpus, so the two answers
are worth reading together.

## What "best" means here, and why it is not BIC

BIC selects the model that best *explains* the descriptors. That is the right criterion for 7.4.3,
which is choosing how many components a vocabulary has. It is the wrong one for this task, because
the vocabulary exists to be *used*: 7.4.6 feeds the responsibilities into the HMM's emissions, and
what matters there is whether the components line up with shapes a person would name.

So three numbers are reported for each covariance type and they are allowed to disagree:

    held-out log likelihood   per node, 5-fold. Does the density generalise?
    BIC                       on the full table. What would 7.4.3's criterion pick?
    adjusted Rand / purity    against 4.1.3's five labels. Do the components mean anything?

Purity is computed through a Hungarian assignment rather than by taking each component's most
common label, because the greedy version lets two components both claim `rectangle` and reports a
flattering number for a model that found one shape twice.

## The parameter counts, which is the whole bias-variance argument

22 dimensions, K components:

    spherical   K x (22 + 1)      one variance per component
    diag        K x (22 + 22)     one variance per dimension
    tied        K x 22 + 253      one shared 22x22 covariance
    full        K x (22 + 253)    a 22x22 covariance each

At K = 5 that is 115, 220, 363 and 1,375 free parameters against 12,400 rows. Even `full` is not
obviously starved, which is why this is a real comparison rather than a foregone one - and it is
the opposite situation to 7.3.12, where the smallest state had 303 rows to fit 45 parameters from.

## Standardisation is not optional here

The columns are on wildly different scales - a Hu moment runs to 11, `solidity` sits in [0, 1],
`vertices` is a count that reaches 20. A spherical or tied covariance assumes a shared shape
across dimensions, so on raw columns those two variants would be modelling the units rather than
the data. Every fit below is on z-scored columns, and the scaler is fitted **inside** each fold.

## What it measured

K = 5, 12,400 rows, 22 standardised columns, 5-fold:

    covariance   held-out LL   BIC        params   adj. Rand   purity   iters
    spherical      -19.096     479,008      119      0.0288     0.3485    23
    diag           -10.924     252,764      224      0.1365     0.4019    24
    tied           -19.828     416,460      367      0.1303     0.4584    19
    full            -3.286      55,601    1,379      0.1499     0.4027    20

**All three criteria pick `full`, unanimously.** That is the cleanest answer this task could have
returned and it is the opposite of 7.3.12's, where the continuous HMM's smallest state had 303
rows to fit 45 parameters from and the full covariance was unaffordable. Here 1,379 parameters are
fitted from 12,400 rows - nine rows per parameter - and the columns are correlated enough that
modelling the correlation is worth far more than the parameters cost.

**The margin is not marginal.** `full` beats the next best on held-out likelihood by 7.6 nats a
node, and its BIC is 4.5x lower than `diag`'s despite having six times the parameters. 7.4.1
explains why: `hu_0` through `hu_3` are moments of the same region and move together, `rect_fill`,
`extent` and `solidity` are three normalisations of the same area, and a diagonal covariance is
required to pretend all of that is independent. It is 7.2.4's finding in a different model - the
independence assumption is expensive when the columns are three views of one measurement.

**`tied` is worse than `diag` on held-out likelihood (-19.8 against -10.9) while having more
parameters**, which is the informative failure of the four. A tied covariance says every shape
class has the same shape of scatter and differs only in where it sits. That is false here in a
specific way: circles vary mostly in roundness and rectangles mostly in aspect ratio, so their
clouds are elongated along different axes, and one shared matrix has to average two orientations
into something that fits neither. `tied` also produces the most lopsided partition - one component
of 81 nodes against another of 7,021 - which is the same failure seen from the side.

## The number that matters more than the winner

**The best adjusted Rand against 4.1.3's labels is 0.1499.** Purity peaks at 0.4584, and a
constant "everything is a rectangle" predictor would score 0.480 on this distribution, so **no
covariance type produces components that agree with the declared shape names better than guessing
the majority class.**

That is not a failure of the covariance comparison; it is the finding the comparison exposes. The
mixture is describing the descriptor cloud well - `full` generalises at -3.29 nats a node - and
the structure it finds is simply not the four-way shape distinction a person would draw. 7.4.1
already predicted the mechanism: rectangles average an aspect of 3.26 against ~1.2 for every other
class, so wide tasks and square tasks are further apart in this space than a square task and a
diamond are. **The mixture is finding size and elongation where the labels encode identity.**

`spherical` at ARI 0.0288 is effectively no agreement at all, which confirms the floor was worth
including: with one variance per component the model can only carve the cloud into balls, and the
balls fall along the elongation axis rather than across the shape classes.

7.4.3 asks whether more components help, and 7.4.5 has to do the naming knowing that the
component-to-name map is starting from a weak alignment rather than a strong one.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.features.descriptors import NAMES

SEED = 42

#: The plan names three; `spherical` is included as the floor, so "tied is better than diagonal"
#: can be read against something that is unambiguously too rigid.
COVARIANCES = ("spherical", "diag", "tied", "full")

#: 4.1.3's five declared names. The mixture is not told about these; they are the yardstick.
DEFAULT_K = 5

#: Regularisation added to every covariance diagonal. `full` on 22 correlated columns can find a
#: near-singular component and report an unbounded likelihood; this is the standard floor.
REG = 1e-4


def matrix(table=None):
    """The descriptor table as (X, labels, keys), with non-finite rows dropped.

    A single non-finite cell anywhere in the row makes the whole row unusable for a Gaussian, so
    the drop is by row and the count is reported rather than silently absorbed.
    """
    from src.features.descriptors import load_table

    table = load_table() if table is None else table
    X = table[list(NAMES)].to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=1)
    return (
        X[keep],
        table["label"].to_numpy()[keep],
        table["key"].to_numpy()[keep],
        int((~keep).sum()),
    )


def fit(X, k: int = DEFAULT_K, covariance: str = "full", seed: int = SEED, init: str = "kmeans"):
    """A standardised GMM. The scaler is returned with it so a caller cannot forget to apply it."""
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "gmm",
                GaussianMixture(
                    n_components=k,
                    covariance_type=covariance,
                    reg_covar=REG,
                    random_state=seed,
                    n_init=1,
                    init_params=init,
                    max_iter=200,
                ),
            ),
        ]
    ).fit(X)


def purity(components: np.ndarray, labels: np.ndarray, k: int) -> dict:
    """Best one-to-one component -> label assignment, and the accuracy it buys.

    Hungarian rather than greedy: letting two components both claim `rectangle` would report a
    flattering number for a model that has found one shape twice and missed another entirely.
    """
    from scipy.optimize import linear_sum_assignment

    names = sorted(set(labels))
    table = np.zeros((k, len(names)))
    for component, label in zip(components, labels, strict=True):
        table[component, names.index(label)] += 1

    rows, columns = linear_sum_assignment(-table)
    matched = table[rows, columns].sum()
    return {
        "purity": round(float(matched / max(len(labels), 1)), 4),
        "mapping": {int(r): names[c] for r, c in zip(rows, columns, strict=True)},
        "labels_claimed": len({names[c] for c in columns}),
    }


def evaluate(X, labels, k: int = DEFAULT_K, covariance: str = "full", folds: int = 5) -> dict:
    """Held-out likelihood, BIC, and agreement with the declared labels."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.model_selection import StratifiedKFold

    held_out = []
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, labels):
        model = fit(X[train], k, covariance)
        # `score` is the mean per-sample log likelihood, so folds of different sizes compare.
        held_out.append(float(model.score(X[test])))

    whole = fit(X, k, covariance)
    gmm = whole.named_steps["gmm"]
    scaled = whole.named_steps["scale"].transform(X)
    components = gmm.predict(scaled)

    return {
        "covariance": covariance,
        "k": k,
        "held_out_log_likelihood": round(float(np.mean(held_out)), 4),
        "held_out_std": round(float(np.std(held_out)), 4),
        "bic": round(float(gmm.bic(scaled)), 1),
        "aic": round(float(gmm.aic(scaled)), 1),
        "free_parameters": int(gmm._n_parameters()),
        "converged": bool(gmm.converged_),
        "iterations": int(gmm.n_iter_),
        "adjusted_rand": round(float(adjusted_rand_score(labels, components)), 4),
        "nmi": round(float(normalized_mutual_info_score(labels, components)), 4),
        **purity(components, labels, k),
        "component_sizes": np.bincount(components, minlength=k).tolist(),
    }


def run(k: int = DEFAULT_K, covariances=COVARIANCES, folds: int = 5) -> dict:
    X, labels, _, dropped = matrix()
    rows = [evaluate(X, labels, k, covariance, folds) for covariance in covariances]

    by_likelihood = max(rows, key=lambda r: r["held_out_log_likelihood"])
    by_bic = min(rows, key=lambda r: r["bic"])
    by_agreement = max(rows, key=lambda r: r["adjusted_rand"])
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "dimensions": len(NAMES),
        "k": k,
        "best_by_held_out_likelihood": by_likelihood["covariance"],
        "best_by_bic": by_bic["covariance"],
        "best_by_agreement": by_agreement["covariance"],
        "criteria_agree": len(
            {by_likelihood["covariance"], by_bic["covariance"], by_agreement["covariance"]}
        )
        == 1,
        "by_covariance": {row["covariance"]: row for row in rows},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.k, COVARIANCES, args.folds), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

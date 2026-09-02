"""Phase 8.1 - K-means over the shape descriptors, and what the spherical assumption costs.

    python -m src.cluster.kmeans

7.4.2 already fitted a Gaussian mixture to this table and reported that `full` covariance beat
`spherical` on every criterion. K-means is the *hard, spherical, equal-weight* limit of that
mixture: one shared isotropic variance, no covariance, and a point assigned wholly to its
nearest centre. So this task is not a fresh question about whether the descriptors cluster - 7.4
settled that they cluster badly - it is a question about **how much of the mixture's advantage
came from the parts K-means throws away.**

That framing matters because K-means is the algorithm anyone reaches for first, and the honest
thing to report is the size of the gap to the model that was already fitted, on the same rows,
the same scaling and the same yardstick.

## The design

    K                    5 (4.1.3's declared vocabulary) and 6 (7.4.3's elbow), so the numbers
                         line up with both of the K's the project has already argued for.
    initialisation       k-means++ and random, ten restarts each, because 7.4.4 measured a
                         3.35-nat spread from the seed alone on this very table and a
                         single-restart number here would be unreproducible for the same reason.
    scaling              one `StandardScaler`, as everywhere in 7.4. K-means minimises Euclidean
                         distance, so leaving `hu_0` at ~12 and `circularity` at ~0.2 unscaled
                         would make the partition a function of the Hu moments and nothing else.

Scored by adjusted Rand index and Hungarian purity against the four annotated labels, against
7.4.2's mixture and against the majority baseline, plus inertia and the per-cluster descriptor
means that say *what* each cluster is.

## What the yardsticks are for

Purity alone rises with K and would make K=6 look better than K=5 for free. ARI is chance-
corrected and does not. The majority baseline (always `rectangle`, 0.4796) is the number a
partition has to beat to be worth computing at all, and 7.4.2's mixture reached 0.4731 purity
and 0.1499 ARI - below it. **A clustering that cannot beat a constant is a finding, not a
failure of the run**, and it is the one this table has produced twice already.

## What it measured

12,400 shapes, 693 pages, 0 rows dropped, 22 standardised descriptors, ten restarts a cell.

    K   init         ARI      purity   labels claimed   silhouette   smallest cluster
    5   k-means++   0.0717    0.4485         4            0.2238            2
    5   random      0.0556    0.4140         4            0.2235          100
    6   k-means++   0.0535    0.4133         4            0.2203            1
    6   random      0.0280    0.3528         4            0.1796           39

    7.4.2's Gaussian mixture, K = 5, full covariance   ARI 0.1499   purity 0.4731
    majority baseline (always `rectangle`)                          purity 0.4796

**The spherical assumption costs more than half the mixture's already-poor agreement**: 0.0717
against 0.1499 ARI, on the same rows, the same scaler and the same yardstick. What K-means drops
relative to the mixture is per-component covariance and soft assignment, and dropping them
halves the result - which is the answer to the question this task exists to ask.

**And neither model beats a constant.** The best purity here is 0.4485 against a majority
baseline of 0.4796. Answering `rectangle` for all 12,400 shapes is more accurate than the best
partition either algorithm found. That is now the third independent method to land under the
baseline on this table, after 7.4.2's mixture and 7.4.3's whole K sweep.

## The clusters are the same three that 7.4.5 found, arrived at by a different algorithm

    cluster    n      dominant     share   circularity   solidity   aspect
      0      2,827   rectangle     0.80       0.033       0.219      4.34
      1      5,660   rectangle     0.46       0.532       0.880      1.68
      2      3,788   diamond       0.37       0.250       0.671      1.38
      3        123   circle        0.48       0.338       0.719      1.09
      4          2   circle        0.50       0.593       0.855      1.04

**Cluster 0 is 2,827 nodes - 22.8% of the corpus - at circularity 0.033 and solidity 0.219, and
it is 80% `rectangle`.** That is 7.4.5's damage axis exactly: a perimeter enormous for the area
and a hull far larger than the shape, which is a fragmented outline rather than a geometry. It
is also the *purest* cluster in the table, and what it is pure in is breakage. A hard spherical
partition with no covariance structure finds the same dominant axis a full-covariance mixture
found, which promotes that finding from a property of the GMM to **a property of the table**.

The remaining structure is one clean cluster (1: round, solid, low aspect - where the circles
went, though still only 46% rectangle by count because rectangles are half the corpus) and one
middle cluster (2) that is 37% diamond and holds a third of everything.

## Only three of the five clusters are real

**Cluster 4 has two members and cluster 3 has 123.** At K = 6 the smallest cluster is a single
node. So the K-means partition of this corpus is a three-way split plus two outlier magnets, and
the K that was asked for is not the K that was delivered - the same collapse 7.4.3 documented,
where components past K = 9 landed on individual outliers and dragged the BIC minimum with them.

This is also where the initialisation difference comes from, and it is the opposite of the usual
story. **k-means++ scores better (+0.0161 ARI at K = 5) while producing the smaller degenerate
clusters** (2 members against random's 100), because seeding proportional to squared distance is
precisely a procedure for finding outliers and putting a centre on one. Random initialisation
gives more balanced and worse partitions. Neither is good; they fail differently.

## The seed does not matter here, and that is worth stating

    ARI across five seeds        0.0392 - 0.0529, spread 0.0137
    agreement between seeds      mean ARI 0.8788, worst pair 0.7689

7.4.4 measured a 3.35-nat spread from the seed alone when fitting a mixture to this table, so a
single-seed number here would have been suspect by default. It is not: with ten restarts the
partition is stable to within 0.014 ARI and different seeds agree with each other at 0.88.
**K-means is reproducible on this table and the mixture is not**, which is the one respect in
which the simpler model is the better-behaved one - and it does not rescue the result, because a
stable answer that loses to a constant is still an answer that loses to a constant.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.features.descriptors import NAMES
from src.parse.vocab import matrix, purity

SEED = 42

#: 4.1.3's declared vocabulary size, and 7.4.3's measured elbow.
KS = (5, 6)

INITS = ("k-means++", "random")

#: 7.4.4 measured a 3.35-nat spread from the seed alone on this table; one restart is not a
#: measurement of the algorithm, it is a measurement of one draw.
RESTARTS = 10


def fit(X, k: int = 6, init: str = "k-means++", n_init: int = RESTARTS, seed: int = SEED):
    """A standardised K-means. The scaler travels with it so a caller cannot forget to apply it."""
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=k, init=init, n_init=n_init, random_state=seed, max_iter=300),
            ),
        ]
    ).fit(X)


def cluster_means(X, assignment: np.ndarray, k: int) -> list[dict]:
    """Per-cluster size and mean descriptor, in the original units rather than the scaled ones.

    Scaled means are what the algorithm optimised; unscaled means are what a reader can check
    against a picture. 7.4.5 needed exactly this to notice that three of its six components had
    a mean circularity of 0.06.
    """
    out = []
    for c in range(k):
        rows = X[assignment == c]
        if not len(rows):
            out.append({"cluster": c, "size": 0})
            continue
        means = rows.mean(axis=0)
        out.append(
            {
                "cluster": c,
                "size": int(len(rows)),
                **{name: round(float(means[i]), 4) for i, name in enumerate(NAMES)},
            }
        )
    return out


def label_mix(assignment: np.ndarray, labels: np.ndarray, k: int) -> list[dict]:
    """Which annotated labels ended up in each cluster."""
    out = []
    for c in range(k):
        got = labels[assignment == c]
        names, counts = np.unique(got, return_counts=True)
        order = np.argsort(-counts)
        out.append(
            {
                "cluster": c,
                "size": int(len(got)),
                "mix": {str(names[i]): int(counts[i]) for i in order},
                "dominant": str(names[order[0]]) if len(order) else None,
                "dominant_share": round(float(counts[order[0]] / len(got)), 4) if len(got) else 0.0,
            }
        )
    return out


def evaluate(X, labels, k: int, init: str, seed: int = SEED) -> dict:
    """One K-means fit, scored every way this task reports."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    model = fit(X, k=k, init=init, seed=seed)
    assignment = model.named_steps["kmeans"].labels_
    scaled = model.named_steps["scale"].transform(X)
    hungarian = purity(assignment, labels, k)
    return {
        "k": k,
        "init": init,
        "inertia": round(float(model.named_steps["kmeans"].inertia_), 2),
        "iterations": int(model.named_steps["kmeans"].n_iter_),
        "ari": round(float(adjusted_rand_score(labels, assignment)), 4),
        "purity": float(hungarian["purity"]),
        "labels_claimed": int(hungarian["labels_claimed"]),
        "silhouette": round(
            float(silhouette_score(scaled, assignment, sample_size=5000, random_state=seed)), 4
        ),
        "smallest_cluster": int(np.bincount(assignment, minlength=k).min()),
        "clusters": label_mix(assignment, labels, k),
        "means": cluster_means(X, assignment, k),
    }


def seed_spread(X, labels, k: int = 6, init: str = "k-means++", seeds=range(5)) -> dict:
    """How much the answer moves when only the seed moves.

    Reported because 7.4.4 found the seed worth nearly half of the biggest modelling decision on
    this table, and a single-seed ARI would inherit that without saying so.
    """
    from sklearn.metrics import adjusted_rand_score

    runs, assignments = [], []
    for s in seeds:
        model = fit(X, k=k, init=init, seed=int(s))
        a = model.named_steps["kmeans"].labels_
        assignments.append(a)
        runs.append(
            {
                "seed": int(s),
                "inertia": round(float(model.named_steps["kmeans"].inertia_), 2),
                "ari": round(float(adjusted_rand_score(labels, a)), 4),
            }
        )
    pairwise = [
        round(float(adjusted_rand_score(assignments[i], assignments[j])), 4)
        for i in range(len(assignments))
        for j in range(i + 1, len(assignments))
    ]
    aris = [r["ari"] for r in runs]
    return {
        "runs": runs,
        "ari_spread": round(max(aris) - min(aris), 4),
        "agreement_between_seeds": {
            "mean": round(float(np.mean(pairwise)), 4),
            "min": round(float(min(pairwise)), 4),
        },
    }


def majority_baseline(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return round(float(counts.max() / len(labels)), 4)


def sweep(ks=KS, inits=INITS) -> dict:
    X, labels, keys, dropped = matrix()
    cells = [evaluate(X, labels, k, init) for k in ks for init in inits]
    best = max(cells, key=lambda c: c["ari"])
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "pages": int(len({k.split(":")[0] for k in keys})),
        "majority_baseline": majority_baseline(labels),
        "cells": cells,
        "best": {k: v for k, v in best.items() if k not in ("clusters", "means")},
        "seed_spread": seed_spread(X, labels),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    result = sweep()
    text = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

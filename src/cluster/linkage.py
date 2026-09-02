"""Phase 8.4 - four linkage rules, and the one number that tells them apart honestly.

    python -m src.cluster.linkage

Single, complete, average and Ward differ only in how the distance between two *clusters* is
defined from the distances between their *points*: nearest pair, furthest pair, mean pair, and
the increase in within-cluster sum of squares. Everything else - the data, the scaling, the
agglomerative procedure - is identical, so this is the cleanest single-variable comparison in
Phase 8.

## What each rule is expected to do, so the measurement can contradict it

    single      chains. It merges on the nearest pair, so a thread of intermediate points joins
                two otherwise distant clusters. On a corpus with 12,400 points and a heavy tail
                of broken outlines, chaining is not a theoretical risk.
    complete    the opposite bias: it merges on the furthest pair, so it prefers compact clusters
                of roughly equal diameter and will split a genuinely elongated one.
    average     between the two, and the rule with the highest cophenetic correlation by
                construction on most data, because it is the one that most nearly *is* the mean
                distance it is being compared against.
    ward        minimises within-cluster variance, which makes it 8.1's objective read
                bottom-up, and the only one of the four that is not defined on distances alone.

## The two yardsticks disagree on purpose

    cophenetic correlation   how faithfully the tree reproduces the pairwise distances. It is a
                             property of the *tree*, and it is what a linkage rule is entitled
                             to be judged on in its own terms.
    ARI and purity at K = 4  whether the cut agrees with the shape labels. It is a property of
                             the *corpus*, and no linkage rule promised it.

Reporting both is the point. A rule can win the first and lose the second, and 8.2 has already
shown that on this table the well-separated structure is not the labelled structure - so a
comparison on cophenetic correlation alone would recommend whichever rule best describes a
geometry that is mostly outline damage.

**The balance of the cut is reported beside both**, because single linkage's characteristic
failure is not a low score but a partition of 12,397 points and three singletons, which can post
a respectable cophenetic correlation while being useless.

## What it measured

12,400 shapes, 22 standardised descriptors, one cut at K = 4, four rules.

    rule        cophenetic     ARI     purity   silhouette   largest cluster   singletons
    average       0.9291     0.0003    0.4799     0.8359         12,389            2
    complete      0.8202    -0.0011    0.4783     0.6903         12,380            2
    single        0.7771     0.0000    0.4797     0.9563         12,397            3
    ward          0.4737     0.0148    0.4615     0.4052          3,444            1

    majority baseline (always `rectangle`)   0.4796

## The two rankings are exactly inverted

    by cophenetic correlation    average > complete > single > ward
    by agreement with the labels    ward > average > single > complete

**The rule that describes this geometry best is the rule that partitions it worst, and the
reverse.** Ward reproduces less than half the variance of the distances it was built from and is
the only rule whose cut agrees with the shape labels at all; average reproduces 93% of that
variance and its cut is worth 0.0003 ARI - three ten-thousandths, which is chance.

## Three of the four rules did not produce a partition

Single, complete and average all put **more than 99.8% of the corpus in one cluster** and spend
their remaining three on 1-18 nodes. That is not a poor clustering; it is the absence of one.
Single linkage's 12,397/1/1/1 is textbook chaining, and on a corpus with a heavy tail of
fragmented outlines the chain has no gaps to stop at. Complete and average, whose biases are
supposed to be the opposite, arrive at the same place: at K = 4 the only cuts they can find are
peels of extreme outliers.

**Ward is the only rule that produced four clusters** (3,444 / 8,836 / 119 / 1), which is why it
is recommended despite losing every unsupervised column.

## Both unsupervised criteria reward the degeneracy, and by large margins

    single linkage, 12,397 + three singletons     silhouette 0.9563   purity 0.4797
    ward, an actual four-way partition            silhouette 0.4052   purity 0.4615

**Single linkage's silhouette is 0.9563.** Three points sitting far from a single enormous blob
give almost every point a near-zero within-cluster penalty and a large between-cluster distance,
which is what a silhouette rewards. A reader choosing a linkage rule by silhouette would choose
the one that clustered nothing, by a margin of 0.55.

**Purity does the same thing more quietly.** Average linkage scores 0.4799 and single 0.4797,
both *above* Ward's 0.4615 and both at or above the 0.4796 majority baseline - because a
Hungarian assignment mapping one 12,389-node cluster to `rectangle` **is** the majority
classifier, exactly reproduced. Every task in 8.x has reported purity against that baseline for
this reason; here is the case where the two coincide to four decimal places and the mechanism is
visible.

## What this settles for the rest of Phase 8

8.2 found the silhouette peaking at a K whose ARI was -0.0001. This task finds it peaking on a
rule that made no clusters. **Neither result is a fluke of one metric: on this table, internal
validity criteria systematically prefer the partitions that carry the least information**, because
what they measure - compactness and separation - is maximised by isolating the outliers that 8.1,
8.3 and 7.4.5 all identified as fragmented outlines rather than shapes. 8.8 computes three more of
these criteria and inherits the caveat rather than discovering it.

The recommendation is **Ward**, on the grounds that it is the only rule that partitions, and with
the explicit note that it wins nothing else. 8.3's cut levels stand.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.cluster.hierarchy import cophenetic, cut, tree
from src.cluster.kmeans import label_mix, majority_baseline
from src.parse.vocab import matrix, purity

#: The plan names exactly these four.
METHODS = ("single", "complete", "average", "ward")

#: 8.2's recommendation, and hdbpmn's true class count. One K, so the rules are the only variable.
K = 4


def evaluate(X, labels, method: str, k: int = K) -> dict:
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    from src.cluster.hierarchy import scaled

    Z = tree(X, method=method)
    assignment = cut(Z, k)
    sizes = np.bincount(assignment, minlength=k)
    return {
        "method": method,
        "cophenetic": cophenetic(Z, X),
        "ari": round(float(adjusted_rand_score(labels, assignment)), 4),
        "purity": float(purity(assignment, labels, k)["purity"]),
        "silhouette": round(
            float(silhouette_score(scaled(X), assignment, sample_size=5000, random_state=42)), 4
        ),
        "sizes": [int(s) for s in sizes],
        "largest_share": round(float(sizes.max() / sizes.sum()), 4),
        "singletons": int((sizes == 1).sum()),
        "clusters": label_mix(assignment, labels, k),
    }


def compare(X, labels, methods=METHODS, k: int = K) -> list[dict]:
    return [evaluate(X, labels, m, k) for m in methods]


def run(methods=METHODS, k: int = K) -> dict:
    X, labels, keys, dropped = matrix()
    rows = compare(X, labels, methods, k)
    usable = [r for r in rows if r["largest_share"] < 0.95]
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "k": k,
        "majority_baseline": majority_baseline(labels),
        "methods": rows,
        "best_cophenetic": max(rows, key=lambda r: r["cophenetic"])["method"],
        "best_ari": max(rows, key=lambda r: r["ari"])["method"],
        # A rule whose largest cluster holds 95% of the corpus has not produced a partition, so
        # it is excluded from the recommendation rather than allowed to win on another column.
        "recommended": max(usable, key=lambda r: r["ari"])["method"] if usable else None,
        "degenerate": [r["method"] for r in rows if r["largest_share"] >= 0.95],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    result = run()
    text = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

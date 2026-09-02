"""Phase 8.8 - three internal validity indices, and the partition all three prefer.

    python -m src.cluster.validity

The plan asks for silhouette, Davies-Bouldin and Calinski-Harabasz as a metrics table. 8.2 and
8.4 have already found the first of those preferring, respectively, a K whose agreement with the
shape labels was -0.0001 and a linkage rule that put 12,397 of 12,400 points in one cluster. So
the question this task can usefully answer is **not** which partition is best - none of the three
indices can see a shape label - but **whether the other two make the same mistake.**

They are constructed differently enough that they need not:

    silhouette          per point, (b - a) / max(a, b) over its own and its nearest other
                        cluster. Bounded, and dominated by points far from everything.
    Davies-Bouldin      mean over clusters of the worst pairwise ratio of spread to separation.
                        Lower is better. It is a *worst-case* index, so one badly overlapping
                        pair spoils it however good the rest are.
    Calinski-Harabasz   between-cluster over within-cluster dispersion, scaled by degrees of
                        freedom. Unbounded and rises with K almost mechanically, which is
                        exactly the failure mode to watch for.

## The design

Both families from earlier in the phase, on one table: K-means at every K from 2 to 15, and each
of 8.4's four linkage rules at K = 4. Every index computed on the same standardised space, with
ARI against the shape labels and the smallest cluster size carried in the same row.

**The ARI column is not a fourth index.** It is the answer key. An internal index that ranks
partitions the way ARI does is telling the truth about this corpus; one that ranks them the
opposite way is measuring compactness faithfully and compactness is not what the corpus is made
of. The whole value of the table is in that comparison, and it is why a metrics table without the
answer key would have been decoration.

## What it measured

18 partitions - K-means at K = 2..15 and four linkage rules at K = 4 - over 12,400 shapes.

    partition        silhouette   Davies-Bouldin   Calinski-Harabasz     ARI    smallest
    single   k=4       0.9563         0.0547              426.5        0.0000       1
    average  k=4       0.8359         0.4104              611.3        0.0003       1
    complete k=4       0.6903         0.2839              568.4       -0.0011       1
    kmeans   k=3       0.4300         1.0773            4,025.4        0.0028     119
    kmeans   k=2       0.4168         1.1858            4,882.5       -0.0001   3,237
    ward     k=4       0.4052         0.8565            3,068.3        0.0148       1
    kmeans   k=4       0.2374         1.4203            3,396.7        0.0823      65
    kmeans   k=15       0.1636        1.3371            2,198.2        0.1359       1

    ranked by silhouette, best first. The ARI column is the answer key.

## Not one of the three indices is even neutral

    index               Spearman with ARI    p        its best partition
    silhouette              -0.8411        0.0000     single k=4
    Davies-Bouldin          -0.7049        0.0011     single k=4
    Calinski-Harabasz       -0.0774        0.7602     kmeans k=2

    best by ARI                                       kmeans k=15

**Two of the three are strongly and significantly anti-correlated with the truth, and the third
is noise.** Silhouette at -0.84 is not a weak index on this table; it is a reliable one pointed
the wrong way, and a reader could do better by minimising it. Davies-Bouldin, built on an
entirely different construction - a worst-case ratio of spread to separation rather than a
per-point margin - reproduces the same error at -0.70, which rules out the explanation that
silhouette has some idiosyncratic flaw.

**Both name `single k=4` as the best partition in the table.** That partition is 12,397 nodes,
one node, one node and one node. Davies-Bouldin scores it **0.0547**, where anything under about
0.5 is normally read as excellent. Three points far from one enormous blob is the global optimum
of both indices, and it contains no information at all.

## Calinski-Harabasz is not neutral either; it is two opposite errors cancelling

    within K-means only     silhouette -0.7055   DB -0.4242   CH -0.7187
    within linkage only     silhouette -0.4000   DB -0.8000   CH +0.8000

Pooled, CH reads -0.08 with p = 0.76, which looks like an index that simply carries no signal.
Split, it is the **only index in the table that rejects the degenerate partitions**: it scores the
three collapsed linkage rules at 426-611 against K-means' 2,200-4,900 and correctly picks `ward`
among the four, at +0.80. Its between-over-within dispersion ratio has an explicit penalty for
clusters that hold nothing, which neither of the others does.

And within K-means it is the worst of the three at -0.72, because it falls almost mechanically
with K (4,882 at K = 2 down to 2,198 at K = 15) while ARI rises. The two effects are close enough
in size to cancel, so **the pooled coefficient describes neither half and would have been the only
number a single-line metrics table reported.**

## What the table is actually good for

Nothing here says these indices are broken. They measure compactness and separation, they measure
them correctly, and 8.1, 8.3, 8.5 and 7.4.5 have between them established that the compact,
well-separated structure in these 22 columns is **outline integrity and extraction failure, not
shape**. An index that faithfully finds the best-separated partition finds the one that isolates
the most damaged outlines, and that partition is uninformative about shape by construction.

The practical rule that survives, and it is the one thing to carry out of 8.x:

**On this table, an internal validity index should be used only to reject partitions, never to
select one.** CH's rejection of the collapsed linkage rules is correct and useful; every
selection any of the three made was wrong. Where a label exists, use it; where none exists - which
is 8.10's situation - expect these numbers to prefer whichever partition has isolated the
artefacts, and check the smallest cluster size before believing any of them.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.cluster.hierarchy import cut, scaled, tree
from src.cluster.kmeans import fit
from src.cluster.linkage import METHODS
from src.parse.vocab import matrix, purity

SEED = 42

K_RANGE = tuple(range(2, 16))

LINKAGE_K = 4

#: Silhouette is O(n^2); the other two are not. One fixed subsample keeps the column comparable
#: across rows rather than merely cheap.
SILHOUETTE_SAMPLE = 5000

INDICES = ("silhouette", "davies_bouldin", "calinski_harabasz")

#: Which direction is "better" for each index, so a ranking can be built without hard-coding it
#: three times over.
BETTER = {"silhouette": "high", "davies_bouldin": "low", "calinski_harabasz": "high"}


def indices(Xs: np.ndarray, assignment: np.ndarray, seed: int = SEED) -> dict:
    """All three, on one standardised space."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    if len(set(assignment.tolist())) < 2:
        return dict.fromkeys(INDICES)
    return {
        "silhouette": round(
            float(
                silhouette_score(
                    Xs, assignment, sample_size=min(SILHOUETTE_SAMPLE, len(Xs)), random_state=seed
                )
            ),
            4,
        ),
        "davies_bouldin": round(float(davies_bouldin_score(Xs, assignment)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(Xs, assignment)), 1),
    }


def score_row(Xs, X, labels, assignment, name: str) -> dict:
    from sklearn.metrics import adjusted_rand_score

    k = len(set(assignment.tolist()))
    sizes = np.bincount(assignment - assignment.min(), minlength=k)
    return {
        "partition": name,
        "k": k,
        **indices(Xs, assignment),
        "ari": round(float(adjusted_rand_score(labels, assignment)), 4),
        "purity": float(purity(assignment, labels, k)["purity"]),
        "smallest_cluster": int(sizes.min()),
        "largest_share": round(float(sizes.max() / sizes.sum()), 4),
    }


def kmeans_rows(Xs, X, labels, ks=K_RANGE) -> list[dict]:
    rows = []
    for k in ks:
        model = fit(X, k=k)
        rows.append(score_row(Xs, X, labels, model.named_steps["kmeans"].labels_, f"kmeans k={k}"))
    return rows


def linkage_rows(Xs, X, labels, methods=METHODS, k: int = LINKAGE_K) -> list[dict]:
    return [score_row(Xs, X, labels, cut(tree(X, method=m), k), f"{m} k={k}") for m in methods]


def agreement(rows: list[dict]) -> dict:
    """Does each index rank partitions the way the answer key does?

    Spearman rather than Pearson: the indices are on wildly different scales and only their
    ordering is being compared.
    """
    from scipy.stats import spearmanr

    usable = [r for r in rows if r.get("silhouette") is not None]
    ari = [r["ari"] for r in usable]
    out = {}
    for index in INDICES:
        values = [r[index] for r in usable]
        rho, p = spearmanr(values, ari)
        # Flip the sign for lower-is-better indices, so a positive number always means
        # "this index agrees with the answer key".
        sign = -1.0 if BETTER[index] == "low" else 1.0
        out[index] = {
            "spearman_with_ari": round(float(sign * rho), 4),
            "p_value": round(float(p), 4),
            "best_partition": (max if BETTER[index] == "high" else min)(
                usable, key=lambda r: r[index]
            )["partition"],
        }
    out["best_by_ari"] = max(usable, key=lambda r: r["ari"])["partition"]
    return out


def agreement_within_families(rows: list[dict]) -> dict:
    """The same correlation computed inside each family, because a pooled one can cancel.

    An index that ranks the K-means K's backwards and the linkage rules correctly has two
    opposite effects in one number, and the pooled coefficient reports their sum rather than
    either. Both are real and they mean different things, so both are computed.
    """
    families = {
        "kmeans_only": [r for r in rows if r["partition"].startswith("kmeans")],
        "linkage_only": [r for r in rows if not r["partition"].startswith("kmeans")],
    }
    return {name: agreement(group) for name, group in families.items() if len(group) > 2}


def run(ks=K_RANGE) -> dict:
    X, labels, keys, dropped = matrix()
    Xs = scaled(X)
    rows = kmeans_rows(Xs, X, labels, ks) + linkage_rows(Xs, X, labels)
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "partitions": rows,
        "agreement_with_the_answer_key": agreement(rows),
        "agreement_within_families": agreement_within_families(rows),
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

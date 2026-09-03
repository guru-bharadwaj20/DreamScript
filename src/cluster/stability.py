"""Phase 8.9 - how much of each partition survives being asked again on different rows.

    python -m src.cluster.stability

The plan asks for bootstrap resampling agreement in ARI. That is a different question from every
other one in Phase 8: not *is this partition right* - 8.1 through 8.5 have answered that, and the
answer is no - but **is it the same partition twice.** A clustering can be stable and wrong,
and a stable wrong answer is more dangerous than an unstable one, because it looks like a
finding.

## The design

    resampling     a bootstrap draw of n rows with replacement, refit from scratch, then compare
                   the new assignment with the reference assignment **on the rows that appear in
                   both**. Comparing on the resample alone would score a partition against itself.
    duplicates     a bootstrap draw contains repeats, and a repeated point is trivially assigned
                   the same cluster twice. Every comparison is made on the *unique* shared rows,
                   which is the difference between measuring stability and measuring that
                   `x == x`.
    label switching   ARI is invariant to permutation of cluster ids, which is exactly why it is
                   the right statistic here: two runs that found the same three groups in a
                   different order agree completely and should score 1.
    draws          25 per setting. The quantity being estimated is a mean of a bounded statistic,
                   and 25 draws put its standard error well under the differences being reported.

## Why K-means and Ward are both measured

They fail differently. 8.1 found K-means stable across seeds on this table (0.879 mean pairwise
ARI) but landing on outlier magnets; 8.3 found Ward producing a singleton at K = 4 that never
leaves. **A deterministic algorithm is not automatically a stable one** - Ward has no seed, so
every point of instability it shows is instability with respect to the *sample*, which is the
kind that matters for a claim about a corpus.

## The floor this has to be read against

A partition that puts everything in one cluster is perfectly stable. So stability is reported
beside the smallest cluster size and the ARI against the labels, and a high stability with a
degenerate partition is read as what it is. 8.4 has three such partitions on record.

## What it measured

12,400 shapes, 25 bootstrap draws per setting, compared on the unique shared rows.

    algorithm    K    mean ARI    sd      min      max     smallest   ARI vs labels
    kmeans       3     0.9875   0.0084   0.9683   0.9995      119        0.0028
    kmeans       4     0.8842   0.1479   0.4884   0.9788       65        0.0823
    kmeans       6     0.7940   0.0809   0.5679   0.9493        1        0.0535
    ward         3     0.8015   0.0405   0.7289   0.8737      120        0.0148
    ward         4     0.6623   0.1476   0.4243   0.8536        1        0.0148
    ward         6     0.5540   0.1339   0.3068   0.7422        1        0.0843

**These partitions are stable.** K-means at K = 3 recurs at 0.9875 across independent resamples
of the corpus, and even the worst row recurs at 0.55. Whatever these algorithms are finding in the
descriptor table, they find the same thing again when shown different rows - so nothing in 8.1
through 8.5 is an artefact of which 12,400 shapes happened to be photographed.

That is the useful half of the result and it is worth stating plainly, because the rest of Phase 8
has been a sequence of negatives and this is not one. The structure is real. It is simply not
shape.

## Stability is a fourth criterion pointing the wrong way, and this time the sample is too small

    Spearman between stability and agreement with the labels    -0.5508   (p = 0.2574)
    the most stable partition's ARI against the labels           0.0028
    the most informative partition's stability                   0.5540

The point estimates line up with 8.2, 8.4 and 8.8 exactly. **The most reproducible partition in the
table, K-means at K = 3 at 0.9875, agrees with the shape labels at 0.0028** - it is the partition
8.2 already identified as the one both unsupervised criteria chose and the one that knows nothing.
**The most informative partition, Ward at K = 6, is the least stable of the six at 0.5540.**

But the correlation is computed over six rows and **p = 0.2574, so it is not significant and is
not claimed as a finding.** Three earlier tasks measured this relationship on samples large enough
to support a claim - 8.8's silhouette reached -0.84 at p < 0.001 over eighteen partitions. This
task is consistent with them and does not independently establish anything, which is the honest
description of a six-row correlation however suggestive its point estimate.

## Ward has no seed and is the less stable algorithm anyway

    kmeans   0.9875 / 0.8842 / 0.7940
    ward     0.8015 / 0.6623 / 0.5540

**Ward is less stable than K-means at every K, by 0.19 to 0.24**, and Ward is deterministic: given
the same rows it returns the same tree every time, which the tests assert. So none of its
instability is initialisation, and all of it is sensitivity to *which rows are in the sample*.

That is the more serious kind. K-means' variation across seeds can be bought down with restarts,
and 8.1 measured ten restarts buying a seed spread of 0.0137. Sample sensitivity cannot be bought
down at all - it is a statement about how much the answer depends on the corpus, and Ward's
agglomerative merges commit early and irreversibly on whichever pair happens to be closest.

**K-means at K = 4 is the exception that shows what fragility looks like from the inside**: mean
0.8842 with a standard deviation of 0.1479 and a worst draw of 0.4884, the widest spread of any
K-means row. K = 4 is 8.2's recommendation and the K-means row with the highest agreement with the
labels, and it is also the one whose partition moves most between resamples. The informative
partition is the fragile one.

## What this settles

Two things, in opposite directions.

**Nothing in 8.1-8.5 is a sampling artefact.** Every partition recurs, most of them strongly, so
the damage axis, the collapsed linkage rules and the outlier rind are properties of the corpus
rather than of this particular draw of it.

**And stability cannot be used to choose among them.** The most stable partition here is
uninformative and the most informative is the least stable, at least as point estimates; with six
rows this task cannot say more, and 8.8 has already said it at a sample size that can. **Stability
belongs in the same category as silhouette and Davies-Bouldin on this table: useful for rejecting
a partition that will not reproduce, useless for selecting the one that means something.**
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.cluster.hierarchy import cut, tree
from src.cluster.kmeans import fit
from src.parse.vocab import matrix

SEED = 42

#: 3 is 8.2's elbow and silhouette peak, 4 its recommendation, 6 is 8.3's ARI peak.
KS = (3, 4, 6)

DRAWS = 25

ALGORITHMS = ("kmeans", "ward")


def assign(X, k: int, algorithm: str, seed: int = SEED) -> np.ndarray:
    if algorithm == "kmeans":
        return fit(X, k=k, seed=seed).named_steps["kmeans"].labels_
    if algorithm == "ward":
        return cut(tree(X), k)
    raise ValueError(f"algorithm must be one of {list(ALGORITHMS)}; got {algorithm!r}")


def one_draw(X, reference: np.ndarray, k: int, algorithm: str, rng) -> float | None:
    """One bootstrap resample, refit, and the ARI on the unique rows the two share.

    The reference assignment is read off the rows themselves rather than predicted, so this
    measures whether the *partition* recurs, not whether a fitted model generalises.
    """
    from sklearn.metrics import adjusted_rand_score

    draw = rng.integers(0, len(X), size=len(X))
    unique = np.unique(draw)
    if len(unique) < k:
        return None

    resampled = assign(X[draw], k, algorithm, seed=int(rng.integers(1 << 30)))
    # First occurrence of each unique row, so a row repeated three times votes once.
    first = {int(row): position for position, row in reversed(list(enumerate(draw)))}
    order = np.array([first[int(row)] for row in unique])
    return float(adjusted_rand_score(reference[unique], resampled[order]))


def stability(X, k: int, algorithm: str, draws: int = DRAWS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    reference = assign(X, k, algorithm)
    scores = [one_draw(X, reference, k, algorithm, rng) for _ in range(draws)]
    scores = [s for s in scores if s is not None]
    sizes = np.bincount(reference, minlength=k)
    return {
        "algorithm": algorithm,
        "k": k,
        "draws": len(scores),
        "mean_ari": round(float(np.mean(scores)), 4),
        "std_ari": round(float(np.std(scores)), 4),
        "min_ari": round(float(min(scores)), 4),
        "max_ari": round(float(max(scores)), 4),
        "smallest_cluster": int(sizes.min()),
        "largest_share": round(float(sizes.max() / sizes.sum()), 4),
    }


def run(ks=KS, algorithms=ALGORITHMS, draws: int = DRAWS) -> dict:
    from sklearn.metrics import adjusted_rand_score

    X, labels, keys, dropped = matrix()
    rows = []
    for algorithm in algorithms:
        for k in ks:
            row = stability(X, k, algorithm, draws)
            row["ari_against_labels"] = round(
                float(adjusted_rand_score(labels, assign(X, k, algorithm))), 4
            )
            rows.append(row)
    best = max(rows, key=lambda r: r["mean_ari"])
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "draws": draws,
        "stability": rows,
        "most_stable": f"{best['algorithm']} k={best['k']}",
        "stability_vs_informativeness": informativeness(rows),
    }


def informativeness(rows: list[dict]) -> dict:
    """Does the partition that recurs most often also agree with the labels most?

    8.2, 8.4 and 8.8 each found an unsupervised criterion ranking partitions opposite to the
    answer key. Stability is a fourth such criterion and it is entitled to a different answer, so
    the correlation is computed rather than assumed.
    """
    from scipy.stats import spearmanr

    rho, p = spearmanr([r["mean_ari"] for r in rows], [r["ari_against_labels"] for r in rows])
    return {
        "spearman": round(float(rho), 4),
        "p_value": round(float(p), 4),
        "most_stable_partition_ari": max(rows, key=lambda r: r["mean_ari"])["ari_against_labels"],
        "most_informative_partition_stability": max(rows, key=lambda r: r["ari_against_labels"])[
            "mean_ari"
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    result = run(draws=args.draws)
    text = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

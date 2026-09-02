"""Phase 8.2 - choosing K rigorously, and what happens when the rigorous criteria disagree.

    python -m src.cluster.choosek

The plan asks for elbow and silhouette and a chosen K. 7.4.3 ran the same exercise for a
Gaussian mixture and found that the criterion which *looked* most rigorous - the BIC minimum -
was an artefact of components collapsing onto single outliers, and that the elbow of the same
curve gave the defensible answer. This task asks whether K-means, with a different objective and
a different degeneracy, reproduces that or contradicts it.

## The three criteria, and why all three are reported

    inertia elbow      within-cluster sum of squares falls monotonically with K by construction,
                       so it has no minimum and the elbow is the only thing it can offer. Read as
                       the K past which splitting stops buying compactness.
    silhouette         a bounded, chance-uncorrected measure of separation that *can* peak. It is
                       computed on the scaled space the algorithm optimised, because a silhouette
                       over unscaled columns would score the Hu moments' geometry rather than the
                       partition's.
    ARI against labels the yardstick the first two cannot see. It is not a way of choosing K in
                       deployment - there are no labels there - but it is the only one of the
                       three that knows what a shape is, and its disagreement with the others is
                       the finding this task is for.

A fourth column, the smallest cluster size, is carried because 8.1 found a two-member cluster at
K = 5 and a one-member cluster at K = 6. **A criterion that keeps improving while the partition
is dissolving into singletons is not measuring what its name suggests**, and the only way to see
that is to print the sizes next to it.

## Elbow, stated precisely

The elbow is the K of maximum discrete curvature - the largest second difference - which is the
same definition 7.4.3 used, kept identical so the two sweeps can be read side by side. It is a
heuristic and it is sensitive to where the sweep stops; both facts are why the raw minimum and
the ARI peak are printed beside it rather than instead of it.

## What it measured

12,400 shapes, K from 2 to 15, ten restarts each, silhouette on the scaled space.

     K    inertia   silhouette     ARI    purity   smallest cluster
     2    195,722     0.4168    -0.0001   0.4470       3,237
     3    165,392     0.4300     0.0028   0.4450         119
     4    149,722     0.2374     0.0823   0.4525          65
     5    137,143     0.2238     0.0717   0.4485           2
     6    127,394     0.2203     0.0535   0.4133           1
    10     96,832     0.1761     0.1128   0.4238           2
    13     83,539     0.1735     0.1331   0.4077           1
    15     78,281     0.1636     0.1359   0.3681           1

    inertia elbow          K = 3
    silhouette peak        K = 3
    ARI peak               K = 15
    largest K with no cluster under ten members    K = 4
    recommended            K = 4

## The elbow and the silhouette agree, and they agree on a partition that knows nothing

**Both unsupervised criteria choose K = 3, and at K = 3 the ARI against the shape labels is
0.0028.** At K = 2, where the silhouette is 0.4168, the ARI is **-0.0001** - the partition with
the cleanest separation this table can produce is, to four decimal places, exactly independent of
what shape was drawn. That is the whole result. It is not that the criteria are unreliable; they
are measuring separation faithfully, and the well-separated structure in these 22 columns is not
shape.

**The two curves move in opposite directions at exactly the point where the split happens.**
Between K = 3 and K = 4 the silhouette collapses 0.4300 -> 0.2374, its largest single drop in the
sweep, while the ARI jumps 0.0028 -> 0.0823, its largest single rise. The K at which the
partition starts to know something about shape is the K at which its separation falls apart, and
a reader with only the silhouette panel would stop one step before the first informative
partition.

## The ARI peak is where the sweep stopped, not where the answer is

ARI climbs again to 0.1359 at K = 15 and is still rising. It is not a better partition: at K = 15
two clusters hold fewer than ten nodes and the smallest holds one, so the gain comes from
carving single outliers into their own clusters, each of which is trivially pure. **Purity, which
is not chance-corrected, tells the truth here by falling** - 0.4525 at K = 4 down to 0.3681 at
K = 15 - because it cannot be bought with singletons the way ARI can.

This is 7.4.3's finding in a different currency. There the BIC minimum at K = 13 was an artefact
of components collapsing onto outliers; here the ARI maximum at K = 15 is the same artefact
wearing the opposite sign, and the same guard catches both: **print the smallest cluster size next
to the criterion.** The partition ceases to be a partition at K = 5, where one cluster holds two
nodes.

## The chosen K, and why it is not the one either criterion named

**K = 4** - the best-scoring K among those that are still partitions. It is the purity maximum
(0.4525), the ARI maximum below the dissolution point (0.0823), and the largest K at which no
cluster holds fewer than ten nodes. It is also, for what it is worth, the number of shape classes
hdbpmn actually contains, which nothing in the sweep was told.

Two caveats attach to it and neither is small. **Its purity of 0.4525 is still below the majority
baseline of 0.4796**, so K = 4 is the best of a set of partitions none of which beats answering
`rectangle` every time. And it disagrees with 7.4.3's elbow of 6 for the Gaussian mixture on the
same table, which is the expected consequence of a different objective with a different
degeneracy rather than a contradiction - **the K that is right depends on the model, and reporting
"the" K for a corpus would have been the error this task was designed to avoid.**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.cluster.kmeans import fit, majority_baseline
from src.parse.vocab import matrix, purity

SEED = 42

#: 2 is the smallest partition that exists; 15 matches 7.4.3's sweep so the two can be compared.
K_RANGE = tuple(range(2, 16))

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p8_choose_k.png"

#: Silhouette is O(n^2) in memory over 12,400 rows; sklearn's subsample is used with a fixed seed
#: so the column is comparable across K rather than merely cheap.
SILHOUETTE_SAMPLE = 5000


def elbow(ks, values) -> int:
    """The K of maximum curvature - where the curve stops falling fast.

    Identical to 7.4.3's definition on purpose. A minimum is a different thing and is reported
    separately, because a curve still falling at the end of the sweep has its minimum at wherever
    the sweep was stopped and that is a fact about the sweep.
    """
    ks = list(ks)
    if len(values) < 3:
        return ks[0]
    second = np.diff(values, n=2)
    return ks[int(np.argmax(second)) + 1]


def sweep(X, labels, ks=K_RANGE, seed: int = SEED) -> list[dict]:
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    rows = []
    for k in ks:
        model = fit(X, k=k, seed=seed)
        assignment = model.named_steps["kmeans"].labels_
        scaled = model.named_steps["scale"].transform(X)
        sizes = np.bincount(assignment, minlength=k)
        rows.append(
            {
                "k": int(k),
                "inertia": round(float(model.named_steps["kmeans"].inertia_), 2),
                "silhouette": round(
                    float(
                        silhouette_score(
                            scaled, assignment, sample_size=SILHOUETTE_SAMPLE, random_state=seed
                        )
                    ),
                    4,
                ),
                "ari": round(float(adjusted_rand_score(labels, assignment)), 4),
                "purity": float(purity(assignment, labels, k)["purity"]),
                "smallest_cluster": int(sizes.min()),
                "clusters_under_ten": int((sizes < 10).sum()),
            }
        )
    return rows


def choose(rows: list[dict]) -> dict:
    """The three criteria's answers, side by side, and the one this task recommends."""
    ks = [r["k"] for r in rows]
    inertia_elbow = elbow(ks, [r["inertia"] for r in rows])
    silhouette_peak = max(rows, key=lambda r: r["silhouette"])["k"]
    ari_peak = max(rows, key=lambda r: r["ari"])["k"]
    intact = [r for r in rows if r["smallest_cluster"] >= 10]
    ceiling = max((r["k"] for r in intact), default=None)
    return {
        "inertia_elbow": inertia_elbow,
        "silhouette_peak": silhouette_peak,
        "ari_peak": ari_peak,
        "criteria_agree": len({inertia_elbow, silhouette_peak, ari_peak}) == 1,
        "largest_k_with_no_cluster_under_ten": ceiling,
        # The recommendation is the best-scoring K among the ones that are still partitions.
        # Taking the ARI peak outright would recommend a K at which a cluster holds one node.
        "recommended": max(intact, key=lambda r: r["ari"])["k"] if intact else inertia_elbow,
    }


def figure(rows: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in rows]
    picks = choose(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in (
        (axes[0], "inertia", "inertia (elbow)"),
        (axes[1], "silhouette", "silhouette (peak)"),
        (axes[2], "ari", "ARI against labels"),
    ):
        ax.plot(ks, [r[key] for r in rows], marker="o", color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.grid(alpha=0.3)
    for ax, k in (
        (axes[0], picks["inertia_elbow"]),
        (axes[1], picks["silhouette_peak"]),
        (axes[2], picks["ari_peak"]),
    ):
        ax.axvline(k, color="#d62728", linestyle="--", label=f"K = {k}")
        ax.legend()

    # The sizes are drawn on the ARI panel because that is where a reader is most likely to be
    # tempted by a high-K answer that has already dissolved.
    twin = axes[2].twinx()
    twin.plot(ks, [r["smallest_cluster"] for r in rows], color="#7f7f7f", linestyle=":")
    twin.set_ylabel("smallest cluster", color="#7f7f7f")
    twin.set_yscale("log")

    fig.suptitle("8.2 - three rigorous criteria, three different K", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(ks=K_RANGE) -> dict:
    X, labels, keys, dropped = matrix()
    rows = sweep(X, labels, ks)
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "majority_baseline": majority_baseline(labels),
        "sweep": rows,
        "chosen": choose(rows),
        "figure": str(figure(rows)),
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

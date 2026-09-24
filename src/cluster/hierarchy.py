"""Phase 8.3 - a Ward dendrogram, and whether the hierarchy the plan hoped for is there.

    python -m src.cluster.hierarchy

The plan states a hypothesis rather than a procedure. It expects the tree to nest:

    shapes -> {box-like, circular, pointed} -> {rectangle, rounded-rect, square, ...}

That is a falsifiable claim about this table and it is what this task tests. A dendrogram can
always be drawn; the question is whether cutting it at three gives three *geometric families*
and whether cutting deeper refines them into *shapes* rather than re-cutting the same axis.

8.1 and 8.2 have already made the outcome likely: a flat spherical partition found outline
damage rather than shape, and both unsupervised criteria chose a K whose agreement with the
labels was 0.003. A hierarchy built from the same distances is built over the same structure.
What it can add is *where* in the tree shape starts to appear, if it appears at all - a flat
partition cannot answer that and a tree can.

## The design

    linkage        Ward on the standardised descriptors. Ward minimises the within-cluster sum
                   of squares at every merge, which makes it the agglomerative counterpart of
                   8.1's objective and therefore the one whose disagreement with 8.1 would be
                   informative rather than merely different.
    cuts           2, 3, 4, 6 and 8 clusters. Three is the plan's middle tier; four is 8.2's
                   recommendation and the true class count; the rest bracket them.
    nesting test   for each cut, whether each cluster's *children* at the next cut split it by
                   label or re-split it by the same axis - which is the difference between a
                   taxonomy and a ranking.

## Cophenetic correlation, and why it is here

The cophenetic correlation is the correlation between the original pairwise distances and the
distances implied by the tree. It is the one number that says whether a dendrogram is a fair
picture of the data at all, as opposed to a fair picture of the linkage rule. **A tree can be
perfectly well-formed and still misrepresent the geometry it was built from**, and quoting cut
levels off a tree with a low cophenetic correlation would be quoting the drawing.

## What it measured

12,400 shapes, Ward on 22 standardised descriptors, **cophenetic correlation 0.4737**.

    cut    ARI     purity   sizes
     2   0.0116    0.4631   3,444 / 8,956
     3   0.0148    0.4616   3,444 / 8,836 / 120
     4   0.0148    0.4615   3,444 / 8,836 / 119 / 1
     6   0.0843    0.4410   691 / 2,753 / 2,472 / 6,364 / 119 / 1
     8   0.0468    0.3720   ... / 18 / ... / 1

    majority baseline (always `rectangle`)   0.4796

## The plan's hypothesis is refuted at the first split

The plan expected the three-way cut to give `{box-like, circular, pointed}`. It gives:

    cluster    n       dominant    circularity   solidity   aspect
       0     3,444    rectangle      0.045        0.284      4.09
       1     8,836    rectangle      0.441        0.811      1.46
       2       120    circle         0.343        0.723      1.07

**The root merge of the entire hierarchy separates broken outlines from intact ones**, and going
from two clusters to three only peels off a 120-node oddity. Cluster 0 is 27.8% of the corpus at
circularity 0.045 and solidity 0.284 - a perimeter enormous for its area and a hull far larger
than the shape - and it is 78% `rectangle`. This is 8.1's cluster 0 and 7.4.5's components 2, 3
and 5, found for a third time by a third algorithm. **The top of the tree is not
box/round/pointed; it is damaged/intact**, and no cut level can recover a taxonomy from a root
split that is about image quality.

## Deeper cuts refine the same axis instead of naming shapes

That is what the refinement test was for. Of every parent cluster at every transition, the ones
whose children have *different* dominant labels are:

    2 -> 3    one of two parents split by label (the 120-node peel)
    3 -> 4    one of three (and the new child is a single node)
    4 -> 6    one of four - the informative one: 8,836 splits into a diamond-dominant 2,472
    6 -> 8    **none of six**

**By the eight-way cut, not one parent is splitting by label any more.** Every further merge
level subdivides clusters that keep the label they already had, which is the difference between
a taxonomy and a ranking: the tree is sorting shapes along a continuum of outline integrity and
cutting that continuum at finer and finer thresholds.

The one place the hierarchy earns its keep is 4 -> 6, where the intact cluster finally splits
into a diamond-dominant group (2,472 nodes, 44% diamond, circularity 0.277) and a rounder one
(6,364, 50% circularity). **ARI peaks there at 0.0843** - six times the K = 4 figure, and still
an eighth of what a supervised model on the same columns achieves. A flat partition could not
have told us that shape first appears at the *fourth* level down; this is the one question a
dendrogram answered that 8.1 could not.

## Two things that disqualify the deeper cuts

**A singleton appears at K = 4** and never leaves, so the four-way Ward cut is three clusters and
one node. By K = 8 the tree has spent a whole cluster on **18 nodes at rect_aspect 163.9 with two
vertices** - degenerate slivers from failed contour extraction, which are real rows in the table
and are exactly what an agglomerative rule will isolate first once the broad structure is used up.

**The cophenetic correlation is 0.4737.** The tree reproduces less than half the variance of the
distances it was built from, so the dendrogram is a loose picture of this geometry rather than a
faithful one. That is not a bug in the run - Ward optimises variance rather than distance
fidelity, and 8.4 measures exactly how much fidelity that costs against the three rules that do
optimise distance - but it does mean the cut levels above should be read as *where a variance
criterion chose to cut*, not as natural joints in the data. Quoting them as natural joints would
be quoting the drawing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.cluster.kmeans import label_mix, majority_baseline
from src.features.descriptors import NAMES
from src.parse.vocab import matrix, purity
from src.utils.figures import save as _figsave

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p8_dendrogram.png"

#: Three is the plan's middle tier, four is 8.2's recommendation and hdbpmn's true class count.
CUTS = (2, 3, 4, 6, 8)

METHOD = "ward"


def scaled(X) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(X)


def tree(X, method: str = METHOD):
    """The linkage matrix, over standardised columns."""
    from scipy.cluster.hierarchy import linkage

    return linkage(scaled(X), method=method)


def cut(Z, k: int) -> np.ndarray:
    """A flat k-cluster assignment, renumbered from zero so it matches 8.1's conventions."""
    from scipy.cluster.hierarchy import fcluster

    return fcluster(Z, t=k, criterion="maxclust") - 1


def cophenetic(Z, X) -> float:
    """How faithfully the tree reproduces the distances it was built from."""
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    correlation, _ = cophenet(Z, pdist(scaled(X)))
    return round(float(correlation), 4)


def descriptor_means(
    X,
    assignment: np.ndarray,
    k: int,
    columns=("circularity", "solidity", "rect_aspect", "vertices"),
) -> list:
    """Only the columns a reader can picture, rather than all 22."""
    index = {name: i for i, name in enumerate(NAMES)}
    out = []
    for c in range(k):
        rows = X[assignment == c]
        out.append(
            {
                "cluster": c,
                "size": int(len(rows)),
                **{
                    name: round(float(rows[:, index[name]].mean()), 4) if len(rows) else None
                    for name in columns
                },
            }
        )
    return out


def refinement(parent: np.ndarray, child: np.ndarray, labels: np.ndarray) -> list[dict]:
    """For each parent cluster, what its children did to it.

    The plan's hypothesis is that a deeper cut turns a *family* into *shapes*. The test is
    whether the children of one parent have different dominant labels. If every child of a
    parent keeps the parent's dominant label, the deeper cut has re-split the same axis at finer
    resolution - a ranking, not a taxonomy.
    """
    out = []
    for p in sorted(set(parent.tolist())):
        mask = parent == p
        kids = sorted(set(child[mask].tolist()))
        dominants = []
        for kid in kids:
            got = labels[mask & (child == kid)]
            names, counts = np.unique(got, return_counts=True)
            dominants.append(str(names[int(np.argmax(counts))]))
        got = labels[mask]
        names, counts = np.unique(got, return_counts=True)
        out.append(
            {
                "parent": int(p),
                "size": int(mask.sum()),
                "parent_dominant": str(names[int(np.argmax(counts))]),
                "children": len(kids),
                "child_dominants": dominants,
                "split_by_label": len(set(dominants)) > 1,
            }
        )
    return out


def levels(Z, X, labels, cuts=CUTS) -> list[dict]:
    from sklearn.metrics import adjusted_rand_score

    rows = []
    for k in cuts:
        assignment = cut(Z, k)
        sizes = np.bincount(assignment, minlength=k)
        rows.append(
            {
                "k": int(k),
                "ari": round(float(adjusted_rand_score(labels, assignment)), 4),
                "purity": float(purity(assignment, labels, k)["purity"]),
                "smallest_cluster": int(sizes.min()),
                "clusters": label_mix(assignment, labels, k),
                "means": descriptor_means(X, assignment, k),
            }
        )
    return rows


def figure(Z, labels, path: Path = FIGURE, cuts=CUTS) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # The merge heights of the last few merges are exactly the thresholds that produce each cut,
    # so the cut lines are read off the tree rather than guessed.
    heights = {k: (Z[-k, 2] + Z[-k + 1, 2]) / 2 for k in cuts if k > 1}

    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True, color_threshold=heights[4], ax=ax)
    for k, h in heights.items():
        ax.axhline(h, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(ax.get_xlim()[1], h, f" cut {k}", va="center", fontsize=8)
    ax.set_title(f"8.3 - Ward dendrogram over {len(labels):,} shapes (last 30 merges)")
    ax.set_ylabel("merge distance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(cuts=CUTS) -> dict:
    X, labels, keys, dropped = matrix()
    Z = tree(X)
    rows = levels(Z, X, labels, cuts)
    refinements = {
        f"{a}->{b}": refinement(cut(Z, a), cut(Z, b), labels)
        for a, b in zip(cuts, cuts[1:], strict=False)
    }
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "method": METHOD,
        "majority_baseline": majority_baseline(labels),
        "cophenetic_correlation": cophenetic(Z, X),
        "levels": rows,
        "refinement": refinements,
        "figure": str(figure(Z, labels, cuts=cuts)),
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

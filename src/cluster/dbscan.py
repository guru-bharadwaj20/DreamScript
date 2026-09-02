"""Phase 8.5 - DBSCAN, and what the shapes outside the vocabulary actually turn out to be.

    python -m src.cluster.dbscan

The plan asks DBSCAN to "discover outlier and novel shapes outside the vocabulary" and wants an
outlier gallery. Two of those words are doing a lot of work and the task is built to keep them
apart:

    outlier   a point in a low-density region of the 22-column descriptor space. DBSCAN can find
              these and that is all it can find.
    novel     a shape whose *geometry* is unlike anything the four annotated names cover.

**These coincide only if the dominant axis of the descriptor space is shape**, and 8.1, 8.3 and
7.4.5 have each independently found that it is not - it is outline integrity. So the honest
prediction going in is that the outliers will be badly extracted rectangles rather than exotic
shapes, and the gallery is built from real crops precisely so that prediction can be checked by
looking rather than argued about.

## Choosing eps

DBSCAN's `eps` is not a free parameter to be tuned against a score - there is no score here - so
it is chosen the standard way: the elbow of the sorted k-th nearest-neighbour distance curve,
with `k = min_samples`. That curve is reported, not just the chosen value, because on a corpus
with a heavy density tail the elbow can be shallow and a reader is entitled to see how shallow.

A small sweep around the chosen value is run as well. **A clustering whose noise fraction swings
from 3% to 40% over a factor of two in eps is not reporting a property of the corpus**, and the
only way to know is to vary it.

## min_samples

Set to 2 x dimensionality, which is the usual rule for a space this wide, and the alternative is
reported. It matters less than eps here and pretending otherwise would be padding.

## Why DBSCAN is expected to behave like 8.4's single linkage

DBSCAN is single-linkage clustering with a density threshold and a noise label. 8.4 found single
linkage putting 12,397 of 12,400 points in one cluster because the fragmented-outline tail forms
a continuous bridge rather than a gap. The density threshold is exactly the mechanism that can
cut that bridge, so **the interesting question is whether it does** - and if it does, at what
cost in noise fraction.

## What it measured

12,400 shapes, `min_samples` 44, eps read off the k-distance elbow at **4.5301**.

    eps      clusters   noise   noise share   largest cluster
    2.2650      2         688      5.55%          11,645
    3.3975      1         227      1.83%          12,173
    4.5301      1         142      1.15%          12,258
    6.7951      1          70      0.56%          12,330
    9.0601      1          43      0.35%          12,357

    min_samples halved to 22, same eps:   1 cluster, 1.10% noise

**DBSCAN never produced a vocabulary.** At every setting but the smallest it returns one cluster
holding 98-99% of the corpus plus a thin rind of noise, which is 8.4's single-linkage result with
a noise label attached - as predicted, since DBSCAN is single linkage with a density threshold.
The threshold does not cut the fragmented-outline bridge; it only shaves the ends off it. Halving
`min_samples` moves the noise share by five hundredths of a percent, so eps is the only lever and
the sweep is the honest way to report it: **the noise fraction swings 16-fold, 0.35% to 5.55%,
across a four-fold change in eps**, which is why no single number from this method describes the
corpus.

The elbow itself is weakly determined. The k-distance curve runs 0.40 / 1.46 median / 3.09 at the
95th percentile / 113.46 maximum, so the chosen 4.53 sits out past p95 in a tail dominated by a
few dozen points. That is a property of this corpus, not a failure of the construction, and it is
the reason the elbow yields 1.15% noise rather than a substantial partition.

## The outliers are not what the design predicted, and the gallery is why

The prediction was that the outliers would be badly extracted rectangles - the damage axis that
8.1, 8.3 and 7.4.5 all found dominating this table. **The enrichment says the opposite:**

    label       corpus share   outlier share   enrichment
    diamond        17.3%          27.5%          1.59
    circle         26.1%          38.7%          1.49
    freeform        8.7%           5.6%          0.65
    rectangle      48.0%          28.2%          0.59

`rectangle` is *depleted* among the outliers, and so is `freeform` - the one class that would
most plausibly have meant "novel shape". The outliers are enriched in the two classes with the
most regular geometry.

**And the mean of one column would have described 118 of the 142 backwards.** Outlier
`rect_aspect` averages 23.28 against a clustered 1.95, which reads as "the outliers are elongated
slivers". The *medians* are 1.07 against 1.36 - the typical outlier is more square than the
typical clustered shape. Both facts are true of different subsets, so the subsets are reported
separately:

    kind              n     share   labels                          vertices   defects
    sliver (asp > 5)  24    16.9%   rectangle 24                       2.00      0.13
    compact           118   83.1%   circle 55, diamond 39,             9.79      4.21
                                    rectangle 16, freeform 8
    clustered      12,258     -                                        8.77      3.36

The 24 slivers are extraction failures and every one is labelled `rectangle`: two vertices,
circularity 0.012, aspect above 5. They are the same family 8.3's eight-way cut isolated at
aspect 163.9.

**The other 118 are the finding.** At the same circularity as the corpus (0.335 against 0.331)
they carry **more vertices (9.79 against 8.77) and more convexity defects (4.21 against 3.36)** -
extra internal structure inside an otherwise ordinary outline. The gallery says what that
structure is: **BPMN gateways with their X and + markers, and events with message, timer and
terminate glyphs inside the circle.**

## What that means, and it is the most useful thing Phase 8 has produced

The plan asked for "novel shapes outside the vocabulary". They exist, and they are not new
geometry - **they are BPMN's own iconography, which hdbpmn's four labels flatten.** An exclusive
gateway is annotated `diamond` and a timer event is annotated `circle`, exactly like a plain
gateway and a plain event, and the descriptor table can tell them apart even though the label
cannot. That is why `diamond` and `circle` are the enriched classes and why `freeform` is not:
the outliers are shapes the annotation *under*-describes, not shapes the geometry cannot handle.

This reframes several earlier results without contradicting any. 7.4.8 found both vocabularies
reaching about half a supervised ceiling; part of what the supervised model is picking up is
presumably this sub-structure, which no four-name scheme can reward. And it is a concrete
recommendation for Phase 10: **where the annotation says `diamond`, the marker inside it is
recoverable and carries the gateway's semantics** - which is the difference between an exclusive
and a parallel branch, and therefore between two different generated programs.

The honest limit is the count. 118 nodes out of 12,400 is what a density threshold tuned to a
1.15% noise rind can surface; it is a demonstration that the signal is there and separable, not a
detector. Building one is a supervised problem with the markers as its labels, and nothing in
this corpus provides those labels today.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.cluster.hierarchy import scaled
from src.features.descriptors import NAMES
from src.parse.vocab import matrix

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p8_outlier_gallery.png"

#: The usual rule for a space this wide, with the alternative reported beside it.
MIN_SAMPLES = 2 * len(NAMES)

#: Multipliers applied to the k-distance elbow, so the sweep is centred on the chosen value.
EPS_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)

GALLERY_ROWS, GALLERY_COLS = 4, 10


def k_distances(X, k: int = MIN_SAMPLES) -> np.ndarray:
    """Sorted distance to the k-th nearest neighbour - the curve eps is read off."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    return np.sort(distances[:, -1])


def eps_elbow(curve: np.ndarray) -> float:
    """The knee of the sorted k-distance curve: the point furthest from the chord of its ends.

    The second-difference definition 8.2 uses is too noisy on 12,400 raw distances; the
    furthest-from-chord construction is the standard one for this particular curve and is stable
    without smoothing.
    """
    n = len(curve)
    x = np.arange(n, dtype=float)
    x0, y0, x1, y1 = 0.0, float(curve[0]), float(n - 1), float(curve[-1])
    # perpendicular distance from each point to the chord joining the two ends
    numerator = np.abs((y1 - y0) * x - (x1 - x0) * curve + x1 * y0 - y1 * x0)
    denominator = np.hypot(y1 - y0, x1 - x0)
    return float(curve[int(np.argmax(numerator / denominator))])


def evaluate(Xs, labels, eps: float, min_samples: int = MIN_SAMPLES) -> dict:
    from sklearn.cluster import DBSCAN

    assignment = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(Xs)
    noise = assignment == -1
    clusters = sorted(set(assignment.tolist()) - {-1})
    sizes = [int((assignment == c).sum()) for c in clusters]
    return {
        "eps": round(float(eps), 4),
        "min_samples": min_samples,
        "clusters": len(clusters),
        "noise": int(noise.sum()),
        "noise_share": round(float(noise.mean()), 4),
        "largest_cluster": max(sizes) if sizes else 0,
        "largest_share": round(float(max(sizes) / len(assignment)), 4) if sizes else 0.0,
        "sizes": sizes[:10],
        "noise_label_mix": _mix(labels[noise]),
        "clustered_label_mix": _mix(labels[~noise]),
    }


def _mix(labels: np.ndarray) -> dict:
    names, counts = np.unique(labels, return_counts=True)
    total = max(int(counts.sum()), 1)
    return {str(n): round(float(c / total), 4) for n, c in zip(names, counts, strict=True)}


def enrichment(noise_mix: dict, corpus_mix: dict) -> dict:
    """How much each label is over- or under-represented among the outliers.

    A gallery of 500 crops is not a measurement. This is: if the outliers were novel shapes,
    `freeform` should be enriched and `rectangle` depleted.
    """
    return {
        name: round(noise_mix.get(name, 0.0) / share, 3)
        for name, share in corpus_mix.items()
        if share
    }


def outlier_direction(X, assignment: np.ndarray) -> dict:
    """Where the outliers sit on the columns 8.1 and 8.3 found the corpus splitting on."""
    index = {name: i for i, name in enumerate(NAMES)}
    noise = assignment == -1
    out = {}
    for name in ("circularity", "solidity", "rect_aspect", "defect_count", "vertices"):
        column = X[:, index[name]]
        out[name] = {
            "outliers": round(float(column[noise].mean()), 4) if noise.any() else None,
            "clustered": round(float(column[~noise].mean()), 4) if (~noise).any() else None,
        }
    return out


#: Above this aspect ratio a "shape" is a sliver, which on this corpus means an extraction
#: failure rather than a drawing. The clustered p90 is 2.90, so 5 is comfortably outside it.
SLIVER_ASPECT = 5.0


def outlier_kinds(X, assignment: np.ndarray, labels: np.ndarray) -> dict:
    """Split the outliers into extraction failures and everything else.

    The mean `rect_aspect` of the outliers is 23.3 against a clustered 1.9, which reads as
    "outliers are elongated". The median says 1.07 against 1.36, which reads as the opposite.
    Both are true of different subsets, so the subsets are separated rather than averaged.
    """
    index = {name: i for i, name in enumerate(NAMES)}
    aspect = X[:, index["rect_aspect"]]
    noise = assignment == -1
    sliver = noise & (aspect > SLIVER_ASPECT)
    compact = noise & (aspect <= SLIVER_ASPECT)
    out = {"outliers": int(noise.sum())}
    for name, mask in (("sliver", sliver), ("compact", compact)):
        out[name] = {
            "n": int(mask.sum()),
            "share_of_outliers": round(float(mask.sum() / max(noise.sum(), 1)), 4),
            "labels": _counts(labels[mask]),
            **{
                column: round(float(X[mask, index[column]].mean()), 4) if mask.any() else None
                for column in ("vertices", "defect_count", "circularity")
            },
        }
    # An eps large enough to cluster everything, or small enough to cluster nothing, leaves one
    # of these empty; the row is reported as null rather than raising out of a summary function.
    out["clustered"] = {
        column: round(float(X[~noise, index[column]].mean()), 4) if (~noise).any() else None
        for column in ("vertices", "defect_count", "circularity")
    }
    out["aspect"] = {
        "outlier_mean": _stat(aspect[noise], np.mean),
        "outlier_median": _stat(aspect[noise], np.median),
        "clustered_mean": _stat(aspect[~noise], np.mean),
        "clustered_median": _stat(aspect[~noise], np.median),
        "clustered_p90": _stat(aspect[~noise], lambda a: np.percentile(a, 90)),
    }
    return out


def _stat(values: np.ndarray, function) -> float | None:
    return round(float(function(values)), 4) if len(values) else None


def _counts(labels: np.ndarray) -> dict:
    names, counts = np.unique(labels, return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts, strict=True)}


def gallery(keys, assignment, X, path: Path = FIGURE) -> Path:
    """Real crops of the strongest outliers, so the prediction can be checked by looking."""
    import matplotlib

    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt

    from src.features.descriptors import CROP_PAD
    from src.ir.model import Diagram
    from src.preprocess import layers as ly
    from src.preprocess.exif import load

    noise = np.flatnonzero(assignment == -1)
    if not len(noise):
        raise ValueError("no outliers to draw")
    # Furthest from the corpus centre first: the most extreme outliers, not an arbitrary slice.
    centre = X.mean(axis=0)
    spread = X.std(axis=0)
    spread[spread == 0] = 1.0
    distance = np.linalg.norm((X[noise] - centre) / spread, axis=1)
    chosen = noise[np.argsort(-distance)][: GALLERY_ROWS * GALLERY_COLS]

    wanted: dict[str, list[str]] = {}
    for index in chosen:
        page, element = keys[index].split(":", 1)
        wanted.setdefault(page, []).append(element)

    patches = []
    for page, elements in wanted.items():
        path_ir = ROOT / "data" / "processed" / "ir" / "hdbpmn" / f"{page}.ir.json"
        if not path_ir.is_file():
            continue
        diagram = Diagram.load(path_ir)
        image_path = ROOT / diagram.meta["image"]
        if not image_path.is_file():
            continue
        original = load(image_path, grayscale=True)
        gray, mask = ly.prepare(image_path)
        scale = gray.shape[1] / original.shape[1]
        boxes = {n.id: n.bbox for n in diagram.nodes if n.bbox}
        for element in elements:
            if element not in boxes:
                continue
            x, y, w, h = (v * scale for v in boxes[element])
            px, py = CROP_PAD * w, CROP_PAD * h
            x0, y0 = max(0, int(x - px)), max(0, int(y - py))
            x1 = min(mask.shape[1], int(x + w + px))
            y1 = min(mask.shape[0], int(y + h + py))
            if x1 - x0 > 4 and y1 - y0 > 4:
                patches.append(cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8) * 255, (64, 64)))

    fig, axes = plt.subplots(
        GALLERY_ROWS, GALLERY_COLS, figsize=(GALLERY_COLS * 1.0, GALLERY_ROWS * 1.05)
    )
    for axis, patch in zip(axes.ravel(), patches + [None] * axes.size, strict=False):
        axis.axis("off")
        if patch is not None:
            axis.imshow(255 - patch, cmap="gray", vmin=0, vmax=255)
    fig.suptitle("8.5 - the strongest DBSCAN outliers", y=1.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(factors=EPS_FACTORS) -> dict:
    from sklearn.cluster import DBSCAN

    X, labels, keys, dropped = matrix()
    Xs = scaled(X)

    curve = k_distances(Xs)
    eps = eps_elbow(curve)
    sweep = [evaluate(Xs, labels, eps * f) for f in factors]
    alternative = evaluate(Xs, labels, eps, min_samples=len(NAMES))

    chosen = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, n_jobs=-1).fit_predict(Xs)
    corpus_mix = _mix(labels)
    return {
        "rows": int(len(X)),
        "dropped_non_finite": dropped,
        "min_samples": MIN_SAMPLES,
        "eps_elbow": round(float(eps), 4),
        "k_distance_curve": {
            "min": round(float(curve[0]), 4),
            "median": round(float(np.median(curve)), 4),
            "p95": round(float(np.percentile(curve, 95)), 4),
            "max": round(float(curve[-1]), 4),
        },
        "corpus_label_mix": corpus_mix,
        "sweep": sweep,
        "min_samples_halved": alternative,
        "chosen": evaluate(Xs, labels, eps),
        "outlier_enrichment": enrichment(_mix(labels[chosen == -1]), corpus_mix),
        "outlier_direction": outlier_direction(X, chosen),
        "outlier_kinds": outlier_kinds(X, chosen, labels),
        "figure": str(gallery(keys, chosen, X)),
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

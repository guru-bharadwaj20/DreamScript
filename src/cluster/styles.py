"""Phase 8.6 - clustering scribes by how they write, and whether the clusters are the hand.

    python -m src.cluster.styles --build   # writes data/features/scribe_styles.parquet
    python -m src.cluster.styles           # the clustering, from the built table

The plan asks for style clusters over stroke statistics - slant, thickness, curvature, spacing -
so that 8.7 can route text crops to a per-cluster OCR head. 7.4.7 already established the two
facts this task has to work between:

    a rectangle identifies its writer at 8.6x chance, so scribe identity is real
    per-scribe density models cost 7.93 nats, because 59 rectangles cannot fit 44 parameters

and concluded that **grouping many writers into a few style clusters is the form that could
work**, because it buys parameters per cluster rather than per person. This task builds those
clusters. It is the one place in Phase 8 where the clustering has a downstream consumer rather
than a yardstick, which changes what has to be proved: not that the partition agrees with a
label, but that it is stable, that it is about the hand, and that it is not about the page.

## The statistics, and what each is measuring

Computed over 3.2.8's **text layer** - the writing, separated from the drawn shapes - because a
statistic pooled over box outlines and handwriting together would measure how someone draws
rectangles, which 7.4.7 has already done.

    thickness       twice the mean distance transform on the ink: the pen's stroke half-width in
                    working-width pixels, which is the nib and the pressure.
    slant           the shear angle whose vertical projection profile has the highest variance.
                    Upright writing has its strokes aligned at 0; a right-leaning hand peaks at a
                    positive shear. This is the classical estimator and it is chosen over a
                    gradient histogram because it is insensitive to stroke thickness.
    curvature       mean absolute turning per unit contour length. A printed hand turns in sharp
                    corners and long straights; a cursive one turns continuously.
    spacing         median nearest-neighbour gap between text components, in units of component
                    height, so it is a *relative* spacing and does not encode how large the
                    photograph is.
    height, aspect, density, components_per_area   the size and texture of the writing.

Every one is a ratio or a normalised length. Nothing carries the resolution of the photograph,
which matters because hdbpmn's writers did not share a camera.

## The two controls, and why the task is worthless without them

**Exercise.** Each hdbpmn page is one writer doing one of a small set of exercises. If the style
clusters track the exercise rather than the writer, they are clusters of *content* - how much
text a task requires, how big the boxes are - wearing a handwriting label. The mutual information
between cluster and exercise is reported against the mutual information between cluster and
scribe.

**Page.** A scribe with several pages should land in one cluster. If their pages scatter, the
partition is measuring the photograph rather than the hand, and 8.7 cannot route by it because at
inference there is no scribe, only a page. Reported as the share of multi-page scribes whose
pages agree.

Neither control can be passed by construction, and both are the kind that 7.4.7's own positive
control caught a real bug with.

## What it measured

692 hdbpmn pages, **105 scribes**, a median of 8 pages each, 4 writers with only one.

     K   silhouette   sizes                     smallest
     2     0.1965     60 / 45                      45
     3    *0.2080*    41 / 12 / 52                 12
     4     0.1807     10 / 34 / 20 / 41            10
     5     0.2032     13 / 47 / 20 / 17 / 8         8
     6     0.2112     13 / 46 / 20 / 17 / 8 / 1     1

**K = 3**, the best-separated partition with at least three clusters and none smaller than five
writers - K = 6 scores marginally higher and puts one scribe alone, which 8.7 cannot fit a model
inside. The silhouettes are low across the board (0.18-0.21) and that is the correct reading:
handwriting style is a continuum, and these are three regions of it rather than three kinds.

    cluster  scribes  curvature  height  aspect  spacing  density  thickness  slant
       0       41       0.301    0.0258   1.09    1.56     0.235     3.086     0.26
       1       12       0.343    0.0180   1.56    1.93     0.256     3.103     2.13
       2       52       0.373    0.0189   1.13    1.39     0.285     2.911    -0.69

Cluster 0 writes **large and straight** - the tallest components in the corpus at 0.0258 of page
height, the lowest curvature. Cluster 2 writes **small, dense and round** - the highest curvature
and density, the tightest spacing, and it is half the corpus. Cluster 1 is 12 writers who write
**small, wide and far apart**: aspect 1.56 against 1.09 and 1.13, spacing 1.93 against 1.39.

## The plan named the two statistics that turned out not to matter

    column                ANOVA F across the three clusters
    curvature                    56.04
    height                       50.20
    aspect                       35.21
    spacing                      20.17
    density                      18.83
    components_per_area           7.44
    slant                         3.01   (p = 0.054)
    thickness                     2.30   (p = 0.106)

**`slant` and `thickness` are the only two columns that do not significantly separate the
clusters, and they are the first two the plan names.** Thickness is a pen and a photograph, and
hdbpmn's writers mostly used similar pens; the variation that survives is between cameras rather
than between hands.

Slant has a specific and reportable failure: **53.9% of pages measure exactly 0**. The estimator
is the classical one - the shear whose vertical projection profile has the highest variance - and
it needs long vertical strokes to have a peak to find. BPMN labels are two or three short words
inside a box, so on most pages the profile is nearly flat and the search returns its centre. The
estimator is not wrong; it is being asked for a statistic this corpus's writing is too short to
support. A line-level corpus like IAM would supply it and hdbpmn does not.

What does separate the writers is **how round and how large they write**, which is the pair that
needs the least text to measure.

## The two controls: one passes cleanly, one passes with a caveat

    cluster against exercise, AMI            -0.0080
    cluster against scribe, AMI               0.3039

**The clusters have nothing whatever to do with the exercise** - the AMI is negative, which is
the chance-corrected way of saying no relationship at all. Eleven modelling prompts produce
eleven kinds of content, and none of it reached these columns. That was the control most likely
to fail, because a prompt requiring more text changes spacing and component density directly, and
it did not fail.

    a scribe's pages, all in one cluster       0.2772   (chance 0.0159, so 17.4x)
    a scribe's pages, in their modal cluster   0.7927   (chance 0.3333, so 2.4x)

Both readings of the page control are reported because they say different things. Requiring *all*
of a writer's pages to agree is all-or-nothing and a writer with ten pages fails on one stray:
27.7% pass, against a 1.6% chance rate. **The quantity 8.7 actually needs is the modal share -
can a page be routed to its own writer's style head - and that is 0.7927 against a chance rate of
0.3333.**

So four pages in five route correctly and **one in five does not**. That is the caveat this task
hands forward, and it bounds what 8.7 can achieve: a per-cluster OCR head that is better than the
global one on its own pages will still see roughly a fifth of its traffic misrouted, and the gain
has to survive that.

## What this delivers, and what it does not

The plan's definition of done asks for at least three style clusters and an OCR gain. **The
clusters are here, they are about the hand rather than the exercise, and pages route to them at
2.4x chance.** The OCR gain is 8.7's measurement and is reported there, because measuring it
requires fine-tuning recognition heads that this task does not build - and because a gain quoted
by the task that chose the clusters would be a gain quoted by an interested party.

For 7.4.7 this closes the loop it opened. Per-scribe density models cost 7.93 nats because 59
rectangles cannot fit 44 parameters; **the three clusters here hold 41, 12 and 52 writers**, so a
per-cluster model has between 12 and 52 times the data a per-writer model had. Whether that is
enough is exactly what 8.7 measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "features" / "scribe_styles.parquet"
FIGURE = ROOT / "reports" / "figures" / "p8_style_clusters.png"
IR_DIR = ROOT / "data" / "processed" / "ir" / "hdbpmn"

#: The plan asks for at least three.
KS = (2, 3, 4, 5, 6)

#: Shear angles searched by the slant estimator, in degrees from vertical.
SHEARS = tuple(range(-45, 46, 3))

NAMES: tuple[str, ...] = (
    "thickness",
    "slant",
    "curvature",
    "spacing",
    "height",
    "aspect",
    "density",
    "components_per_area",
)

#: A page with almost no separated writing cannot support any of these statistics.
MIN_TEXT_PIXELS = 400
MIN_COMPONENTS = 4


def slant(mask: np.ndarray, shears=SHEARS) -> float:
    """The shear angle whose vertical projection profile has the highest variance.

    Sheared upright, a hand's vertical strokes stack into the same columns and the profile
    spikes; sheared any other way they smear. Insensitive to stroke thickness, which a gradient
    histogram is not.
    """
    import cv2

    height, width = mask.shape
    ink = mask.astype(np.uint8)
    best, best_variance = 0.0, -1.0
    for degrees in shears:
        shift = np.tan(np.radians(degrees))
        matrix = np.array([[1.0, shift, -shift * height / 2.0], [0.0, 1.0, 0.0]])
        warped = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST)
        profile = warped.sum(axis=0).astype(float)
        variance = float(profile.var())
        if variance > best_variance:
            best, best_variance = float(degrees), variance
    return best


def thickness(mask: np.ndarray) -> float:
    """Twice the mean distance to the nearest background pixel, over the ink."""
    import cv2

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[mask]
    return float(2.0 * values.mean()) if values.size else 0.0


def curvature(mask: np.ndarray) -> float:
    """Mean absolute turning angle per unit contour length, over every text contour."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    turns, length = 0.0, 0.0
    for contour in contours:
        points = contour[:, 0, :].astype(float)
        if len(points) < 4:
            continue
        deltas = np.diff(points, axis=0, append=points[:1])
        segments = np.hypot(deltas[:, 0], deltas[:, 1])
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        change = np.abs(np.diff(angles, append=angles[:1]))
        change = np.minimum(change, 2 * np.pi - change)
        turns += float(change.sum())
        length += float(segments.sum())
    return float(turns / length) if length > 1 else 0.0


def component_stats(mask: np.ndarray) -> dict:
    """Size, shape and spacing of the separated text components."""
    import cv2

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    # Row 0 is the background.
    boxes = stats[1:]
    centres = centroids[1:]
    if len(boxes) < MIN_COMPONENTS:
        return {}

    widths = boxes[:, cv2.CC_STAT_WIDTH].astype(float)
    heights = boxes[:, cv2.CC_STAT_HEIGHT].astype(float)
    height = float(np.median(heights))
    if height <= 0:
        return {}

    from scipy.spatial import cKDTree

    distances, _ = cKDTree(centres).query(centres, k=2)
    gap = float(np.median(distances[:, 1]))

    page_area = float(mask.shape[0] * mask.shape[1])
    return {
        "spacing": gap / height,
        "height": height / float(mask.shape[0]),
        "aspect": float(np.median(widths / np.maximum(heights, 1.0))),
        # Ink over the area of the boxes the ink sits in - how solidly a hand fills its own
        # letterforms. Dividing by the components' pixel counts instead would be identically 1.
        "density": float(boxes[:, cv2.CC_STAT_AREA].sum() / max((widths * heights).sum(), 1.0)),
        "components_per_area": len(boxes) / page_area * 1e4,
    }


def page_style(ir_path: Path) -> dict | None:
    """Every statistic for one page, or None if the text layer is too thin to support them."""
    from src.ir.model import Diagram
    from src.preprocess import layers as ly

    diagram = Diagram.load(ir_path)
    image_path = ROOT / diagram.meta["image"]
    if not image_path.is_file():
        return None
    gray, mask = ly.prepare(image_path)
    text = ly.separate(mask, gray).text
    if int(text.sum()) < MIN_TEXT_PIXELS:
        return None

    stats = component_stats(text)
    if not stats:
        return None
    return {
        "page": ir_path.name.replace(".ir.json", ""),
        "scribe": diagram.meta.get("scribe_id"),
        "exercise": diagram.meta.get("exercise"),
        "thickness": thickness(text),
        "slant": slant(text),
        "curvature": curvature(text),
        **stats,
    }


def build(limit: int | None = None):
    """Every page's statistics, in parallel, written to the feature table."""
    import pandas as pd

    from src.utils.parallel import pmap

    paths = sorted(IR_DIR.glob("*.ir.json"))[:limit]
    rows = [row for row in pmap(page_style, paths) if row]
    table = pd.DataFrame(rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TABLE, index=False)
    return table


def load_table():
    import pandas as pd

    if not TABLE.is_file():
        raise FileNotFoundError(f"{TABLE} not built; run --build")
    return pd.read_parquet(TABLE)


def per_scribe(table=None):
    """One row per scribe: the median of their pages, so one odd photograph cannot move them."""
    table = load_table() if table is None else table
    grouped = table.groupby("scribe")[list(NAMES)].median()
    counts = table.groupby("scribe").size()
    return grouped.to_numpy(dtype=float), grouped.index.to_numpy(), counts.to_numpy()


def cluster(X, k: int, seed: int = SEED):
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=25, random_state=seed)),
        ]
    ).fit(X)


def sweep(X, ks=KS) -> list[dict]:
    from sklearn.metrics import silhouette_score

    rows = []
    for k in ks:
        model = cluster(X, k)
        assignment = model.named_steps["kmeans"].labels_
        scaled = model.named_steps["scale"].transform(X)
        sizes = np.bincount(assignment, minlength=k)
        rows.append(
            {
                "k": int(k),
                "silhouette": round(float(silhouette_score(scaled, assignment)), 4),
                "inertia": round(float(model.named_steps["kmeans"].inertia_), 2),
                "sizes": [int(s) for s in sizes],
                "smallest": int(sizes.min()),
            }
        )
    return rows


def _mutual_information(a, b) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    return round(float(adjusted_mutual_info_score(a, b)), 4)


def controls(table, scribes: np.ndarray, assignment: np.ndarray) -> dict:
    """Is the partition about the hand, the exercise, or the photograph?"""
    lookup = dict(zip(scribes, assignment, strict=True))
    page_cluster = table["scribe"].map(lookup).to_numpy()

    # Per-page clustering, fitted independently, so page agreement is a real test rather than a
    # restatement of the fact that scribe rows were medians.
    from sklearn.preprocessing import StandardScaler

    page_X = StandardScaler().fit_transform(table[list(NAMES)].to_numpy(dtype=float))
    from sklearn.cluster import KMeans

    page_assignment = KMeans(
        n_clusters=len(set(assignment.tolist())), n_init=25, random_state=SEED
    ).fit_predict(page_X)

    k = len(set(assignment.tolist()))
    agree, multi, modal = 0, 0, []
    for _, group in table.assign(_c=page_assignment).groupby("scribe"):
        if len(group) < 2:
            continue
        multi += 1
        agree += int(group["_c"].nunique() == 1)
        modal.append(group["_c"].value_counts().iloc[0] / len(group))

    # Two readings of the same control, because "all of a scribe's pages agree" is all-or-nothing
    # and a writer with ten pages fails it on one stray. The modal share is the quantity 8.7
    # actually needs - can a page be routed to its writer's style head - and the chance rate for
    # each is computed rather than eyeballed.
    pages = table.groupby("scribe").size()
    pages = pages[pages > 1].to_numpy().astype(float)
    chance_agree = float(np.mean((1.0 / k) ** (pages - 1.0)))

    return {
        "cluster_vs_exercise_ami": _mutual_information(page_cluster, table["exercise"].to_numpy()),
        "cluster_vs_scribe_ami": _mutual_information(page_cluster, table["scribe"].to_numpy()),
        "multi_page_scribes": multi,
        "pages_agree_share": round(agree / multi, 4) if multi else None,
        "pages_agree_share_at_chance": round(chance_agree, 6),
        "pages_modal_share": round(float(np.mean(modal)), 4) if modal else None,
        "pages_modal_share_at_chance": round(1.0 / k, 4),
        "page_vs_scribe_clustering_ami": _mutual_information(page_assignment, page_cluster),
    }


def profile(X, assignment: np.ndarray, k: int) -> list[dict]:
    out = []
    for c in range(k):
        rows = X[assignment == c]
        out.append(
            {
                "cluster": c,
                "scribes": int(len(rows)),
                **{
                    name: round(float(rows[:, i].mean()), 4) if len(rows) else None
                    for i, name in enumerate(NAMES)
                },
            }
        )
    return out


def figure(X, assignment, rows, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    pairs = [("slant", "thickness"), ("curvature", "spacing"), ("height", "density")]
    for ax, (a, b) in zip(axes, pairs, strict=True):
        i, j = NAMES.index(a), NAMES.index(b)
        for c in sorted(set(assignment.tolist())):
            mask = assignment == c
            ax.scatter(X[mask, i], X[mask, j], s=18, alpha=0.75, label=f"c{c}")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"8.6 - {len(X)} scribes in {len(rows)} style clusters", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(k: int | None = None) -> dict:
    table = load_table()
    X, scribes, counts = per_scribe(table)
    rows = sweep(X)
    if k is None:
        # The plan requires at least three; among those, the best-separated partition with no
        # cluster smaller than five scribes, since 8.7 has to fit a model inside each one.
        usable = [r for r in rows if r["k"] >= 3 and r["smallest"] >= 5]
        k = max(usable, key=lambda r: r["silhouette"])["k"] if usable else 3

    model = cluster(X, k)
    assignment = model.named_steps["kmeans"].labels_
    return {
        "pages": int(len(table)),
        "scribes": int(len(scribes)),
        "pages_per_scribe": {
            "median": float(np.median(counts)),
            "max": int(counts.max()),
            "single_page": int((counts == 1).sum()),
        },
        "sweep": rows,
        "chosen_k": int(k),
        "clusters": profile(X, assignment, k),
        "controls": controls(table, scribes, assignment),
        "assignment": {str(s): int(c) for s, c in zip(scribes, assignment, strict=True)},
        "figure": str(figure(X, assignment, profile(X, assignment, k))),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.build:
        table = build(args.limit)
        print(f"built {len(table)} pages -> {TABLE}")
        return 0

    result = run(args.k)
    text = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

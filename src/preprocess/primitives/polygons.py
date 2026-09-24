"""Phase 3.2.3 - polygon approximation and vertex counts.

Douglas-Peucker reduces a contour of hundreds of points to the handful of corners a person
would say it has. The vertex count is then the single most informative number about a shape:
four for a box or a diamond, many for a circle, three when a diamond loses a corner to the
approximation.

**The tolerance is the whole method.** Douglas-Peucker takes one parameter - how far a point may
sit from the simplified line before it must be kept - and it is expressed here as a fraction of
the contour's perimeter so it scales with the shape. Too small and hand wobble registers as
extra corners; too large and a diamond flattens into a triangle. Rather than pick one value and
hope, this module sweeps it and writes the histogram contributing.md 3.2.3 asks for, so the choice is
visible: `reports/figures/p3_vertex_histogram.png`.

The default is 0.02 and it is inherited from Phase 2.2.4's geometry classifier, where the same
sweep was done against known shapes.

The sweep is reported twice, over all contours and over closed shapes only, because a page of
BPMN is mostly *handwriting*: of 19,514 contours on 60 pages, only 1,213 are closed shapes, and
a tolerance chosen to fit the letters would be chosen to fit the wrong thing.

**The result is a warning, and it is worth reading before Phase 4 builds features from it.**
Even restricted to those 1,213 real shapes, the share recovered as quadrilaterals is:

    epsilon    0.005   0.01   0.02   0.04   0.08
    4-gons      0.2%   3.1%  11.3%  21.5%  36.1%

At no tolerance does a majority of hand-drawn boxes come back as four-sided. Raising the
tolerance buys 4-gons only by flattening everything - at 0.08 there are already 286 triangles,
which are diamonds and boxes that have lost a corner. Vertex count is therefore a **weak**
feature on hand-drawn shapes, not the decisive one it is on rendered ones, and this is the same
fact that held the geometric shape classifier of Phase 2.2.4 to kappa 0.30. Phase 4 should treat
it as one weak signal among several rather than as the shape.

    python -m src.preprocess.primitives.polygons --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import cv2
import numpy as np

from src.preprocess.primitives.contours import Contour
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURE = ROOT / "reports" / "figures" / "p3_vertex_histogram.png"

DEFAULT_EPSILON = 0.02

#: The sweep reported in the histogram.
SWEEP = (0.005, 0.01, 0.02, 0.04, 0.08)

#: A contour must enclose at least this many pixels to be a shape rather than a letter. At the
#: 1400px working width used by `collect`, a drawn box is tens of thousands of pixels and a
#: character is a few hundred.
MIN_SHAPE_AREA = 2000.0


def approximate(points: np.ndarray, epsilon_frac: float = DEFAULT_EPSILON) -> np.ndarray:
    """Simplified polygon as an (N, 2) array of corners."""
    contour = np.asarray(points, np.int32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon_frac * perimeter, True).reshape(-1, 2)


def vertex_count(points: np.ndarray, epsilon_frac: float = DEFAULT_EPSILON) -> int:
    return int(len(approximate(points, epsilon_frac)))


def interior_angles(polygon: np.ndarray) -> np.ndarray:
    """Interior angle at each vertex, in degrees. Useful for telling a diamond from a box."""
    points = np.asarray(polygon, np.float64).reshape(-1, 2)
    n = len(points)
    if n < 3:
        return np.array([])
    angles = []
    for i in range(n):
        incoming = points[i] - points[i - 1]
        outgoing = points[(i + 1) % n] - points[i]
        norms = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        if norms == 0:
            angles.append(180.0)
            continue
        cosine = np.clip(np.dot(-incoming, outgoing) / norms, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosine))))
    return np.array(angles)


def describe(contour: Contour, epsilon_frac: float = DEFAULT_EPSILON) -> dict:
    polygon = approximate(contour.points, epsilon_frac)
    angles = interior_angles(polygon)
    return {
        "index": contour.index,
        "vertices": int(len(polygon)),
        "epsilon": epsilon_frac,
        "min_angle": round(float(angles.min()), 1) if len(angles) else 0.0,
        "max_angle": round(float(angles.max()), 1) if len(angles) else 0.0,
        "closed": contour.closed,
        "bbox": contour.bbox,
    }


def sweep_counts(contours: list[Contour], sweep=SWEEP) -> dict[float, Counter]:
    return {
        epsilon: Counter(vertex_count(c.points, epsilon) for c in contours) for epsilon in sweep
    }


def figure(counts: dict[float, Counter], total: int) -> object:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(counts), figsize=(3.3 * len(counts), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    limit = 14
    for ax, (epsilon, counter) in zip(axes, sorted(counts.items()), strict=True):
        xs = list(range(3, limit + 1))
        ys = [counter.get(x, 0) for x in xs[:-1]] + [
            sum(v for k, v in counter.items() if k >= limit)
        ]
        ax.bar(xs, ys, color="#4C78A8")
        ax.set_title(f"epsilon = {epsilon:g}")
        ax.set_xlabel("vertices")
        ax.set_xticks([3, 4, 5, 6, 8, 10, 12, 14])
        ax.set_xticklabels(["3", "4", "5", "6", "8", "10", "12", "14+"])
        four = counter.get(4, 0)
        ax.axvline(4, color="#E45756", linestyle=":", linewidth=1)
        ax.text(
            0.97,
            0.95,
            f"4-gons: {four / total:.0%}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
    axes[0].set_ylabel(f"contours (n = {total})")
    fig.suptitle("Douglas-Peucker tolerance against recovered vertex count", fontsize=11)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, FIGURE, dpi=130)
    plt.close(fig)
    return FIGURE


def collect(limit: int = 60) -> list[Contour]:
    """Contours from real pages, for the sweep."""
    from src.ir.model import SUFFIX, Diagram
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.exif import load
    from src.preprocess.primitives import contours as ct
    from src.preprocess.rules import suppress
    from src.utils.parallel import pmap

    paths = sorted((ROOT / "data" / "processed" / "ir" / "hdbpmn").glob(f"*{SUFFIX}"))[:limit]

    def one(path):
        diagram = Diagram.load(path)
        image = ROOT / diagram.meta["image"]
        if not image.is_file():
            return []
        gray = load(image, grayscale=True)
        scale = 1400 / max(gray.shape)
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = denoise(suppress(binarize(median(gray))))
        return ct.drawn_outlines(ct.extract(mask))

    return [c for group in pmap(one, paths, prefer="threads") for c in group]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args(argv)

    contours = collect(args.limit)
    if not contours:
        print("no contours; run the hdBPMN converter first", file=sys.stderr)
        return 1

    # Two populations, and only one of them is what the tolerance is chosen for. Most contours
    # on a page of BPMN are letters; the shapes are the closed ones above a size. Sweeping over
    # everything measures handwriting, which is not the question.
    shapes = [c for c in contours if c.closed and c.area >= MIN_SHAPE_AREA]
    counts = sweep_counts(contours)
    shape_counts = sweep_counts(shapes) if shapes else {}
    path = figure(shape_counts or counts, len(shapes) if shapes else len(contours))

    print(f"wrote {path.relative_to(ROOT)}")
    print(json.dumps({"contours": len(contours), "closed_shapes": len(shapes)}, indent=2))
    for title, group, population in (
        ("all contours (mostly handwriting)", counts, len(contours)),
        ("closed shapes only", shape_counts, len(shapes)),
    ):
        if not population:
            continue
        print(f"\n  {title}, n = {population}")
        print(
            f"{'epsilon':>9} {'3':>6} {'4':>6} {'5':>6} {'6-9':>6} {'10+':>6}  {'4-gon share':>12}"
        )
        for epsilon, counter in sorted(group.items()):
            buckets = [
                counter.get(3, 0),
                counter.get(4, 0),
                counter.get(5, 0),
                sum(v for k, v in counter.items() if 6 <= k <= 9),
                sum(v for k, v in counter.items() if k >= 10),
            ]
            share = counter.get(4, 0) / population
            print(f"{epsilon:9g} " + " ".join(f"{b:6d}" for b in buckets) + f"  {share:11.1%}")

    checks = {
        "figure_written": path.is_file(),
        "sweep_covers_five_tolerances": len(counts) == len(SWEEP),
        "default_recovers_quadrilaterals": counts[DEFAULT_EPSILON].get(4, 0) > 0,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

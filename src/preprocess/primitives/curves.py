"""Phase 3.2.5 - curvature, corners, and the line-versus-curve ratio.

The question this answers is the one that separates a diamond from a circle when vertex counting
fails, which - per 3.2.3 - it usually does on hand-drawn shapes: **is the outline made of
straight runs meeting at corners, or is it curving continuously?**

Curvature is measured along the contour by the *turning angle* between the chord arriving at a
point and the chord leaving it, both taken over a window of several points. Using a window
rather than adjacent pixels is what makes this usable: on a rasterised line, consecutive pixels
turn by 45 degrees constantly, so pixel-level curvature is nothing but staircase noise. The
window is a fraction of the contour's own length, so it scales with the shape.

Three numbers come out, and they are the honest answer to the vertex-count problem:

    straightness   share of the outline turning less than `STRAIGHT_DEG` per window
    corners        local maxima of turning above `CORNER_DEG`, one per corner
    mean_curvature average absolute turn per window

Measured on ideal outlines, the separation is clean where 3.2.3's vertex count was not:

    shape       straightness   corners   straight/curved
    box             0.72          4           6.5
    diamond         0.73          4           7.4
    ellipse         0.39          0           0.6
    circle          0.00          0           0.0

Note that `mean_curvature` is ~14 for *all four* - a closed outline turns 360 degrees in total
whatever its shape, so the average turn per window says nothing. It is the *distribution* that
carries the shape: whether the turning is spread evenly (a circle) or concentrated at a few
points with straight runs between them (a polygon). That is why `straightness` and `corners` are
the outputs and the mean is reported only for completeness.

This distinction survives wobble far better than counting vertices does, because it is a
property of the whole outline rather than of where an approximation happened to put its
breakpoints.

    python -m src.preprocess.primitives.curves <image>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

#: Window over which a turn is measured, as a fraction of the contour's point count.
WINDOW_FRAC = 0.04
MIN_WINDOW = 3

#: A window turning less than this is straight; more than this is a corner. Between the two is
#: a curve. Measured on ideal shapes: a box's sides turn under 2 degrees per window and its
#: corners turn 90; a circle of 150px radius turns about 15 degrees per window everywhere.
STRAIGHT_DEG = 8.0
CORNER_DEG = 40.0

#: Two corner candidates closer than this share of the contour are the same corner.
CORNER_MERGE_FRAC = 0.05


def turning(points: np.ndarray, window: int | None = None, *, closed: bool = True) -> np.ndarray:
    """Absolute turning angle in degrees at each point.

    `closed=True` treats the points as a loop, which is right for a contour. `closed=False` is
    for an **open** path - a skeleton branch, an arrow's shaft - where wrapping would invent a
    turn between the two ends of the stroke and report it as a corner. The first and last
    `window` points of an open path have no turn defined and come back as zero.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 5:
        return np.zeros(n)
    if window is None:
        window = max(MIN_WINDOW, int(WINDOW_FRAC * n))
    window = min(window, n // 2 - 1) if n > 2 * MIN_WINDOW else MIN_WINDOW

    if closed:
        incoming = pts - np.roll(pts, window, axis=0)
        outgoing = np.roll(pts, -window, axis=0) - pts
    else:
        index = np.arange(n)
        before = np.clip(index - window, 0, n - 1)
        after = np.clip(index + window, 0, n - 1)
        incoming = pts - pts[before]
        outgoing = pts[after] - pts
    angle_in = np.arctan2(incoming[:, 1], incoming[:, 0])
    angle_out = np.arctan2(outgoing[:, 1], outgoing[:, 0])
    delta = np.degrees(angle_out - angle_in)
    # Fold to [-180, 180]: a turn of 350 degrees one way is 10 degrees the other.
    delta = (delta + 180.0) % 360.0 - 180.0
    result = np.abs(delta)
    if not closed:
        result[:window] = 0.0
        result[n - window :] = 0.0
    return result


def corners(points: np.ndarray, window: int | None = None) -> np.ndarray:
    """Indices of the corners: local maxima of turning above the corner threshold.

    Non-maximum suppression is by arc distance rather than by value, because a hand-drawn
    corner is rounded and spreads its turn over several points; without merging, one corner is
    reported three times.
    """
    angles = turning(points, window)
    n = len(angles)
    if n == 0:
        return np.array([], int)
    candidates = np.flatnonzero(angles >= CORNER_DEG)
    if not len(candidates):
        return np.array([], int)

    merge = max(2, int(CORNER_MERGE_FRAC * n))
    kept: list[int] = []
    for index in sorted(candidates, key=lambda i: -angles[i]):
        if all(min(abs(index - other), n - abs(index - other)) > merge for other in kept):
            kept.append(int(index))
    return np.array(sorted(kept), int)


def describe(points: np.ndarray, window: int | None = None) -> dict:
    angles = turning(points, window)
    if not len(angles):
        return {"straightness": 0.0, "corners": 0, "mean_curvature": 0.0, "points": 0}
    found = corners(points, window)
    return {
        "points": int(len(angles)),
        "straightness": round(float(np.mean(angles < STRAIGHT_DEG)), 4),
        "corners": int(len(found)),
        "corner_indices": found.tolist(),
        "mean_curvature": round(float(np.mean(angles)), 2),
        "max_curvature": round(float(np.max(angles)), 2),
    }


def line_vs_curve_ratio(points: np.ndarray, window: int | None = None) -> float:
    """Straight length over curved length. High for polygons, near zero for circles."""
    angles = turning(points, window)
    if not len(angles):
        return 0.0
    straight = float(np.count_nonzero(angles < STRAIGHT_DEG))
    curved = float(np.count_nonzero((angles >= STRAIGHT_DEG) & (angles < CORNER_DEG)))
    return straight / curved if curved else float("inf") if straight else 0.0


def skeleton_curvature(mask: np.ndarray) -> dict:
    """Curvature statistics of a shape's *skeleton* rather than its outline.

    The outline of a stroke is a closed loop around it; the skeleton is the stroke itself. For
    an open stroke - an arrow, a line - the skeleton is what has meaning, and this is what
    3.2.6 uses to find the two converging strokes of an arrowhead.
    """
    from src.preprocess.thinning import branch_points, end_points, thin

    skeleton = thin(mask)
    return {
        "skeleton_pixels": int(skeleton.sum()),
        "branch_points": int(branch_points(skeleton).sum()),
        "end_points": int(end_points(skeleton).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.exif import load
    from src.preprocess.primitives import contours as ct
    from src.preprocess.rules import suppress

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    args = ap.parse_args(argv)

    if not args.image.is_file():
        print(f"no such image: {args.image}", file=sys.stderr)
        return 1
    gray = load(args.image, grayscale=True)
    scale = 1400 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = denoise(suppress(binarize(median(gray))))
    shapes = [c for c in ct.drawn_outlines(ct.extract(mask)) if c.closed and c.area > 2000]

    print(json.dumps({"shapes": len(shapes), **skeleton_curvature(mask)}, indent=2))
    for contour in shapes[:10]:
        print(f"  {contour.bbox}  {describe(contour.points)}")
    return 0 if shapes else 1


if __name__ == "__main__":
    sys.exit(main())

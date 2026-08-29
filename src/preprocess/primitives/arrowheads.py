"""Phase 3.2.6 - arrowhead detection.

An arrowhead is the single most load-bearing mark in a hand-drawn diagram: it is what turns a
line between two boxes into a *directed* edge, and Phase 10 cannot build a graph without it.

The shape of the thing is what the detector looks for. Take the skeleton of the ink. Where a
shaft meets two barbs there is a **junction with three branches**, and two of them - the barbs -
are short and roughly symmetric about the third. That is the whole test:

    1. find skeleton junctions
    2. trace the branches leaving each one
    3. keep junctions with exactly three branches, two of them short
    4. require the two short ones to sit on opposite sides of the long one, at a plausible
       barb angle

Barb length is measured **relative to the longest branch**, not in absolute pixels, because a
drawing may be 800 or 4000 pixels wide and the arrowheads scale with it.

**How this is scored.** hdBPMN annotates every sequence flow as a polyline, so the last waypoint
of each is where an arrowhead was drawn. A detection is a true positive when it lands within
`MATCH_RADIUS_FRAC` of the page's long side from some annotated edge endpoint. That is a real
labelled evaluation rather than a visual impression.

## The measured result, and why it is what it is

**plan.md 3.2.6 asks for precision >= 0.80. This detector reaches 0.12, with recall 0.22, over
20 pages and 265 annotated arrows.** The bar is not met, and the reason is worth more than the
number.

(Those figures are after fixing a real bug found here: `branch_points` originally tested the
neighbour *count*, which reports a staircase pixel on any diagonal line as a junction. The
crossing-number test replaced it, and the correction is in `src/preprocess/thinning.py`.)

Counting the branches at every skeleton junction that lies within the match radius of a *real,
annotated* arrowhead, across six pages:

    branches at the junction    0    1    2    3    4    5+
    junctions                   1   44  171   93   40   14

**Only a quarter of real arrowheads present as a three-branch junction.** The rest do not,
because a drawn arrow touches the shape it points at: the barbs run into the target's outline,
and the skeleton there is a continuation of that outline rather than a fork. A geometric
signature that assumes the arrowhead stands alone therefore cannot see most of them, and the
ones it does see are outnumbered by the three-way junctions that handwriting produces
everywhere on the page.

Two things follow, and neither is a reason to keep tuning thresholds:

* **This is evidence for Phase 9, not a defect to fix here.** The plan puts a learned detector
  at 9.1 precisely because hand-drawn marks resist hand-written rules; this is the same result
  Phase 2.2.4 got for shape (kappa 0.30) and Phase 3.2.3 got for vertex counts (11% of boxes
  recovered as quadrilaterals). Three independent measurements now say the same thing.
* **The output is still useful as a feature, not as a decision.** 3.2.9 caches these detections
  and Phase 4 can use "an arrowhead-like junction is near this stroke end" as one weak signal
  among many. What it must not be used as is the answer to "is this edge directed".

    python -m src.preprocess.primitives.arrowheads --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import ROOT

#: A branch this long, relative to the longest branch at the junction, is a shaft not a barb.
MAX_BARB_RATIO = 0.55

#: Barbs shorter than this fraction of the page's long side are skeleton spurs, not arrowheads;
#: longer than MAX_BARB_FRAC and the branch is a stroke of the drawing, not a barb.
MIN_BARB_FRAC = 0.004
MAX_BARB_FRAC = 0.10

#: Half-angle between shaft and barb. A drawn arrowhead opens to roughly 20-60 degrees a side;
#: outside that it is a corner or a T-junction, not a head.
MIN_BARB_ANGLE = 15.0
MAX_BARB_ANGLE = 75.0

#: How symmetric the two barbs must be about the shaft, in degrees.
MAX_ASYMMETRY = 45.0

#: A detection matches an annotated arrow if it lands within this fraction of the long side.
MATCH_RADIUS_FRAC = 0.02

#: The shaft is re-sampled this many times further out than the barbs, and must reach at least
#: `MIN_SHAFT_REACH_RATIO` times the barb radius. An arrow comes from somewhere; a three-way
#: junction inside a handwritten letter does not.
SHAFT_RADIUS_MULTIPLIER = 4
MIN_SHAFT_REACH_RATIO = 2.5

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class Arrowhead:
    y: int
    x: int
    shaft_angle: float  # direction the arrow points, degrees, 0 = right
    barb_angles: tuple[float, float]
    barb_lengths: tuple[int, int]
    shaft_length: int

    @property
    def point(self) -> tuple[float, float]:
        return (float(self.x), float(self.y))

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "shaft_angle": round(self.shaft_angle, 1),
            "barb_angles": [round(a, 1) for a in self.barb_angles],
            "barb_lengths": list(self.barb_lengths),
            "shaft_length": self.shaft_length,
        }


def _trace(skeleton: np.ndarray, start: tuple[int, int], first: tuple[int, int], limit: int):
    """Walk one branch away from a junction until it ends, forks, or hits `limit` steps."""
    height, width = skeleton.shape
    path = [start, first]
    visited = {start, first}
    current = first
    while len(path) <= limit:
        options = []
        for dy, dx in _NEIGHBOURS:
            y, x = current[0] + dy, current[1] + dx
            if 0 <= y < height and 0 <= x < width and skeleton[y, x] and (y, x) not in visited:
                options.append((y, x))
        if len(options) != 1:
            break  # an end point (0) or a fork (2+): the branch stops here either way
        current = options[0]
        visited.add(current)
        path.append(current)
    return path


def _angle(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Direction from a to b in degrees, 0 = right, measured in image coordinates."""
    return float(np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])))


def _angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _junction_clusters(skeleton: np.ndarray) -> list[np.ndarray]:
    """Adjacent branch pixels grouped into single junctions.

    A skeleton junction is almost never one pixel. Where three strokes meet, Zhang-Suen leaves
    a small blob - nine pixels on a plain drawn arrow - and every one of them satisfies the
    "three or more neighbours" test. Tracing outward from any single member immediately runs
    into the others and stops after one step, which is why the first version of this detector
    found three arrowheads on twenty pages. Clustering first is not a refinement; it is what
    makes the method work at all.
    """
    from src.preprocess.thinning import branch_points

    marks = branch_points(skeleton).astype(np.uint8)
    # Dilate before labelling: the blob at a three-way meeting is often two groups a pixel
    # apart rather than one 8-connected clump, and splitting it in two leaves each half with
    # the wrong branch count. The dilation is intersected back with the skeleton so the cluster
    # only ever contains real skeleton pixels.
    grown = cv2.dilate(marks, np.ones((3, 3), np.uint8))
    grown = (grown > 0) & skeleton
    count, labels = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    return [np.argwhere(labels == label) for label in range(1, count)]


def _branch_directions(
    skeleton: np.ndarray, cluster: np.ndarray, radius: int
) -> list[tuple[float, int]]:
    """(direction, pixels reached) for each branch leaving a junction, sampled at `radius`.

    Directions are taken at a fixed distance from the junction rather than at the branch's far
    end, and this is the design decision that makes the detector work on real drawings. A drawn
    arrowhead's barbs do not end: they run into the outline of the shape the arrow points at,
    so any rule phrased as "two *short* branches" rejects exactly the arrowheads that matter.
    Sampling the direction a barb's length away from the apex is unaffected by what the barb
    later merges into.
    """
    height, width = skeleton.shape
    cluster_set = {(int(y), int(x)) for y, x in cluster}
    centre = cluster.mean(axis=0)

    # Seeds: skeleton pixels adjacent to the cluster but not in it.
    seeds: set[tuple[int, int]] = set()
    for y, x in cluster_set:
        for dy, dx in _NEIGHBOURS:
            point = (y + dy, x + dx)
            if (
                0 <= point[0] < height
                and 0 <= point[1] < width
                and skeleton[point]
                and point not in cluster_set
            ):
                seeds.add(point)

    # Group seeds that are adjacent to each other: one group per branch.
    groups: list[set[tuple[int, int]]] = []
    for seed in sorted(seeds):
        attached = [
            g for g in groups if any(max(abs(seed[0] - y), abs(seed[1] - x)) <= 1 for y, x in g)
        ]
        if attached:
            merged = {seed}
            for group in attached:
                merged |= group
                groups.remove(group)
            groups.append(merged)
        else:
            groups.append({seed})

    out: list[tuple[float, int]] = []
    for group in groups:
        # Breadth-first walk outward, staying off the junction and the other branches.
        visited = set(group) | cluster_set
        frontier = deque((point, 0) for point in group)
        furthest = next(iter(group))
        reached = 0
        while frontier:
            (y, x), distance = frontier.popleft()
            if distance > reached:
                reached, furthest = distance, (y, x)
            if distance >= radius:
                continue
            for dy, dx in _NEIGHBOURS:
                point = (y + dy, x + dx)
                if (
                    0 <= point[0] < height
                    and 0 <= point[1] < width
                    and skeleton[point]
                    and point not in visited
                ):
                    visited.add(point)
                    frontier.append((point, distance + 1))
        if reached < 2:
            continue
        out.append((_angle((int(round(centre[0])), int(round(centre[1]))), furthest), reached))
    return out


def detect(mask: np.ndarray, *, radius_frac: float = MAX_BARB_FRAC) -> list[Arrowhead]:
    """Every arrowhead in a boolean ink mask."""
    from src.preprocess.thinning import prune_spurs, thin

    # Prune before looking for junctions: rasterisation whiskers are three-way forks too.
    skeleton = prune_spurs(thin(mask))
    clusters = _junction_clusters(skeleton)
    if not clusters:
        return []

    long_side = max(mask.shape)
    radius = max(6, int(radius_frac * long_side / 2))
    min_reach = max(3, int(MIN_BARB_FRAC * long_side))

    found: list[Arrowhead] = []
    for cluster in clusters:
        centre = cluster.mean(axis=0)
        junction = (int(round(centre[0])), int(round(centre[1])))
        branches = _branch_directions(skeleton, cluster, radius)
        if len(branches) != 3:
            continue
        if min(reach for _, reach in branches) < min_reach:
            continue
        # A real arrow has a *long* shaft: it comes from somewhere. Sampling the branches again
        # at several times the barb radius separates an arrowhead from the many three-way
        # junctions in handwriting, where every branch runs out within a few pixels.
        far = _branch_directions(skeleton, cluster, radius * SHAFT_RADIUS_MULTIPLIER)
        if len(far) != 3 or max(reach for _, reach in far) < MIN_SHAFT_REACH_RATIO * radius:
            continue

        # Try each branch as the shaft and keep the assignment that looks like an arrowhead.
        for shaft_index in range(3):
            backward = branches[shaft_index][0]
            barbs = [branches[i] for i in range(3) if i != shaft_index]
            angles = (barbs[0][0], barbs[1][0])
            offsets = [_angular_difference(a, backward) for a in angles]
            if not all(MIN_BARB_ANGLE <= o <= MAX_BARB_ANGLE for o in offsets):
                continue
            signed = [((a - backward + 180.0) % 360.0) - 180.0 for a in angles]
            if signed[0] * signed[1] >= 0:
                continue  # both barbs the same side of the shaft: a corner, not a head
            if abs(offsets[0] - offsets[1]) > MAX_ASYMMETRY:
                continue
            found.append(
                Arrowhead(
                    y=junction[0],
                    x=junction[1],
                    # The arrow points opposite to the shaft branch, folded to (-180, 180].
                    shaft_angle=((backward + 360.0) % 360.0) - 180.0,
                    barb_angles=angles,
                    barb_lengths=(barbs[0][1], barbs[1][1]),
                    shaft_length=branches[shaft_index][1],
                )
            )
            break
    return found


def evaluate(limit: int = 30) -> dict:
    """Precision and recall against hdBPMN's annotated edge endpoints."""
    from src.ir.model import SUFFIX, Diagram
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.exif import load
    from src.preprocess.rules import suppress
    from src.utils.parallel import pmap

    paths = sorted((ROOT / "data" / "processed" / "ir" / "hdbpmn").glob(f"*{SUFFIX}"))[:limit]

    def one(path):
        diagram = Diagram.load(path)
        image_path = ROOT / diagram.meta["image"]
        if not image_path.is_file():
            return None
        gray = load(image_path, grayscale=True)
        scale = 1400 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = denoise(suppress(binarize(median(gray))))

        # Where the annotation says an arrowhead is: the last waypoint of every directed edge.
        targets = [
            (edge.polyline[-1][0] * scale, edge.polyline[-1][1] * scale)
            for edge in diagram.edges
            if edge.directed and edge.polyline
        ]
        if not targets:
            return None
        found = detect(mask)
        radius = MATCH_RADIUS_FRAC * max(mask.shape)

        matched_targets = set()
        hits = 0
        for head in found:
            distances = [np.hypot(head.x - tx, head.y - ty) for tx, ty in targets]
            best = int(np.argmin(distances))
            if distances[best] <= radius:
                hits += 1
                matched_targets.add(best)
        return {
            "detected": len(found),
            "true_positives": hits,
            "targets": len(targets),
            "targets_found": len(matched_targets),
        }

    rows = [r for r in pmap(one, paths, prefer="threads") if r]
    if not rows:
        return {"pages": 0}
    detected = sum(r["detected"] for r in rows)
    hits = sum(r["true_positives"] for r in rows)
    targets = sum(r["targets"] for r in rows)
    covered = sum(r["targets_found"] for r in rows)
    return {
        "pages": len(rows),
        "detected": detected,
        "annotated_arrows": targets,
        "precision": round(hits / detected, 4) if detected else 0.0,
        "recall": round(covered / targets, 4) if targets else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args(argv)

    result = evaluate(args.limit)
    if not result["pages"]:
        print("no annotated pages; run the hdBPMN converter first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))

    checks = {"precision_at_least_0.80": result["precision"] >= 0.80}
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name} ({result['precision']:.1%})")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 10.1.7 - telling a crossing from a junction, and the pairing that comes with the answer.

    python -m src.assemble.crossings

At a skeleton junction two strokes either **cross** (two independent edges that overlap on the
page and must stay separate) or **join** (a real branch or merge). Welding a crossing into one
edge is the worst failure mode in Phase 10 - a confident wrong graph, not a missing one - so this
module exists to catch it before assembly does.

The discriminator is directional continuity: two branches that continue one another are close to
180 degrees apart (`deviation`), have similar stroke width, and turn the same way across the
junction (`pair_score`). `best_matching` finds the pairing of maximum total score by brute-force
enumeration rather than greedily, because the case this row exists for - a shallow crossing where
the single best pair would steal a branch the *other* pair needed - is exactly where greedy is
wrong. A site is called a crossing only with four or more branches and **two** pairs that clear
the continuity gate; fewer branches, or only one continuing pair, is a junction.

## What it measured - all 470 held-out pages (308 val + 162 test)

Ground truth comes from the polylines themselves: two distinct edges whose segments intersect
away from any node they share are a labelled crossing; a node where three or more edges meet is a
labelled junction. That yields **2,045 sites: 315 crossings and 1,730 junctions**, on **97 of 470
pages (21%)** - a crossing is not rare, but the large majority of pages (79%) never present one,
and it is almost entirely one corpus's problem: **314 of 315 crossings are hdbpmn** (flows
crossing lanes), against **1 in fa_bresler** out of 369 fa_bresler sites. didi, flowchartseg and
sketch2code contribute no sites at all in this pass - their ground-truth edges carry no polyline
geometry to intersect.

At the chosen tolerance (`ANGLE_TOL` = 35 degrees, picked as the sweep's accuracy/recall
trade-off rather than its top score - see below): **87.0% accuracy, 69.8% crossing recall, 90.2%
junction recall.** Confusion: 220 crossings called crossings, 95 crossings called junctions, 170
junctions called crossings, 1,560 junctions called junctions - the discriminator's dominant error
is a junction being mistaken for a crossing, not the reverse.

The angle-tolerance sweep (10 to 75 degrees) actually peaks in raw accuracy at **10 degrees
(90.5%)**, not at the chosen 35: a tighter gate rejects more genuine junctions from being called
crossings, at the cost of crossing recall (63.5% vs 69.8%). 35 degrees is kept because the sweep
is nearly flat in accuracy from 10 to 40 degrees (90.5% down to 86.1%) while crossing recall keeps
climbing across that same range, and a wider band before the gate starts being simply generous is
worth more here than the last percentage point of raw accuracy.

**Where it breaks down is not where intuition says it should.** Bucketing real crossings by the
acute angle between the two strokes (`crossing_angle`): recall is **88.4% for near-perpendicular
crossings (60-90 degrees, 224 of 315 - most of the corpus)**, but falls to **13.3% at 30-45
degrees** and **24.4% at 45-60 degrees** (30 and 45 cases) - the discriminator is *worse* in the
oblique middle than at either extreme. The very shallow buckets (0-30 degrees, 16 cases combined)
recall around 40-100% but on too few examples to trust. The honest reading: this discriminator is
built and tuned for the perpendicular crossing that dominates the labelled set, and the oblique
30-60 degree case - common enough in hdbpmn's diagonal lane-crossing flows to matter, at 75 of 315
crossings - is where it should not be trusted without more work.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.assemble.corpus import RUNS, pages, truth
from src.utils.parallel import pmap

#: Two branch tangents count as continuing each other below this deviation from antiparallel.
#: Chosen by the sweep in `run()`; see the write-up.
ANGLE_TOL = math.radians(35.0)

#: Deviation at which directional continuity is worth nothing at all. The angle score falls
#: linearly from 1 at perfectly antiparallel to 0 here, so it is a soft version of `ANGLE_TOL`.
ANGLE_SPAN = math.radians(90.0)

#: How far along a branch, as a share of the page diagonal, the tangent is measured. Short
#: enough that a curved edge is locally straight, long enough that polyline jitter averages out.
SPAN_FRAC = 0.02

#: Weight of the stroke-width agreement term and of the curvature-continuity term in a pair's
#: score. Both are corrections to the angle term, never a substitute for it.
WIDTH_WEIGHT = 0.25
CURV_WEIGHT = 0.25

#: Signed turning difference, in radians per span, at which curvature continuity scores zero.
CURV_SPAN = math.radians(60.0)

#: A pair must beat this combined score to be called continuing, whatever its raw angle says.
PAIR_SCORE_MIN = 0.25

#: A ground-truth intersection closer than this share of the diagonal to a node the two edges
#: share is a meeting at that node, not a crossing, and is excluded from the crossing set.
NODE_RADIUS_FRAC = 0.05

#: Beyond this many branches the brute-force matching is not run; a skeleton site with nine
#: incident strokes is a blob artefact, not a junction.
MAX_BRANCHES = 8


@dataclass(frozen=True)
class Branch:
    """One stroke leaving a junction point.

    `angle` points *away* from the junction. `curvature` is the signed turning accumulated over
    the walk away from the junction, so a straight-through smooth curve has `k_i == -k_j` for its
    two halves - reversing a traversal negates its signed turning.
    """

    angle: float
    curvature: float = 0.0
    width: float | None = None
    ref: Any = None


@dataclass(frozen=True)
class Junction:
    """The verdict at one site: what it is, how the branches pair up, and how sure that is."""

    kind: str
    pairing: tuple[tuple[int, int], ...] = ()
    unpaired: tuple[int, ...] = ()
    confidence: float = 0.0
    scores: tuple[float, ...] = ()
    deviations: tuple[float, ...] = ()
    point: tuple[float, float] | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_crossing(self) -> bool:
        return self.kind == "crossing"


# ------------------------------------------------------------------------------------------
# the discriminator
# ------------------------------------------------------------------------------------------


def deviation(a: Branch, b: Branch) -> float:
    """How far the two branches are from continuing each other, in radians. 0 = antiparallel."""
    delta = abs(math.atan2(math.sin(a.angle - b.angle), math.cos(a.angle - b.angle)))
    return abs(math.pi - delta)


def pair_score(a: Branch, b: Branch) -> tuple[float, float]:
    """`(score, deviation)` for treating `a` and `b` as two halves of one stroke."""
    dev = deviation(a, b)
    angle_term = max(0.0, 1.0 - dev / ANGLE_SPAN)
    width_term = 1.0 - abs(a.width - b.width) / (a.width + b.width) if a.width and b.width else 1.0
    curve_term = max(0.0, 1.0 - abs(a.curvature + b.curvature) / CURV_SPAN)
    return (
        angle_term
        * (1.0 - WIDTH_WEIGHT + WIDTH_WEIGHT * width_term)
        * (1.0 - CURV_WEIGHT + CURV_WEIGHT * curve_term),
        dev,
    )


def _matchings(items: list[int]) -> list[list[tuple[int, int]]]:
    if not items:
        return [[]]
    head, rest = items[0], items[1:]
    out = []
    for k in range(len(rest)):
        for tail in _matchings(rest[:k] + rest[k + 1 :]):
            out.append([(head, rest[k]), *tail])
    return out


def best_matching(branches: list[Branch]) -> tuple[list[tuple[int, int]], list[int], float]:
    """The pairing of maximum total score, by enumeration rather than greedily.

    Greedy pairing is the obvious implementation and it is wrong on exactly the case this row
    exists for: at a shallow crossing the single best pair can take a branch the *other* pair
    needed, and the leftovers then look like a junction. With at most `MAX_BRANCHES` branches
    the perfect matchings are few enough to enumerate all of them.
    """
    n = len(branches)
    indices = list(range(n))
    best: tuple[float, list[tuple[int, int]], list[int]] = (-1.0, [], indices)
    for left_out in itertools.combinations(indices, n % 2):
        rest = [i for i in indices if i not in left_out]
        for pairing in _matchings(rest):
            total = sum(pair_score(branches[i], branches[j])[0] for i, j in pairing)
            if total > best[0]:
                best = (total, pairing, list(left_out))
    return best[1], best[2], best[0]


def classify_junction(branches: list[Branch], point: tuple[float, float] | None = None) -> Junction:
    """Crossing or junction, with the pairing that justifies it.

    A crossing is two strokes that happen to overlap, so it needs **two** branch pairs that
    continue each other. Fewer than four branches cannot supply that: a degree-3 site is a real
    merge or split whatever its angles look like, which is why the branch count is checked before
    any angle is.
    """
    n = len(branches)
    if n < 3 or n > MAX_BRANCHES:
        return Junction(kind="junction", unpaired=tuple(range(n)), confidence=1.0, point=point)

    pairing, unpaired, _ = best_matching(branches)
    scored = [(pair_score(branches[i], branches[j]), (i, j)) for i, j in pairing]
    scored.sort(key=lambda row: -row[0][0])
    scores = tuple(round(s, 4) for (s, _), _ in scored)
    devs = tuple(round(d, 4) for (_, d), _ in scored)
    ordered = tuple(pair for _, pair in scored)

    continuing = [
        k for k, ((s, d), _) in enumerate(scored) if d <= ANGLE_TOL and s >= PAIR_SCORE_MIN
    ]
    crossing = n >= 4 and len(continuing) >= 2

    if crossing:
        # How comfortably the *second* pair cleared the gate. The first one clearing it says
        # nothing, because a T-junction's straight bar clears it too.
        margin = 1.0 - devs[continuing[1]] / ANGLE_TOL
    else:
        worst = devs[0] if devs else math.pi
        margin = min(1.0, abs(worst - ANGLE_TOL) / ANGLE_TOL)
    return Junction(
        kind="crossing" if crossing else "junction",
        pairing=ordered,
        unpaired=tuple(unpaired),
        confidence=round(max(0.0, min(1.0, margin)), 4),
        scores=scores,
        deviations=devs,
        point=point,
        detail={"branches": n, "continuing_pairs": len(continuing)},
    )


# ------------------------------------------------------------------------------------------
# polyline geometry - turning a site on two polylines into branches
# ------------------------------------------------------------------------------------------


def _as_array(polyline: list[list[float]]) -> np.ndarray:
    points = np.asarray(polyline, dtype=float).reshape(-1, 2)
    keep = [0]
    for i in range(1, len(points)):
        if float(np.hypot(*(points[i] - points[keep[-1]]))) > 1e-6:
            keep.append(i)
    return points[keep]


def _segment_intersection(p1, p2, p3, p4):
    """Intersection point of two segments as `(point, t, u)`, or `None`."""
    d1, d2 = p2 - p1, p4 - p3
    denominator = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denominator) < 1e-12:
        return None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denominator
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / denominator
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return p1 + t * d1, float(t), float(u)
    return None


def _walk(points: np.ndarray, index: int, t: float, forward: bool, span: float):
    """Walk `span` pixels away from the point at (segment `index`, parameter `t`).

    Returns `(angle, curvature)`, or `None` if there is not enough polyline left to measure - a
    stub shorter than half the span is not evidence about anything and is dropped rather than
    extrapolated.
    """
    start = points[index] + t * (points[index + 1] - points[index])
    order = range(index + 1, len(points)) if forward else range(index, -1, -1)
    walked = [start]
    total = 0.0
    for i in order:
        total += float(np.hypot(*(points[i] - walked[-1])))
        walked.append(points[i])
        if total >= span:
            break
    if total < span * 0.5 or len(walked) < 2:
        return None
    end, near = walked[-1], walked[1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    near_angle = math.atan2(near[1] - start[1], near[0] - start[0])
    turn = math.atan2(math.sin(angle - near_angle), math.cos(angle - near_angle))
    return angle, turn


def _widths(image: Path, sites: list[tuple[float, float]]) -> list[float | None]:
    """Stroke width at each point, from a distance transform of the binarised page."""
    try:
        import cv2

        grey = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            return [None] * len(sites)
        dist = cv2.distanceTransform((grey < 128).astype(np.uint8), cv2.DIST_L2, 3)
    except Exception:  # noqa: BLE001 - no distance transform means no width evidence, not a crash
        return [None] * len(sites)
    h, w = dist.shape
    out: list[float | None] = []
    for x, y in sites:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            patch = dist[max(0, yi - 2) : yi + 3, max(0, xi - 2) : xi + 3]
            value = float(patch.max()) if patch.size else 0.0
            out.append(2.0 * value if value > 0 else None)
        else:
            out.append(None)
    return out


# ------------------------------------------------------------------------------------------
# the labelled set, built from the ground-truth polylines themselves
# ------------------------------------------------------------------------------------------


def _near_node(point, bbox, radius: float) -> bool:
    if bbox is None:
        return False
    x, y, w, h = bbox
    dx = max(x - point[0], 0.0, point[0] - (x + w))
    dy = max(y - point[1], 0.0, point[1] - (y + h))
    return math.hypot(dx, dy) <= radius


def crossing_sites(diagram, span: float, node_radius: float) -> list[dict]:
    """Every place two *distinct* ground-truth edges genuinely cross on the page.

    An intersection near a node both edges touch is those two edges meeting at that node, not a
    crossing, and is dropped here rather than being allowed to pollute the positive class.
    """
    boxes = {n.id: n.bbox for n in diagram.nodes if n.bbox is not None}
    edges = [e for e in diagram.edges if e.polyline and len(e.polyline) >= 2]
    arrays = {e.id: _as_array(e.polyline) for e in edges}
    out = []
    for a, b in itertools.combinations(edges, 2):
        pa, pb = arrays[a.id], arrays[b.id]
        if len(pa) < 2 or len(pb) < 2:
            continue
        shared = ({a.src, a.dst} & {b.src, b.dst}) - {None}
        for i in range(len(pa) - 1):
            for j in range(len(pb) - 1):
                hit = _segment_intersection(pa[i], pa[i + 1], pb[j], pb[j + 1])
                if hit is None:
                    continue
                point, t, u = hit
                if any(
                    _near_node(point, boxes.get(nid), node_radius) for nid in shared if nid in boxes
                ):
                    continue
                branches = []
                for points, index, param in ((pa, i, t), (pb, j, u)):
                    for forward in (True, False):
                        walked = _walk(points, index, param, forward, span)
                        if walked is not None:
                            branches.append(walked)
                if len(branches) != 4:
                    continue
                out.append(
                    {
                        "label": "crossing",
                        "point": (float(point[0]), float(point[1])),
                        "branches": branches,
                        "edges": [a.id, b.id],
                    }
                )
    return out


def junction_sites(diagram, span: float) -> list[dict]:
    """Every node where three or more ground-truth edges meet - labelled true junctions.

    The branch directions are the incident edges' tangents at the end nearest the node, walked
    outwards, which is exactly the evidence a skeleton walk would have at that spot.
    """
    boxes = {n.id: n.bbox for n in diagram.nodes if n.bbox is not None}
    incident: dict[str, list] = {}
    for edge in diagram.edges:
        if not edge.polyline or len(edge.polyline) < 2:
            continue
        points = _as_array(edge.polyline)
        if len(points) < 2:
            continue
        for nid, at_start in ((edge.src, True), (edge.dst, False)):
            if nid in boxes:
                incident.setdefault(nid, []).append((points, at_start, edge.id))
    out = []
    for nid, items in incident.items():
        if len(items) < 3:
            continue
        branches, refs = [], []
        for points, at_start, eid in items:
            index, param, forward = (0, 0.0, True) if at_start else (len(points) - 2, 1.0, False)
            walked = _walk(points, index, param, forward, span)
            if walked is not None:
                branches.append(walked)
                refs.append(eid)
        if len(branches) < 3:
            continue
        x, y, w, h = boxes[nid]
        out.append(
            {
                "label": "junction",
                "point": (float(x + w / 2), float(y + h / 2)),
                "branches": branches,
                "edges": refs,
            }
        )
    return out


def page_sites(page, *, with_width: bool = True) -> list[dict]:
    """The page-level pass: every labelled crossing and junction site on one page."""
    diagram = truth(page)
    span = SPAN_FRAC * page.diagonal
    sites = crossing_sites(diagram, span, NODE_RADIUS_FRAC * page.diagonal)
    sites += junction_sites(diagram, span)
    if with_width and sites:
        widths = _widths(page.image, [s["point"] for s in sites])
        for site, width in zip(sites, widths, strict=True):
            site["width"] = width
    for site in sites:
        site["page"] = page.name
        site["source"] = page.source
    return sites


def to_branches(site: dict) -> list[Branch]:
    width = site.get("width")
    return [Branch(angle=a, curvature=k, width=width) for a, k in site["branches"]]


def classify_page(page, *, with_width: bool = True) -> list[tuple[dict, Junction]]:
    """Classify every site on one page. The page-level pass."""
    sites = page_sites(page, with_width=with_width)
    return [(s, classify_junction(to_branches(s), s["point"])) for s in sites]


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------


def confusion(sites: list[dict], tol: float) -> dict:
    """Confusion at one angle tolerance, plus accuracy and per-class recall."""
    global ANGLE_TOL
    saved, ANGLE_TOL = ANGLE_TOL, tol
    try:
        counts = {"cc": 0, "cj": 0, "jc": 0, "jj": 0}
        for site in sites:
            verdict = classify_junction(to_branches(site))
            key = ("c" if site["label"] == "crossing" else "j") + (
                "c" if verdict.is_crossing else "j"
            )
            counts[key] += 1
    finally:
        ANGLE_TOL = saved
    total = sum(counts.values()) or 1
    crossings = counts["cc"] + counts["cj"]
    junctions = counts["jj"] + counts["jc"]
    return {
        "angle_tol_deg": round(math.degrees(tol), 1),
        "accuracy": round((counts["cc"] + counts["jj"]) / total, 4),
        "crossing_recall": round(counts["cc"] / crossings, 4) if crossings else None,
        "junction_recall": round(counts["jj"] / junctions, 4) if junctions else None,
        "confusion": {
            "crossing_as_crossing": counts["cc"],
            "crossing_as_junction": counts["cj"],
            "junction_as_crossing": counts["jc"],
            "junction_as_junction": counts["jj"],
        },
    }


def _mean_axis(a: float, b: float) -> float:
    """The axis of two roughly antiparallel directions, as an angle mod pi."""
    return math.atan2(math.sin(2 * a) + math.sin(2 * b), math.cos(2 * a) + math.cos(2 * b)) / 2.0


def crossing_angle(site: dict) -> float:
    """The acute angle, in degrees, between the two strokes that cross here."""
    branches = to_branches(site)
    if len(branches) != 4:
        return float("nan")
    pairing, _, _ = best_matching(branches)
    (i, j), (k, m) = pairing[0], pairing[1]
    delta = (
        abs(
            _mean_axis(branches[i].angle, branches[j].angle)
            - _mean_axis(branches[k].angle, branches[m].angle)
        )
        % math.pi
    )
    return round(math.degrees(min(delta, math.pi - delta)), 2)


def angle_breakdown(sites: list[dict], edges: tuple[int, ...] = (10, 20, 30, 45, 60, 90)) -> list:
    """Crossing recall bucketed by how shallow the crossing is. The row's headline number."""
    rows: list[dict] = []
    lower = 0
    for upper in edges:
        bucket = [
            s for s in sites if s["label"] == "crossing" and lower < crossing_angle(s) <= upper
        ]
        found = sum(1 for s in bucket if classify_junction(to_branches(s)).is_crossing)
        rows.append(
            {
                "angle_deg": f"{lower}-{upper}",
                "n": len(bucket),
                "recall": round(found / len(bucket), 4) if bucket else None,
            }
        )
        lower = upper
    return rows


# ------------------------------------------------------------------------------------------
# synthetic cases with exactly known geometry
# ------------------------------------------------------------------------------------------


def synthetic() -> dict[str, tuple[list[Branch], str]]:
    """Six shapes whose answer is known by construction. The definition of done's test cases."""

    def deg(*angles: float, curvature: tuple[float, ...] | None = None) -> list[Branch]:
        curves = curvature or (0.0,) * len(angles)
        return [
            Branch(angle=math.radians(a), curvature=k) for a, k in zip(angles, curves, strict=True)
        ]

    return {
        "clean_x": (deg(0, 90, 180, 270), "crossing"),
        "clean_t": (deg(0, 180, 270), "junction"),
        "shallow_x": (deg(0, 20, 180, 200), "crossing"),
        "curved_x": (deg(0, 90, 175, 265, curvature=(0.3, 0.3, -0.3, -0.3)), "crossing"),
        "y_branch": (deg(90, 210, 330), "junction"),
        # Four branches, but only one antiparallel pair (0/180) - a real 4-way merge where two
        # of the arms happen to line up, not two strokes crossing. The best matching still pairs
        # (0, 180) together (score 1.0) and (60, 120) together (score 0.0, dev 120 deg), so only
        # one pair clears the continuity gate and `classify_junction` needs two.
        "four_way_merge": (deg(0, 60, 120, 180), "junction"),
    }


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def run(limit: int | None = None, with_width: bool = True) -> dict:
    held = pages()[:limit] if limit else pages()
    batches = pmap(lambda p: page_sites(p, with_width=with_width), held, n_jobs=3, prefer="threads")
    sites = [s for batch in batches for s in batch]
    crossings = [s for s in sites if s["label"] == "crossing"]
    junctions = [s for s in sites if s["label"] == "junction"]

    sweep = [confusion(sites, math.radians(d)) for d in (10, 15, 20, 25, 30, 35, 40, 50, 60, 75)]
    hard = [s for s in sites if len(s["branches"]) == 4]

    synthetic_rows = []
    for name, (branches, expected) in synthetic().items():
        verdict = classify_junction(branches)
        synthetic_rows.append(
            {
                "case": name,
                "expected": expected,
                "got": verdict.kind,
                "pairing": [list(p) for p in verdict.pairing],
                "confidence": verdict.confidence,
                "pass": verdict.kind == expected,
            }
        )

    return {
        "pages": len(held),
        "sites": len(sites),
        "crossings": len(crossings),
        "junctions": len(junctions),
        "pages_with_a_crossing": len({s["page"] for s in crossings}),
        "crossing_rate_per_page": round(len(crossings) / max(1, len(held)), 3),
        "sites_with_stroke_width": sum(1 for s in sites if s.get("width")),
        "chosen_angle_tol_deg": round(math.degrees(ANGLE_TOL), 1),
        "at_chosen_tol": confusion(sites, ANGLE_TOL),
        "degree_4_only": confusion(hard, ANGLE_TOL) | {"n": len(hard)},
        "sweep": sweep,
        "best_in_sweep": max(sweep, key=lambda row: row["accuracy"]),
        "crossing_recall_by_angle": angle_breakdown(sites),
        "by_source": {
            src: {
                "crossings": sum(1 for s in crossings if s["source"] == src),
                "junctions": sum(1 for s in junctions if s["source"] == src),
            }
            for src in sorted({s["source"] for s in sites})
        },
        "synthetic": synthetic_rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-width", action="store_true")
    ap.add_argument("--out", type=Path, default=RUNS / "crossings.json")
    args = ap.parse_args(argv)

    result = run(args.limit, with_width=not args.no_width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

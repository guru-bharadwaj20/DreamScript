"""Phase 10.1.5 - which way does the arrow point.

    python -m src.assemble.direction

A traced polyline between two nodes is undirected: the same ink supports `u -> v` and `v -> u`,
and every downstream consumer - 7.3's roles, 10.2's validators, Phase 16's export - needs the
answer. The plan names three sources of evidence and asserts the combination. They are
implemented and **ablated separately**, because the value of the row is finding out what each one
is worth, and one of the three turns out to carry almost all of it.

## The decision, and why chance is *not* exactly 0.5 everywhere

Every scorer here is **antisymmetric**: swapping the two candidate endpoints negates its score.
That is not decoration, it is what makes the ablation readable - a feature that cannot express a
preference for "the first endpoint" cannot inherit the accuracy of the IR's own edge ordering.
Candidates are canonicalised to `sorted((src, dst))` before anything is scored, and `id_order` -
always choose the lower node id - is reported as a control precisely so that any residual leak
through the annotation order would be visible. On hdbpmn it is: **0.4865**, a coin, as designed.
**On fa_bresler it measures 0.7006 - nowhere near 0.5.** State-machine node ids are assigned in an
order that itself correlates with which state is the source 70% of the time; canonicalising by
`sorted((src, dst))` does not remove that, it just relabels it. This is a real property of the
fa_bresler corpus, not a bug in the canonicalisation, and every fa_bresler number below has to be
read against a 0.70 floor, not a 0.50 one.

The five features, all signed so that positive means "the first endpoint is the source":

    prior_x, prior_y   the unit vector from one node's centre to the other's
    head_det           9.1's `arrowhead` class: head mass at the far end minus head mass at
                       the near end
    head_geo           3.2.6's geometric arrowheads, the same difference, weighted by how well
                       the head's shaft angle agrees with the polyline's tangent there
    role               +1 per Phase 7 `start` at the tail or `terminal` at the head, -1 each way
                       round, halved into [-1, 1]

They are combined as a **log-odds sum**: one no-intercept logistic regression over the feature
subset, fitted on the detector's `val` split and reported on `test`. No intercept, because an
intercept is exactly the "prefer the first endpoint" term antisymmetry forbids. Each ablation is
that same fit over its own subset, so the rows are comparable and none of them is tuned on the
numbers below. The prior's weights are what the corpus says they are, not what the plan assumed.

## What the flow prior actually is, and why the plan's sentence is half wrong

The plan says hand-drawn flowcharts "overwhelmingly run top-to-bottom and left-to-right". Measured
over 7,579 hdbpmn and 1,338 fa_bresler edges on the **training** pages - deliberately not the pages
anything is scored on - only half of that is true:

    hdbpmn        left-to-right 76.8%  (85.7% of horizontal-major edges)   top-to-bottom 50.8%
    fa_bresler    left-to-right 73.4%  (83.1% of horizontal-major edges)   top-to-bottom 56.4%

**BPMN is a left-to-right notation and has no vertical convention at all**: 50.8% is a coin. The
vertical prior only appears once the horizontal one is spent - among edges whose displacement is
mostly vertical, 62.0% of hdbpmn edges run downwards (57.2% for fa_bresler) - which is why the
fitted `prior_y` weight ends up small next to `prior_x` rather than the dominant term the plan
expected. Assuming the sentence would have produced a worse model than measuring it.

## What it measured

Ground-truth polylines with ground-truth endpoints, so tracing and binding error are excluded and
this is direction error alone. **flowchartseg contributes nothing: its held-out pages carry zero
edges**, so there is no third corpus here and pooling one in would have been inventing it. `test`
split, 1,996 hdbpmn and 314 fa_bresler edges:

    features                       hdbpmn    fa_bresler
    all (head_det+head_geo+prior+role)   0.9108    0.7611   <- all three
    head_det+prior+role                  0.9108    0.7516
    head (head_det+head_geo)             0.8793    0.4777
    head_det                             0.8803    0.7006
    prior+role                           0.7956    0.7516
    prior                                0.7956    0.7325   <- the control that costs nothing
    head_geo                             0.6453    0.4777
    role                                 0.5681    0.7293
    id_order                             0.4865    0.7006   <- far from 0.5 on fa_bresler; see above
    random                               0.4890    0.5064

**hdbpmn meets the 0.90 bar at 0.9108. fa_bresler does not, at 0.7611, and most of the gap has one
cause.** The `arrowhead` detection class is derived from hdBPMN's polyline waypoints and exists on
hdbpmn pages only - `head_det` fits to a weight of exactly 0.0 on fa_bresler because there is
nothing to fit, and its "accuracy" there (0.7006) is just the `id_order` leak described above, not
evidence. So on fa_bresler the combined model runs on the prior, roles and the geometric detector
only, and 0.7611 is what that combination is worth once the id_order floor is subtracted back out
(0.7611 - 0.7006 = 0.0605 of genuine lift). This is a class-coverage hole in 9.1.2, not a modelling
failure in 10.1.5, and it is the single change that would move the number most.

**Against the free `prior` control, the full combination adds 0.1152 on hdbpmn and only 0.0286 on
fa_bresler** - and against the *honest* fa_bresler floor (`id_order` at 0.7006, not 0.5) that
0.0286 is most of what there is to add. Roles add nothing on top of the prior for hdbpmn
(`prior+role` == `prior` to four decimal places, 0.7956 both) but add 0.0191 for fa_bresler
(0.7516 vs 0.7325) and a full 0.1911 on top of the arrowhead detectors alone there (`head+role`
0.6688 vs `head` 0.4777) - on fa_bresler, where heads barely work at all, roles are doing real
work; on hdbpmn, where the detector already saturates most of the signal, they are redundant.

**The two arrowhead detectors are not interchangeable, and the direction of the gap is what the
plan expected but the size is smaller than assumed.** 3.2.6's geometric detector reaches 0.6453
alone on hdbpmn - real signal, well above the 0.4865 coin - against 0.8803 for the detection
class, and adding `head_geo` to `head_det` **costs** 0.0010 of hdbpmn accuracy rather than adding
anything (`head` 0.8793 vs `head_det` 0.8803): the geometric detector's false positives outvote
its true ones often enough to be a small net negative once the learned class is already present.
3.2.6 already reported precision 0.11 at recall 0.22 and explained why (a drawn barb merges into
the outline of the shape it touches, so most real arrowheads are not three-branch skeleton
junctions); this is that same weakness measured on the task the detector exists to serve. On
fa_bresler, where it is the *only* head evidence available and there is no `head_det` to compare
against, `head_geo` alone measures 0.4777 - **below the id_order floor of 0.7006**, so on this
corpus the geometric detector alone is actively worse than doing nothing and guessing the
canonical order; it only becomes useful once combined with the prior and roles.

**The honest summary: hdbpmn passes on the strength of one detector, fa_bresler does not pass, and
the two corpora cannot be pooled or compared at face value because their `id_order` floors are
0.30 apart.** `head_det` alone (0.8803) accounts for essentially all of hdbpmn's 0.9108, with
`prior`+`role` closing the remaining ~3 points. fa_bresler has no `head_det` signal at all; what
it has is a prior worth 0.7325, roles worth a further 0.0191-0.1911 depending what else is
present, and a geometric detector that is a net negative alone and only additive in combination -
and even the best fa_bresler combination (`all`, 0.7611) is barely two points of *genuine* lift
above the corpus's own id-order coincidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.assemble.corpus import RUNS, Page, centre, detections, pages, truth
from src.ir.model import Edge, Node
from src.parse.roles import ROLE_TO_STATE
from src.utils.config import ROOT
from src.utils.parallel import pmap

PRIMITIVES = ROOT / "data" / "interim" / "primitives_detect"

#: An arrowhead counts towards a polyline end if it lies within this share of the page diagonal
#: of it. 0.02 is roughly a node's stroke width at 896px; wider and a head at one waypoint of a
#: dense BPMN page starts voting on its neighbour's edge.
HEAD_RADIUS = 0.02

#: Head mass decays as `exp(-(d / (HEAD_RADIUS * diagonal))**2)`, so a head exactly on the
#: endpoint is worth 1 and one at the radius is worth 0.37. Soft, because a hand-drawn barb's
#: centroid is not its tip.
HEAD_DECAY = 1.0

#: A geometric arrowhead's vote is scaled by `max(0, cos(shaft - tangent))`: 3.2.6 gives the
#: direction the head points, and a head that points back along the polyline is not this edge's.
GEO_ANGLE_FLOOR = 0.0

#: The two Phase 7 states that constrain direction at all. A `start` has no incoming edge and a
#: `terminal` no outgoing one; every other state says nothing about which way an edge runs.
DIRECTED_STATES: dict[str, float] = {"start": 1.0, "terminal": -1.0}

#: The feature names, in the order the weight vector is indexed. Frozen: an ablation is a subset
#: of these strings and a reordering would silently re-label every fitted weight.
FEATURES: tuple[str, ...] = ("prior_x", "prior_y", "head_det", "head_geo", "role")

#: The ablations reported, plan-first: each source alone, each pair, all three. `head_det` and
#: `head_geo` are one source ("arrowhead evidence") split in two because they are two different
#: detectors of the same mark and the contrast is the point.
ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("head_det", ("head_det",)),
    ("head_geo", ("head_geo",)),
    ("head", ("head_det", "head_geo")),
    ("prior", ("prior_x", "prior_y")),
    ("role", ("role",)),
    ("head+prior", ("head_det", "head_geo", "prior_x", "prior_y")),
    ("head+role", ("head_det", "head_geo", "role")),
    ("prior+role", ("prior_x", "prior_y", "role")),
    ("head_det+prior+role", ("head_det", "prior_x", "prior_y", "role")),
    ("all", FEATURES),
)

#: Fit on the detector's `val` pages, report on its `test` pages. Both are held out of detector
#: training; splitting them again keeps the fusion weights off the numbers they are quoted with.
FIT_SPLIT = "val"
REPORT_SPLIT = "test"

#: Newton steps for the no-intercept logistic. Five features and a well-conditioned Hessian; it
#: converges in three or four and the extra steps cost nothing.
NEWTON_STEPS = 40

#: Ridge added to the Hessian diagonal, so a feature that is identically zero on a corpus (as
#: `head_det` is on fa_bresler) leaves a weight of 0 rather than a singular solve.
RIDGE = 1e-3


# ------------------------------------------------------------------------------------------
# the decision
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Direction:
    """One resolved edge: the decision, and the per-source scores that produced it."""

    src: str
    dst: str
    #: Log-odds of the chosen orientation against its reverse. Always >= 0.
    logodds: float
    #: Signed per-feature score for the *canonical* orientation, before weighting.
    features: dict[str, float]
    #: Weighted per-feature contribution to the canonical orientation's log-odds.
    contributions: dict[str, float]
    #: True when the canonical orientation (lower node id first) was rejected.
    flipped: bool

    @property
    def pair(self) -> tuple[str, str]:
        return (self.src, self.dst)


@dataclass
class Evidence:
    """Everything about a page that direction resolution reads, gathered once per page."""

    diagonal: float
    #: `(x, y, weight)` per detector `arrowhead` box, weight being its detection score.
    det_heads: list[tuple[float, float, float]] = field(default_factory=list)
    #: `(x, y, shaft_angle_radians)` per 3.2.6 geometric arrowhead.
    geo_heads: list[tuple[float, float, float]] = field(default_factory=list)
    #: Fitted log-odds weights, keyed by feature name. Missing keys weigh nothing.
    weights: dict[str, float] = field(default_factory=dict)


def _oriented(edge: Edge, u: Node, v: Node) -> list[list[float]] | None:
    """The polyline with its first point at `u`'s end, or None if it cannot be oriented.

    The IR stores waypoints in source-to-target order, which is the answer. Orientation is
    therefore re-derived from the node centres and never read off the stored order.
    """
    line = edge.polyline
    if not line or len(line) < 2 or u.bbox is None or v.bbox is None:
        return None
    uc, vc = centre(u.bbox), centre(v.bbox)
    head, tail = line[0], line[-1]
    forward = math.dist(head, uc) + math.dist(tail, vc)
    backward = math.dist(head, vc) + math.dist(tail, uc)
    return list(line) if forward <= backward else list(reversed(line))


def _tangent(line: list[list[float]], at_end: bool) -> float:
    """Angle of the polyline where it meets an endpoint, pointing *outwards*, in radians."""
    if at_end:
        (x0, y0), (x1, y1) = line[-2], line[-1]
    else:
        (x0, y0), (x1, y1) = line[1], line[0]
    return math.atan2(y1 - y0, x1 - x0)


def _mass(
    heads: list[tuple[float, float, float]],
    point: list[float],
    diagonal: float,
    tangent: float | None = None,
) -> float:
    """Strongest head vote at `point`. The max, not the sum: a cluster is still one arrowhead."""
    radius = max(HEAD_RADIUS * diagonal, 1e-6)
    best = 0.0
    for head in heads:
        x, y = head[0], head[1]
        distance = math.hypot(x - point[0], y - point[1])
        weight = math.exp(-HEAD_DECAY * (distance / radius) ** 2)
        if tangent is not None:
            weight *= max(GEO_ANGLE_FLOOR, math.cos(head[2] - tangent))
        else:
            weight *= head[2]
        best = max(best, weight)
    return best


def features(edge: Edge, nodes: dict[str, Node], evidence: Evidence) -> dict[str, float] | None:
    """The five signed features for the canonical orientation, or None if the edge is unusable.

    Canonical is `sorted((edge.src, edge.dst))`; positive means the first of those is the source.
    Every feature negates under a swap of the two, which is what pins chance at 0.5.
    """
    if edge.src is None or edge.dst is None or edge.src == edge.dst:
        return None
    first, second = sorted((edge.src, edge.dst))
    u, v = nodes.get(first), nodes.get(second)
    if u is None or v is None:
        return None
    line = _oriented(edge, u, v)
    if line is None:
        return None

    ux, uy = centre(u.bbox)
    vx, vy = centre(v.bbox)
    span = math.hypot(vx - ux, vy - uy)
    prior_x, prior_y = ((vx - ux) / span, (vy - uy) / span) if span > 1e-6 else (0.0, 0.0)

    tail, head = line[0], line[-1]
    det = _mass(evidence.det_heads, head, evidence.diagonal) - _mass(
        evidence.det_heads, tail, evidence.diagonal
    )
    geo = _mass(evidence.geo_heads, head, evidence.diagonal, _tangent(line, True)) - _mass(
        evidence.geo_heads, tail, evidence.diagonal, _tangent(line, False)
    )

    role = (
        DIRECTED_STATES.get(ROLE_TO_STATE.get(u.semantic_role, ""), 0.0)
        - DIRECTED_STATES.get(ROLE_TO_STATE.get(v.semantic_role, ""), 0.0)
    ) / 2.0
    return {
        "prior_x": prior_x,
        "prior_y": prior_y,
        "head_det": det,
        "head_geo": geo,
        "role": role,
    }


def resolve(edge: Edge, nodes: dict[str, Node], evidence: Evidence) -> Direction | None:
    """Decide which way `edge` runs between its two endpoints, and say why.

    Returns None for an edge that carries no usable geometry, rather than guessing: a caller that
    cannot see the difference between "runs u to v" and "there was nothing to go on" will report
    a direction accuracy that includes coin flips.
    """
    scores = features(edge, nodes, evidence)
    if scores is None:
        return None
    first, second = sorted((edge.src, edge.dst))
    contributions = {k: evidence.weights.get(k, 0.0) * value for k, value in scores.items()}
    total = sum(contributions.values())
    flipped = total < 0.0
    return Direction(
        src=second if flipped else first,
        dst=first if flipped else second,
        logodds=abs(total),
        features=scores,
        contributions=contributions,
        flipped=flipped,
    )


# ------------------------------------------------------------------------------------------
# page-level convenience
# ------------------------------------------------------------------------------------------


def _geometric_heads(page: Page) -> list[tuple[float, float, float]]:
    """3.2.6's arrowheads for a page, from the cache the primitives already live in."""
    try:
        from src.preprocess.cache import load_or_compute

        primitives, _ = load_or_compute(page.name, page.image, PRIMITIVES)
    # A page whose primitives cannot be built has no geometric heads.
    except Exception:  # noqa: BLE001
        return []
    return [
        (float(a["x"]), float(a["y"]), math.radians(float(a["shaft_angle"])))
        for a in getattr(primitives, "arrowheads", []) or []
    ]


def evidence_for(page: Page, weights: dict[str, float] | None = None) -> Evidence:
    """Gather one page's direction evidence: detector heads, geometric heads, and the weights."""
    det = [
        (*centre(box["bbox"]), float(box["score"]))
        for box in detections(page)
        if box["cls"] == "arrowhead"
    ]
    return Evidence(
        diagonal=page.diagonal,
        det_heads=det,
        geo_heads=_geometric_heads(page),
        weights=dict(weights or {}),
    )


def resolve_page(page: Page, weights: dict[str, float] | None = None) -> list[Direction]:
    """Resolve every ground-truth edge on `page`. The convenience wrapper around `resolve`."""
    diagram = truth(page)
    nodes = {n.id: n for n in diagram.nodes}
    evidence = evidence_for(page, weights)
    out = []
    for edge in diagram.edges:
        decision = resolve(edge, nodes, evidence)
        if decision is not None:
            out.append(decision)
    return out


# ------------------------------------------------------------------------------------------
# fitting the log-odds sum
# ------------------------------------------------------------------------------------------


def fit(matrix: np.ndarray, labels: np.ndarray, steps: int = NEWTON_STEPS) -> np.ndarray:
    """No-intercept logistic regression by Newton's method, ridge-stabilised.

    No intercept on purpose. An intercept is a constant preference for the canonical orientation,
    which is precisely the "always pick the first endpoint" term that antisymmetry rules out;
    allowing one would let a subset with no usable feature drift off 0.5 for free.
    """
    matrix = np.asarray(matrix, float)
    y = np.asarray(labels, float)
    weights = np.zeros(matrix.shape[1])
    for _ in range(steps):
        probability = 1.0 / (1.0 + np.exp(-np.clip(matrix @ weights, -30, 30)))
        gradient = matrix.T @ (y - probability) - RIDGE * weights
        hessian = (matrix * (probability * (1 - probability))[:, None]).T @ matrix
        hessian += RIDGE * np.eye(matrix.shape[1])
        step = np.linalg.solve(hessian, gradient)
        weights += step
        if float(np.abs(step).max()) < 1e-9:
            break
    return weights


def _rows(page: Page) -> list[dict[str, Any]]:
    """One row per usable ground-truth edge: its features and whether the canonical way is right."""
    diagram = truth(page)
    nodes = {n.id: n for n in diagram.nodes}
    evidence = evidence_for(page)
    out = []
    for edge in diagram.edges:
        scores = features(edge, nodes, evidence)
        if scores is None:
            continue
        first, _ = sorted((edge.src, edge.dst))
        out.append({"source": page.source, "y": float(edge.src == first), **scores})
    return out


def collect(split: str, n_jobs: int = 3) -> list[dict[str, Any]]:
    held = pages((split,))
    return [
        row for page_rows in pmap(_rows, held, n_jobs=n_jobs, prefer="threads") for row in page_rows
    ]


def _matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    return np.array([[row[name] for name in names] for row in rows], dtype=float)


def _accuracy(rows: list[dict[str, Any]], names: tuple[str, ...], weights: np.ndarray) -> float:
    """Share of edges whose direction is right. A score of exactly 0 keeps the canonical order,
    which is the `id_order` control and is measured, not assumed, to be a coin."""
    if not rows:
        return float("nan")
    total = _matrix(rows, names) @ weights
    predicted = (total >= 0.0).astype(float)
    return float((predicted == np.array([r["y"] for r in rows])).mean())


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def flow_prior(split: str = "train") -> dict[str, dict[str, float]]:
    """The corpus's actual flow convention, per source, from pages nothing is scored on.

    Measured rather than assumed: the plan asserts top-to-bottom and left-to-right, and only one
    of those two is true of this corpus.
    """
    out: dict[str, dict[str, float]] = {}
    for page in pages((split,)):
        diagram = truth(page)
        nodes = {n.id: n for n in diagram.nodes}
        bucket = out.setdefault(
            page.source,
            {
                "edges": 0,
                "right": 0,
                "down": 0,
                "h_major": 0,
                "h_right": 0,
                "v_major": 0,
                "v_down": 0,
            },
        )
        for edge in diagram.edges:
            u, v = nodes.get(edge.src), nodes.get(edge.dst)
            if u is None or v is None or u.bbox is None or v.bbox is None:
                continue
            (ux, uy), (vx, vy) = centre(u.bbox), centre(v.bbox)
            dx, dy = vx - ux, vy - uy
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            bucket["edges"] += 1
            bucket["right"] += dx > 0
            bucket["down"] += dy > 0
            if abs(dx) >= abs(dy):
                bucket["h_major"] += 1
                bucket["h_right"] += dx > 0
            else:
                bucket["v_major"] += 1
                bucket["v_down"] += dy > 0
    summary = {}
    for source, b in sorted(out.items()):
        n = max(1, b["edges"])
        summary[source] = {
            "edges": b["edges"],
            "left_to_right": round(b["right"] / n, 4),
            "top_to_bottom": round(b["down"] / n, 4),
            "right_given_horizontal": round(b["h_right"] / max(1, b["h_major"]), 4),
            "down_given_vertical": round(b["v_down"] / max(1, b["v_major"]), 4),
        }
    return summary


def run(n_jobs: int = 3) -> dict:
    fit_rows = collect(FIT_SPLIT, n_jobs)
    report_rows = collect(REPORT_SPLIT, n_jobs)
    sources = sorted({row["source"] for row in report_rows})

    ablations: dict[str, dict[str, float]] = {}
    weights_out: dict[str, dict[str, float]] = {}
    for label, names in ABLATIONS:
        ablations[label] = {}
        weights_out[label] = {}
        for source in sources:
            train = [r for r in fit_rows if r["source"] == source]
            test = [r for r in report_rows if r["source"] == source]
            weights = fit(_matrix(train, names), np.array([r["y"] for r in train]))
            ablations[label][source] = round(_accuracy(test, names, weights), 4)
            weights_out[label][source] = {
                name: round(float(w), 4) for name, w in zip(names, weights, strict=True)
            }

    rng = np.random.default_rng(42)
    controls = {"id_order": {}, "random": {}}
    for source in sources:
        test = [r for r in report_rows if r["source"] == source]
        y = np.array([r["y"] for r in test])
        controls["id_order"][source] = round(float(y.mean()), 4)
        controls["random"][source] = round(float((rng.integers(0, 2, len(y)) == y).mean()), 4)

    best = ABLATIONS[-1][0]
    return {
        "fit_split": FIT_SPLIT,
        "report_split": REPORT_SPLIT,
        "edges": {s: sum(1 for r in report_rows if r["source"] == s) for s in sources},
        "fit_edges": {s: sum(1 for r in fit_rows if r["source"] == s) for s in sources},
        "flow_prior": flow_prior(),
        "ablation": ablations,
        "controls": controls,
        "weights": weights_out,
        "target": 0.90,
        "meets_target": {s: bool(ablations[best][s] >= 0.90) for s in sources},
        # The two comparisons the row is actually about.
        "beats_prior_by": {
            s: round(ablations[best][s] - ablations["prior"][s], 4) for s in sources
        },
        "head_det_minus_head_geo": {
            s: round(ablations["head_det"][s] - ablations["head_geo"][s], 4) for s in sources
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RUNS / "direction.json")
    args = ap.parse_args(argv)

    result = run(args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "weights"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

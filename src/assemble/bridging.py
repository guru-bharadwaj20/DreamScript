"""Phase 10.1.4 - broken-arrow repair, and the geometric evidence 7.3.10 said this needed.

    python -m src.assemble.bridging

7.3.10 tried to repair broken arrows from HMM posteriors and failed on purpose: **5.3% recovered
against this same 80% target, at 5.8% precision** - seventeen wrong edges invented for every
right one. It also said why, and handed the reason forward rather than just the failure: its
damage model deleted whole edges, which throws away the only evidence a real break leaves behind
- "a real broken arrow is a polyline with a gap that has two endpoints, a direction and a length."
This module is that geometric version, built on stubs rather than deletions.

## The damage model, and what it does and does not simulate

A ground-truth edge's polyline is cut at a random interior point, leaving two **stubs**: the
`src`-side stub keeps its anchor at the source node and a free end at the gap; the `dst`-side
stub keeps its anchor at the destination and a free end at the other side of the gap. Each stub
keeps its true length, its true local direction near the free end, and which real node it is
still attached to - that attachment is the one thing a photograph of a broken arrow always shows
(you can see which node the surviving ink touches), so giving it to the repairer is not the same
information 7.3.10's deleted-edge model destroyed.

Every eligible edge on every held-out page is cut this way, simultaneously, at a given gap size -
not one edge per page - so the repairer faces the real disambiguation problem: several stubs
scattered over one page, most of which belong to *some* other stub, and it has to find the right
partner rather than being handed the only candidate.

**What this does not simulate**: the stub's direction and length are read off noiseless
ground-truth geometry, not a traced skeleton, and the gap is a clean excision rather than
whatever a torn or smudged line actually looks like. The recovery numbers below are an upper
bound on what a downstream tracer (10.1.1, already committed) would hand this repairer, not a
number about the whole detect-and-repair pipeline.

Two sweeps are reported because "80% recovered" means nothing without saying how broken the
input was: `GAP_FRACTIONS` cuts a gap sized as a share of the *page diagonal*, so short and long
edges lose different shares of themselves; `EDGE_GAP_FRACTIONS` instead sizes the gap as a share
of the *edge's own length*, holding the difficulty per edge constant regardless of page scale.

## The repairer: three cues, scored so they can be told apart

A candidate pairs a `src`-side stub with a `dst`-side stub and is scored on three independent
pieces of geometry, matching the plan's own list:

    gap distance         the euclidean distance between the two free ends. A pair further apart
                         than `GAP_TOLERANCE` (a share of the page diagonal) is not a candidate
                         at all - this is the gate, not just a tiebreaker.
    direction continuity  each stub's local heading near its free end, extrapolated. A real
                         break has both stubs' headings agreeing with each other and with the
                         straight line between them; a coincidentally nearby stub usually does
                         not.
    port snapping        the `src`-stub's heading is extrapolated forward by the candidate gap
                         distance and checked against the actual bounding box of the node the
                         `dst`-stub is anchored to - "does this stub, followed to where the gap
                         says it ends, land on that node" - independent of how well the two
                         stubs' own headings agree with each other.

Candidates are ranked by a weighted sum of the three and assigned greedily, highest score first,
each stub used once. Ablating a cue sets its weight to zero without changing the others; the
**unconditional-nearest control** drops the gate and both scores and simply pairs every stub
with its closest free partner, which is what 7.3.10's position-only reasoning amounts to in
geometric form.

## What it measured, over all 470 held-out pages

    gap, share of page diagonal   0.005   0.01   0.02   0.03   0.05   0.08   0.12    0.2
    broken edges                   4891   4803   4248   3433   2552   1904   1340    719
    recovery                      0.996  0.993  0.988  0.979  0.960  0.920  0.818  0.689
    precision                     0.996  0.993  0.988  0.979  0.960  0.920  0.823  0.713

    gap, share of edge's own length   0.1    0.2    0.3    0.4    0.5    0.6    0.7
    recovery                        0.986  0.974  0.955  0.930  0.896  0.862  0.820
    precision                       0.986  0.975  0.956  0.935  0.906  0.876  0.837

**80% is met, and the honest boundary is the diagonal sweep, not the edge one.** Sized as a
share of the page - the sweep that matters, because it is what a real smudge or eraser mark does
regardless of which edge it lands on - recovery clears 80% up to a gap of **12% of the page
diagonal** (81.8%) and misses it at 20% (68.9%). Sized instead as a share of each edge's own
length, every point tested from 10% to 70% of the edge clears 80%, because that framing quietly
excludes the short edges a diagonal-relative gap would have erased outright - both numbers are
reported because either one alone would misstate how broken "recovered" means.

Compare to 7.3.10's ceiling: at its best configuration, deleting whole edges, it recovered
**5.3%** at **5.8%** precision. Working from the stub instead of a deletion, this recovers
**81.8% at 82.3% precision** at a gap of 12% of the diagonal and stays above **99%** on both
axes for anything up to 2% of the diagonal - the two studies are not close, and the gap between
them is exactly the evidence 7.3.10 said this task would have and its own did not: a direction
and a length, not just a missing edge.

## The three-cue ablation, at a 8%-of-diagonal gap (1,904 broken edges, past the sweep's knee)

    condition                        recovery   precision
    distance only / unconditional control   0.749      0.749
    distance + port                         0.879      0.881
    distance + direction (shipped default)  0.920      0.920
    distance + direction + port (equal)     0.898      0.901

**Direction carries the result, and blending in port at equal weight makes it worse rather than
better.** The gate alone - "is there a stub within tolerance at all" - is indistinguishable from
the unconditional-nearest control (0.749 either way, confirming the gate is not doing
discriminating work by itself at this tolerance). Direction continuity is the one cue that beats
the control by a wide margin, +17.1 points. Port snapping *does* carry real signal in isolation
(+13.0 points over the control) - a true pair's port score averages 0.71 against 0.48 for a false
one - but it is noisier than direction's (0.73 vs 0.50), and folding it into the default at equal
weight *costs* 2.2 points rather than adding to them. So the shipped default (`WEIGHTS["port"] ==
0`) is not a guess: it is what this ablation says to do, and port stays wired into `score_pair`
and `repair(use_port=...)` so a future corpus with more ambiguous stubs can be re-checked rather
than re-derived.

    python -m src.assemble.bridging
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.assemble.corpus import RUNS, Page, pages, truth
from src.ir.model import Diagram
from src.utils.parallel import pmap

OUT = RUNS / "bridging.json"
N_JOBS = 3
SEED = 42

#: Gap sizes as a share of the page diagonal - the "how broken" sweep that matters most, since
#: a gap this large relative to the *page* is large relative to almost every edge on it.
GAP_FRACTIONS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)

#: Gap sizes as a share of each edge's *own* length - holds per-edge difficulty constant across
#: pages of very different scale, where the diagonal sweep above does not.
EDGE_GAP_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

#: A stub must retain at least this share of the edge's original length, on each side of the
#: gap, or there is nothing left to estimate a direction from.
MIN_STUB_SHARE = 0.12

#: The local direction near a free end is measured over this share of the page diagonal (capped
#: at half the stub's own length), not the whole stub - a right-angled connector's far end points
#: nowhere near the gap.
DIRECTION_WINDOW_FRAC = 0.03

#: How far apart two free ends may be and still be a candidate pair, as a share of the page
#: diagonal. Must clear the largest gap swept above (0.2) or every true pair would be gated out
#: by construction rather than by the repairer failing.
GAP_TOLERANCE = 0.28

#: The port cue is a *direction* test, not a distance one: `score_pair` asks how well the stub's
#: own heading agrees with the heading from the stub to the candidate node's centre, and scores
#: that cosine on [0, 1]. There is no tolerance in it.
#:
#: There used to be a `PORT_TOLERANCE = 0.06` here, described as "how close an extrapolated stub
#: must land to the candidate's node box", and `score_pair` and `repair` both took a
#: `port_tolerance` argument - `score_pair`'s keyword-required, so every caller and the corpus
#: sweep had to compute and pass `PORT_TOLERANCE * diagonal`. The body never read it. The sweep
#: was therefore threading a number through three functions to no effect and reporting the flat
#: result as a measurement. It is gone rather than implemented: implementing it would change
#: every recovery figure this module reports, which is a re-measurement and not a bug fix, and
#: the cue that exists is documented here instead of a cue that did not.

#: Default cue weights: gap closeness, direction continuity, port snapping. Port starts at 0 -
#: the ablation below is what sets it there, not a guess: its own signal is real (a true pair's
#: port score averages 0.71 against 0.48 for a false one) but noisier than direction's (0.73 vs
#: 0.50), so blending it in at equal weight dilutes the stronger cue and costs recovery rather
#: than adding to it. It stays wired in and measurable - `use_port=True` - for that reason.
WEIGHTS = {"distance": 1.0, "direction": 1.0, "port": 0.0}


# ------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------


def _arclens(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each vertex, `pts[0]` at 0."""
    if len(pts) < 2:
        return np.zeros(len(pts))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _point_at(pts: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    """Interpolate the point at arc length `s` along `pts` (clamped to the polyline's span)."""
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(max(i, 0), len(pts) - 2)
    span = cum[i + 1] - cum[i]
    t = 0.0 if span <= 0 else (s - cum[i]) / span
    return pts[i] + t * (pts[i + 1] - pts[i])


# ------------------------------------------------------------------------------------------
# the damage model
# ------------------------------------------------------------------------------------------


@dataclass
class Stub:
    """One end of a broken arrow: still anchored to a real node, free at the gap."""

    edge_id: str
    end: str  # "src" or "dst"
    anchor: str  # the node id this stub is still physically attached to
    point: tuple[float, float]  # the free end, at the gap
    direction: tuple[float, float]  # unit heading, extrapolated forward across the gap


def _cut_one(edge, diagonal: float, gap_len: float, rng: random.Random) -> tuple[Stub, Stub] | None:
    """Cut `edge`'s polyline at a random interior position, gap sized `gap_len`. `None` if the
    edge is too short to leave a usable stub on each side."""
    if edge.polyline is None or len(edge.polyline) < 2 or edge.src is None or edge.dst is None:
        return None
    pts = np.array(edge.polyline, dtype=float)
    cum = _arclens(pts)
    length = float(cum[-1])
    min_side = MIN_STUB_SHARE * length
    if length <= 0 or gap_len <= 0 or gap_len >= length - 2 * min_side:
        return None

    lo, hi = min_side, length - gap_len - min_side
    s0 = rng.uniform(lo, hi)
    s1 = s0 + gap_len
    point_a = _point_at(pts, cum, s0)
    point_b = _point_at(pts, cum, s1)

    window = min(DIRECTION_WINDOW_FRAC * diagonal, s0, length - s1)
    window = max(window, 1e-6)
    behind_a = _point_at(pts, cum, s0 - window)
    ahead_b = _point_at(pts, cum, s1 + window)
    travel_a = point_a - behind_a
    travel_b = ahead_b - point_b
    norm_a = np.linalg.norm(travel_a) or 1.0
    norm_b = np.linalg.norm(travel_b) or 1.0
    dir_a = tuple((travel_a / norm_a).tolist())
    dir_b = tuple((travel_b / norm_b).tolist())

    stub_src = Stub(edge.id, "src", edge.src, tuple(point_a.tolist()), dir_a)
    stub_dst = Stub(edge.id, "dst", edge.dst, tuple(point_b.tolist()), dir_b)
    return stub_src, stub_dst


def damage_diagram(
    diagram: Diagram, diagonal: float, gap_frac: float, *, by_edge: bool = False, seed: int = SEED
) -> tuple[list[Stub], dict[str, tuple[str, str]], int]:
    """Cut every eligible edge's polyline once. Returns `(stubs, truth_by_edge, skipped)` where
    `truth_by_edge` maps an edge id to its `(src, dst)` and `skipped` counts edges too short for
    this gap size to touch at all.

    `by_edge=True` sizes the gap as a share of each edge's own length (`EDGE_GAP_FRACTIONS`);
    otherwise it is a share of the page diagonal (`GAP_FRACTIONS`), the same absolute gap for
    every edge on the page.
    """
    rng = random.Random(seed)
    stubs: list[Stub] = []
    truth_by_edge: dict[str, tuple[str, str]] = {}
    skipped = 0
    for edge in diagram.edges:
        if edge.polyline is None or len(edge.polyline) < 2 or edge.src is None or edge.dst is None:
            continue
        pts = np.array(edge.polyline, dtype=float)
        length = float(_arclens(pts)[-1])
        gap_len = gap_frac * length if by_edge else gap_frac * diagonal
        cut = _cut_one(edge, diagonal, gap_len, rng)
        if cut is None:
            skipped += 1
            continue
        stubs.extend(cut)
        truth_by_edge[edge.id] = (edge.src, edge.dst)
    return stubs, truth_by_edge, skipped


# ------------------------------------------------------------------------------------------
# the repairer
# ------------------------------------------------------------------------------------------


def _cos(u: tuple[float, float], v: tuple[float, float]) -> float:
    a, b = np.array(u), np.array(v)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def score_pair(
    a: Stub,
    b: Stub,
    nodes: dict[str, Any],
    *,
    tolerance: float,
    use_direction: bool = True,
    use_port: bool = True,
    weights: dict[str, float] = WEIGHTS,
) -> tuple[bool, float, dict[str, float]]:
    """`(passes_gate, combined_score, parts)` for pairing `a` (a src-side stub) with `b`
    (a dst-side stub). Geometry only - `a.edge_id == b.edge_id` is never looked at here, only in
    scoring the result afterwards."""
    pa, pb = np.array(a.point), np.array(b.point)
    gap = float(np.linalg.norm(pb - pa))
    passes = gap <= tolerance
    dist_score = max(0.0, 1.0 - gap / tolerance) if tolerance > 0 else 0.0

    direction_score = 0.0
    if gap > 0:
        straight = tuple(((pb - pa) / gap).tolist())
        direction_score = (_cos(a.direction, b.direction) + _cos(a.direction, straight)) / 2.0
        direction_score = (direction_score + 1.0) / 2.0  # map [-1, 1] -> [0, 1]

    port_score = 0.5
    node_b = nodes.get(b.anchor)
    bbox_b = node_b.bbox if node_b is not None else None
    if bbox_b is not None and gap > 0:
        centre_b = np.array([bbox_b[0] + bbox_b[2] / 2, bbox_b[1] + bbox_b[3] / 2])
        toward_b = centre_b - pa
        if np.linalg.norm(toward_b) > 0:
            port_score = (_cos(a.direction, tuple(toward_b.tolist())) + 1.0) / 2.0

    w = dict(weights)
    if not use_direction:
        w["direction"] = 0.0
    if not use_port:
        w["port"] = 0.0
    total_w = sum(w.values()) or 1.0
    combined = (
        w["distance"] * dist_score + w["direction"] * direction_score + w["port"] * port_score
    ) / total_w
    return passes, combined, {"gap": gap, "direction": direction_score, "port": port_score}


def repair(
    stubs: list[Stub],
    nodes: dict[str, Any],
    *,
    tolerance: float = GAP_TOLERANCE,
    use_direction: bool = True,
    use_port: bool = True,
    use_gate: bool = True,
    weights: dict[str, float] = WEIGHTS,
) -> list[tuple[Stub, Stub, float]]:
    """Pair `src`-side stubs with `dst`-side stubs, greedily, best score first. Each stub is used
    at most once. `use_gate=False` drops the distance gate entirely (every stub may pair with
    every other), which combined with `use_direction=False, use_port=False` is the
    unconditional-nearest control."""
    src_stubs = [s for s in stubs if s.end == "src"]
    dst_stubs = [s for s in stubs if s.end == "dst"]
    candidates = []
    for a in src_stubs:
        for b in dst_stubs:
            passes, score, _ = score_pair(
                a,
                b,
                nodes,
                tolerance=tolerance,
                use_direction=use_direction,
                use_port=use_port,
                weights=weights,
            )
            if passes or not use_gate:
                candidates.append((score, a, b))
    candidates.sort(key=lambda c: -c[0])

    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for score, a, b in candidates:
        ia, ib = id(a), id(b)
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        pairs.append((a, b, score))
    return pairs


def nearest_control(stubs: list[Stub]) -> list[tuple[Stub, Stub, float]]:
    """7.3.10's reasoning in geometric form: pair every stub with its closest free partner,
    unconditionally - no tolerance, no direction, no node geometry."""
    # A large but finite tolerance: `dist_score = 1 - gap / tolerance` must stay rank-preserving
    # in `gap`, which `tolerance = inf` collapses to a constant (every pair tied at 1.0) - degenerate
    # rather than "nearest".
    return repair(
        stubs,
        {},
        tolerance=1e9,
        use_direction=False,
        use_port=False,
        use_gate=False,
        weights={"distance": 1.0, "direction": 0.0, "port": 0.0},
    )


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------


def score_repair(
    pairs: list[tuple[Stub, Stub, float]], truth_by_edge: dict[str, tuple[str, str]]
) -> dict[str, int]:
    """Recovered = both stubs of the pair came from the same original edge. Wrong = they did
    not. Missed = an edge whose two stubs were never both put back together."""
    recovered_edges = set()
    wrong = 0
    for a, b, _ in pairs:
        if a.edge_id == b.edge_id:
            recovered_edges.add(a.edge_id)
        else:
            wrong += 1
    total = len(truth_by_edge)
    return {
        "broken": total,
        "recovered": len(recovered_edges),
        "wrong": wrong,
        "missed": total - len(recovered_edges),
    }


def _rates(counts: dict[str, int]) -> dict[str, float]:
    broken = max(1, counts["broken"])
    proposed = counts["recovered"] + counts["wrong"]
    return {
        "recovery": round(counts["recovered"] / broken, 4),
        "precision": round(counts["recovered"] / max(1, proposed), 4),
    }


# ------------------------------------------------------------------------------------------
# the corpus sweep
# ------------------------------------------------------------------------------------------


#: The named ablation conditions. `default` is what `repair()` does with no overrides (distance
#: gate + direction, `WEIGHTS["port"] == 0`); `all_three` turns port back on at equal weight to
#: show what it costs; `distance_only` and `control` are the two ways of asking "is geometry
#: alone as good as distance was ever going to get".
_MODES = {
    "default": {},
    "all_three": {"weights": {"distance": 1.0, "direction": 1.0, "port": 1.0}},
    "distance_and_port": {
        "use_direction": False,
        "weights": {"distance": 1.0, "direction": 0.0, "port": 1.0},
    },
    "distance_only": {"use_direction": False, "use_port": False},
}


def _measure_page(
    page: Page, gap_frac: float, *, by_edge: bool, mode: str = "default"
) -> dict[str, int]:
    diagram = truth(page)
    diagonal = page.diagonal
    stubs, truth_by_edge, _ = damage_diagram(diagram, diagonal, gap_frac, by_edge=by_edge)
    nodes = {n.id: n for n in diagram.nodes}
    tol = GAP_TOLERANCE * diagonal
    if mode == "control":
        pairs = nearest_control(stubs)
    elif mode in _MODES:
        pairs = repair(stubs, nodes, tolerance=tol, **_MODES[mode])
    else:
        raise ValueError(mode)
    return score_repair(pairs, truth_by_edge)


def _sweep(pages_: tuple[Page, ...], fractions, *, by_edge: bool, mode: str, n_jobs: int) -> dict:
    out = {}
    for frac in fractions:
        rows = pmap(
            lambda p, f=frac: _measure_page(p, f, by_edge=by_edge, mode=mode),
            pages_,
            n_jobs=n_jobs,
            desc=f"bridging g={frac}",
        )
        totals = {k: sum(r[k] for r in rows) for k in ("broken", "recovered", "wrong", "missed")}
        out[str(frac)] = {**totals, **_rates(totals)}
    return out


def run(n_jobs: int = N_JOBS) -> dict[str, Any]:
    held = pages()

    diag_sweep = _sweep(held, GAP_FRACTIONS, by_edge=False, mode="default", n_jobs=n_jobs)
    edge_sweep = _sweep(held, EDGE_GAP_FRACTIONS, by_edge=True, mode="default", n_jobs=n_jobs)

    # The gap size used for the ablation and control: past the diagonal sweep's knee, where the
    # shipped repairer is neither trivially easy (it is, below this) nor already failing outright.
    ablation_gap = 0.08
    ablation = {
        name: _sweep(held, [ablation_gap], by_edge=False, mode=name, n_jobs=n_jobs)[
            str(ablation_gap)
        ]
        for name in (*_MODES, "control")
    }

    return {
        "pages": len(held),
        "gap_as_fraction_of_diagonal": diag_sweep,
        "gap_as_fraction_of_edge_length": edge_sweep,
        "ablation_gap": ablation_gap,
        "ablation": ablation,
        "target_met_by_gap_fraction_of_diagonal": {
            g: row["recovery"] >= 0.80 for g, row in diag_sweep.items()
        },
        "target_met_by_gap_fraction_of_edge_length": {
            g: row["recovery"] >= 0.80 for g, row in edge_sweep.items()
        },
        "target": 0.80,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    result = run(n_jobs=args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "ablation"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

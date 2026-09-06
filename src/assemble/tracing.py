"""Phase 10.1.3 - following the skeleton between node boundaries, and what that recovers.

    python -m src.assemble.tracing
    python -m src.assemble.tracing --limit 40        # a quick pass while iterating

The plan's line is *follow skeleton polylines between node boundaries*, and the whole task is in
the word *between*: a connector is what is left of the drawing once the nodes are taken out of it.
So the tracer erases the detected boxes from the shape layer, thins what remains, walks the
resulting one-pixel graph, and hands each walk's two ends back to the nearest box.

## The pipeline, and the two decisions inside it

    1  shape layer      3.1's mask through 3.2's text/shape split, so a label inside a box is
                        not mistaken for a connector. The detector image is at most 1280px and
                        `layers.WORKING_WIDTH` is 1400, so `prepare` does not resize and the
                        mask is already in detector pixels - the one frame Phase 10 works in.
    2  erase the nodes  every box, inflated by `BOX_PAD` of its own short side, is zeroed. The
                        inflation is the first decision: a box drawn tight around a rectangle
                        leaves its own outline behind, and an outline is a closed loop that
                        every connector touching that node is then welded to.
    3  thin + prune     Zhang-Suen, then `prune_spurs`, because a ragged stroke edge thins into
                        whiskers and a whisker is a phantom three-way junction.
    4  walk             elementary paths between skeleton pixels of degree != 2, then chained
                        *through* junctions by direction - the second decision, below.
    5  attach           each end to the nearest box boundary within `ATTACH_TOL` of the page
                        diagonal, or to nothing.

**Chaining through junctions is what makes this a tracer rather than a segment detector.** Split
at every junction and a single arrow becomes four stubs, because its head is a junction and so is
every place it crosses another line. At each junction the incoming path directions are compared
and the two that are most nearly collinear are joined, if they turn by less than `MAX_TURN`;
everything else stays separate. A crossing is then traced straight through, and a T-junction where
one line genuinely ends into another is not joined into a spurious corner.

## Direction and gaps are not this task

The walk gives a polyline with an arbitrary orientation - whichever end it started from - and
`src` / `dst` are that order, not the arrow's. **10.1.5 decides direction; nothing here should be
read as one.** A stroke broken in two is traced as two edges with a dangling end each, which is
exactly the evidence 10.1.4 needs, and is left that way.

## How it is scored

An edge is *recovered* if its endpoint pair matches a ground-truth edge's pair. Traced endpoints
are detected boxes, so each is first mapped to the ground-truth node it overlaps by IoU >= 0.5;
pairs are compared **unordered** because direction is not this task's, and matched one-to-one and
greedily so that tracing one edge twice cannot be counted twice. flowchartseg's 132 IR files carry
**zero** edges, so it cannot contribute a recall and is used only as a false-positive count.

**The ceiling run replaces the detected boxes with the ground-truth ones and changes nothing
else.** That contrast is the number worth having: it separates "the tracer cannot follow this
stroke" from "the detector put the box in the wrong place", and those two failures need different
fixes in different tasks.

## What it measured

470 held-out pages; 4,712 ground-truth edges, all on hdbpmn (242 pages, 4,082 of them) and
fa_bresler (96 pages, 630 of them) - flowchartseg's 132 pages carry none.

    boxes            edges traced   recall   precision      F1   dangling ends
    detected               17,785   0.2328      0.0626   0.0987          0.5223
    ground truth           19,357   0.2455      0.0606   0.0972          0.5143

    per source (detected -> ground truth)
    hdbpmn        16,085 -> 17,659 traced   recall 0.1325 -> 0.1475   precision 0.0336 -> 0.0341
    fa_bresler     1,433 ->  1,430 traced   recall 0.8825 -> 0.8810   precision 0.3880 -> 0.3881
    flowchartseg     267 ->    268 traced, 0 GT edges - all of them false by construction

**Recall is poor and precision is worse, and the honest headline is that this is an
over-segmentation problem, not a missing-edge problem.** The tracer emits **17,785 polylines
against 4,712 ground-truth edges** - almost four traced fragments for every real one - and only
1,097 of them land on a correct pair. A pipeline that shattered every real connector into 3-4
pieces and matched none of the extra pieces to anything would produce exactly this shape of
result: recall in the low twenties because *some* fragment of most edges still attaches correctly
at least once, and precision near six percent because the other three fragments count against it.

**The ceiling shows the detector is nearly irrelevant to this failure.** Perfect boxes move recall
from **0.2328 to 0.2455**, a 5.5% relative gain, and precision does not move with it - it is
**slightly worse** with perfect boxes (0.0606 against 0.0626), because removing the node correctly
still leaves the same fragmented, crossing-heavy ink behind for the walk to over-segment. That is
the opposite of what a detector-limited pipeline would show, and it means re-running this after
9.1's detector improves would buy under six percent of recall - the loss is almost entirely in the
walk, not the boxes.

**Dangling ends are 52.23% of all ends detected, 51.43% with perfect boxes - essentially unmoved
by box quality**, which is further evidence the failure is upstream of attachment. An end that
attaches to no box within 6% of the diagonal is either a stroke that stops short of its node -
10.1.4's gap - or a fragment of node outline that was never a connector; with over-segmentation
this severe, most dangling ends are the latter: pieces of a shattered connector or a leftover box
contour that never reaches anything.

**fa_bresler traces far better than hdbpmn at every setting** (recall 0.8825 against 0.1325
detected - a 6.7x gap), and the reason is drawing density: fa_bresler pages average 4.9 nodes and
9.5 edges; hdbpmn pages average 18.3 nodes and 19.5 edges, with connectors that run in bundles down
the same corridor and cross each other repeatedly. Every crossing is a junction where the
collinear-continuation rule has to guess, and hdbpmn supplies 16,085 of the 17,785 detected traces
- 90% of all output - almost entirely on the corpus dense enough to fragment. **Density, not
source, is what the tracer is sensitive to, and hdbpmn is where nearly all of the over-segmentation
lives.**

**flowchartseg contributes 267 edges that are all false**, because its IR has no edges at all.
That is not a measured false-positive *rate* - the pages certainly have connectors drawn on them -
so it is reported as a count and excluded from precision, which is computed only over the 338
pages where a ground-truth pair exists to be right about. It is also, at 267 traces over 132 dense
pages, far smaller a contributor to the false-positive total than hdbpmn's own over-tracing.

**The confidence is a heuristic and is not calibrated.** It averages the share of ends attached
with the polyline's straightness (chord over arc), which ranks a clean two-ended trace above a
one-ended squiggle and is worth exactly that much. 9.3.7's lesson applies unchanged: use it as a
rank, never as a probability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np

from src.assemble.corpus import RUNS, Page, corners, detections, iou, pages, truth
from src.ir.model import Diagram, Edge, Node

#: A node box is erased inflated by this share of its own short side. Below about 0.03 the box's
#: own drawn outline survives and welds every connector touching that node into one loop.
BOX_PAD = 0.06

#: Detections below this score are not treated as nodes. `corpus.CONF_FLOOR` is 0.05 so that
#: assembly can see the weak boxes; a tracer that erased all of them would erase the page.
MIN_SCORE = 0.25

#: Whiskers shorter than this many pixels are pruned off the skeleton before the walk.
SPUR_LENGTH = 6

#: A traced polyline shorter than this share of the page diagonal is dropped: at 1500px that is
#: 30 pixels, which is shorter than any connector and the length of a leftover box corner.
MIN_LENGTH = 0.02

#: Two paths meeting at a junction are joined only if the walk turns by less than this. 60 deg
#: traces a crossing straight through and refuses to round a genuine T into a corner.
MAX_TURN = 60.0

#: An end attaches to a box whose boundary is within this share of the page diagonal.
ATTACH_TOL = 0.06

#: Douglas-Peucker tolerance for the emitted polyline, as a share of the diagonal.
SIMPLIFY = 0.004

#: A detected box is identified with a ground-truth node above this IoU, for scoring only. The
#: tracer never sees ground truth.
MATCH_IOU = 0.5

_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ------------------------------------------------------------------------------------------
# the connector mask
# ------------------------------------------------------------------------------------------


def shape_layer(page: Page) -> np.ndarray:
    """The page's drawing ink, text removed, in detector pixels."""
    from src.preprocess.layers import prepare, separate

    gray, mask = prepare(page.image)
    return separate(mask, gray).shape


def erase_nodes(mask: np.ndarray, boxes: list[dict], pad: float | None = None) -> np.ndarray:
    """Zero every node box, inflated by `pad` of its short side. What is left is connector ink."""
    pad = BOX_PAD if pad is None else pad
    out = mask.astype(bool).copy()
    height, width = out.shape
    for box in boxes:
        x, y, w, h = box["bbox"]
        grow = pad * max(1.0, min(w, h))
        x1 = max(0, int(math.floor(x - grow)))
        y1 = max(0, int(math.floor(y - grow)))
        x2 = min(width, int(math.ceil(x + w + grow)))
        y2 = min(height, int(math.ceil(y + h + grow)))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = False
    return out


# ------------------------------------------------------------------------------------------
# walking the skeleton
# ------------------------------------------------------------------------------------------


def _skeleton_pixels(mask: np.ndarray) -> set[tuple[int, int]]:
    from src.preprocess.thinning import prune_spurs, thin

    skeleton = prune_spurs(thin(mask), SPUR_LENGTH)
    ys, xs = np.nonzero(skeleton)
    return {(int(y), int(x)) for y, x in zip(ys, xs, strict=True)}


def _clusters(keys: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Group touching non-degree-2 pixels: a fat junction is one junction, not four."""
    label: dict[tuple[int, int], int] = {}
    index = 0
    for pixel in sorted(keys):
        if pixel in label:
            continue
        stack, index = [pixel], index + 1
        while stack:
            current = stack.pop()
            if current in label:
                continue
            label[current] = index
            y, x = current
            stack.extend(
                p for d in _NEIGHBOURS if (p := (y + d[0], x + d[1])) in keys and p not in label
            )
    return label


def elementary_paths(pixels: set[tuple[int, int]]) -> list[dict]:
    """Every run of degree-2 skeleton pixels between two junction/endpoint clusters."""
    degree = {
        p: sum(1 for d in _NEIGHBOURS if (p[0] + d[0], p[1] + d[1]) in pixels) for p in pixels
    }
    keys = {p for p, d in degree.items() if d != 2}
    label = _clusters(keys)
    paths: list[dict] = []
    seen: set[tuple] = set()
    for start in sorted(keys):
        for step in _NEIGHBOURS:
            first = (start[0] + step[0], start[1] + step[1])
            if first not in pixels:
                continue
            if first in keys:
                if label[first] == label[start]:
                    continue
                walk = [start, first]
            else:
                walk, previous, current = [start, first], start, first
                while current not in keys:
                    nexts = [
                        p
                        for d in _NEIGHBOURS
                        if (p := (current[0] + d[0], current[1] + d[1])) in pixels and p != previous
                    ]
                    if len(nexts) != 1:
                        break
                    previous, current = current, nexts[0]
                    walk.append(current)
                if current not in keys:
                    continue
            # Every arm is walked from both ends (once as `start`'s neighbour, once as
            # `first`'s), so the same physical path is found twice, forward and reversed.
            # The token must be identical either way or the reversed half survives dedup and
            # doubles every edge count downstream - compare the whole pixel run, not a sampled
            # midpoint, which disagrees with itself on reversal whenever the walk is even-length.
            forward, backward = tuple(walk), tuple(reversed(walk))
            token = min(forward, backward)
            if token in seen:
                continue
            seen.add(token)
            paths.append({"pixels": walk, "ends": (label[walk[0]], label[walk[-1]])})
    return paths


def _direction(walk: list[tuple[int, int]], at_start: bool) -> tuple[float, float]:
    """The unit vector pointing *away* from one end of a path, over up to 8 pixels."""
    span = walk[: min(9, len(walk))] if at_start else walk[-min(9, len(walk)) :][::-1]
    dy = span[-1][0] - span[0][0]
    dx = span[-1][1] - span[0][1]
    norm = math.hypot(dx, dy) or 1.0
    return (dx / norm, dy / norm)


def chain(paths: list[dict], max_turn: float | None = None) -> list[list[tuple[int, int]]]:
    """Join elementary paths through junctions, following the straightest continuation.

    At each junction the incoming directions are compared pairwise and only *mutually* best,
    near-collinear pairs are joined. A four-way crossing therefore traces as two straight lines
    and a T-junction as a line plus a stub that ends on it, which is what the drawing means.
    """
    max_turn = MAX_TURN if max_turn is None else max_turn
    at: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for i, path in enumerate(paths):
        at[path["ends"][0]].append((i, True))
        at[path["ends"][1]].append((i, False))

    joined: dict[tuple[int, bool], tuple[int, bool]] = {}
    limit = math.cos(math.radians(max_turn))
    for arms in at.values():
        if len(arms) < 2:
            continue
        vectors = {a: _direction(paths[a[0]]["pixels"], a[1]) for a in arms}
        scores: dict[tuple, float] = {}
        for i, a in enumerate(arms):
            for b in arms[i + 1 :]:
                if a[0] == b[0]:
                    continue
                # Straight-through means the two outgoing directions are opposed.
                scores[(a, b)] = -(vectors[a][0] * vectors[b][0] + vectors[a][1] * vectors[b][1])
        best: dict[tuple, tuple] = {}
        for (a, b), score in sorted(scores.items(), key=lambda kv: -kv[1]):
            if score < limit or a in best or b in best:
                continue
            best[a], best[b] = b, a
        for a, b in best.items():
            if a not in joined and b not in joined:
                joined[a], joined[b] = b, a

    traces, done = [], set()
    for i in range(len(paths)):
        if i in done:
            continue
        # Walk back to a free arm first, so a chain is emitted once and whole rather than from
        # wherever the enumeration happened to enter it.
        head, seen = (i, True), {i}
        while (following := joined.get(head)) is not None and following[0] not in seen:
            seen.add(following[0])
            head = (following[0], not following[1])

        walk: list[tuple[int, int]] = []
        cursor: tuple[int, bool] | None = head
        while cursor is not None and cursor[0] not in done:
            done.add(cursor[0])
            piece = paths[cursor[0]]["pixels"]
            piece = piece if cursor[1] else piece[::-1]
            walk.extend(piece if not walk else piece[1:])
            cursor = joined.get((cursor[0], not cursor[1]))
        if len(walk) >= 2:
            traces.append(walk)
    return traces


# ------------------------------------------------------------------------------------------
# attaching ends to boxes
# ------------------------------------------------------------------------------------------


def box_distance(point: tuple[float, float], bbox: list[float]) -> float:
    """Distance from a point to a box's boundary region; 0 inside."""
    x1, y1, x2, y2 = corners(bbox)
    dx = max(x1 - point[0], 0.0, point[0] - x2)
    dy = max(y1 - point[1], 0.0, point[1] - y2)
    return math.hypot(dx, dy)


def attach(point: tuple[float, float], boxes: list[dict], tolerance: float) -> str | None:
    """The id of the nearest box within `tolerance` pixels, or None."""
    best, best_distance = None, tolerance
    for box in boxes:
        d = box_distance(point, box["bbox"])
        if d <= best_distance:
            best, best_distance = box["id"], d
    return best


def _simplify(walk: list[tuple[int, int]], epsilon: float) -> list[list[float]]:
    import cv2

    points = np.array([[x, y] for y, x in walk], dtype=np.float32).reshape(-1, 1, 2)
    reduced = cv2.approxPolyDP(points, max(epsilon, 1.0), False).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in reduced]


def _straightness(polyline: list[list[float]]) -> float:
    arc = sum(math.dist(polyline[i], polyline[i + 1]) for i in range(len(polyline) - 1))
    chord = math.dist(polyline[0], polyline[-1])
    return float(chord / arc) if arc > 0 else 0.0


# ------------------------------------------------------------------------------------------
# the public API
# ------------------------------------------------------------------------------------------


def trace(page: Page, node_boxes: list[dict], *, mask: np.ndarray | None = None) -> list[Edge]:
    """Connector polylines between `node_boxes`, as IR edges.

    `node_boxes` is `[{"id": str, "bbox": [x, y, w, h]}]` in detector pixels. `src`/`dst` are the
    ids the two ends attached to, `None` where an end attached to nothing, and their **order is
    the walk's, not the arrow's** - 10.1.5 owns direction.
    """
    ink = shape_layer(page) if mask is None else mask
    remainder = erase_nodes(ink, node_boxes)
    if not remainder.any():
        return []
    traces = chain(elementary_paths(_skeleton_pixels(remainder)))

    minimum = MIN_LENGTH * page.diagonal
    tolerance = ATTACH_TOL * page.diagonal
    edges: list[Edge] = []
    for walk in traces:
        polyline = _simplify(walk, SIMPLIFY * page.diagonal)
        if len(polyline) < 2 or len(walk) < minimum:
            continue
        src = attach(tuple(polyline[0]), node_boxes, tolerance)
        dst = attach(tuple(polyline[-1]), node_boxes, tolerance)
        attached = (src is not None) + (dst is not None)
        edges.append(
            Edge(
                id=f"traced_{len(edges):04d}",
                src=src,
                dst=dst,
                directed=False,  # 10.1.5's job; an undirected polyline is what the walk knows.
                polyline=polyline,
                confidence=round(0.5 * (attached / 2.0) + 0.5 * _straightness(polyline), 4),
                attrs={"phase": "10.1.3", "pixels": len(walk)},
            )
        )
    return edges


def node_boxes(page: Page, min_score: float | None = None) -> list[dict]:
    """The detected boxes as tracer input - arrowheads excluded, they are not nodes."""
    min_score = MIN_SCORE if min_score is None else min_score
    return [
        {"id": f"d{i:03d}", "bbox": row["bbox"], "cls": row["cls"], "score": row["score"]}
        for i, row in enumerate(detections(page))
        if row["score"] >= min_score and row["cls"] != "arrowhead"
    ]


def trace_page(page: Page, *, mask: np.ndarray | None = None) -> list[Edge]:
    """`trace` with the boxes pulled from the detector cache."""
    return trace(page, node_boxes(page), mask=mask)


def to_diagram(page: Page, boxes: list[dict], edges: list[Edge]) -> Diagram:
    """Wrap a trace as an IR `Diagram`, with every open end recorded as unresolved."""
    diagram = Diagram(
        id=page.name,
        diagram_type="flowchart",
        nodes=[
            Node(id=b["id"], shape=b.get("cls", "freeform"), bbox=list(b["bbox"])) for b in boxes
        ],
        edges=list(edges),
        meta={"phase": "10.1.3", "frame": "detect-pixels", "directed": False},
    )
    diagram.sync_unresolved()
    return diagram


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------


def _identify(boxes: list[dict], diagram: Diagram) -> dict[str, str]:
    """Map each traced box id onto the ground-truth node it overlaps, by IoU."""
    out = {}
    for box in boxes:
        best, score = None, MATCH_IOU
        for node in diagram.nodes:
            if node.bbox is None:
                continue
            value = iou(box["bbox"], node.bbox)
            if value >= score:
                best, score = node.id, value
        if best is not None:
            out[box["id"]] = best
    return out


def score_page(edges: list[Edge], boxes: list[dict], diagram: Diagram) -> dict:
    """Recall/precision of endpoint pairs, matched unordered and one-to-one."""
    identity = _identify(boxes, diagram)
    truth_pairs: list[frozenset] = [
        frozenset((e.src, e.dst))
        for e in diagram.edges
        if e.src is not None and e.dst is not None and e.src != e.dst
    ]
    remaining: dict[frozenset, int] = defaultdict(int)
    for pair in truth_pairs:
        remaining[pair] += 1

    matched = 0
    ends = 2 * len(edges)
    dangling = sum((e.src is None) + (e.dst is None) for e in edges)
    for edge in edges:
        a, b = identity.get(edge.src or ""), identity.get(edge.dst or "")
        if a is None or b is None or a == b:
            continue
        pair = frozenset((a, b))
        if remaining.get(pair, 0) > 0:
            remaining[pair] -= 1
            matched += 1
    return {
        "traced": len(edges),
        "truth": len(truth_pairs),
        "matched": matched,
        "ends": ends,
        "dangling": dangling,
    }


def _totals(rows: list[dict]) -> dict:
    traced = sum(r["traced"] for r in rows)
    truths = sum(r["truth"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    # Precision is only defined where a ground-truth pair could have been matched, so pages with
    # no ground-truth edges are excluded from it rather than counted as all-wrong.
    scored = [r for r in rows if r["truth"]]
    traced_scored = sum(r["traced"] for r in scored)
    recall = matched / truths if truths else float("nan")
    precision = matched / traced_scored if traced_scored else float("nan")
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    ends = sum(r["ends"] for r in rows)
    return {
        "pages": len(rows),
        "traced": traced,
        "truth_edges": truths,
        "matched": matched,
        "recall": round(recall, 4) if truths else None,
        "precision": round(precision, 4) if traced_scored else None,
        "f1": round(f1, 4) if truths else None,
        "dangling_share": round(sum(r["dangling"] for r in rows) / ends, 4) if ends else 0.0,
    }


def run(limit: int | None = None, n_jobs: int = 4) -> dict:
    """Trace every held-out page twice - detected boxes, then ground-truth boxes - and score."""
    from src.utils.parallel import pmap

    held = pages()[:limit] if limit else pages()

    def one(page: Page) -> dict:
        diagram = truth(page)
        ink = shape_layer(page)
        detected = node_boxes(page)
        ideal = [
            {"id": n.id, "bbox": list(n.bbox), "cls": n.shape}
            for n in diagram.nodes
            if n.bbox is not None
        ]
        out = {"page": page.name, "source": page.source}
        for tag, boxes in (("detected", detected), ("truth", ideal)):
            edges = trace(page, boxes, mask=ink)
            out[tag] = score_page(edges, boxes, diagram)
            out[tag]["boxes"] = len(boxes)
        return out

    rows = pmap(one, held, n_jobs=n_jobs, prefer="threads", desc="tracing")
    sources = sorted({r["source"] for r in rows})
    return {
        "pages": len(rows),
        "by_source": {s: sum(1 for r in rows if r["source"] == s) for s in sources},
        "boxes": {
            "detected": {
                "overall": _totals([r["detected"] for r in rows]),
                "by_source": {
                    s: _totals([r["detected"] for r in rows if r["source"] == s]) for s in sources
                },
            },
            "truth": {
                "overall": _totals([r["truth"] for r in rows]),
                "by_source": {
                    s: _totals([r["truth"] for r in rows if r["source"] == s]) for s in sources
                },
            },
        },
        "constants": {
            "box_pad": BOX_PAD,
            "min_score": MIN_SCORE,
            "min_length": MIN_LENGTH,
            "max_turn": MAX_TURN,
            "attach_tol": ATTACH_TOL,
            "match_iou": MATCH_IOU,
        },
        "note": (
            "flowchartseg IR carries zero edges, so its traced count is a false-positive count "
            "and it contributes nothing to recall or precision. Direction is 10.1.5; gaps are "
            "10.1.4."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=lambda p: RUNS / p, default=RUNS / "tracing.json")
    args = ap.parse_args(argv)

    result = run(args.limit, args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

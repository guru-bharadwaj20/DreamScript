"""Phase 4.1.7 - connectivity: the graph the drawing implies, before Phase 10 builds one.

    conn_mean_degree    2 * edges / nodes
    conn_self_loops     connectors that leave a shape and come back to it
    conn_cycle_count    edges - nodes + components, the cyclomatic number
    conn_components     how many disconnected pieces the drawing is in

State machines have self-loops and flowcharts do not; a wireframe is a pile of unconnected
boxes and a process diagram is one connected piece. These are graph statistics, but this is not
graph assembly - Phase 10 does that, with OCR, arrow direction and the IR behind it. What
happens here is deliberately cruder and needs no learned component: **a strand of connector ink
is an edge between the shapes it touches.**

## Touching is decided by rectangle, not by tracing

Each shape's box, padded by `TOUCH_FRAC` of the page, is intersected with the labelled connector
components. Every strand appearing in that band is an edge end at that shape. This is one array
slice per shape rather than a path trace per strand, which matters because 4.2.2 runs it over
thousands of pages, and it fails in one predictable way: a connector that *passes over* a shape
on its way somewhere else is counted as touching it. On an orthogonal-routed flowchart that is
rare; on a dense ER diagram it is not, and it shows up below as an over-counted degree.

A strand touching one shape is a self-loop, but only if it is long enough to have gone
somewhere: `MIN_LOOP_FRAC` of the page's long side. Without that floor every stub of ink left
by a broken arrow becomes a self-loop, and state machines stop being distinguishable by the
feature that is supposed to identify them.

## Measured against the ground-truth graphs

Over 366 synthetic pages of the three graph-like types, mean degree against the truth:

    mean signed error   -0.69
    mean absolute error  0.71
    within 0.5 of truth  0.451

Degree is biased low by about two thirds of an edge per node, for the reason 4.1.1 already
established: connectors that cross or run together merge into one strand. That bias is
consistent, and a consistently low degree still orders the types correctly.

**Self-loop detection, on the other hand, does not work, and the failure is not a near miss.**
Detected loops per page against true loops per page:

    type            true mean   detected mean   true any   detected any
    state_machine     0.704         0.118        0.704        0.096
    flowchart         0.000         0.073        0.000        0.064
    er_diagram        0.000         0.734        0.000        0.496
    circuit           0.000         2.079        0.000        0.712
    wireframe         0.000         0.663        0.000        0.300

The one type that has self-loops is the type with the **fewest** detected. A circuit, which has
none at all, averages 2.08. The feature is not measuring loops; it is measuring **strands that
reach exactly one shape** - and on a circuit or a wireframe almost every strand does, because
the symbols and widgets those pages are made of are not closed regions in the first place
(4.1.1 measured circuit node count at -3.66). Meanwhile a real self-loop on a state machine is
a small arc drawn against its own circle, which fuses with that circle's ink and never becomes a
separate strand at all.

So `conn_self_loops` is renamed by its measurement rather than its intention: it is a count of
dangling connector ends. That is a genuine and fairly strong signal - it separates circuits
(2.08) and ER diagrams (0.73) from flowcharts (0.07) - but **it is not the self-loop count, and
nothing downstream may read it as one.** Recovering a real self-loop needs the arc separated
from the circle it touches, which is Phase 9's detector and Phase 10's assembly, not a
connected-component count. The name is kept so the plan row and the feature agree; this
paragraph is the correction.

`conn_cycle_count` has a limitation of the same kind, and it is structural rather than a
threshold. **A cycle in a drawing encloses area**, and enclosed area is exactly what `context`
calls a shape - so a flowchart's loop-back arrow, together with the boxes it returns past, is
read as one large region, and the connectors that formed the cycle are inside it and gone. The
feature therefore reports 0 on the clearest possible drawn cycle, which a test pins so that
nobody reads a zero as evidence of a tree. Cycles are recoverable from an assembled graph, which
is 10.1, not from ink.

`conn_components` is the most reliable member and the least interesting: it is close to "how
many separate pieces of ink are there", which 3.2.1 could already answer.

    python -m src.features.connectivity --pages 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from src.features import structural as st
from src.features.context import PageContext

NAMES = ("conn_mean_degree", "conn_self_loops", "conn_cycle_count", "conn_components")

#: A shape's box is grown by this share of the page's long side before asking which strands
#: touch it. Large enough to bridge the gap a hand leaves between an arrow and a box; small
#: enough not to reach the next shape.
TOUCH_FRAC = 0.02

#: A strand returning to the shape it started from must be at least this long to be a loop
#: rather than a stub of a broken connector.
MIN_LOOP_FRAC = 0.05


def _labelled_strands(context: PageContext) -> tuple[np.ndarray, dict[int, float]]:
    """Connector components as a label image, with each one's diagonal in pixels."""
    if context.connector_mask is None or not context.connector_mask.any():
        return np.zeros(context.mask.shape, np.int32), {}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        context.connector_mask.astype(np.uint8), 8
    )
    floor = st.MIN_STRAND_FRAC * context.long_side
    sizes = {}
    for i in range(1, count):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        diagonal = float(np.hypot(w, h))
        if diagonal >= floor:
            sizes[i] = diagonal
    return labels, sizes


def graph(context: PageContext) -> dict:
    """Nodes, undirected edges and self-loops implied by which strands touch which shapes."""
    labels, sizes = _labelled_strands(context)
    pad = int(round(TOUCH_FRAC * context.long_side))
    height, width = context.mask.shape

    touches: dict[int, set[int]] = {label: set() for label in sizes}
    for index, region in enumerate(context.regions):
        x, y, w, h = region.bbox
        band = labels[
            max(0, y - pad) : min(height, y + h + pad), max(0, x - pad) : min(width, x + w + pad)
        ]
        for label in np.unique(band):
            if int(label) in touches:
                touches[int(label)].add(index)

    edges: set[tuple[int, int]] = set()
    loops = 0
    loop_floor = MIN_LOOP_FRAC * context.long_side
    for label, ends in touches.items():
        if len(ends) >= 2:
            ordered = sorted(ends)
            # One strand joining three shapes is a junction, not three separate edges: the
            # pairs it implies are recorded, and 10.1 is where a real routing is recovered.
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    edges.add((a, b))
        elif len(ends) == 1 and sizes[label] >= loop_floor:
            loops += 1

    return {"nodes": len(context.regions), "edges": sorted(edges), "self_loops": loops}


def components(node_count: int, edges: list[tuple[int, int]]) -> int:
    """Connected components of the implied graph, isolated shapes included."""
    if node_count == 0:
        return 0
    parent = list(range(node_count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(node_count)})


def extract(context: PageContext) -> dict[str, float]:
    built = graph(context)
    nodes = built["nodes"]
    if nodes == 0:
        return dict.fromkeys(NAMES, float("nan"))

    edges = built["edges"]
    pieces = components(nodes, edges)
    return {
        "conn_mean_degree": 2.0 * len(edges) / nodes,
        "conn_self_loops": float(built["self_loops"]),
        # Cyclomatic number: how many independent cycles the graph holds. Never negative, and
        # 0 for any tree, which is what a well-drawn flowchart without a loop back is.
        "conn_cycle_count": float(max(0, len(edges) - nodes + pieces)),
        "conn_components": float(pieces),
    }


def _true_degree(item: dict) -> float:
    nodes = len(item["nodes"])
    return 2.0 * len(item["edges"]) / nodes if nodes else float("nan")


def _true_loops(item: dict) -> int:
    return sum(1 for edge in item["edges"] if edge.get("src") == edge.get("dst"))


def _row(item: dict) -> dict | None:
    from src.features import context as ctx
    from src.utils.config import ROOT

    path = ROOT / item["file"]
    if not path.is_file():
        return None
    features = extract(ctx.from_image(path, item["id"]))
    return {
        "diagram_type": item["diagram_type"],
        "true_degree": _true_degree(item),
        "true_loops": _true_loops(item),
        **features,
    }


def evaluate(limit: int = 600, n_jobs: int | None = None) -> dict:
    """Mean degree and self-loops against the synthetic ground-truth graphs."""
    from src.utils.parallel import pmap

    rows = [
        r for r in pmap(_row, st._synthetic_pages(limit), n_jobs=n_jobs, desc="connectivity") if r
    ]
    if not rows:
        return {"pages": 0}

    graph_like = [
        r for r in rows if r["diagram_type"] in st.GRAPH_LIKE and np.isfinite(r["conn_mean_degree"])
    ]
    error = np.array([r["conn_mean_degree"] - r["true_degree"] for r in graph_like], float)
    types = sorted({r["diagram_type"] for r in rows})
    return {
        "pages": len(rows),
        "degree": {
            "pages": len(graph_like),
            "mean_signed_error": round(float(np.mean(error)), 2),
            "mean_absolute_error": round(float(np.mean(np.abs(error))), 2),
            "within_half": round(float(np.mean(np.abs(error) <= 0.5)), 3),
        },
        "self_loops_by_type": {
            kind: {
                "true_mean": round(
                    float(np.mean([r["true_loops"] for r in rows if r["diagram_type"] == kind])), 3
                ),
                "detected_mean": round(
                    float(
                        np.nanmean(
                            [r["conn_self_loops"] for r in rows if r["diagram_type"] == kind]
                        )
                    ),
                    3,
                ),
                "true_any": round(
                    float(
                        np.mean([r["true_loops"] > 0 for r in rows if r["diagram_type"] == kind])
                    ),
                    3,
                ),
                "detected_any": round(
                    float(
                        np.nanmean(
                            [r["conn_self_loops"] > 0 for r in rows if r["diagram_type"] == kind]
                        )
                    ),
                    3,
                ),
            }
            for kind in types
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker count")
    ap.add_argument("image", nargs="?", type=Path)
    args = ap.parse_args(argv)

    if args.image:
        from src.features import context as ctx

        print(json.dumps(extract(ctx.from_image(args.image)), indent=2))
        return 0

    result = evaluate(args.pages, args.jobs)
    if not result["pages"]:
        print("no synthetic corpus; run the 1.3.7 generator first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

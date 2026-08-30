"""Phase 4.1.1 - structural counts: how many nodes, edges, arrowheads and labels.

The four cheapest numbers on a page, and the ones every ratio in 4.1.2 is built from:

    node_count        closed drawn shapes big enough to be a node (a `context.Region`)
    edge_count        strands of connector ink joining them
    arrowhead_count   3.2.6's detector, used with its measured precision in mind
    text_block_count  groups of writing on the text layer

The claim being tested is that flowcharts are arrow-dense and wireframes are not.

## What these counts are actually worth

Counts taken from an image are not counts taken from a graph, and the gap is measurable, so it
is measured rather than assumed. Against the ground-truth graphs of the synthetic corpus from
1.3.7, over 375 pages of the three types whose drawing *is* their graph:

    node count     exact on 42.4% of pages, within one on 60.0%, mean signed error -0.50
    edge count     exact on 24.5% of pages, within one on 43.7%, mean signed error -1.28

The headline hides the only thing worth knowing, which is that neither number is one number:

    type            nodes exact   within one      edges exact   within one
    flowchart          0.896        0.968            0.592        0.840
    state_machine      0.232        0.584            0.120        0.376
    er_diagram         0.144        0.248            0.024        0.096

**On flowcharts both counts are usable**, and flowcharts are the type this pipeline is built
around. The other two fail for reasons specific to their notation rather than for want of
tuning: a state machine draws an accepting state as two concentric circles, and both circles
are closed outlines of a plausible size, so every accept state counts twice (+0.85 nodes); an ER
diagram hangs attribute ovals off its entities, and those ovals overlap the entity box often
enough to fuse with it, so both are lost (-2.14 nodes, -4.38 edges). One is a double count and
the other a merge, and they push the error in opposite directions - no threshold fixes both.

Two types are excluded from the headline entirely, because for them the comparison is not
meaningful rather than merely hard. A wireframe's ground-truth graph has no edges at all while
its drawing is full of lines - buttons, fields, rules - so the counted number is right and the
reference number is zero. A circuit's components are symbols, not closed boxes, so `node_count`
sees 3.66 fewer nodes than the graph has, and would see none of them however the area floor
were set. Both are still *features* - a page with 17 strands and no closed shapes is extremely
informative about diagram type - they are simply not measurable against a graph.

## The one thing that mattered more than any threshold

`context.connector_ink` pads each shape by a stroke width before subtracting it. Without that
padding, a shape welded to its connectors survives only as its *inner* edge, so a closed ring
of its own outline stays in the connector layer, touching every arrow that arrives - and the
whole page becomes a single strand. Measured on these 375 pages, that one change moved edge
count from **+5.48 mean signed error, 4.3% exact** to **-1.28 and 24.5% exact**. The remaining
undercount is genuine merging: two arrows that cross, or run alongside each other into the same
box, are one strand, exactly as 3.2.1 predicted for components.

`MIN_STRAND_FRAC` was then chosen from a sweep on 187 pages rather than by eye:

    floor    exact   within one   mean signed   mean absolute
    0.000    0.086      0.332        -0.10          2.59
    0.005    0.166      0.390        -0.93          2.70
    0.010    0.235      0.428        -1.45          2.77
    0.020    0.257      0.439        -2.09          2.93
    0.030    0.251      0.390        -2.86          3.26
    0.050    0.011      0.155        -4.10          4.13
    0.080    0.016      0.139        -4.35          4.35

No floor at all is the unbiased choice and has the lowest absolute error, and it is not the one
taken: on a clean render every speck is a strand, and 3.1.6's denoising is tuned for photographs
rather than for this. 0.01 nearly triples the exact-match rate for one extra half-edge of bias,
and past 0.02 the cost is real connectors between adjacent boxes. The bias is recorded here so
4.1.2's ratios can be read knowing the denominator runs about one edge light.

`arrowhead_count` is included because 4.1.2's ratios want it, but 3.2.6 measured that detector
at **precision 0.11, recall 0.22** on 265 real arrows, so on a real page this feature largely
counts skeleton junctions. It is kept unmodified as an input for 4.2.6 to rank: if a feature
this noisy still carries mutual information about diagram type, that is worth knowing, and if
it does not, the ranking is the honest place to drop it. Phase 9.1's learned detector is what
replaces it.

    python -m src.features.structural --pages 600
    python -m src.features.structural --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from src.features.context import PageContext

NAMES = ("node_count", "edge_count", "arrowhead_count", "text_block_count")

#: A strand of connector ink shorter than this share of the page's long side is a fragment left
#: by binarization, an arrow barb, or a label the text layer did not claim - not an edge. Chosen
#: from the sweep in the docstring as the value at which the count is unbiased, not the value
#: with the smallest error.
MIN_STRAND_FRAC = 0.01

#: The sweep reported in the docstring.
STRAND_SWEEP = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)


def strands(context: PageContext, min_frac: float = MIN_STRAND_FRAC) -> list[dict]:
    """Connected pieces of connector ink long enough to join two things."""
    if context.connector_mask is None or not context.connector_mask.any():
        return []
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        context.connector_mask.astype(np.uint8), 8
    )
    floor = min_frac * context.long_side
    out = []
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if float(np.hypot(w, h)) < floor:
            continue
        out.append(
            {
                "bbox": [x, y, w, h],
                "area": int(stats[i, cv2.CC_STAT_AREA]),
                "centroid": [float(centroids[i][0]), float(centroids[i][1])],
            }
        )
    return out


def text_blocks(context: PageContext) -> list[list[int]]:
    """Writing grouped into words and lines, by 3.2.7's own grouping rule."""
    from src.preprocess.primitives import text as tx

    if not context.text_boxes:
        return []
    return tx.group_into_lines(list(context.text_boxes))


def extract(context: PageContext) -> dict[str, float]:
    return {
        "node_count": float(len(context.regions)),
        "edge_count": float(len(strands(context))),
        "arrowhead_count": float(len(context.arrowheads)),
        "text_block_count": float(len(text_blocks(context))),
    }


def _synthetic_pages(limit: int) -> list[dict]:
    from src.utils.config import ROOT

    index = ROOT / "data" / "processed" / "synthetic" / "index.json"
    if not index.is_file():
        return []
    items = json.loads(index.read_text(encoding="utf-8"))
    # Every nth item, so all five diagram types are represented rather than the first type only.
    step = max(1, len(items) // limit)
    return items[::step][:limit]


def _score_one(item: dict) -> dict | None:
    from src.features import context as ctx
    from src.utils.config import ROOT

    path = ROOT / item["file"]
    if not path.is_file():
        return None
    counted = extract(ctx.from_image(path, item["id"]))
    return {
        "id": item["id"],
        "diagram_type": item["diagram_type"],
        "true_nodes": len(item["nodes"]),
        "true_edges": len(item["edges"]),
        "nodes": int(counted["node_count"]),
        "edges": int(counted["edge_count"]),
    }


#: The types whose drawing *is* their graph, and where counting nodes in the image can be
#: compared with counting them in the ground truth without an apples-to-oranges problem. A
#: wireframe's graph has no edges at all while its drawing is full of lines, and a circuit's
#: components are symbols rather than closed boxes - see the module docstring.
GRAPH_LIKE = ("flowchart", "state_machine", "er_diagram")


def _agreement(rows: list[dict], kind: str) -> dict:
    error = np.array([r[kind] - r[f"true_{kind}"] for r in rows], float)
    return {
        "pages": len(rows),
        "exact": round(float(np.mean(error == 0)), 3),
        "within_one": round(float(np.mean(np.abs(error) <= 1)), 3),
        "mean_signed_error": round(float(np.mean(error)), 2),
        "mean_absolute_error": round(float(np.mean(np.abs(error))), 2),
    }


def evaluate(limit: int = 200, n_jobs: int | None = None) -> dict:
    """Counted nodes and edges against the ground-truth graphs of the synthetic corpus."""
    from src.utils.parallel import pmap

    items = _synthetic_pages(limit)
    rows = [r for r in pmap(_score_one, items, n_jobs=n_jobs, desc="structural counts") if r]
    if not rows:
        return {"pages": 0}

    graph_like = [r for r in rows if r["diagram_type"] in GRAPH_LIKE]
    per_type = {}
    for kind in sorted({r["diagram_type"] for r in rows}):
        subset = [r for r in rows if r["diagram_type"] == kind]
        per_type[kind] = {
            "nodes": _agreement(subset, "nodes"),
            "edges": _agreement(subset, "edges"),
        }
    return {
        "pages": len(rows),
        "nodes": _agreement(graph_like, "nodes"),
        "edges": _agreement(graph_like, "edges"),
        "per_type": per_type,
    }


def _sweep_one(item: dict) -> tuple[int, list[int]] | None:
    from src.features import context as ctx
    from src.utils.config import ROOT

    path = ROOT / item["file"]
    if not path.is_file():
        return None
    context = ctx.from_image(path, item["id"])
    return len(item["edges"]), [len(strands(context, floor)) for floor in STRAND_SWEEP]


def sweep_strand_floor(limit: int = 300, n_jobs: int | None = None) -> list[dict]:
    """Edge-count agreement at each candidate floor. This is where MIN_STRAND_FRAC comes from."""
    from src.utils.parallel import pmap

    items = [i for i in _synthetic_pages(limit) if i["diagram_type"] in GRAPH_LIKE]
    rows = [r for r in pmap(_sweep_one, items, n_jobs=n_jobs, desc="strand floor sweep") if r]
    out = []
    for position, floor in enumerate(STRAND_SWEEP):
        error = np.array([counted[position] - truth for truth, counted in rows], float)
        out.append(
            {
                "floor": floor,
                "pages": len(rows),
                "exact": round(float(np.mean(error == 0)), 3),
                "within_one": round(float(np.mean(np.abs(error) <= 1)), 3),
                "mean_signed_error": round(float(np.mean(error)), 2),
                "mean_absolute_error": round(float(np.mean(np.abs(error))), 2),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker count")
    ap.add_argument("--sweep", action="store_true", help="sweep the strand-length floor")
    ap.add_argument("image", nargs="?", type=Path, help="score one image instead")
    args = ap.parse_args(argv)

    if args.image:
        from src.features import context as ctx

        print(json.dumps(extract(ctx.from_image(args.image)), indent=2))
        return 0

    if args.sweep:
        print(json.dumps(sweep_strand_floor(args.pages, args.jobs), indent=2))
        return 0

    result = evaluate(args.pages, args.jobs)
    if not result["pages"]:
        print("no synthetic corpus; run the 1.3.7 generator first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

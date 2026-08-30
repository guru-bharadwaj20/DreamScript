"""Phase 4.1.2 - ratios: the counts from 4.1.1 made independent of how much was drawn.

    line_to_curve    straight outline length over curved outline length
    arrows_per_node  arrowhead candidates per drawn shape
    text_per_node    text blocks per drawn shape
    edges_per_node   connector strands per drawn shape

A count says how big the drawing is; a ratio says what kind of drawing it is. Two flowcharts of
four and forty boxes have wildly different `node_count` and nearly the same `edges_per_node`,
which is the property a classifier can use. Everything here is one of 4.1.1's counts divided by
another, so nothing new is extracted - only made scale-free.

## Dividing by zero is a modelling decision, not an accident

Every ratio here has a denominator that is legitimately zero on some real page: a photograph of
a circuit has no closed shapes, so `node_count` is 0 and all four of these are undefined. The
choice made is to **return `nan` and let 4.2.3 impute it**, not to return 0.0.

The distinction matters more than it looks. 0.0 is a claim - "this page has no edges per node" -
and it is indistinguishable from a page that genuinely has isolated boxes, which is a real and
different thing. `nan` says "this page cannot answer the question", 4.2.3's missingness
indicator preserves *that* as its own feature, and a page with no shapes at all turns out to be
strongly informative about diagram type. Squashing it to zero throws that away and quietly
merges two populations.

Measured over 600 synthetic pages, all four are undefined on the same **7.8%** of pages - they
share the `node_count == 0` denominator, and a page with no closed shape at all is mostly a
circuit whose symbols are open squiggles.

## Which of these actually separate the types

Per-type medians over the same 600 pages, taken over the pages where the ratio is defined:

    type            line_to_curve  arrows_per_node  text_per_node  edges_per_node
    flowchart           1.43            0.000            0.00           0.83
    state_machine       0.20            0.000            0.40           0.50
    er_diagram          0.88            0.125            0.89           0.48
    circuit             0.77            0.000            1.00           5.00
    wireframe           2.50            0.000            0.50           8.00

Two of the four earn their place immediately. **`edges_per_node` separates the drawn-graph types
from the rest by an order of magnitude** - 0.48 to 0.83 for flowcharts, state machines and ER
diagrams against 5.0 for circuits and 8.0 for wireframes - which is 4.1.1's "a wireframe's
drawing is full of lines that its graph does not have" turned into a scale-free number.
**`line_to_curve` orders the types by how round their nodes are**: state machines at 0.20 (a
state is a circle) through flowcharts at 1.43 to wireframes at 2.50 (everything is a rectangle).
That is exactly the box-versus-circle signal 3.2.5 was built to give, and 3.2.3 showed vertex
counting could not.

`arrows_per_node` has a median of **0.000 for four of the five types**: half the pages have no
detected arrowhead at all, which is 3.2.6's recall-0.22 detector showing up as a nearly dead
feature. It is kept unchanged for 4.2.6 to rank rather than dropped here on suspicion - the
ranking is the honest place for that - and it is expected to rank last.

    python -m src.features.ratios --pages 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.features import structural as st
from src.features.context import PageContext

NAMES = ("line_to_curve", "arrows_per_node", "text_per_node", "edges_per_node")

#: A straightness this close to 1 leaves no curved length to divide by; the ratio is capped
#: rather than allowed to run to infinity, because a feature table cannot hold `inf` and the
#: scalers in 4.2.4 would be destroyed by one page that did.
MAX_RATIO = 100.0


def ratio(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, or `nan` when the question does not apply. Never `inf`."""
    if denominator <= 0:
        return float("nan")
    return float(min(numerator / denominator, MAX_RATIO))


def straight_share(context: PageContext) -> float:
    """Perimeter-weighted share of outline that is straight, over every drawn shape.

    Weighted by perimeter rather than averaged per shape: a page with one large box and six
    tiny circles is mostly straight line, and an unweighted mean would call it curved.
    """
    if not context.regions:
        return float("nan")
    weights = np.array([r.perimeter for r in context.regions], float)
    values = np.array([r.straightness for r in context.regions], float)
    if weights.sum() <= 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def extract(context: PageContext) -> dict[str, float]:
    counts = st.extract(context)
    nodes = counts["node_count"]
    straight = straight_share(context)
    return {
        "line_to_curve": ratio(straight, 1.0 - straight) if np.isfinite(straight) else float("nan"),
        "arrows_per_node": ratio(counts["arrowhead_count"], nodes),
        "text_per_node": ratio(counts["text_block_count"], nodes),
        "edges_per_node": ratio(counts["edge_count"], nodes),
    }


def _row(item: dict) -> dict | None:
    from src.features import context as ctx
    from src.utils.config import ROOT

    path = ROOT / item["file"]
    if not path.is_file():
        return None
    row = {"diagram_type": item["diagram_type"]}
    row.update(extract(ctx.from_image(path, item["id"])))
    return row


def evaluate(limit: int = 600, n_jobs: int | None = None) -> dict:
    """Per-type medians and how often each ratio is undefined."""
    from src.utils.parallel import pmap

    items = st._synthetic_pages(limit)
    rows = [r for r in pmap(_row, items, n_jobs=n_jobs, desc="ratios") if r]
    if not rows:
        return {"pages": 0}

    def median_of(subset: list[dict], name: str) -> float:
        values = np.array([r[name] for r in subset], float)
        values = values[np.isfinite(values)]
        return round(float(np.median(values)), 3) if len(values) else float("nan")

    types = sorted({r["diagram_type"] for r in rows})
    return {
        "pages": len(rows),
        "undefined_share": {
            name: round(float(np.mean([not np.isfinite(r[name]) for r in rows])), 3)
            for name in NAMES
        },
        "median_by_type": {
            kind: {
                name: median_of([r for r in rows if r["diagram_type"] == kind], name)
                for name in NAMES
            }
            for kind in types
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker count")
    ap.add_argument("image", nargs="?", type=Path, help="score one image instead")
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

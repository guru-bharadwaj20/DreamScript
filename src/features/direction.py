"""Phase 4.1.8 - directionality: which way the connectors run.

    dir_flow_axis          +1 all vertical, -1 all horizontal, 0 balanced
    dir_angle_entropy      how spread the connector angles are, 0 (one direction) to 1 (all)
    dir_axis_aligned       share of connector length within 12 degrees of an axis

A flowchart flows downward, a wireframe's rules run across, and a circuit goes both ways at
once. All three read **connector** segments only - 3.2.4's segments whose midpoint is outside
every shape - so the box outlines, which are axis-aligned on every page whatever the diagram
type, cannot drown out the signal. Measured with the outlines left in, `dir_axis_aligned` is
0.93 for every type on every page, which is a measurement of rectangles and not of flow.

Angles are length-weighted. An unweighted histogram counts a 12-pixel fragment the same as the
200-pixel arrow it broke off, and after binarization there are many more of the former.

## Direction is modulo 180, and that is not a limitation of the code

A line has no arrowhead. 3.2.6 established that the arrowhead detector runs at precision 0.11,
so there is nothing available to tell "down" from "up" - only "vertical" from "horizontal".
`dir_flow_axis` is therefore an *axis*, not a *flow*: it says a flowchart is laid out vertically
and cannot say it runs downward. plan.md names this feature "dominant flow axis" and the axis is
what is delivered; the direction needs 9.1, and Phase 10 is where it becomes an ordering.

## What it measures, per type

Medians over 600 synthetic pages; all three are undefined on 1.2% of them, the pages with no
connector segment left after the shapes are excluded:

    type            flow_axis   angle_entropy   axis_aligned
    flowchart          +0.169        0.383          0.953
    state_machine      -0.511        0.351          0.394
    er_diagram         -0.107        0.588          0.770
    circuit            -0.131        0.366          0.922
    wireframe          -0.045        0.254          0.999

`dir_flow_axis` is the only positive number in the column for flowcharts, which is the plan's
claim surviving in the weak form the measurement can support: **flowcharts are the one type laid
out vertically**, at +0.169 against -0.045 to -0.511 for everything else. It is a small number
because an orthogonally routed arrow spends as much length going across as going down; what
separates the type is the sign, not the size. State machines are strongly horizontal (-0.511),
their transition arcs sweeping sideways between circles.

This column only exists because the shapes are excluded properly. Before `connector_segments`
padded the region boxes by a stroke width, the outer edge of every drawn rectangle counted as a
connector, and the same table read -0.100 for flowcharts - **the wrong sign**, and the plan's
claim would have been recorded as refuted by what was really a measurement of box outlines.

The other two split the types the first one cannot:

* **`dir_axis_aligned` separates state machines (0.394) from wireframes (0.999)**, with
  flowcharts and circuits above 0.92: curved transitions against ruled boxes. This is the
  cleanest ordering in the family.
* **`dir_angle_entropy` isolates ER diagrams at 0.588** against 0.254-0.383 for everything else.
  An ER diagram fans attribute lines out from an entity at every angle available; a wireframe
  uses exactly two, and scores lowest.

    python -m src.features.direction --pages 600
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

NAMES = ("dir_flow_axis", "dir_angle_entropy", "dir_axis_aligned")

#: Angle bins over 0-180 degrees. Ten degrees a bin, centred on the axes the way 3.2.4's
#: histogram is, so a horizontal line falls in the middle of a bin rather than on its edge.
BINS = 18

#: Within this many degrees of an axis counts as axis-aligned. 3.2.4's own tolerance.
AXIS_TOL = 12.0


def _weighted_angles(context: PageContext) -> tuple[np.ndarray, np.ndarray]:
    """(angles in degrees 0-180, lengths) for every connector segment."""
    segments = context.connector_segments
    if not segments:
        return np.zeros(0), np.zeros(0)
    angles = np.array([s["angle"] % 180.0 for s in segments], float)
    lengths = np.array([s["length"] for s in segments], float)
    return angles, lengths


def entropy(angles: np.ndarray, lengths: np.ndarray, bins: int = BINS) -> float:
    """Shannon entropy of the length-weighted angle histogram, scaled to 0-1."""
    if not len(angles):
        return float("nan")
    # Bins centred on 0 and 90: the first bin straddles the wrap-around, as in 3.2.4.
    half = 180.0 / bins / 2.0
    shifted = (angles + half) % 180.0
    histogram, _ = np.histogram(shifted, bins=bins, range=(0.0, 180.0), weights=lengths)
    total = histogram.sum()
    if total <= 0:
        return float("nan")
    share = histogram / total
    share = share[share > 0]
    # max(0, ...) only to keep a single-bin page at 0.0 rather than -0.0.
    return float(max(0.0, -np.sum(share * np.log(share)) / np.log(bins)))


def extract(context: PageContext) -> dict[str, float]:
    angles, lengths = _weighted_angles(context)
    if not len(angles):
        return dict.fromkeys(NAMES, float("nan"))

    radians = np.radians(angles)
    vertical = float(np.sum(lengths * np.abs(np.sin(radians))))
    horizontal = float(np.sum(lengths * np.abs(np.cos(radians))))
    total = vertical + horizontal
    near_axis = np.minimum(angles, 180.0 - angles) <= AXIS_TOL
    near_axis |= np.abs(angles - 90.0) <= AXIS_TOL
    return {
        "dir_flow_axis": (vertical - horizontal) / total if total else float("nan"),
        "dir_angle_entropy": entropy(angles, lengths),
        "dir_axis_aligned": float(lengths[near_axis].sum() / lengths.sum()),
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
    from src.utils.parallel import pmap

    rows = [r for r in pmap(_row, st._synthetic_pages(limit), n_jobs=n_jobs, desc="direction") if r]
    if not rows:
        return {"pages": 0}
    types = sorted({r["diagram_type"] for r in rows})
    return {
        "pages": len(rows),
        "undefined_share": {
            name: round(float(np.mean([not np.isfinite(r[name]) for r in rows])), 3)
            for name in NAMES
        },
        "median_by_type": {
            kind: {
                name: round(
                    float(np.nanmedian([r[name] for r in rows if r["diagram_type"] == kind])), 3
                )
                for name in NAMES
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

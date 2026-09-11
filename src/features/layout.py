"""Phase 4.1.4 - layout geometry: where the shapes sit, not what they are.

    layout_node_density     shapes per unit of page, page size divided out
    layout_nn_distance      mean distance to the nearest other shape, in page widths
    layout_grid_score       share of shapes sharing a row or column with another
    layout_row_regularity   how evenly spaced the rows are
    layout_col_regularity   how evenly spaced the columns are

The claim is that a wireframe is a grid and a flowchart is a tree: both may hold six boxes, but
in a wireframe they line up. Nothing here looks at a single shape's geometry - only at the
arrangement - so this family is intact even where 4.1.3 fails, which matters given that 4.1.3
does not work on photographs.

## Everything is divided by the page's long side

A feature that changes when the photograph is taken from further away is not a feature. Every
distance here is a fraction of the page's long side and every count is per unit of normalised
area, so the same drawing at 900px and at 4,000px gives the same numbers - which the tests pin
directly rather than by inspection.

## Alignment needs a tolerance, and the tolerance is what it measures

Two shapes are in the same row when their centres agree vertically to within `ALIGN_TOL` of the
page's long side. At 0.02 that is 19px on a 960px page, about a third of a drawn box's height.
The whole family stands on that number, so it is swept rather than assumed - median grid score
per type over 600 synthetic pages:

    tolerance   flowchart   state_machine   er_diagram   circuit   wireframe
      0.005       1.000        0.667          0.449       0.000      0.000
      0.010       1.000        0.833          0.750       0.000      0.000
      0.020       1.000        1.000          1.000       0.000      0.000
      0.040       1.000        1.000          1.000       0.000      0.000
      0.080       1.000        1.000          1.000       0.000      0.000

The feature is **completely insensitive to the tolerance** across a sixteen-fold range, which is
the useful thing the sweep says: the two populations are not near the boundary, they are at 1
and 0. 0.02 is kept because it is the middle of the flat region and because 3.1.7 leaves up to a
degree of residual skew on a real page, which moves a box on the far side of the sheet further
than 0.01 of the page.

## What it measures - which is not what the plan assumed

Medians over 600 synthetic pages:

    type            density   nn_distance   grid_score   row_reg   col_reg
    flowchart          5.33      0.186          1.000      0.825     0.000
    state_machine      6.43      0.179          1.000      0.350     0.738
    er_diagram        10.29      0.118          1.000      0.493     0.622
    circuit            1.32      0.452          0.000      0.965     0.660
    wireframe          2.47      0.142          0.000      0.730     0.293

contributing.md's rationale for this family was "wireframes are grid-like", and **the measurement says
the opposite**: wireframes score 0.000 and flowcharts 1.000. Both halves of that have a cause
worth recording.

A flowchart is a *vertical chain*, and a chain is a column - every box shares its column with
the box above it, so `layout_grid_score` saturates at 1.0. It was built to find grids and what
it actually finds is *any* regular arrangement, which a chain is.

Wireframes score zero for a different and less flattering reason: 4.1.1 measured node count on
wireframes at -2.38 against the truth, because a wireframe's panels nest and fuse rather than
sit apart, so the median wireframe here contributes about two regions. Two shapes rarely share a
row, and with fewer than three rows `_regularity` is undefined. **The feature is not measuring
the wireframe's layout; it is measuring how little of the wireframe survived region
extraction.** That is a real signal for a classifier - the density column separates wireframes
and circuits from the rest as clearly as anything else here - but it is not the signal the plan
named, and Phase 5 should not be told it is.

`layout_nn_distance` and `layout_node_density` are the honest members: circuits sit at 0.452 and
1.32 against 0.118-0.186 and 5.33-10.29 for the drawn-graph types, and that ordering is stable.
Row and column regularity are the weakest, are undefined on any page with fewer than three
rows, and are left for 4.2.5 to prune on the evidence rather than dropped here on suspicion.

    python -m src.features.layout --pages 600
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

NAMES = (
    "layout_node_density",
    "layout_nn_distance",
    "layout_grid_score",
    "layout_row_regularity",
    "layout_col_regularity",
)

#: Two shapes are in the same row (or column) when their centres agree to this share of the
#: page's long side. See the docstring: chosen wider than the sweep's best, to survive skew.
ALIGN_TOL = 0.02


def centres(context: PageContext) -> np.ndarray:
    """Shape centres in page-long-side units, so nothing here depends on resolution."""
    if not context.regions:
        return np.zeros((0, 2))
    return np.array([r.centre for r in context.regions], float) / context.long_side


def nearest_neighbour_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(0)
    difference = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(difference, axis=2)
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1)


def _groups(values: np.ndarray, tolerance: float) -> list[list[int]]:
    """Indices grouped by a one-dimensional coordinate, single-linkage within `tolerance`."""
    order = np.argsort(values)
    groups: list[list[int]] = []
    for index in order:
        if groups and abs(values[index] - values[groups[-1][-1]]) <= tolerance:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return groups


def _regularity(positions: list[float]) -> float:
    """1 for evenly spaced lines, 0 for erratic ones. Undefined for fewer than three."""
    if len(positions) < 3:
        return float("nan")
    gaps = np.diff(sorted(positions))
    if gaps.mean() <= 0:
        return float("nan")
    # 1 - coefficient of variation, floored at 0: a spacing that varies as much as it measures
    # is not "negatively regular", it is simply irregular.
    return float(max(0.0, 1.0 - gaps.std() / gaps.mean()))


def extract(context: PageContext, tolerance: float = ALIGN_TOL) -> dict[str, float]:
    points = centres(context)
    if len(points) == 0:
        return dict.fromkeys(NAMES, float("nan"))

    # Normalised page area: 1.0 x (short side / long side), so density is per unit of page.
    area = (context.height * context.width) / (context.long_side**2)
    rows = _groups(points[:, 1], tolerance)
    columns = _groups(points[:, 0], tolerance)
    shared = sum(len(group) for group in rows + columns if len(group) > 1)

    neighbours = nearest_neighbour_distances(points)
    return {
        "layout_node_density": float(len(points) / area),
        "layout_nn_distance": float(neighbours.mean()) if len(neighbours) else float("nan"),
        "layout_grid_score": float(min(shared / len(points), 1.0)),
        "layout_row_regularity": _regularity([float(np.mean(points[group, 1])) for group in rows]),
        "layout_col_regularity": _regularity(
            [float(np.mean(points[group, 0])) for group in columns]
        ),
    }


def _row(item: dict) -> dict | None:
    from src.features import context as ctx
    from src.utils.config import ROOT

    path = ROOT / item["file"]
    if not path.is_file():
        return None
    context = ctx.from_image(path, item["id"])
    row = {"diagram_type": item["diagram_type"]}
    row.update(extract(context))
    row["sweep"] = [extract(context, t)["layout_grid_score"] for t in SWEEP]
    return row


#: Alignment tolerances reported in the docstring.
SWEEP = (0.005, 0.01, 0.02, 0.04, 0.08)


def evaluate(limit: int = 600, n_jobs: int | None = None) -> dict:
    """Per-type medians, and the tolerance sweep that `ALIGN_TOL` was chosen against."""
    from src.utils.parallel import pmap

    rows = [r for r in pmap(_row, st._synthetic_pages(limit), n_jobs=n_jobs, desc="layout") if r]
    if not rows:
        return {"pages": 0}

    def median(subset: list[dict], key: str) -> float:
        values = np.array([r[key] for r in subset], float)
        values = values[np.isfinite(values)]
        return round(float(np.median(values)), 3) if len(values) else float("nan")

    types = sorted({r["diagram_type"] for r in rows})
    sweep = []
    for position, tolerance in enumerate(SWEEP):
        medians = {
            kind: round(
                float(
                    np.nanmedian([r["sweep"][position] for r in rows if r["diagram_type"] == kind])
                ),
                3,
            )
            for kind in types
        }
        sweep.append({"tolerance": tolerance, "grid_score_median": medians})
    return {
        "pages": len(rows),
        "median_by_type": {
            kind: {
                name: median([r for r in rows if r["diagram_type"] == kind], name) for name in NAMES
            }
            for kind in types
        },
        "alignment_sweep": sweep,
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

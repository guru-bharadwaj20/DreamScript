"""Phase 4.1.9 - containment: how deeply the shapes are nested inside one another.

    contain_nested_count   shapes drawn inside another shape
    contain_nested_share   that count as a fraction of all shapes
    contain_max_depth      deepest nesting on the page

Wireframes nest - a panel holds a row, a row holds a button - and flowcharts do not. This is
the shortest family in Phase 4 because the work was already done: `context` recomputes depth and
parent over the surviving regions rather than inheriting the contour tree, precisely so that a
number read here means "drawn inside" and not "an artefact of the fusion shell that was
dropped". All three features are a count over `Region.depth`.

## Two things that are deliberately not counted

**A shape inside a shape whose outline broke is not nested.** If a panel's border is interrupted
so that it never closes, it is not a region, and everything drawn in it is at depth 0. The
feature undercounts on damaged pages and there is no repair here; 3.3.3's failure gallery has
the pages where that happens.

**A label inside a box is not nesting.** Text is on the other layer by 3.2.8 and never becomes a
region, so a flowchart with writing in every box still scores zero. That is the intended
reading: 4.1.6 measures labels-in-shapes, this measures shapes-in-shapes, and keeping them
apart is why the two-layer split exists.

## What it measures, per type

Medians over 600 synthetic pages, with the mean alongside because the median of a count that is
usually zero hides everything:

    type            nested (median / mean)   share (mean)   max depth (mean)
    flowchart            0.0 / 0.00             0.000           0.000
    circuit              0.0 / 0.00             0.000           0.000
    er_diagram           0.0 / 0.25             0.028           0.121
    state_machine        1.0 / 0.90             0.156           0.546
    wireframe            0.0 / 0.95             0.164           0.602

The claim in plan.md is "wireframes nest; flowcharts don't", and half of it holds exactly:
**flowcharts and circuits never nest, on any of the 600 pages, on any of the three features.**
A feature that is identically zero for two of five classes is a strong negative signal, which is
worth as much to a classifier as a strong positive one.

The other half is wrong, and the type that nests most is not the wireframe. **State machines
have the highest median and the highest share**, because an accepting state is drawn as two
concentric circles - a shape inside a shape by any definition. 4.1.1 met the same drawing and
called it an error, a double count of nodes; here the identical geometry is signal. Both
readings are right - one symbol, two outlines - and it is a clean illustration of why 2.1.4's
vocabulary keeps `double-circle` as a shape of its own instead of folding it into `circle`.

Wireframes do nest, at a mean of 0.95, but their median is 0: on half the pages the panel border
never closes as a region, or the panel and its contents fuse. 4.1.1 measured the same weakness
as a node count of -2.38 against the truth. So the feature confirms the plan's claim only on the
pages where the wireframe survived region extraction, and `contain_nested_share` cannot separate
a wireframe (0.164) from a state machine (0.156) at all.

    python -m src.features.containment --pages 600
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

NAMES = ("contain_nested_count", "contain_nested_share", "contain_max_depth")


def extract(context: PageContext) -> dict[str, float]:
    if not context.regions:
        return dict.fromkeys(NAMES, float("nan"))
    depths = np.array([region.depth for region in context.regions], int)
    return {
        "contain_nested_count": float(np.count_nonzero(depths > 0)),
        "contain_nested_share": float(np.mean(depths > 0)),
        "contain_max_depth": float(depths.max()),
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

    rows = [
        r for r in pmap(_row, st._synthetic_pages(limit), n_jobs=n_jobs, desc="containment") if r
    ]
    if not rows:
        return {"pages": 0}
    types = sorted({r["diagram_type"] for r in rows})
    return {
        "pages": len(rows),
        "by_type": {
            kind: {
                name: {
                    "median": round(
                        float(np.nanmedian([r[name] for r in rows if r["diagram_type"] == kind])),
                        3,
                    ),
                    "mean": round(
                        float(np.nanmean([r[name] for r in rows if r["diagram_type"] == kind])), 3
                    ),
                }
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

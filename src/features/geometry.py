"""Phase 4.1.5 - global geometry: three cheap numbers about the page as a whole.

    global_aspect        image width over height
    global_ink_coverage  share of the page that is ink
    global_bbox_fill     share of the page inside the drawing's own bounding box

These cost nothing - no contours, no regions, no skeleton - and they are the only features in
Phase 4 that still exist when every other family returns `nan` because nothing was detected.
That makes them the floor under the feature vector rather than its most interesting part.

## `global_aspect` is a camera feature, and it will leak

Aspect ratio is not a property of the diagram. It is a property of whoever held the phone, and
in this corpus that correlates with the source dataset, which correlates almost perfectly with
diagram type. Measured over 60 images sampled from each source that is present on disk:

    source         n    median aspect   range
    sketch2code    60       0.822       0.385 - 1.780
    chaos          60       1.264       0.406 - 3.553
    hdbpmn         60       1.361       0.689 - 4.096

Every sketch2code image is a wireframe and every hdBPMN image is a flowchart, so a classifier
handed `global_aspect` can score above chance by learning **portrait means wireframe** - a fact
about two data collections, not about diagrams. 1.2's scribe-disjoint splits do not protect
against this at all: the split is disjoint by writer, and both sides of it contain the same
sources.

The feature is kept, and it is flagged here and in the model card rather than dropped, for two
reasons. It is genuinely predictive on the corpus as it exists, so removing it silently would
make the Phase 5 numbers look worse without making them more honest; and 14's ablations are the
place to measure what it is worth - the difference between accuracy with and without it is a
direct measurement of how much of the classifier is reading the camera. **`global_aspect` must
be in the first ablation Phase 14 runs.**

## The other two

`global_ink_coverage` is the share of the page that survived 3.1 as ink, which is a measure of
how much was drawn and, on a bad photograph, of how much was not removed - 3.3.3 found a page
where 84% of the "ink" was the desk. `global_bbox_fill` is the share of the page inside the
drawing's own bounding box: a sketch in one corner scores low, a diagram that fills the sheet
scores near 1. Both are honest page properties and neither is a camera artefact, because both
are ratios within the frame rather than a property of the frame.

Medians over 600 synthetic pages:

    type            aspect   ink_coverage   bbox_fill
    flowchart        1.067      0.0134        0.405
    state_machine    1.286      0.0150        0.415
    er_diagram       1.286      0.0204        0.298
    circuit          1.323      0.0168        0.569
    wireframe        0.809      0.0223        0.859

The synthetic corpus reproduces the leak rather than escaping it: 1.3.7's generator picks a page
shape per diagram type, so `global_aspect` is **0.809 for every wireframe and 1.067 for every
flowchart** here too. A model trained on this corpus can read the type off the page shape
without looking at the drawing, and it will score well doing it. This is the second independent
route to the same wrong answer, and it is why the ablation matters more than the flag.

`global_bbox_fill` is the useful one: **0.859 for wireframes against 0.298 for ER diagrams**, a
real statement about how the drawing occupies the sheet. Ink coverage separates the extremes by
less than a factor of two here, and it is expected to be worth more on photographs, where it
also picks up whatever the page detector failed to crop.

    python -m src.features.geometry --pages 600
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

NAMES = ("global_aspect", "global_ink_coverage", "global_bbox_fill")


def ink_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) around every ink pixel, or None on a blank page."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(columns):
        return None
    return (
        int(columns[0]),
        int(rows[0]),
        int(columns[-1] - columns[0] + 1),
        int(rows[-1] - rows[0] + 1),
    )


def extract(context: PageContext) -> dict[str, float]:
    box = ink_bbox(context.mask)
    return {
        "global_aspect": float(context.width / context.height) if context.height else float("nan"),
        "global_ink_coverage": float(context.mask.sum()) / context.page_area,
        # A blank page has no drawing, so the question "how much of the page does the drawing
        # cover" has no answer - `nan`, not 0.0, on the 4.1.2 rule.
        "global_bbox_fill": (box[2] * box[3]) / context.page_area if box else float("nan"),
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

    rows = [r for r in pmap(_row, st._synthetic_pages(limit), n_jobs=n_jobs, desc="geometry") if r]
    if not rows:
        return {"pages": 0}
    types = sorted({r["diagram_type"] for r in rows})
    return {
        "pages": len(rows),
        "median_by_type": {
            kind: {
                name: round(
                    float(np.nanmedian([r[name] for r in rows if r["diagram_type"] == kind])), 4
                )
                for name in NAMES
            }
            for kind in types
        },
    }


def aspect_by_source(per_source: int = 60, n_jobs: int | None = None) -> dict:
    """The leakage measurement in the docstring: aspect ratio grouped by data source.

    Reads image headers only - no decoding - so it costs a millisecond a file.
    """
    import pandas as pd
    from PIL import Image

    from src.utils.config import ROOT
    from src.utils.parallel import pmap

    manifest = ROOT / "data" / "processed" / "manifest.parquet"
    if not manifest.is_file():
        return {"images": 0}
    frame = pd.read_parquet(manifest)
    sample = pd.concat(
        [
            group.sample(min(per_source, len(group)), random_state=42)
            for _, group in frame.groupby("source")
        ]
    )

    def one(record: dict) -> tuple[str, float] | None:
        path = ROOT / record["path"]
        if not path.is_file():
            return None
        try:
            width, height = Image.open(path).size
        except OSError:
            return None
        return (record["source"], width / height)

    rows = [r for r in pmap(one, sample.to_dict("records"), n_jobs=n_jobs) if r]
    sources = sorted({source for source, _ in rows})
    return {
        "images": len(rows),
        "by_source": {
            source: {
                "n": sum(1 for s, _ in rows if s == source),
                "median_aspect": round(float(np.median([a for s, a in rows if s == source])), 3),
                "min": round(float(min(a for s, a in rows if s == source)), 3),
                "max": round(float(max(a for s, a in rows if s == source)), 3),
            }
            for source in sources
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker count")
    ap.add_argument("--aspect-by-source", action="store_true", help="the leakage measurement")
    ap.add_argument("image", nargs="?", type=Path)
    args = ap.parse_args(argv)

    if args.image:
        from src.features import context as ctx

        print(json.dumps(extract(ctx.from_image(args.image)), indent=2))
        return 0
    if args.aspect_by_source:
        print(json.dumps(aspect_by_source(n_jobs=args.jobs), indent=2))
        return 0

    result = evaluate(args.pages, args.jobs)
    if not result["pages"]:
        print("no synthetic corpus; run the 1.3.7 generator first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 4.1.6 - text statistics: how much writing there is and where it sits.

    text_area_frac      share of the ink that is writing rather than drawing
    text_label_length   mean width of a text block, in page widths
    text_inside_share   share of text blocks whose centre falls inside a drawn shape

ER diagrams are label-heavy and circuits are not; a flowchart writes inside its boxes and an
ER diagram writes on its edges. All three read the *text layer* from 3.2.8 and never re-propose
regions, so whatever 3.2.7 decided is writing is what is counted here - one answer, not two.

## These features inherit a known error, in a known direction

3.2.7 measured its text proposals two ways. On synthetic pages with exact layers it reaches
**text F1 0.94 while keeping 100% of the shape ink**; on 20 real photographs the same method
claims 0.46 of what a human would call writing, against 0.16 of the connector ink. So on a
photograph roughly half the writing is missed and a sixth of the connectors are wrongly claimed.

That maps onto these three features as a **downward bias on `text_area_frac`** and a **noise
floor under `text_inside_share`**, both worse on photographs than on renders. It does not make
them useless - a page with no writing and a page covered in it are still far apart - but it does
mean a small difference in `text_area_frac` between two types is not evidence of anything.
No correction is applied here: correcting a bias whose size is known only on synthetic data
would move the number without making it more true.

`text_label_length` is a **width, not a character count**. Reading the label needs 9.3's OCR,
which is four phases away, and a width in page units is what geometry can honestly supply: it
separates a one-word state name from a sentence in a process box without pretending to know
what either says.

## What it measures, per type

Medians over 600 synthetic pages. `text_label_length` and `text_inside_share` are undefined on
**18.7%** of pages, which are the pages where the text layer came back empty:

    type            area_frac   label_length   inside_share
    flowchart         0.0038       0.0146          1.000
    state_machine     0.0201       0.0211          0.667
    er_diagram        0.1503       0.0436          0.714
    circuit           0.1034       0.0742          0.000
    wireframe         0.0147       0.0553          0.000

`text_inside_share` does what the family was built for, and does it cleanly: **1.000 for
flowcharts, whose labels live inside their boxes, against 0.000 for circuits and wireframes**,
whose annotations sit beside a symbol or a field. ER diagrams and state machines land between at
0.71 and 0.67. That is a three-way split on one feature, and it is the strongest single signal
Phase 4 has produced so far.

`text_label_length` and `text_area_frac` disagree with the plan's expectation in an instructive
way. ER diagrams *are* the most label-heavy type by area fraction (0.150) as predicted - but
**circuits come second at 0.103 while flowcharts come last at 0.004**, which is the reverse of
what "flowcharts are full of labelled boxes" suggests. The cause is not the writing; it is
3.2.8's split. A flowchart's label sits inside a box, touching nothing, and is claimed as text
cleanly - but so little of it survives relative to the heavy box outlines that its *share of
ink* is tiny. A circuit is mostly thin symbols, so the same amount of writing is a much larger
share of very little ink. **`text_area_frac` is measuring how much drawing there is as much as
how much writing**, and it should be read next to 4.1.5's ink coverage rather than alone - which
is a question for 4.2.5's correlation pruning to settle on the evidence.

    python -m src.features.textstats --pages 600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.features import structural as st
from src.features.context import PageContext, inside

NAMES = ("text_area_frac", "text_label_length", "text_inside_share")


def blocks(context: PageContext) -> list[tuple[int, int, int, int]]:
    """Text grouped into words and lines, as boxes. One block is one label."""
    boxes = list(context.text_boxes)
    if not boxes:
        return []
    grouped = []
    for group in st.text_blocks(context):
        xs = [boxes[i][0] for i in group]
        ys = [boxes[i][1] for i in group]
        x1 = max(boxes[i][0] + boxes[i][2] for i in group)
        y1 = max(boxes[i][1] + boxes[i][3] for i in group)
        grouped.append((min(xs), min(ys), x1 - min(xs), y1 - min(ys)))
    return grouped


def extract(context: PageContext) -> dict[str, float]:
    ink = float(context.mask.sum())
    labels = blocks(context)

    if not labels:
        # No writing at all is a fact, not a gap: the share of ink that is text is genuinely
        # zero. But the mean length of no labels, and where no labels sit, are not.
        return {
            "text_area_frac": 0.0 if ink else float("nan"),
            "text_label_length": float("nan"),
            "text_inside_share": float("nan"),
        }

    widths = np.array([w for _, _, w, _ in labels], float) / context.long_side
    centres = [(x + w / 2, y + h / 2) for x, y, w, h in labels]
    within = sum(
        1 for point in centres if any(inside(region.bbox, *point) for region in context.regions)
    )
    return {
        "text_area_frac": float(context.text_mask.sum()) / ink if ink else float("nan"),
        "text_label_length": float(widths.mean()),
        "text_inside_share": within / len(labels),
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

    rows = [r for r in pmap(_row, st._synthetic_pages(limit), n_jobs=n_jobs, desc="text") if r]
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
                    float(np.nanmedian([r[name] for r in rows if r["diagram_type"] == kind])), 4
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

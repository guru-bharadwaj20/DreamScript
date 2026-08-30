r"""Phase 4.1.3 - shape mix: what fraction of the drawn shapes are boxes, diamonds, ovals.

    shape_frac_rectangle   four-cornered and fills its own box
    shape_frac_diamond     four-cornered and fills about half of it
    shape_frac_ellipse     no corners, longer than it is tall
    shape_frac_circle      no corners, as tall as it is wide
    shape_frac_freeform    everything the four rules above decline to name

Diamonds mean flowchart or ER; circles mean state machine; a page of nothing but rectangles is
a wireframe. The five fractions sum to 1 by construction, so one of them is redundant and 4.2.5
is expected to say so - it is left in because *which* one is redundant depends on the corpus,
and dropping the wrong one by hand loses information.

## The rule, and the two measurements it needs

3.2.3 measured that **only 11% of hand-drawn closed shapes recover as 4-gons** under
Douglas-Peucker, so a rule reading `vertices == 4` classifies almost nothing. Two measurements
replace it, and it takes both:

* `rect_fill` - area over the area of the shape's own **minimum-area rectangle**. This is
  rotation-invariant, and it answers "is this four-sided?": a box and a diamond both score 1.00,
  an ellipse 0.785, a star 0.34.
* `extent` - area over the *axis-aligned* box. This is deliberately **not** rotation-invariant,
  because **a diamond is a rotated square**. Nothing intrinsic to the outline separates the two;
  the only difference between a box and a diamond is how it sits on the page, so the feature
  that separates them has to be one that changes when you turn it. An ideal box reads 0.99 and
  an ideal diamond 0.50.

Using `rect_fill` alone would be the tidier design and it is wrong: it calls every diamond a
rectangle. Using `extent` alone is what an earlier version did, and it fails the other way -
on a photograph the page is not square to the camera, so a drawn box tilts a few degrees and
its axis-aligned box grows around it. Measured on the 2.2.4 crops, `extent` reads 0.54-0.75 for
shapes a person called a rectangle. So `rect_fill` decides *four-sided* and `extent` decides
*which*.

The consequence is that this family depends on 3.1.3 and 3.1.7 having done their job. Measured
on a synthetic box rotated in place:

    rotation    extent   rect_fill   verdict
      0 deg      0.99      1.00      rectangle
      5 deg      0.84      0.99      rectangle
     10 deg      0.73      0.99      rectangle
     20 deg      0.59      0.99      diamond
     45 deg      0.48      1.00      diamond

A rectangle survives about 15 degrees of tilt and then becomes a diamond, which is not a bug to
be fixed but the definition being applied consistently. 3.1.7 leaves a **median residual skew of
0.28 degrees** on a rectified page, so this is safe after preprocessing and unusable without it.

## Measured against human labels: it does not work on photographs

The 30 blind-labelled crops from 2.2.4 are the only shape ground truth in this repository that a
person produced, so that is what this is scored on: crop each labelled element out of its own
page by its IR bounding box, run the region extractor over the crop, classify the largest
region. `rounded-rect` folds into `rectangle` - no geometric rule recovers a corner radius from
a drawn box, and 2.2.4 already found the BPMN convention does not match what the pen drew.

    accuracy 11/30 = 0.367, against a majority-class baseline of 0.400

**The rule is worse than calling everything a rectangle.** That is the result. Tuning does not
rescue it either: a grid search over both thresholds, the corner limit and the aspect split -
1,530 combinations on these same 30 crops - tops out at 0.467, and reaches it by labelling half
the sample rectangle.

    truth \ predicted   rectangle  diamond  circle  ellipse  freeform
    rectangle (12)           4         2        0        0        6
    diamond (4)              0         4        0        0        0
    circle (4)               2         1        1        0        0
    ellipse (2)              2         0        0        0        0
    freeform (8)             2         3        1        0        2

The reason is in the distributions, and it is the same reason 3.2.6 gave up on arrowhead
precision: **the classes overlap on the measurement**, so there is no operating point rather
than no good threshold.

    label        n   rect_fill min   median   max
    rectangle    8       0.49         0.70    0.78
    circle       4       0.60         0.75    0.79
    ellipse      2       0.77         0.78    0.79
    diamond      4       0.54         0.60    0.63
    freeform     6       0.57         0.66    0.83

A hand-drawn circle and a hand-drawn rectangle have the same median fill on this corpus. The
one bright spot is fragile: **diamond recall is 1.00 - all four are found - at precision 0.40**,
because six other crops share the window. Tightening the window to 0.60 drops recall to 2 of 4
and precision *falls* to 0.33, which is what a distribution overlap looks like when squeezed.

Two further facts to carry into Phase 9: **6 of the 30 crops yield no closed region at all** -
the outline is broken, nothing encloses area, and they are scored `freeform` rather than skipped
- and on clean renders the rule is exact, which the unit tests pin. This is a photograph
problem, not an algorithm one, and 9.1's learned detector is the answer to it.

What survives is weaker than a shape classifier and may still be useful: these are **fractions
over a whole page**. A page-level mix can carry a signal that no single verdict does - 40% of
shapes landing in the diamond window is a different page from 0%, even when any one of them is
a coin flip. Whether that survives into accuracy is 4.2.6's ranking and Phase 5's job to say,
not this module's to assume.

    python -m src.features.shapes --evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.features.context import PageContext, Region

#: The five names this family reports. A subset of 2.1.4's frozen `SHAPES` vocabulary: the
#: distinctions Phase 4 can make from geometry alone, with everything else in `freeform`.
KINDS = ("rectangle", "diamond", "ellipse", "circle", "freeform")
NAMES = tuple(f"shape_frac_{kind}" for kind in KINDS)

#: Filling this much of its own *minimum-area* rectangle makes an outline a quadrilateral of
#: some kind. An ideal box and an ideal diamond both score 1.00 here - see the docstring - so
#: this test decides "four-sided", never which of the two it is. Set at half rather than at the
#: two thirds an ideal shape suggests: a hand-drawn box on a photograph reads 0.66-0.78 and a
#: hand-drawn diamond 0.54-0.63, and a gate that excludes the diamonds excludes the one verdict
#: this family exists to make.
QUAD_FILL_MIN = 0.50

#: Share of its *axis-aligned* box, which is the only thing that separates a rectangle from a
#: diamond, because a diamond is a rotated square. An ideal box is 0.99 and an ideal diamond
#: 0.50; on a photograph the same shapes read 0.54-0.75 and 0.41-0.56.
RECT_EXTENT_MIN = 0.62
DIAMOND_EXTENT = (0.28, 0.60)

#: An ellipse fills pi/4 = 0.785 of its rectangle. Anything smooth in this window is an oval;
#: the aspect ratio then decides whether it is a circle.
ROUND_FILL = (0.60, 0.90)

#: Long side over short side below which an oval is "as round as it is tall".
CIRCLE_ASPECT = 1.30

#: More turning maxima than this and the outline is not smooth, whatever its fill says. Two
#: rather than zero: a real pen closes a circle with a visible join, and that join is a corner.
MAX_ROUND_CORNERS = 2


def classify(region: Region) -> str:
    """One of `KINDS` for a single drawn shape."""
    fill, extent, corners = region.rect_fill, region.extent, region.corners

    if corners <= MAX_ROUND_CORNERS and ROUND_FILL[0] <= fill <= ROUND_FILL[1]:
        return "circle" if region.rect_aspect < CIRCLE_ASPECT else "ellipse"
    if fill >= QUAD_FILL_MIN:
        # Four-sided. Which four-sided shape is a question about the page, not the outline.
        if extent >= RECT_EXTENT_MIN:
            return "rectangle"
        if DIAMOND_EXTENT[0] <= extent <= DIAMOND_EXTENT[1]:
            return "diamond"
    return "freeform"


def mix(context: PageContext) -> dict[str, float]:
    """The five fractions. `nan` everywhere when there is no shape to describe."""
    if not context.regions:
        return dict.fromkeys(NAMES, float("nan"))
    counted = [classify(region) for region in context.regions]
    total = len(counted)
    return {f"shape_frac_{kind}": counted.count(kind) / total for kind in KINDS}


def extract(context: PageContext) -> dict[str, float]:
    return mix(context)


# ---------------------------------------------------------------------------------------
# Evaluation against the 30 human-labelled crops from 2.2.4
# ---------------------------------------------------------------------------------------

#: The human vocabulary of 2.2.4 mapped onto the five names above. `rounded-rect` is a
#: rectangle here because no geometric rule recovers a corner radius from a drawn box.
LABEL_MAP = {
    "rectangle": "rectangle",
    "rounded-rect": "rectangle",
    "diamond": "diamond",
    "circle": "circle",
    "ellipse": "ellipse",
    "freeform": "freeform",
}

#: Padding around a cropped element, as a share of its own size, so the outline is not clipped.
CROP_PAD = 0.12


def _crop_regions(page_id: str, bbox: list[float]) -> list[Region]:
    """Regions found inside one IR bounding box, cropped out of its own page."""
    import cv2

    from src.features import context as ctx
    from src.preprocess import layers as ly
    from src.preprocess.exif import load
    from src.utils.config import ROOT

    from src.ir.model import Diagram  # isort: skip

    diagram = Diagram.load(ROOT / "data" / "processed" / "ir" / "hdbpmn" / f"{page_id}.ir.json")
    image_path = ROOT / diagram.meta["image"]
    if not image_path.is_file():
        return []

    original = load(image_path, grayscale=True)
    gray, mask = ly.prepare(image_path)
    # The IR is in original pixels and the mask is at the working width: move the box, do not
    # move the pixels. 3.1.3 made the same choice for the same reason.
    scale = gray.shape[1] / original.shape[1]
    x, y, w, h = (value * scale for value in bbox)
    pad_x, pad_y = CROP_PAD * w, CROP_PAD * h
    x0 = max(0, int(x - pad_x))
    y0 = max(0, int(y - pad_y))
    x1 = min(mask.shape[1], int(x + w + pad_x))
    y1 = min(mask.shape[0], int(y + h + pad_y))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    patch = mask[y0:y1, x0:x1]
    # Cropping can cut a stroke at the border, which leaves the outline open; a one-pixel
    # border of background closes nothing but keeps `findContours` off the image edge.
    patch = cv2.copyMakeBorder(patch.astype(np.uint8), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    return ctx.build_regions(patch.astype(bool), min_area_frac=0.02)


def _score_one(row: dict) -> dict | None:
    from src.utils.config import ROOT

    from src.ir.model import Diagram  # isort: skip

    page_id, element_id = row["key"].split(":", 1)
    path = ROOT / "data" / "processed" / "ir" / "hdbpmn" / f"{page_id}.ir.json"
    if not path.is_file():
        return None
    node = next((n for n in Diagram.load(path).nodes if n.id == element_id), None)
    if node is None or node.bbox is None:
        return None
    regions = _crop_regions(page_id, node.bbox)
    if not regions:
        return {"key": row["key"], "truth": LABEL_MAP[row["shape"]], "predicted": "freeform"}
    largest = max(regions, key=lambda r: r.area)
    return {
        "key": row["key"],
        "truth": LABEL_MAP[row["shape"]],
        "predicted": classify(largest),
        "extent": round(largest.extent, 3),
        "corners": largest.corners,
        "aspect": round(largest.aspect, 3),
    }


def evaluate(n_jobs: int | None = None) -> dict:
    """Accuracy and confusion against the 30 blind human labels from 2.2.4."""
    import pandas as pd

    from src.utils.config import ROOT
    from src.utils.parallel import pmap

    sample = ROOT / "data" / "annotations" / "human_shape_sample.csv"
    if not sample.is_file():
        return {"items": 0}
    labelled = pd.read_csv(sample).to_dict("records")
    rows = [r for r in pmap(_score_one, labelled, n_jobs=n_jobs, desc="shape crops") if r]
    if not rows:
        return {"items": 0}

    kinds = sorted({r["truth"] for r in rows} | {r["predicted"] for r in rows})
    confusion = {
        truth: {
            predicted: sum(1 for r in rows if r["truth"] == truth and r["predicted"] == predicted)
            for predicted in kinds
        }
        for truth in kinds
    }
    correct = sum(1 for r in rows if r["truth"] == r["predicted"])
    return {
        "items": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 3),
        "confusion": confusion,
        "errors": [r for r in rows if r["truth"] != r["predicted"]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluate", action="store_true", help="score against the human labels")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker count")
    ap.add_argument("image", nargs="?", type=Path)
    args = ap.parse_args(argv)

    if args.image:
        from src.features import context as ctx

        print(json.dumps(extract(ctx.from_image(args.image)), indent=2))
        return 0

    result = evaluate(args.jobs)
    if not result["items"]:
        print("no human shape sample; see 2.2.4", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

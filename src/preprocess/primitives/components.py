"""Phase 3.2.1 - connected components and their statistics.

The first question about a cleaned page is "how many separate pieces of ink are there, and
where". Everything downstream is organised by component: a shape is usually one, a word is
several, and an arrow that touches two boxes fuses all three into one - which is a fact worth
knowing rather than a problem to hide.

Each component carries the statistics Phase 4 will turn into features:

    area          ink pixels
    bbox          x, y, w, h
    aspect        w / h
    extent        area / bbox area - how much of its box the shape fills
    solidity      area / convex hull area - how "dented" the outline is
    perimeter     contour length
    circularity   4 pi area / perimeter^2 - 1 for a circle, lower for anything else
    holes         how many enclosed background regions it has

`holes` is the one that is not obvious and is worth the extra pass: a drawn box has one hole
and a scribble has none, which separates "an outline" from "a mark" more reliably than any
shape statistic. It is counted from the contour hierarchy rather than by flood fill, so a hole
inside a hole is not double-counted.

**Two things seen on the first real page this was run against**, both worth carrying forward:

* the three-way parallel branch came back as *one* component, because each arrow touches the
  boxes at both of its ends. That is the normal case, not a failure - Phase 10 separates shapes
  from connectors - but it means component count is not shape count and must never be used as
  one;
* thin pen strokes in the *handwriting* were broken by binarization, so a word came through as
  disconnected fragments. Shapes are drawn with a firmer hand than text and survive; text does
  not. Phase 9's OCR sees the text layer from 3.2.8 rather than these components, and that is
  where the loss will show up.

    python -m src.preprocess.primitives.components <image>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

#: Components below this share of the page are noise that survived 3.1.6.
MIN_AREA_FRAC = 5e-5


@dataclass
class Component:
    label: int
    area: int
    bbox: list[int]  # x, y, w, h
    centroid: list[float]
    aspect: float
    extent: float
    solidity: float
    perimeter: float
    circularity: float
    holes: int
    touches_border: bool
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _hole_count(mask: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    # RETR_CCOMP gives two levels: outer boundaries, then holes. A hole has a parent.
    return int(sum(1 for row in hierarchy[0] if row[3] != -1))


def extract(mask: np.ndarray, min_area_frac: float = MIN_AREA_FRAC) -> list[Component]:
    """Every component of a boolean ink mask, largest first."""
    binary = mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = mask.shape
    min_area = max(1.0, min_area_frac * mask.size)

    out: list[Component] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        piece = (labels[y : y + h, x : x + w] == label).astype(np.uint8)

        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        outline = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(outline, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(outline))) or float(area)

        out.append(
            Component(
                label=label,
                area=area,
                bbox=[x, y, w, h],
                centroid=[float(centroids[label][0]), float(centroids[label][1])],
                aspect=round(w / h, 4) if h else 0.0,
                extent=round(area / (w * h), 4) if w and h else 0.0,
                # Clamped: for a thin diagonal stroke the *pixel count* can exceed the
                # *polygon area* of its own convex hull, because a polygon's area does not
                # include the width of its boundary pixels. Measured at 1.03 on a 3px line.
                solidity=round(min(area / hull_area, 1.0), 4) if hull_area else 0.0,
                perimeter=round(perimeter, 2),
                circularity=(
                    round(4 * np.pi * area / (perimeter * perimeter), 4) if perimeter else 0.0
                ),
                holes=_hole_count(piece),
                touches_border=bool(x == 0 or y == 0 or x + w >= width or y + h >= height),
            )
        )
    out.sort(key=lambda c: -c.area)
    return out


def summarise(components: list[Component]) -> dict:
    if not components:
        return {"components": 0}
    areas = np.array([c.area for c in components], float)
    return {
        "components": len(components),
        "total_ink": int(areas.sum()),
        "largest_area": int(areas.max()),
        "median_area": int(np.median(areas)),
        "with_holes": int(sum(c.holes > 0 for c in components)),
        "touching_border": int(sum(c.touches_border for c in components)),
    }


def main(argv: list[str] | None = None) -> int:
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.exif import load
    from src.preprocess.rules import suppress

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    args = ap.parse_args(argv)

    if not args.image.is_file():
        print(f"no such image: {args.image}", file=sys.stderr)
        return 1
    gray = load(args.image, grayscale=True)
    # Rule suppression before counting, not after: on squared paper the grid fuses most of the
    # page into one component, and every statistic below is then a statistic about the paper.
    mask = denoise(suppress(binarize(median(gray))))
    components = extract(mask)
    print(json.dumps(summarise(components), indent=2))
    for component in components[:10]:
        print(
            f"  #{component.label:<4d} area {component.area:<7d} bbox {component.bbox} "
            f"holes {component.holes} solidity {component.solidity}"
        )
    return 0 if components else 1


if __name__ == "__main__":
    sys.exit(main())

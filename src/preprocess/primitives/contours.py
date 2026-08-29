"""Phase 3.2.2 - contours and their nesting.

A connected component says *where* ink is. Its contours say what shape its boundary has, and -
through the hierarchy - what is inside what.

The hierarchy is the reason to use `RETR_TREE` rather than the simpler retrieval modes. A
hand-drawn diagram is full of nesting that carries meaning:

    a box with a word written inside it        outer contour, then the letters at depth 2
    an accepting state's double circle         two nested rings at depths 0 and 1
    a pool containing lanes containing tasks   three levels, and the depth *is* the containment

Phase 10 needs that containment to build a graph. Recovering it later from bounding-box overlap
is guesswork; reading it off the contour tree is exact.

Each contour carries the fields a later phase actually uses: its points, its depth in the tree,
its parent, whether it is closed, and its area and perimeter. Nothing is classified here.

    python -m src.preprocess.primitives.contours <image>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

#: Contours shorter than this many points cannot describe a shape.
MIN_POINTS = 5

#: Contours enclosing less than this share of the page are noise or single characters.
MIN_AREA_FRAC = 2e-5

#: 4*pi*A/P^2 above which a contour is treated as enclosing something rather than tracing a
#: stroke. Drawn box 0.79, drawn circle 0.85, 3px line 0.033.
CLOSED_RATIO = 0.10

#: Two contours are the two edges of one drawn stroke when the inner encloses at least this
#: much of the outer's area. A genuinely nested shape encloses far less.
SAME_STROKE_AREA_RATIO = 0.75


def _isoperimetric(area: float, perimeter: float) -> float:
    return 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0


@dataclass
class Contour:
    index: int
    points: np.ndarray  # (N, 2) int32
    depth: int
    parent: int
    area: float
    perimeter: float
    closed: bool
    bbox: list[int] = field(default_factory=list)

    @property
    def children(self) -> list[int]:
        return list(self._children)

    _children: list[int] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "depth": self.depth,
            "parent": self.parent,
            "area": round(self.area, 2),
            "perimeter": round(self.perimeter, 2),
            "closed": self.closed,
            "bbox": self.bbox,
            "n_points": int(len(self.points)),
            "children": self.children,
        }


def extract(
    mask: np.ndarray, *, min_area_frac: float = MIN_AREA_FRAC, min_points: int = MIN_POINTS
) -> list[Contour]:
    """Every contour of a boolean mask, with its depth in the containment tree."""
    found, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or not len(found):
        return []
    hierarchy = hierarchy[0]  # next, previous, first child, parent

    # Depth by walking up to the root. The tree is shallow - three or four levels at most - so
    # the repeated walk is cheaper than building an index.
    def depth_of(index: int) -> int:
        depth = 0
        parent = hierarchy[index][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        return depth

    min_area = max(1.0, min_area_frac * mask.size)
    contours: dict[int, Contour] = {}
    for index, points in enumerate(found):
        if len(points) < min_points:
            continue
        area = float(cv2.contourArea(points))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(points)
        contours[index] = Contour(
            index=index,
            points=points.reshape(-1, 2),
            depth=depth_of(index),
            parent=int(hierarchy[index][3]),
            area=area,
            perimeter=float(cv2.arcLength(points, True)),
            # A contour that traces a closed outline encloses real area; one that traces a
            # *stroke* doubles back on itself, so its perimeter is twice the stroke's length
            # while its area is only length x width. Measured: a drawn box scores 0.79 on
            # 4*pi*A/P^2 and a 3px line scores 0.033, so 0.10 sits well clear of both.
            closed=bool(_isoperimetric(area, float(cv2.arcLength(points, True))) >= CLOSED_RATIO),
            bbox=[int(x), int(y), int(w), int(h)],
        )

    for index, contour in contours.items():
        if contour.parent in contours:
            contours[contour.parent]._children.append(index)
    return sorted(contours.values(), key=lambda c: -c.area)


def outer(contours: list[Contour]) -> list[Contour]:
    return [c for c in contours if c.depth == 0]


def drawn_outlines(contours: list[Contour]) -> list[Contour]:
    """One contour per *drawn* outline, with the stroke's inner edge dropped.

    This is the correction that makes the hierarchy usable. `RETR_TREE` sees a pen stroke as
    two contours - the outside of the ink and the inside of it - so a box drawn inside another
    box comes back at depths 0, 1, 2 and 3 rather than 0 and 1. Reading containment off the raw
    depth therefore double-counts every level.

    A pair is recognised by area: the inner edge of a 3px stroke encloses ~96% of the outer
    edge's area, while a genuinely nested shape encloses a small fraction of its container.
    """
    by_index = {c.index: c for c in contours}
    inner_edges = set()
    for contour in contours:
        for child_index in contour.children:
            child = by_index.get(child_index)
            if child and child.area >= SAME_STROKE_AREA_RATIO * contour.area:
                inner_edges.add(child_index)
    return [c for c in contours if c.index not in inner_edges]


def nested_pairs(contours: list[Contour]) -> list[tuple[int, int]]:
    """(outer index, inner index) for every direct containment - a double circle, a lane."""
    return [(c.parent, c.index) for c in contours if c.parent != -1]


def summarise(contours: list[Contour]) -> dict:
    if not contours:
        return {"contours": 0}
    depths = [c.depth for c in contours]
    return {
        "contours": len(contours),
        "outer": sum(d == 0 for d in depths),
        "max_depth": max(depths),
        "closed": sum(c.closed for c in contours),
        "nested_pairs": len(nested_pairs(contours)),
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
    mask = denoise(suppress(binarize(median(load(args.image, grayscale=True)))))
    contours = extract(mask)
    print(json.dumps(summarise(contours), indent=2))
    for contour in contours[:10]:
        print(f"  {contour.to_dict()}")
    return 0 if contours else 1


if __name__ == "__main__":
    sys.exit(main())

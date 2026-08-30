"""Phase 4 - the page context every feature family reads from.

Phase 3 ends with a bag of primitives: components, contours, line segments, arrowhead
candidates, text boxes, and the two-layer split. Phase 4 turns those into a fixed-length vector
of numbers. Between the two sits one problem worth solving once: **a feature family should not
have to decide what counts as a drawn shape.**

Nine families are about to ask overlapping questions - how many nodes, what fraction are
rectangles, how far apart are they, how deeply are they nested. If each one re-derives "node"
from contours, nine slightly different definitions of a node go into the same feature vector,
and a change to one of them silently moves features in the others. So the definition lives
here, is made once per page, and is what every family in `src/features/` is handed.

## What a region is, and what it is not

A `Region` is a *closed* contour on the shape layer that encloses at least `MIN_AREA_FRAC` of
the page, taken from the outline set that 3.2.2's `drawn_outlines` has already de-duplicated -
so the inner and outer edge of one pen stroke are one region, not two. Each carries the
geometry the families need: box shape, fill, solidity, its Douglas-Peucker vertex count, and
the curvature description from 3.2.5 that 3.2.3 showed vertex counting alone cannot replace.

Everything on the shape layer that is *not* inside a region is **connector ink**: arrows,
lines, and whatever else joins the shapes. This is a subtraction rather than a detection, which
means it inherits every failure of the region test - a box whose outline broke into two arcs
does not become a region, and its ink becomes "connector". That is stated here because it is
the main way a feature vector from this context can be wrong, and it is why 4.1.1 measures node
count against a ground-truth graph rather than trusting it.

Regions and connectors are both derived from the *shape* layer, and text statistics from the
*text* layer, so 3.2.8's partition is what keeps handwriting out of the geometry features.

    python -m src.features.context <image>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

#: A contour must enclose this share of the page to be a drawn shape rather than a letter or a
#: speck. At the 1400px working width of `layers.prepare` this is about 1,400 px - a drawn box
#: is tens of thousands, a character a few hundred. Expressed as a fraction so the same number
#: means the same thing on a test canvas of any size.
MIN_AREA_FRAC = 1e-3

#: Regions closer than this share of the page's long side are treated as touching, which is how
#: 4.1.7 decides a connector reaches a shape.
TOUCH_FRAC = 0.02

#: A *fusion shell*: the outline that appears when a connector joins two shapes into one blob of
#: ink, so `findContours` traces the outside of both boxes and the arrow between them as a single
#: closed curve. 3.2.1 saw this on the first real page it ran against - a three-way branch came
#: back as one component - and it is the single largest source of a wrong node count.
#:
#: A shell is recognised by two properties together, because neither alone is enough:
#: its children explain most of its area, *and* the join between them dents its outline, so it
#: is markedly less solid than its own convex hull. A wireframe panel with boxes drawn inside it
#: also has children explaining its area, but it is a rectangle and scores ~1.0 on solidity, so
#: the second test is what keeps genuine nesting - which 4.1.9 exists to measure - intact.
FUSION_CHILD_RATIO = 0.5
FUSION_SOLIDITY = 0.90


@dataclass
class Region:
    """One drawn shape: a closed outline big enough to be a node."""

    index: int
    bbox: list[int]  # x, y, w, h
    centre: tuple[float, float]
    area: float
    perimeter: float
    extent: float  # area / bbox area
    solidity: float  # area / convex hull area
    aspect: float  # w / h
    vertices: int  # Douglas-Peucker, 3.2.3
    straightness: float  # 3.2.5
    corners: int  # 3.2.5
    min_angle: float
    depth: int  # containment depth, 3.2.2
    parent: int

    def to_dict(self) -> dict:
        out = dict(self.__dict__)
        out["centre"] = list(self.centre)
        return out


@dataclass
class PageContext:
    """Everything Phase 4 is allowed to look at, derived once."""

    id: str
    height: int
    width: int
    mask: np.ndarray  # bool, all ink
    shape_mask: np.ndarray  # bool, drawing
    text_mask: np.ndarray  # bool, writing
    regions: list[Region] = field(default_factory=list)
    connector_mask: np.ndarray | None = None
    segments: list[dict] = field(default_factory=list)
    arrowheads: list[dict] = field(default_factory=list)
    text_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    skeleton: dict = field(default_factory=dict)

    @property
    def page_area(self) -> int:
        return self.height * self.width

    @property
    def long_side(self) -> int:
        return max(self.height, self.width)

    @property
    def touch_px(self) -> float:
        return TOUCH_FRAC * self.long_side

    @property
    def connector_segments(self) -> list[dict]:
        """Segments whose midpoint is not inside any region: the ink that joins shapes."""
        return [s for s in self.segments if not self.inside_any(*midpoint(s))]

    def inside_any(self, x: float, y: float, pad: float = 0.0) -> bool:
        return any(inside(r.bbox, x, y, pad) for r in self.regions)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "shape": [self.height, self.width],
            "regions": len(self.regions),
            "segments": len(self.segments),
            "connector_segments": len(self.connector_segments),
            "arrowheads": len(self.arrowheads),
            "text_boxes": len(self.text_boxes),
            "ink_frac": round(float(self.mask.sum()) / self.page_area, 5),
        }


def midpoint(segment: dict) -> tuple[float, float]:
    return ((segment["x1"] + segment["x2"]) / 2, (segment["y1"] + segment["y2"]) / 2)


def inside(bbox: list[int], x: float, y: float, pad: float = 0.0) -> bool:
    bx, by, bw, bh = bbox
    return (bx - pad) <= x <= (bx + bw + pad) and (by - pad) <= y <= (by + bh + pad)


def densify(points: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    """Resample a closed outline at even spacing, so curvature can be measured on it.

    3.2.2 stores contours with `CHAIN_APPROX_SIMPLE`, which keeps only the endpoints of each
    straight run: a rectangle arrives as four points. 3.2.5's turning angle is measured over a
    window of *points*, so on four points it measures nothing and reports a rectangle as 0.0
    straight. Putting the intermediate points back costs a millisecond and is the difference
    between `straightness` being a shape feature and being a contour-encoding artefact.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 2:
        return points
    loop = np.vstack([points, points[:1]])
    steps = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(distance[-1])
    if total < spacing:
        return points
    wanted = np.arange(0.0, total, spacing)
    return np.stack(
        [np.interp(wanted, distance, loop[:, 0]), np.interp(wanted, distance, loop[:, 1])], axis=1
    )


def _is_fusion_shell(contour, children: list) -> bool:
    """True when this outline is the outside of several shapes joined by connector ink."""
    if len(children) < 2 or contour.area <= 0:
        return False
    if sum(child.area for child in children) / contour.area < FUSION_CHILD_RATIO:
        return False
    hull_area = float(cv2.contourArea(cv2.convexHull(contour.points)))
    solidity = contour.area / hull_area if hull_area else 1.0
    return solidity < FUSION_SOLIDITY


def _nest(regions: list[Region]) -> list[Region]:
    """Depth and parent from containment among the surviving regions.

    The contour tree cannot be reused here: dropping a fusion shell would leave its children at
    depth 1 with a parent that no longer exists, and 4.1.9 would read that as nesting. Depth is
    therefore recounted over what survived, from the boxes themselves.
    """
    by_area = sorted(regions, key=lambda r: -r.area)
    for index, region in enumerate(by_area):
        parent = -1
        for candidate in by_area[:index]:  # strictly larger, so a parent can only be earlier
            if inside(candidate.bbox, *region.centre) and candidate.area > region.area:
                parent = candidate.index  # the smallest such box wins: the list is area-sorted
        region.parent = parent
    depths = {r.index: 0 for r in by_area}
    lookup = {r.index: r for r in by_area}
    for region in by_area:
        depth, walk = 0, region.parent
        while walk in lookup and depth < len(by_area):
            depth += 1
            walk = lookup[walk].parent
        depths[region.index] = depth
        region.depth = depth
    return regions


def build_regions(shape_mask: np.ndarray, min_area_frac: float = MIN_AREA_FRAC) -> list[Region]:
    """Every closed outline on the shape layer that is big enough to be a node."""
    from src.preprocess.primitives import contours as ct
    from src.preprocess.primitives import curves as cu
    from src.preprocess.primitives import polygons as pg

    # `min_points=3` rather than 3.2.2's default of 5: with CHAIN_APPROX_SIMPLE a clean
    # rectangle or diamond is stored as exactly four points, so the default point-count floor
    # drops the very shapes this is meant to find. Point count is an encoding artefact; the
    # area floor below is what actually separates a shape from a speck.
    found = ct.drawn_outlines(ct.extract(shape_mask, min_points=3))
    min_area = min_area_frac * shape_mask.size
    by_index = {c.index: c for c in found}
    children_of = {c.index: [by_index[i] for i in c.children if i in by_index] for c in found}

    out: list[Region] = []
    for contour in found:
        if _is_fusion_shell(contour, children_of[contour.index]):
            continue
        if not contour.closed or contour.area < min_area:
            continue
        x, y, w, h = contour.bbox
        hull_area = float(cv2.contourArea(cv2.convexHull(contour.points))) or contour.area
        described = cu.describe(densify(contour.points))
        polygon = pg.describe(contour)
        out.append(
            Region(
                index=contour.index,
                bbox=[int(x), int(y), int(w), int(h)],
                centre=(x + w / 2, y + h / 2),
                area=float(contour.area),
                perimeter=float(contour.perimeter),
                extent=round(contour.area / (w * h), 4) if w and h else 0.0,
                # Clamped for the same reason 3.2.1 clamps it: a polygon's area excludes the
                # width of its own boundary pixels, so a thin outline can score above 1.
                solidity=round(min(contour.area / hull_area, 1.0), 4) if hull_area else 0.0,
                aspect=round(w / h, 4) if h else 0.0,
                vertices=int(polygon["vertices"]),
                straightness=float(described.get("straightness", 0.0)),
                corners=int(described.get("corners", 0)),
                min_angle=float(polygon["min_angle"]),
                depth=int(contour.depth),
                parent=int(contour.parent),
            )
        )
    return _nest(out)


def stroke_width(mask: np.ndarray) -> float:
    """Pen width in pixels, as twice the distance from the thickest ink to the background."""
    if not mask.any():
        return 1.0
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return max(1.0, 2.0 * float(np.percentile(distance[mask], 90)))


def connector_ink(shape_mask: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Shape-layer ink outside every region's box: what is left is what joins them.

    The boxes are padded by a stroke width and a half before subtracting, and the reason is the
    fusion case above. When a connector welds two shapes into one blob, the outlines that
    survive as regions are the *inner* edges of the pen strokes, so each region's box is a
    stroke narrower than the shape a reader sees. Subtracting the unpadded box leaves a closed
    ring of the shape's own ink behind, and that ring touches the connectors at both ends -
    which merges every edge on the page into one enormous strand and makes `edge_count` 1 on a
    drawing with four arrows. The padding is what keeps a shape's own outline out of the
    connector layer.
    """
    pad = int(np.ceil(1.5 * stroke_width(shape_mask)))
    covered = np.zeros(shape_mask.shape, bool)
    height, width = shape_mask.shape
    for region in regions:
        x, y, w, h = region.bbox
        covered[
            max(0, y - pad) : min(height, y + h + pad), max(0, x - pad) : min(width, x + w + pad)
        ] = True
    return shape_mask & ~covered


def text_boxes_of(text_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes of the writing, read off the text layer rather than proposed again.

    3.2.8 already decided which ink is writing; re-running MSER here would let the two answers
    disagree, and then 4.1.6's `text_area_frac` and `labels_per_node` would be counting
    different things.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), 8)
    return [
        (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        for i in range(1, count)
    ]


def from_masks(
    mask: np.ndarray,
    gray: np.ndarray | None = None,
    *,
    page_id: str = "page",
    min_area_frac: float = MIN_AREA_FRAC,
) -> PageContext:
    """Build a context from an ink mask that has already been cleaned."""
    from src.preprocess import layers as ly
    from src.preprocess.primitives import arrowheads as ah
    from src.preprocess.primitives import segments as sg
    from src.preprocess.thinning import branch_points, end_points, thin

    mask = mask.astype(bool)
    split = ly.separate(mask, gray)
    regions = build_regions(split.shape, min_area_frac)
    skeleton = thin(mask)
    source = gray if gray is not None else np.where(mask, 0, 255).astype(np.uint8)
    height, width = mask.shape
    return PageContext(
        id=page_id,
        height=int(height),
        width=int(width),
        mask=mask,
        shape_mask=split.shape,
        text_mask=split.text,
        regions=regions,
        connector_mask=connector_ink(split.shape, regions),
        segments=[s.to_dict() for s in sg.detect_both(source, split.shape)],
        arrowheads=[a.to_dict() for a in ah.detect(split.shape)],
        text_boxes=text_boxes_of(split.text),
        skeleton={
            "pixels": int(skeleton.sum()),
            "branch_points": int(branch_points(skeleton).sum()),
            "end_points": int(end_points(skeleton).sum()),
        },
    )


def from_image(image_path: Path, page_id: str | None = None) -> PageContext:
    """The whole 3.1 chain, then the 3.2.8 split, then the regions. Seconds per page."""
    from src.preprocess import layers as ly

    image_path = Path(image_path)
    gray, mask = ly.prepare(image_path)
    return from_masks(mask, gray, page_id=page_id or image_path.stem)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    args = ap.parse_args(argv)

    context = from_image(args.image)
    print(json.dumps(context.summary(), indent=2))
    for region in context.regions[:10]:
        print(f"  region {region.index}: {json.dumps(region.to_dict())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

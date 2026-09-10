"""Decide which element wrote each line of text on the page.

`src.ocr.labelcrops` established the finding this module acts on: **the recogniser already
clears S3's target wherever the crop contains its own label** (Activity 0.0621, task 0.0831) and
fails at 0.33-0.72 wherever it does not. So the remaining error is a geometry problem, and it has
exactly three shapes. This module is those three fixes and nothing else.

## 1. A container's title is in its header strip, not in its box

A Lane, Pool or Participant spans the diagram. 9.3.1's inset box hands the recogniser the whole
band, so it reads whichever activity is drawn inside it - which is why ground truth 'claims' came
back as 'insurer' and 'caim officer' as 'students'. Those are not misreadings, they are *other
elements' labels, read correctly*. A BPMN container writes its title in a narrow header strip on
the leading edge, rotated to run along it, so that is what gets cropped, and a block taller than
it is wide is rotated back to horizontal before the recogniser sees it.

## 2. Ownership has to be decided globally, not one line at a time

`labelcrops.own_lines` let every line pick its own best element. Two elements standing close
together therefore compete for the same line and the loser gets nothing, which is the same
failure `labelcrops.assign` already documents from the other direction - and it is why Event and
Flow sit at 0.33 and 0.43 while their crops are individually plausible. Here every element is
given up to `MAX_LINES` slots and the whole element-line matching is solved once by
`linear_sum_assignment`, so a line is taken from a nearer claimant only when that claimant has
nothing else. `prune` then drops any extra line that does not vertically stack with the
element's nearest one, so a spare slot cannot drag in an unrelated phrase.

## 3. An unlocated label is scored as a wrong read, so it should still be read

287 val labels - 12.6% - were located nowhere and scored as empty strings, worth 0.08 of the
headline CER on their own. Refusing to crop them does not make them cheaper, it makes them cost
the maximum. `blocks` therefore runs a second matching with the reach limit removed, and any
element still empty falls back to the conventional place its label would be written. A wrong crop
and no crop score the same; a right one is free.

Nothing here reads a label. Every rule is a function of the page's geometry and its detected text
boxes, and no caller passes the transcript in.
"""

from __future__ import annotations

import numpy as np

#: Classes whose label is a title in a header strip rather than text inside the box.
CONTAINER_PREFIXES = ("Lane", "lane", "pool", "Pool", "Participant", "participant")

#: Classes whose label is written outside the drawn glyph.
EXTERNAL_PREFIXES = (
    "Event",
    "startEvent",
    "endEvent",
    "messageStartEvent",
    "messageEndEvent",
    "messageIntermediateCatchEvent",
    "messageIntermediateThrowEvent",
    "timerStartEvent",
    "timerIntermediateEvent",
    "intermediateEvent",
    "DataObjectReference",
    "dataObjectReference",
    "DataStoreReference",
    "dataStoreReference",
    "Gateway",
    "exclusiveGateway",
    "eventBasedGateway",
    "parallelGateway",
    "TextAnnotation",
)

MAX_LINES = 3
#: Slots for a connector. Cutting this to 1 costs 0.028 CER and to 2 costs 0.006, so an edge
#: label wraps as readily as a node's and the spare slots are earning their keep.
EDGE_LINES = 3
#: How much of a text line must lie inside a shape before that shape owns it outright.
RESERVE = 0.5
#: How far into its own header strip a container's title may start, as a share of strip depth.
EDGE_TOL = 0.5
#: The strip is widened for the containment test only, matching `src.ocr.vertical.WIDEN`.
STRIP_WIDEN = 1.6
#: How near a connector a text line must be to be its label, as a share of page width.
EDGE_REACH = 0.06
#: Cost per unit of page width for a label sitting further along a connector from its source.
START_BIAS = 0.0
BIG = 1e6
#: A block this much taller than it is wide is a title written along a vertical header strip.
ROTATE_ASPECT = 1.5


def kind_of(element: dict) -> str:
    """`container`, `external`, `edge` or `internal` - the four rules this module applies."""
    if element.get("kind") == "edge":
        return "edge"
    name = str(element["id"])
    if name.startswith(CONTAINER_PREFIXES):
        return "container"
    if name.startswith(EXTERNAL_PREFIXES):
        return "external"
    return "internal"


def gap(box, other) -> float:
    """Edge-to-edge distance between two `[x, y, w, h]` boxes; 0 when they touch or overlap."""
    x, y, w, h = box[:4]
    bx, by, bw, bh = other[:4]
    return float(np.hypot(max(x - (bx + bw), bx - (x + w), 0), max(y - (by + bh), by - (y + h), 0)))


def inside(box, block, slack: float = 2.0) -> bool:
    x, y, w, h = box[:4]
    bx, by, bw, bh = block[:4]
    return (
        bx >= x - slack
        and by >= y - slack
        and bx + bw <= x + w + slack
        and by + bh <= y + h + slack
    )


def centre(box) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def polyline_gap(points: np.ndarray, box) -> float:
    """Distance from a text box's centre to the nearest point on a polyline."""
    return polyline_foot(points, box)[0]


def polyline_foot(points: np.ndarray, box) -> tuple[float, float]:
    """`(distance, position)` of the closest point on a polyline, position as arc fraction.

    The position is what separates an edge's own label from its endpoints' labels. Text sitting
    at the very start or end of a connector is almost always the node's - the connector is
    touching that node there - while an edge label is written along the run.
    """
    cx, cy = centre(box)
    point = np.array([cx, cy])
    spans = [float(np.hypot(*(points[i + 1] - points[i]))) for i in range(len(points) - 1)]
    total = sum(spans) or 1.0
    best, at, walked = float("inf"), 0.0, 0.0
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ab = b - a
        length = float(ab @ ab)
        t = 0.0 if length == 0 else float(np.clip((point - a) @ ab / length, 0.0, 1.0))
        distance = float(np.hypot(*(point - (a + t * ab))))
        if distance < best:
            best, at = distance, (walked + t * spans[i]) / total
        walked += spans[i]
    return best, at


def header_strip(box, fraction: float = 0.12, cap: float = 0.45) -> list[float]:
    """The narrow band along a container's leading edge, where BPMN writes its title.

    Left for a horizontal band, top for a vertical one. `cap` keeps the strip from swallowing a
    short container whole - the point is a strip, not a smaller box.
    """
    x, y, w, h = box[:4]
    if w >= h:
        return [x, y, max(8.0, min(fraction * w, cap * h)), h]
    return [x, y, w, max(8.0, min(fraction * h, cap * w))]


def union(boxes) -> list[int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def node_cost(box, line, page_w: float) -> float:
    """9.3.1's node preference, unchanged: inside beats below beats merely near."""
    x, y, w, h = box[:4]
    distance = gap(box, line)
    if distance > max(3.0 * max(w, h), page_w * 0.12):
        return BIG
    bx, by, bw, bh = line[:4]
    centred = abs((bx + bw / 2) - (x + w / 2))
    below = by >= y + h * 0.6 and centred < max(w, bw)
    return (
        distance
        - (10_000 if inside(box, line) else 0)
        - (max(w, h) * 0.5 if below else 0)
        + 0.15 * centred
    )


def overlap_fraction(box, region) -> float:
    """How much of `box` lies inside `region`, as a fraction of the box's own area."""
    x, y, w, h = box[:4]
    rx, ry, rw, rh = region[:4]
    dx = max(0.0, min(x + w, rx + rw) - max(x, rx))
    dy = max(0.0, min(y + h, ry + rh) - max(y, ry))
    return (dx * dy) / max(1.0, w * h)


def container_cost(box, line, page_w: float, share: float = 0.6) -> float:
    """A container may only claim a line that lies in its own header strip.

    Two rules, and both were forced by a nested Participant/Lane pair. **Containment is by area,
    not by the line's centre**: a Participant's title starts at the Participant's own left edge,
    which is *outside* the strip of the Lane drawn inside it, and a centre test let the Lane
    claim it anyway. **Cost is the offset between the two leading edges**, not the distance from
    the strip, because every candidate is inside its strip and so every distance is zero - what
    separates an outer container's title from an inner one's is only that each is written hard
    against its own edge.
    """
    strip = header_strip(box)
    # Containment is tested against a widened strip, because `src.ocr.vertical` detects rotated
    # titles in a widened one and a title that runs a little past the narrow band is still the
    # title. The leading-edge bound below is what actually keeps a nested Lane out.
    x, y, w, h = box[:4]
    wide = (
        [strip[0], strip[1], min(strip[2] * STRIP_WIDEN, w), strip[3]]
        if w >= h
        else [strip[0], strip[1], strip[2], min(strip[3] * STRIP_WIDEN, h)]
    )
    if overlap_fraction(line, wide) < share:
        return BIG
    vertical = box[2] >= box[3]
    offset = abs(line[0] - strip[0]) if vertical else abs(line[1] - strip[1])
    # A title is written *hard against* its own edge. Without this bound a Lane drawn just
    # inside a Participant sits inside the Participant's strip too, and the two swapped titles
    # in 24 of ~80 container crops. Half the strip is the whole tolerance a title needs.
    depth = strip[2] if vertical else strip[3]
    return BIG if offset > EDGE_TOL * depth else offset


def cost_row(element: dict, lines: list, page_w: float, reach: bool = True) -> np.ndarray:
    """One element's cost against every line, under the rule its class earns."""
    role = kind_of(element)
    box = [float(v) for v in element["bbox"]]
    out = np.full(len(lines), BIG)
    for j, line in enumerate(lines):
        if role == "container":
            value = container_cost(box, line, page_w)
        elif role == "edge":
            distance, at = polyline_foot(element["polyline"], line)
            if reach and distance > page_w * EDGE_REACH:
                value = BIG
            else:
                # Position along the run carries nothing, measured in both directions and kept
                # at zero. Penalising both ends - endpoint text belonging to the node the
                # connector touches there - cost 0.009 CER, so edge labels really do sit near
                # an end; biasing toward the source instead, on BPMN's habit of writing a
                # branch condition just after the split, cost 0.013 at 0.05 and 0.035 at 0.15.
                # Only the perpendicular distance is doing any work here.
                value = distance + START_BIAS * page_w * at
        else:
            value = node_cost(box, line, page_w)
            if not reach and value >= BIG:
                value = gap(box, line)
        out[j] = value
    return out


def prune(boxes: list) -> list:
    """Keep the nearest line plus only the lines that vertically stack with it.

    A spare slot with nothing better to take will reach for an unrelated phrase; requiring
    horizontal overlap and a gap under one line height is what a wrapped label actually looks
    like, and is the same test `labelcrops.stack` applies.
    """
    if len(boxes) <= 1:
        return list(boxes)
    boxes = sorted(boxes, key=lambda b: b[1])
    keep = [boxes[0]]
    for box in boxes[1:]:
        kx, ky, kw, kh = keep[-1][:4]
        x, y, w, h = box[:4]
        overlap = min(kx + kw, x + w) - max(kx, x)
        if overlap > 0.4 * min(kw, w) and -0.4 * h <= y - (ky + kh) <= 0.9 * max(kh, h):
            keep.append(box)
    return keep


def match(elements: list[dict], lines: list, page_w: float, reach: bool = True) -> dict:
    """Globally optimal element-to-line matching with up to `MAX_LINES` slots per element."""
    from scipy.optimize import linear_sum_assignment

    if not elements or not lines:
        return {}
    rows = np.vstack([cost_row(e, lines, page_w, reach) for e in elements])
    counts = [EDGE_LINES if kind_of(e) == "edge" else MAX_LINES for e in elements]
    slots = np.repeat(rows, counts, axis=0)
    owner = np.repeat(np.arange(len(elements)), counts)
    # A second or third line is worth having only if it is close, so later slots pay for reach.
    tier = np.concatenate([np.arange(c) for c in counts])[:, None]
    slots = np.where(slots >= BIG, BIG, slots + tier * page_w * 0.01)
    picked: dict[int, list] = {}
    for i, j in zip(*linear_sum_assignment(slots), strict=True):
        if slots[i, j] < BIG:
            picked.setdefault(int(owner[i]), []).append(lines[j])
    return {i: prune(boxes) for i, boxes in picked.items()}


def below_box(element: dict, page_shape) -> list[int]:
    """Where a BPMN external label is written when nothing was detected there."""
    x, y, w, h = (float(v) for v in element["bbox"])
    height, width = page_shape[:2]
    cx = x + w / 2.0
    span = max(3.0 * w, 120.0)
    x0 = max(0.0, min(cx - span / 2.0, width - 8.0))
    y0 = max(0.0, min(y + h + 0.15 * h, height - 8.0))
    return [int(x0), int(y0), int(min(span, width - x0)), int(min(max(h * 0.9, 24.0), height - y0))]


def inset_box(element: dict, inset: float = 0.10) -> list[int]:
    """9.3.1's node crop: the box, inset to drop the drawn outline."""
    x, y, w, h = element["bbox"]
    return [
        int(round(x + inset * w)),
        int(round(y + inset * h)),
        int(round(w * (1 - 2 * inset))),
        int(round(h * (1 - 2 * inset))),
    ]


def edge_fallback(element: dict) -> list[int]:
    """A box straddling the connector's midpoint, which is 9.3.1's original edge rule."""
    points = np.asarray(element["polyline"], dtype=float)
    mid = points[len(points) // 2]
    length = float(np.sum(np.hypot(*np.diff(points, axis=0).T))) if len(points) > 1 else 100.0
    span = max(0.6 * length, 80.0)
    band = max(0.12 * length, 40.0)
    return [int(mid[0] - span / 2), int(mid[1] - band), int(span), int(2 * band)]


def with_rotation(box) -> list[int]:
    """Append the rotate flag: a block taller than it is wide is a title running vertically."""
    x, y, w, h = (int(round(v)) for v in box[:4])
    return [x, y, w, h, int(h > ROTATE_ASPECT * max(1, w))]


def blocks(page_shape, elements: list[dict], lines: list) -> dict[str, list[int]]:
    """`element id -> [x, y, w, h, rotate]` for every element on the page.

    Every element gets a block. The passes run in order of how certain the rule is: containers
    take their header strips, activities keep the inset box that already reads at 0.0621,
    everything that carries its text outside itself competes for what is left, and whatever is
    still empty is rescued without a reach limit and then by convention.
    """
    height, width = page_shape[:2]
    page_area = float(height * width)
    groups: dict[str, list[dict]] = {"container": [], "external": [], "edge": [], "internal": []}
    for element in elements:
        groups[kind_of(element)].append(element)

    out: dict[str, list[int]] = {}
    free = [list(map(int, b[:4])) for b in lines]
    taken: set = set()

    # **Activities reserve first.** They carry the most reliable rule in this module - 9.3.1's
    # inset box reads them at 0.0647 - so where two rules want the same line, theirs wins. Doing
    # this after the containers let a container strip that happens to overlap an activity take
    # that activity's text as its title, which was 43 of ~80 container crops reading body text.
    for element in groups["internal"]:
        out[element["id"]] = with_rotation(inset_box(element))
        if element["bbox"][2] * element["bbox"][3] < 0.20 * page_area:
            box = [float(v) for v in element["bbox"]]
            taken.update(tuple(b) for b in free if overlap_fraction(b, box) >= RESERVE)
    free = [b for b in free if tuple(b) not in taken]

    won = match(groups["container"], free, width)
    for i, boxes in won.items():
        out[groups["container"][i]["id"]] = with_rotation(union(boxes))
        taken.update(tuple(b) for b in boxes)
    # A header strip belongs to its container whether or not the matching used it. Leaving the
    # unclaimed ones on offer is how a sequence flow came to be read as 'customer' and 'claimant'
    # - those are pool titles, sitting in a strip the flow had no business reaching into.
    for element in groups["container"]:
        strip = header_strip([float(v) for v in element["bbox"]])
        taken.update(tuple(b) for b in free if overlap_fraction(b, strip) >= RESERVE)
    free = [b for b in free if tuple(b) not in taken]

    outside = groups["external"] + groups["edge"]
    won = match(outside, free, width)
    for i, boxes in won.items():
        out[outside[i]["id"]] = with_rotation(union(boxes))
        taken.update(tuple(b) for b in boxes)
    free = [b for b in free if tuple(b) not in taken]

    # Rescue: an unlocated label is scored as an empty read, so it is never cheaper to skip one.
    missing = [e for e in outside if e["id"] not in out]
    won = match(missing, free, width, reach=False)
    for i, boxes in won.items():
        out[missing[i]["id"]] = with_rotation(union(boxes))

    for element in elements:
        if element["id"] in out:
            continue
        role = kind_of(element)
        if role == "edge":
            out[element["id"]] = with_rotation(edge_fallback(element))
        elif role == "container":
            out[element["id"]] = with_rotation(header_strip([float(v) for v in element["bbox"]]))
        else:
            out[element["id"]] = with_rotation(below_box(element, (height, width)))
    return out

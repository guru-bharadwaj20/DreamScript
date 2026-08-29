"""Phase 3.2.7 - text region proposal.

A page of hand-drawn diagram carries two completely different kinds of ink, and almost
everything downstream wants only one of them. Phase 9's OCR wants the *writing*; Phases 4 and
10 want the *drawing*. Nothing in the earlier stages separates them - the connected component of
a letter `o` and the connected component of a small circle node are the same object to 3.2.1.

This module proposes where the writing is. It does not remove it; that is 3.2.8.

## What the writing actually looks like, and what that buys

Three properties separate a character from a drawn shape, and each is used as a filter:

* **Size.** A character is a small fraction of the page: on hdBPMN photographs, annotated
  shapes are 5-30% of the long side and characters are 1-3% of it.
* **Stroke-width consistency.** This is the stroke-width transform's contribution. A pen
  writing a word keeps one width; a filled arrowhead or a doubled-over box edge does not. The
  width at a pixel is twice its distance to the background, sampled on the skeleton so the
  taper at a stroke's edge does not pollute the statistic, and what matters is its *coefficient
  of variation* within the candidate, not its absolute value.
* **Company.** A letter is never alone. Text comes in words and words come in lines, so
  candidates are grouped by height and horizontal proximity and a group of one is discarded.
  This filter does more work than the other two combined: it is what stops a box's rounded
  corner - which is small and evenly-stroked - from being read as a letter.

The `height / stroke_width` ratio is the fourth signal and the cheapest: a written character is
a handful of stroke widths tall, while a drawn box is fifty. It is the one filter that survives
a character touching the shape it sits inside.

## How the numbers below were obtained

Two evaluations, because neither one is sufficient alone.

**Synthetic pages** (`synthetic_page`) draw shapes and Hershey-stroked words separately, so the
two ground-truth masks are exact by construction. This measures the separator's mechanism and
is the number reported in plan.md. Hershey fonts are stroked rather than filled, which is the
right model for a pen, but they are regular in a way real handwriting is not, so this is an
upper bound.

**Real hdBPMN pages** have no text annotation, but two things the annotation *does* record stand
in for one: ink well inside an annotated node box is a label, and ink along an annotated edge
polyline is a connector. `weak_agreement` reports the share of each that the proposal claims.

Measured, 12 synthetic pages and 20 hdBPMN photographs:

    synthetic   text recall 0.89   text precision 1.00   shape layer keeps 1.00 of shape ink
    real        writing claimed 0.46          connector ink wrongly claimed 0.16

The real column is the honest one and the two numbers should be read together: the proposal is
about three times more likely to claim writing than to claim a connector, which is what a
*proposal* stage is for, and it is nowhere near the clean separation the synthetic page shows.
Half the writing is missed because a letter touching the shape around it is one component with
that shape (see `refine`), and that is the same fragility Phase 3.2.1 recorded when it noticed
binarization breaks handwriting before it breaks drawing.

    python -m src.preprocess.primitives.text
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

#: Candidate height, as a fraction of the page's long side. Below the floor is speckle; above
#: the ceiling it is a drawn shape, not a character.
MIN_HEIGHT_FRAC = 0.006
MAX_HEIGHT_FRAC = 0.075

#: Width over height. Generous, because MSER happily returns a whole word as one region.
MIN_ASPECT = 0.08
MAX_ASPECT = 12.0

#: Ink inside the candidate's box. A character fills a decent share of its own box; an empty
#: rectangle outline fills very little of it.
MIN_DENSITY = 0.10
MAX_DENSITY = 0.95

#: Height measured in stroke widths. A written character is a few strokes tall; a drawn box is
#: tens. This is the single most discriminative filter here.
MAX_HEIGHT_OVER_STROKE = 22.0
MIN_HEIGHT_OVER_STROKE = 1.5

#: Coefficient of variation of the stroke width within a candidate.
MAX_STROKE_CV = 0.75

#: Grouping into words and lines: two candidates belong together when their heights are within
#: this ratio, their vertical spans overlap, and the horizontal gap between them is at most
#: this many times the taller one's height.
MAX_HEIGHT_RATIO = 2.5
MAX_GAP_OVER_HEIGHT = 1.6
MIN_VERTICAL_OVERLAP = 0.3

#: A group smaller than this is not writing. See the docstring: this filter carries the method.
MIN_GROUP_SIZE = 2

#: A candidate whose ink fits inside a rectangle this many stroke-widths thin is a straight
#: piece of a drawn line, not a character. Binarization breaks a long box edge into short
#: fragments, and without this test every one of those fragments is the right size and the
#: right stroke width to look like a letter.
MAX_LINE_THICKNESS = 2.2

#: Bounding-box diagonal above which a component on a real page is a connector rather than a
#: written label. Used only by `weak_agreement`, to keep the two references apart.
LONG_COMPONENT_PX = 60.0


def stroke_width_map(mask: np.ndarray) -> np.ndarray:
    """Stroke width at every ink pixel: twice the distance to the nearest background pixel.

    This over-reports by exactly one pixel and deliberately keeps that convention: the distance
    transform gives 1 at the pixel next to the background, so a stroke nine pixels across comes
    back as 10. Every use here is a ratio against another length, the offset is the same for
    every stroke on the page, and subtracting one would round the wrong way on even widths.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return 2.0 * distance


def stroke_width_stats(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float]:
    """(median stroke width, coefficient of variation) inside a box.

    Sampled on the skeleton rather than on every ink pixel. Every stroke tapers to width zero at
    its own edge, so averaging over the whole stroke measures the taper as much as the pen, and
    two strokes of the same width but different length then report different statistics.
    """
    from src.preprocess.thinning import thin

    x, y, w, h = box
    patch = mask[y : y + h, x : x + w]
    if not patch.any():
        return 0.0, 0.0
    widths = stroke_width_map(patch)
    spine = thin(patch)
    sampled = widths[spine] if spine.any() else widths[patch]
    if not len(sampled):
        return 0.0, 0.0
    median = float(np.median(sampled))
    mean = float(np.mean(sampled))
    variation = float(np.std(sampled) / mean) if mean else 0.0
    return median, variation


def mser_boxes(gray: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Candidate boxes from MSER, plus the ink components MSER missed.

    MSER is run on the *inverted* grayscale so that dark ink is the bright, stable region. It is
    good at finding a character whose stroke touches a shape, which a connected-component pass
    cannot separate. It is unreliable on the broken, faint strokes a phone photo produces, so
    the components of the ink mask are added as candidates too and the union is de-duplicated.
    """
    long_side = max(gray.shape)
    lo = int((MIN_HEIGHT_FRAC * long_side) ** 2 * MIN_DENSITY)
    hi = int((MAX_HEIGHT_FRAC * long_side) ** 2)
    detector = cv2.MSER_create()
    detector.setMinArea(max(4, lo))
    detector.setMaxArea(max(16, hi))
    _, boxes = detector.detectRegions(255 - gray.astype(np.uint8))

    found = [tuple(int(v) for v in box) for box in boxes]
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        x, y, w, h, _ = stats[index]
        found.append((int(x), int(y), int(w), int(h)))
    return _deduplicate(found)


def _deduplicate(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """MSER returns the same character at several thresholds; keep one box per position."""
    seen: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    for box in boxes:
        key = (box[0] // 4, box[1] // 4, box[2] // 4, box[3] // 4)
        if key not in seen or box[2] * box[3] > seen[key][2] * seen[key][3]:
            seen[key] = box
    return sorted(seen.values())


def _plausible(mask: np.ndarray, box: tuple[int, int, int, int], long_side: int) -> bool:
    x, y, w, h = box
    if h <= 0 or w <= 0:
        return False
    if not (MIN_HEIGHT_FRAC * long_side <= h <= MAX_HEIGHT_FRAC * long_side):
        return False
    if not (MIN_ASPECT <= w / h <= MAX_ASPECT):
        return False
    patch = mask[y : y + h, x : x + w]
    density = float(patch.mean()) if patch.size else 0.0
    if not (MIN_DENSITY <= density <= MAX_DENSITY):
        return False
    width, variation = stroke_width_stats(mask, box)
    if width <= 0 or variation > MAX_STROKE_CV:
        return False
    ratio = h / width
    if not (MIN_HEIGHT_OVER_STROKE <= ratio <= MAX_HEIGHT_OVER_STROKE):
        return False
    return not _is_straight_stroke(patch, width)


def _is_straight_stroke(patch: np.ndarray, width: float) -> bool:
    """Does this candidate's ink lie along a single straight line?"""
    points = np.column_stack(np.nonzero(patch))[:, ::-1].astype(np.float32)
    if len(points) < 5:
        return True
    minor = min(cv2.minAreaRect(points)[1])
    return minor <= MAX_LINE_THICKNESS * width


def _vertical_overlap(a, b) -> float:
    top = max(a[1], b[1])
    bottom = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, bottom - top) / max(1, min(a[3], b[3]))


def group_into_lines(boxes: list[tuple[int, int, int, int]]) -> list[list[int]]:
    """Indices of boxes that belong to the same word or line of writing.

    Union-find over the pairwise "these two are neighbours on a line" relation, which is
    transitive in the way a line of text is: `a` next to `b` next to `c` is one line even though
    `a` and `c` are far apart.
    """
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for position, i in enumerate(order):
        a = boxes[i]
        for j in order[position + 1 : position + 40]:
            b = boxes[j]
            taller = max(a[3], b[3])
            if taller / max(1, min(a[3], b[3])) > MAX_HEIGHT_RATIO:
                continue
            if _vertical_overlap(a, b) < MIN_VERTICAL_OVERLAP:
                continue
            if b[0] - (a[0] + a[2]) > MAX_GAP_OVER_HEIGHT * taller:
                continue
            root_a, root_b = find(i), find(j)
            if root_a != root_b:
                parent[root_a] = root_b

    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def propose(mask: np.ndarray, gray: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
    """Boxes that look like writing, after every filter including grouping."""
    if gray is None:
        gray = np.where(mask, 0, 255).astype(np.uint8)
    long_side = max(mask.shape)
    candidates = [b for b in mser_boxes(gray, mask) if _plausible(mask, b, long_side)]
    if not candidates:
        return []
    kept: list[tuple[int, int, int, int]] = []
    for group in group_into_lines(candidates):
        if len(group) >= MIN_GROUP_SIZE:
            kept.extend(candidates[i] for i in group)
    return sorted(kept)


def to_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """The proposed boxes painted onto a page-sized boolean mask."""
    canvas = np.zeros(shape, bool)
    for x, y, w, h in boxes:
        canvas[y : y + h, x : x + w] = True
    return canvas


def refine(mask: np.ndarray, boxes: list[tuple[int, int, int, int]], share: float = 0.6):
    """The proposed boxes turned into an ink mask, one connected component at a time.

    A box is a rectangle and writing is not, so a connector passing behind a word is inside the
    word's box without being part of the word. Taking whole components instead - and only those
    that lie mostly inside a proposal - removes that contamination: the connector's component
    extends far outside the box and fails the share test.

    The cost is the one case this cannot fix. A letter whose stroke *touches* the shape it sits
    inside is a single component with that shape, so it is either kept entirely or dropped
    entirely, and it is dropped. That is the main source of missed text on real pages, and it is
    a consequence of working from a binary mask rather than of the proposal.
    """
    if not boxes:
        return np.zeros(mask.shape, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    total = np.maximum(stats[:, cv2.CC_STAT_AREA].astype(np.float64), 1.0)

    # Against *one* box, not against the union of them. A page carries hundreds of proposals
    # and their union covers a large part of it, so a long connector can be 60% covered by a
    # dozen unrelated boxes without ever fitting inside any of them. Measured on 20 hdBPMN
    # photographs, testing the union instead of the single box put 23% of annotated connector
    # ink into the text layer; testing one box at a time is what makes the difference.
    best = np.zeros(count, np.float64)
    for x, y, w, h in boxes:
        patch = labels[y : y + h, x : x + w]
        if patch.size == 0:
            continue
        counts = np.bincount(patch.ravel(), minlength=count).astype(np.float64)
        np.maximum(best, counts / total, out=best)

    keep = best >= share
    keep[0] = False
    return keep[labels] & mask


def synthetic_page(seed: int = 0, size: int = 1200) -> dict:
    """A page whose two layers are known exactly: shapes drawn, then words written.

    Hershey fonts are stroke fonts, so `putText` here draws a pen path rather than a filled
    glyph - the right model for a pen, and regular in a way real handwriting is not.
    """
    rng = np.random.default_rng(seed)
    shapes = np.zeros((size, size), np.uint8)
    text = np.zeros((size, size), np.uint8)
    words = ["start", "check", "valid", "retry", "done", "input", "save", "load", "yes", "no"]
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]

    boxes = []
    for row in range(3):
        for column in range(3):
            cx = int(size * (0.2 + 0.3 * column))
            cy = int(size * (0.2 + 0.3 * row))
            w = int(rng.integers(150, 230))
            h = int(rng.integers(70, 110))
            kind = int(rng.integers(0, 3))
            if kind == 0:
                cv2.rectangle(
                    shapes, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), 255, 3
                )
            elif kind == 1:
                cv2.ellipse(shapes, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, 3)
            else:
                points = np.array(
                    [[cx, cy - h // 2], [cx + w // 2, cy], [cx, cy + h // 2], [cx - w // 2, cy]]
                )
                cv2.polylines(shapes, [points], True, 255, 3)
            boxes.append((cx, cy, w, h))

    # Connecting lines, so the page is a diagram rather than nine isolated shapes. They run
    # between the *edges* of neighbouring shapes: a line through the middle of a box would run
    # through the word written there and merge the two layers, which is not what a diagram
    # looks like and would make the ground truth untestable.
    for index, (cx, cy, w, h) in enumerate(boxes):
        row, column = divmod(index, 3)
        if column < 2:
            right = boxes[index + 1]
            cv2.line(shapes, (cx + w // 2, cy), (right[0] - right[2] // 2, right[1]), 255, 3)
        if row < 2:
            below = boxes[index + 3]
            cv2.line(shapes, (cx, cy + h // 2), (below[0], below[1] - below[3] // 2), 255, 3)

    for cx, cy, _, _ in boxes:
        word = words[int(rng.integers(0, len(words)))]
        font = fonts[int(rng.integers(0, len(fonts)))]
        scale = 0.6 + 0.3 * float(rng.random())
        (text_width, text_height), _ = cv2.getTextSize(word, font, scale, 2)
        cv2.putText(
            text,
            word,
            (cx - text_width // 2, cy + text_height // 2),
            font,
            scale,
            255,
            2,
            cv2.LINE_8,
        )

    ink = (shapes > 0) | (text > 0)
    gray = np.where(ink, 40, 245).astype(np.uint8)
    return {"gray": gray, "ink": ink, "shapes": shapes > 0, "text": text > 0}


def score(predicted: np.ndarray, truth: np.ndarray) -> dict:
    from src.preprocess.evalset import f1

    return {k: round(v, 4) for k, v in f1(predicted, truth).items()}


def evaluate(pages: int = 12) -> dict:
    """Text and shape pixel scores on synthetic pages, where the answer is exact."""
    from src.utils.parallel import pmap

    def one(seed: int) -> dict:
        page = synthetic_page(seed)
        boxes = propose(page["ink"], page["gray"])
        proposal = refine(page["ink"], boxes)
        return {
            "boxes": len(boxes),
            "text": score(proposal, page["text"]),
            "shape_kept": score(page["ink"] & ~proposal, page["shapes"]),
        }

    rows = pmap(one, list(range(pages)))
    return {
        "pages": len(rows),
        "boxes": sum(r["boxes"] for r in rows),
        "text_recall": round(float(np.mean([r["text"]["recall"] for r in rows])), 4),
        "text_precision": round(float(np.mean([r["text"]["precision"] for r in rows])), 4),
        "text_f1": round(float(np.mean([r["text"]["f1"] for r in rows])), 4),
        "shape_recall": round(float(np.mean([r["shape_kept"]["recall"] for r in rows])), 4),
        "shape_precision": round(float(np.mean([r["shape_kept"]["precision"] for r in rows])), 4),
    }


def weak_agreement(limit: int = 20) -> dict:
    """Real pages, weak ground truth, taken from two things the annotation does record.

    hdBPMN does not say where the text is, so there is no true recall to measure here. It does
    say two things that stand in for one:

        writing   ink well inside an annotated node box - a label, in almost every case
        drawing   ink along an annotated edge polyline - a connector, and never a label

    The edge polylines are the better half of this: an annotated connector is drawing by
    definition, and nobody writes along one. (An earlier version used the *outline band* of the
    node boxes as the drawing reference and scored much worse - not because the proposal was
    worse, but because a hand-drawn box does not sit where its annotation says to the pixel, and
    a label written near the top of a box lands in that band. The band was measuring annotation
    slack, not text separation.)

    The band around a polyline is reported twice, because it was not obvious which reading was
    right. BPMN labels its sequence flows *along* them, so a word written beside an arrow lands
    in the band and would inflate the drawing figure; splitting the band's ink by component
    length - a connector is long, a label is short - tests that. It made almost no difference
    (0.185 against 0.159), so the band is not meaningfully contaminated and the false claims
    really are connector ink. The hypothesis was wrong and the number stands.
    """
    from src.ir.model import SUFFIX, Diagram
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.exif import load
    from src.preprocess.rules import suppress
    from src.utils.config import ROOT
    from src.utils.parallel import pmap

    paths = sorted((ROOT / "data" / "processed" / "ir" / "hdbpmn").glob(f"*{SUFFIX}"))[:limit]

    def one(path):
        diagram = Diagram.load(path)
        image_path = ROOT / diagram.meta["image"]
        if not image_path.is_file():
            return None
        gray = load(image_path, grayscale=True)
        scale = 1400 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ink = denoise(suppress(binarize(median(gray))))

        inside = np.zeros(ink.shape, bool)
        for node in diagram.nodes:
            if not node.bbox:
                continue
            x, y, w, h = (int(v * scale) for v in node.bbox)
            if w < 20 or h < 20:
                continue
            pad = max(3, int(0.15 * min(w, h)))
            inside[y + pad : y + h - pad, x + pad : x + w - pad] = True

        connectors = np.zeros(ink.shape, np.uint8)
        for edge in diagram.edges:
            if len(edge.polyline or []) < 2:
                continue
            points = (np.asarray(edge.polyline, np.float64) * scale).astype(np.int32)
            cv2.polylines(connectors, [points.reshape(-1, 1, 2)], False, 255, 9)
        if not inside.any() or not connectors.any():
            return None

        proposal = refine(ink, propose(ink, gray))

        # The band around an annotated polyline is not purely drawing: BPMN labels its sequence
        # flows *along* them ("yes", "no"), so a word written beside an arrow lands inside the
        # band. Splitting the band's ink by component length separates the two - a connector is
        # a long component, a label is a short one - and reports both, so the contamination is
        # visible instead of assumed.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        diagonal = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        long_component = np.zeros(count, bool)
        long_component[1:] = diagonal[1:] >= LONG_COMPONENT_PX
        deep = ink & inside
        # A connector that runs across a node box is the annotation's straight line, not the
        # drawn one; excluding the box interiors keeps the two references disjoint.
        outline = ink & (connectors > 0) & ~inside
        connector = outline & long_component[labels]
        return {
            "deep_ink_proposed": float((proposal & deep).sum() / max(1, deep.sum())),
            "outline_ink_proposed": float((proposal & outline).sum() / max(1, outline.sum())),
            "connector_ink_proposed": float((proposal & connector).sum() / max(1, connector.sum())),
        }

    rows = [r for r in pmap(one, paths, prefer="threads") if r]
    if not rows:
        return {"pages": 0}
    return {
        "pages": len(rows),
        "writing_proposed": round(float(np.mean([r["deep_ink_proposed"] for r in rows])), 4),
        "band_proposed": round(float(np.mean([r["outline_ink_proposed"] for r in rows])), 4),
        "drawing_proposed": round(float(np.mean([r["connector_ink_proposed"] for r in rows])), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--real", type=int, default=20)
    args = ap.parse_args(argv)

    result = evaluate(args.pages)
    print(json.dumps(result, indent=2))
    real = weak_agreement(args.real)
    print("real pages (weak ground truth): " + json.dumps(real))

    checks = {
        "text_recall_at_least_0.80": result["text_recall"] >= 0.80,
        "shape_layer_keeps_0.90_of_shape_ink": result["shape_recall"] >= 0.90,
        "writing_proposed_more_than_drawing": (
            real.get("pages", 0) == 0 or real["writing_proposed"] > 2 * real["drawing_proposed"]
        ),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

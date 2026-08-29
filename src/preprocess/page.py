"""Phase 3.1.2 - finding the page in a photograph.

A photo of a diagram is a photo of a *sheet of paper on a desk*: the sheet is a bright convex
quadrilateral with a strong edge against a darker background. Finding it lets everything after
this point work on the page rather than on the desk, the photographer's knee and the window.

The interesting question is how to know the detection is right, and there is a good answer for
free. hdBPMN annotates where every shape is, so **a page detection is correct exactly when the
quadrilateral contains all the annotated content**. That is a real criterion measured against
human labels, not a visual impression, and it catches the failure that matters: a quad that
cuts off part of the diagram.

Not every photo has a findable page. Scans that fill the frame have no border at all, and for
those the honest answer is `None` - meaning "use the whole image" - rather than a made-up
quadrilateral. `detect` returns `None` and the caller decides.

    python -m src.preprocess.page --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from src.preprocess.exif import load
from src.utils.config import ROOT

#: Work at this size regardless of the camera. Page edges are metre-scale features; finding
#: them at 12 megapixels is slower and no more accurate.
WORK_SIDE = 900

#: A quad smaller than this share of the frame is a shape inside the diagram, not the page.
MIN_AREA_FRAC = 0.25

#: Above this, the "page" is the whole frame and there is nothing to crop.
MAX_AREA_FRAC = 0.995


def order_corners(quad: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left.

    By sum and difference of the coordinates, which is orientation-independent: the top-left
    has the smallest x+y and the top-right the smallest y-x, whatever angle the page sits at.
    """
    points = quad.reshape(4, 2).astype(np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(total)],
            points[np.argmin(diff)],
            points[np.argmax(total)],
            points[np.argmax(diff)],
        ],
        dtype=np.float32,
    )


def _candidates(gray: np.ndarray) -> list[np.ndarray]:
    """Contours worth testing, from two views of the image.

    Two, because the two common photographs fail differently: a page on a dark desk has a
    strong intensity edge that Canny finds, while a page on a pale desk barely has one and is
    better separated by Otsu on a heavily blurred copy.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    coarse = cv2.GaussianBlur(gray, (31, 31), 0)
    _, bright = cv2.threshold(coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    found: list[np.ndarray] = []
    for view in (edges, bright):
        contours, _ = cv2.findContours(view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found.extend(sorted(contours, key=cv2.contourArea, reverse=True)[:5])
    return found


#: A page seen through a camera is still nearly a parallelogram. These bounds reject the
#: degenerate trapezoids that Canny finds along the ruled lines of squared paper - which was
#: the actual failure mode: 23 of 120 photos were "detected" as wedges that cut off half the
#: diagram, and every one of them violates at least one of these.
MIN_ANGLE, MAX_ANGLE = 55.0, 125.0
MAX_OPPOSITE_SIDE_RATIO = 2.2
MIN_RECTANGULARITY = 0.60


#: How much brighter the page must be than the surface around it, in grey levels. A sheet of
#: paper photographed on a desk is conspicuously the brightest thing in the frame; a patch of
#: paper surrounded by *more paper* is not, and that is precisely what the surviving false
#: positives were - quads covering 38-54% of a page that filled the whole frame. Testing
#: brightness rather than raising the area threshold keeps a genuine desk photo detectable.
MIN_SURROUND_CONTRAST = 10.0

#: Width of the ring sampled outside the quad, as a fraction of the working image's long side.
SURROUND_RING = 0.05


def _brighter_than_surround(gray: np.ndarray, quad: np.ndarray) -> bool:
    inside = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(inside, quad.reshape(4, 2).astype(np.int32), 255)
    width = max(3, int(SURROUND_RING * max(gray.shape)) | 1)
    grown = cv2.dilate(inside, np.ones((width, width), np.uint8))
    ring = cv2.subtract(grown, inside)
    if ring.sum() == 0:
        return False  # the quad already reaches the frame edge: nothing to compare against
    page = float(gray[inside > 0].mean())
    surround = float(gray[ring > 0].mean())
    return page - surround >= MIN_SURROUND_CONTRAST


#: Each side of a real page sits on a real intensity edge. A quad cutting across the middle of
#: a sheet does not: its sides run over blank paper. The test is the mean gradient along each
#: side, relative to the image's own 90th-percentile gradient, so it does not depend on
#: exposure. All four sides must be supported - three is what a wedge along a ruled line gets.
MIN_EDGE_SUPPORT = 0.18
EDGE_SAMPLES = 64


def _edge_support(gray: np.ndarray, quad: np.ndarray) -> bool:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    reference = float(np.percentile(magnitude, 90)) or 1.0

    points = quad.reshape(4, 2).astype(np.float32)
    height, width = gray.shape
    for i in range(4):
        a, b = points[i], points[(i + 1) % 4]
        ts = np.linspace(0.08, 0.92, EDGE_SAMPLES)[:, None]  # skip the corners
        samples = a + ts * (b - a)
        xs = np.clip(samples[:, 0].astype(int), 0, width - 1)
        ys = np.clip(samples[:, 1].astype(int), 0, height - 1)
        if float(magnitude[ys, xs].mean()) / reference < MIN_EDGE_SUPPORT:
            return False
    return True


#: Grow the accepted quad by this share of the image's long side before returning it.
#: Rectification must not clip ink, and a sheet of paper always has margin beyond the drawing.
#: Four of the twenty cut-content cases overshot by 0.4-2.5% of the long side - a hair - and
#: this converts those from failures into correct detections without loosening any test above.
OUTWARD_MARGIN = 0.03


def _grow(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    points = quad.reshape(4, 2).astype(np.float32)
    centre = points.mean(axis=0)
    offsets = points - centre
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    margin = OUTWARD_MARGIN * max(shape)
    grown = points + offsets / np.maximum(norms, 1e-6) * margin
    grown[:, 0] = np.clip(grown[:, 0], 0, shape[1] - 1)
    grown[:, 1] = np.clip(grown[:, 1], 0, shape[0] - 1)
    return grown


def _plausible_page(quad: np.ndarray) -> bool:
    """Whether four points could be a sheet of paper viewed at an angle."""
    points = quad.reshape(4, 2).astype(np.float64)
    sides = np.array([points[(i + 1) % 4] - points[i] for i in range(4)])
    lengths = np.linalg.norm(sides, axis=1)
    if lengths.min() <= 1e-6:
        return False

    for i in range(4):
        incoming = -sides[i - 1]
        outgoing = sides[i]
        cosine = np.dot(incoming, outgoing) / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        if not MIN_ANGLE <= angle <= MAX_ANGLE:
            return False

    for a, b in ((0, 2), (1, 3)):
        ratio = max(lengths[a], lengths[b]) / min(lengths[a], lengths[b])
        if ratio > MAX_OPPOSITE_SIDE_RATIO:
            return False

    rect = cv2.minAreaRect(points.astype(np.float32))
    rect_area = rect[1][0] * rect[1][1]
    return rect_area > 0 and cv2.contourArea(points.astype(np.float32)) / rect_area >= (
        MIN_RECTANGULARITY
    )


def detect(image: np.ndarray) -> np.ndarray | None:
    """The page as four corner points in image pixels, or None if there is no page to find."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    scale = WORK_SIDE / max(height, width)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame_area = small.shape[0] * small.shape[1]

    best = None
    best_area = 0.0
    for contour in _candidates(small):
        area = cv2.contourArea(contour)
        if area < MIN_AREA_FRAC * frame_area or area > MAX_AREA_FRAC * frame_area:
            continue
        peri = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.04):
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
                and _plausible_page(approx)
                and _brighter_than_surround(small, approx)
                and _edge_support(small, approx)
                and area > best_area
            ):
                best, best_area = approx, area
                break
    if best is None:
        return None
    return order_corners(_grow(best.astype(np.float32) / scale, gray.shape))


def quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(quad.astype(np.float32)))


def contains(quad: np.ndarray, boxes: list[list[float]], *, slack: float = 4.0) -> bool:
    """True if every corner of every box lies inside the quadrilateral."""
    contour = quad.astype(np.float32).reshape(-1, 1, 2)
    for x, y, w, h in boxes:
        for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
            if cv2.pointPolygonTest(contour, (float(px), float(py)), True) < -slack:
                return False
    return True


def evaluate(limit: int = 100) -> dict:
    """Score detection against hdBPMN's own annotations.

    A detection counts as correct when the quad contains every annotated shape. `no_page` is
    not a failure: a scan that fills the frame genuinely has no page boundary, and saying so is
    the right answer.
    """
    from src.ir.model import SUFFIX, Diagram

    ir_dir = ROOT / "data" / "processed" / "ir" / "hdbpmn"
    results = {"total": 0, "detected": 0, "correct": 0, "no_page": 0, "cut_content": []}
    kept_fractions = []
    for path in sorted(ir_dir.glob(f"*{SUFFIX}"))[:limit]:
        diagram = Diagram.load(path)
        image_path = ROOT / diagram.meta["image"]
        if not image_path.is_file():
            continue
        image = load(image_path)
        results["total"] += 1
        quad = detect(image)
        if quad is None:
            results["no_page"] += 1
            continue
        results["detected"] += 1
        boxes = [n.bbox for n in diagram.nodes if n.bbox]
        if contains(quad, boxes):
            results["correct"] += 1
            frame = image.shape[0] * image.shape[1]
            kept_fractions.append(quad_area(quad) / frame)
        else:
            results["cut_content"].append(diagram.id)
    if kept_fractions:
        results["median_page_share_of_frame"] = round(float(np.median(kept_fractions)), 3)
    return results


CAVEAT = """
  Read the two rates together. `handled_correctly` counts a correct decline as a success,
  and it should: hdBPMN is overwhelmingly scans that fill the frame, where there is no page
  boundary at all and cropping could only lose ink. The number that measures *detection* is
  `precision_when_it_crops`, over far fewer cases. A corpus of photographs of paper on desks
  would exercise this properly; this one does not, and Phase 16's uploads are where it first
  matters.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args(argv)

    results = evaluate(args.limit)
    if not results["total"]:
        print("no hdBPMN IR to evaluate against; run the converter first", file=sys.stderr)
        return 1

    accounted = results["correct"] + results["no_page"]
    rate = accounted / results["total"]
    results["handled_correctly"] = round(rate, 3)
    if results["detected"]:
        results["precision_when_it_crops"] = round(results["correct"] / results["detected"], 3)
    print(json.dumps({k: v for k, v in results.items() if k != "cut_content"}, indent=2))
    if results["cut_content"]:
        print(f"  cut content in {len(results['cut_content'])}: {results['cut_content'][:8]}")

    print(CAVEAT)

    checks = {"handles_at_least_90_percent": rate >= 0.90}
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name} ({rate:.1%})")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

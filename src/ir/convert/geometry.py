"""Phase 2.2.2 - guessing a shape from a contour.

Some sources record where a shape is but not what it is: `flowchartseg` publishes a node mask
and nothing else. This module turns a binary region into a vocabulary shape by the same
reasoning a person would use - count the corners, look at the aspect ratio, check whether the
corners sit at the middle of the edges or in them.

It is deliberately simple. Phase 3.2 replaces it with proper primitive extraction and Phase 5
with a trained classifier; what this needs to be is *honest about its own accuracy*, which is
why everything it returns carries a confidence below 1.0 and every node it labels records
`shape_basis: "geometry"`. Phase 2.2.4 measures how often it is right by running it against
hdBPMN, where the true shape is known.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Douglas-Peucker tolerance as a fraction of the contour perimeter. 2% is the value that
#: separates a hand-drawn quadrilateral from a hand-drawn ellipse on this corpus: below it,
#: wobble in a straight pen stroke registers as extra corners; above it, a diamond flattens
#: into a triangle.
EPSILON_FRAC = 0.02

# Thresholds below are the midpoints between class medians measured on the **calibration half**
# of the Phase 2.2.4 sample - 25 hdBPMN diagrams, 404 annotated regions - never on the half the
# agreement report scores. Real ink is nothing like an ideal render, which is why the first
# version of this module, calibrated on drawn-in-code shapes, reached kappa 0.05:
#
#   class          v    circularity  extent  aspect
#   circle         8.5     0.70       0.72    1.03
#   diamond        8.0     0.50       0.49    0.98
#   rectangle      4.0     0.35       0.88    3.00
#   rounded-rect   7.0     0.39       0.78    1.83
#
# The lesson worth keeping: a hand-drawn circle has circularity 0.70, not the 0.89 of a
# rendered one, because the filled boundary of a pen stroke is ragged.

#: Below this a shape leaves half its bounding box empty; only a diamond does that.
DIAMOND_EXTENT = 0.58

#: Above this a shape essentially fills its box: the rectangle family.
RECT_EXTENT = 0.83

#: Circle and ellipse sit well above the rectangle family on circularity (0.70 vs 0.35-0.39).
ROUND_CIRCULARITY = 0.55


def classify_contour(contour: np.ndarray) -> tuple[str, float]:
    """Return (shape, confidence) for one contour, using the frozen shape vocabulary."""
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri <= 0 or area <= 0:
        return "freeform", 0.0

    approx = cv2.approxPolyDP(contour, EPSILON_FRAC * peri, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / h if h else 1.0
    extent = area / float(w * h) if w and h else 0.0
    # 4*pi*A/P^2 is 1 for a circle and drops as the outline gains corners or wobble.
    circularity = 4 * np.pi * area / (peri * peri)
    v = len(approx)

    # A diamond is the only vocabulary shape that leaves half its bounding box empty.
    if extent < DIAMOND_EXTENT:
        return "diamond", 0.6
    # Fills its box: the rectangle family. Four corners means square ones, more means rounded -
    # which is the only signal there is for that distinction, and it is a weak one.
    if extent >= RECT_EXTENT:
        return ("rectangle", 0.6) if v <= 5 else ("rounded-rect", 0.45)
    # Four clean corners but a wasted box: slanted sides.
    if v == 4:
        return "parallelogram", 0.45
    if circularity >= ROUND_CIRCULARITY:
        return ("circle", 0.6) if 0.75 <= aspect <= 1.33 else ("ellipse", 0.55)
    return "rounded-rect", 0.4


def classify_mask(mask: np.ndarray) -> tuple[str, float]:
    """Shape of the largest external contour in a binary mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return "freeform", 0.0
    return classify_contour(max(contours, key=cv2.contourArea))


def classify_crop(gray: np.ndarray) -> tuple[str, float]:
    """Shape of the ink in a grayscale crop, binarised by Otsu.

    Used by the geometric second annotator in Phase 2.2.4, which sees only the pixels inside
    an annotated box and must decide what was drawn there without being told.

    The **fill** step is what makes this work at all. A drawn shape is an outline, not a solid,
    so its external contour traces the ragged outside of a pen stroke and wraps whatever text
    sits inside it. Measured on 820 hdBPMN regions, analysing that contour directly called 327
    of them `ellipse` and reached kappa 0.05 against the annotation. Filling the largest
    external contour into a solid region first - which discards the interior text and collapses
    the two sides of the stroke into one boundary - makes the same measurements comparable to
    the ideal shapes the thresholds were calibrated on.
    """
    if gray.size == 0:
        return "freeform", 0.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Close gaps so a shaky outline is one contour rather than a string of arcs. The kernel
    # scales with the crop: a fixed 3px kernel closes nothing on a 600px photograph of a box.
    span = max(gray.shape)
    k = max(3, (span // 60) | 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "freeform", 0.0
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    # A shape should occupy most of its own box; a stray stroke that happens to be the biggest
    # contour will not, and calling that a rectangle would be worse than admitting ignorance.
    if filled.sum() / 255 < 0.15 * filled.size:
        return "freeform", 0.2
    return classify_mask(filled)

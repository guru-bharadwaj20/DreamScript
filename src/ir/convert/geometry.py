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


def _corner_positions(approx: np.ndarray, x: float, y: float, w: float, h: float) -> float:
    """Mean distance of the polygon's vertices from the bounding-box corners, normalised.

    A rectangle's vertices sit *at* the bbox corners; a diamond's sit at the edge midpoints,
    which is the furthest a convex quadrilateral's vertices can be from them. One number
    separates the two without needing angles.
    """
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    pts = approx.reshape(-1, 2).astype(np.float64)
    diag = float(np.hypot(w, h)) or 1.0
    return float(np.mean([np.min(np.linalg.norm(corners - p, axis=1)) for p in pts]) / diag)


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

    if v == 3:
        return "diamond", 0.35  # a diamond that lost a corner to the approximation
    if v == 4:
        # Vertices at the bbox corners means a rectangle; at the edge midpoints, a diamond.
        if _corner_positions(approx, x, y, w, h) > 0.18:
            return "diamond", 0.7
        # A parallelogram wastes its bounding box; a rectangle fills it. Measured extents on
        # ideal renders: rectangle 0.99, parallelogram 0.76, so 0.85 sits between the two.
        if extent < 0.85:
            return "parallelogram", 0.5
        return "rectangle", 0.7

    # Five or more vertices: a curve, or a many-sided polygon. Thresholds are the midpoints
    # between measured values on ideal renders rather than round numbers -
    # circularity: circle 0.89, octagon 0.81, rounded-rect 0.82, ellipse 0.70;
    # extent:     circle 0.77, octagon 0.70, rounded-rect 0.96, ellipse 0.78.
    if extent > 0.90:
        return "rounded-rect", 0.5  # only a straight-sided shape fills its box this well
    if not 0.75 <= aspect <= 1.33:
        return "ellipse", 0.55
    if circularity >= 0.85:
        return "circle", 0.6
    if circularity >= 0.60:
        # Circle and octagon are 0.89 against 0.81 on *perfect* renders; hand-drawn examples
        # of the two overlap outright. This is the least trustworthy branch here, which is
        # why it returns the lowest confidence in the module.
        return "octagon", 0.35
    return "freeform", 0.3


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
    """
    if gray.size == 0:
        return "freeform", 0.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Close small gaps so a shaky outline is one contour rather than several arcs.
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return classify_mask(binary)

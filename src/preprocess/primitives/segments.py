"""Phase 3.2.4 - line segment detection.

Straight runs are the raw material for three later things: the sides of a polygon, the shaft of
an arrow, and the two short converging strokes that make an arrowhead. This module finds them
and says nothing about what they belong to.

Two detectors, because they fail differently and the difference is worth having:

* **LSD** (Line Segment Detector) works on the *grayscale* image using gradient orientation. It
  finds many short segments with sub-pixel endpoints and does not need a threshold, which makes
  it good at the faint, broken strokes a binarizer loses.
* **Probabilistic Hough** works on the *binary* mask by voting. It finds fewer, longer segments
  and joins collinear pieces across gaps, which makes it good at a box edge broken by a
  crossing arrow.

`detect` runs one; `detect_both` runs both and merges, which is what the arrowhead detector in
3.2.6 uses - it needs the short segments LSD finds and the long ones it does not.

Segments are returned in a consistent orientation (left-to-right, then top-to-bottom) so that
two runs over the same image produce identical output and angles can be compared directly.

    python -m src.preprocess.primitives.segments <image>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Segments shorter than this share of the image's long side are texture, not structure.
MIN_LENGTH_FRAC = 0.01

#: Hough's gap tolerance and vote threshold, as fractions of the same length.
HOUGH_GAP_FRAC = 0.01
HOUGH_VOTES_FRAC = 0.5


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    source: str = "lsd"

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle(self) -> float:
        """Direction in degrees, folded to [0, 180): a segment has no head or tail."""
        return float(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)) % 180.0)

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "length": round(self.length, 2),
            "angle": round(self.angle, 1),
            "source": self.source,
        }


def _orient(x1, y1, x2, y2) -> tuple[float, float, float, float]:
    """Always store the left-most endpoint first, so identical geometry compares equal."""
    if (x1, y1) > (x2, y2):
        return float(x2), float(y2), float(x1), float(y1)
    return float(x1), float(y1), float(x2), float(y2)


def detect_lsd(gray: np.ndarray, min_length_frac: float = MIN_LENGTH_FRAC) -> list[Segment]:
    detector = cv2.createLineSegmentDetector()
    found = detector.detect(gray.astype(np.uint8))[0]
    if found is None:
        return []
    minimum = min_length_frac * max(gray.shape)
    out = []
    for x1, y1, x2, y2 in found.reshape(-1, 4):
        segment = Segment(*_orient(x1, y1, x2, y2), source="lsd")
        if segment.length >= minimum:
            out.append(segment)
    return out


def detect_hough(mask: np.ndarray, min_length_frac: float = MIN_LENGTH_FRAC) -> list[Segment]:
    long_side = max(mask.shape)
    minimum = int(min_length_frac * long_side)
    found = cv2.HoughLinesP(
        mask.astype(np.uint8) * 255,
        rho=1,
        theta=np.pi / 360,
        threshold=max(20, int(HOUGH_VOTES_FRAC * minimum)),
        minLineLength=max(8, minimum),
        maxLineGap=max(2, int(HOUGH_GAP_FRAC * long_side)),
    )
    if found is None:
        return []
    return [Segment(*_orient(*row), source="hough") for row in found.reshape(-1, 4)]


def detect(
    image: np.ndarray, *, method: str = "lsd", min_length_frac: float = MIN_LENGTH_FRAC
) -> list[Segment]:
    """Segments from a grayscale image (`lsd`) or a boolean mask (`hough`)."""
    if method == "lsd":
        gray = image if image.dtype == np.uint8 else (~image.astype(bool)).astype(np.uint8) * 255
        return sort(detect_lsd(gray, min_length_frac))
    if method == "hough":
        mask = image.astype(bool) if image.dtype == bool else image < 128
        return sort(detect_hough(mask, min_length_frac))
    raise ValueError(f"unknown method {method!r}; expected 'lsd' or 'hough'")


def detect_both(gray: np.ndarray, mask: np.ndarray) -> list[Segment]:
    return sort(detect_lsd(gray) + detect_hough(mask))


def sort(segments: list[Segment]) -> list[Segment]:
    return sorted(segments, key=lambda s: (round(s.x1, 3), round(s.y1, 3), round(s.x2, 3)))


def angle_histogram(segments: list[Segment], bins: int = 18) -> np.ndarray:
    """Length-weighted distribution of directions, 10 degrees per bin by default.

    Bins are **centred on the axes**, not started on them. With the obvious binning of [0, 180)
    into eighteen, both directions that matter fall on a boundary: horizontal at 0/180 splits
    between the first bin and the last, and vertical at 90 splits between bins 8 and 9 - a
    rectangle then reported 0.32 + 0.32 horizontal and 0.27 + 0.09 vertical, which is four
    peaks where there are two. Shifting the bin edges by half a bin puts horizontal at the
    centre of bin 0 and vertical at the centre of bin `bins // 2`.

    The histogram is still circular - bin 0 and the last bin remain neighbours - so a consumer
    looking for a peak should wrap, which is what `dominant_direction` does.
    """
    if not segments:
        return np.zeros(bins)
    half_bin = 180.0 / (2 * bins)
    angles = np.array([(s.angle + half_bin) % 180.0 for s in segments])
    weights = np.array([s.length for s in segments])
    counts, _ = np.histogram(angles, bins=bins, range=(0, 180), weights=weights)
    total = counts.sum()
    return counts / total if total else counts


def dominant_direction(segments: list[Segment], bins: int = 18) -> tuple[int, float]:
    """(bin index, share) of the strongest direction, treating the histogram as circular."""
    histogram = angle_histogram(segments, bins)
    if not histogram.any():
        return 0, 0.0
    wrapped = histogram + np.roll(histogram, 1) + np.roll(histogram, -1)
    index = int(np.argmax(wrapped))
    return index, float(wrapped[index])


def axis_aligned_fraction(segments: list[Segment], tolerance: float = 12.0) -> float:
    """Share of segment length within `tolerance` of horizontal or vertical."""
    if not segments:
        return 0.0
    total = sum(s.length for s in segments)
    aligned = sum(
        s.length
        for s in segments
        if min(s.angle, abs(s.angle - 90.0), abs(s.angle - 180.0)) <= tolerance
    )
    return aligned / total if total else 0.0


def summarise(segments: list[Segment]) -> dict:
    if not segments:
        return {"segments": 0}
    lengths = np.array([s.length for s in segments])
    return {
        "segments": len(segments),
        "total_length": round(float(lengths.sum()), 1),
        "median_length": round(float(np.median(lengths)), 1),
        "longest": round(float(lengths.max()), 1),
        "axis_aligned_fraction": round(axis_aligned_fraction(segments), 3),
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
    scale = 1400 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = denoise(suppress(binarize(median(gray))))

    for name, segments in (
        ("lsd", detect_lsd(gray)),
        ("hough", detect_hough(mask)),
    ):
        print(f"{name}: {json.dumps(summarise(segments))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

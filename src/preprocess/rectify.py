"""Phase 3.1.3 - perspective rectification.

Given the four page corners from 3.1.2, warp the page to a top-down view. The homography is
the easy part; the two decisions worth explaining are what size to warp *to*, and what to do
when there is no page.

**Output size.** Using the source image's dimensions would squash a page photographed at an
angle. Instead the target is built from the quadrilateral itself: width from the mean length of
its two horizontal sides, height from its two vertical ones. That preserves the page's real
aspect ratio to within the perspective foreshortening, and it never invents resolution the
photo does not have.

**No page.** `rectify` returns the image unchanged when `quad` is None, and says so in the
returned record. A pipeline that silently warped by an identity homography would look the same
but would hide the fact that nothing was corrected.

Coordinates move. Anything holding boxes in the original frame - the IR, a detector's output -
must be mapped through the same homography, which is what `map_points` is for.

    python -m src.preprocess.rectify --limit 12
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.preprocess import page
from src.preprocess.exif import load
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p3_rectification.png"

#: Never warp to something enormous, whatever the corner arithmetic says.
MAX_SIDE = 4000


@dataclass
class Rectified:
    image: np.ndarray
    homography: np.ndarray | None
    quad: np.ndarray | None
    applied: bool
    size: tuple[int, int] = field(default=(0, 0))

    def map_points(self, points: np.ndarray) -> np.ndarray:
        """Move points from the original frame into the rectified one."""
        if self.homography is None:
            return np.asarray(points, dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def map_box(self, box: list[float]) -> list[float]:
        """An axis-aligned box through a homography is not axis-aligned; this returns the
        bounding box of the warped corners, which is the closest honest answer in IR terms."""
        x, y, w, h = box
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        moved = self.map_points(corners)
        lo = moved.min(axis=0)
        hi = moved.max(axis=0)
        return [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]


def target_size(quad: np.ndarray) -> tuple[int, int]:
    """Output (width, height) implied by the quadrilateral's own side lengths."""
    tl, tr, br, bl = quad.reshape(4, 2).astype(np.float64)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    scale = min(1.0, MAX_SIDE / max(width, height, 1.0))
    return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)


def rectify(image: np.ndarray, quad: np.ndarray | None = None) -> Rectified:
    """Warp the page to a top-down view, or pass the image through untouched."""
    if quad is None:
        quad = page.detect(image)
    if quad is None:
        h, w = image.shape[:2]
        return Rectified(image, None, None, applied=False, size=(w, h))

    ordered = page.order_corners(quad)
    width, height = target_size(ordered)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    homography = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    warped = cv2.warpPerspective(
        image, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return Rectified(warped, homography, ordered, applied=True, size=(width, height))


def qa_grid(limit: int = 12) -> object:
    """A before/after grid over corpus photos, for the visual check plan.md 3.1.3 asks for."""
    from src.ir.model import SUFFIX, Diagram

    ir_dir = ROOT / "data" / "processed" / "ir" / "hdbpmn"
    pairs: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for path in sorted(ir_dir.glob(f"*{SUFFIX}")):
        if len(pairs) >= limit:
            break
        diagram = Diagram.load(path)
        image_path = ROOT / diagram.meta["image"]
        if not image_path.is_file():
            continue
        image = load(image_path)
        result = rectify(image)
        # Show the applied ones first: an unchanged pair proves nothing.
        pairs.append((image, result.image, result.applied))
    pairs.sort(key=lambda p: not p[2])
    pairs = pairs[:limit]
    if not pairs:
        return None

    cell = 260
    columns = 4
    rows = (len(pairs) + columns - 1) // columns
    sheet = np.full((rows * (cell + 26), columns * cell * 2, 3), 255, np.uint8)
    for i, (before, after, applied) in enumerate(pairs):
        for j, img in enumerate((before, after)):
            scale = (cell - 8) / max(img.shape[:2])
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            y = (i // columns) * (cell + 26) + 22
            x = ((i % columns) * 2 + j) * cell + 4
            sheet[y : y + small.shape[0], x : x + small.shape[1]] = small
        label = "rectified" if applied else "no page - passed through"
        cv2.putText(
            sheet,
            label,
            ((i % columns) * 2 * cell + 6, (i // columns) * (cell + 26) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 180) if applied else (120, 120, 120),
            1,
        )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), sheet)
    return FIGURE


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args(argv)

    figure = qa_grid(args.limit)
    if figure is None:
        print("no images to rectify; run the hdBPMN converter first", file=sys.stderr)
        return 1
    print(f"wrote {figure.relative_to(ROOT)}")

    checks = {"qa_grid_written": figure.is_file()}
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

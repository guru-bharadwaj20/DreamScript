"""Phase 3.1.4 - illumination correction.

A photograph of paper is rarely lit evenly. A window on one side, a hand casting a shadow, a
phone's own flash: all of them put a slow brightness ramp across the page, and a global
threshold then eats the strokes on the dark side while keeping paper texture on the bright
side. Correcting the ramp before binarising is what makes 3.1.5 possible at all.

Two steps, in this order:

1. **Background flattening.** Estimate the paper's brightness with a large morphological
   *closing* - a maximum filter, which erases anything darker and smaller than its kernel, and
   ink is exactly that - then divide the image by that estimate. This removes the ramp without
   knowing anything about where the light came from.
2. **CLAHE.** Contrast-limited local equalisation, to restore bite in areas that were flat.
   Contrast-*limited*, because plain histogram equalisation on flattened paper amplifies its
   texture into fake strokes.

The order matters. CLAHE first would equalise the shadow into place and make it permanent.

Success is measured with the same statistic Phase 1 used to *label* these images as shadowed -
the spread of mean brightness across the four quadrants - so the number here is directly
comparable to the number that put them in the `shadow` bucket.

    python -m src.preprocess.illumination
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from src.utils.config import ROOT

#: Kernel for the background estimate, as a fraction of the image's short side. It must be
#: comfortably larger than the widest stroke and smaller than the lighting variation it is
#: meant to capture; 1/16 of the page clears both on this corpus.
BACKGROUND_FRAC = 1 / 16

DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID = 8

#: Phase 1's threshold for calling an image shadowed (src/ingest/chaos_builder.py).
SHADOW_GRADIENT = 28.0


def background(gray: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """The paper, without the ink: a morphological closing with a large kernel."""
    if kernel_size is None:
        kernel_size = max(15, int(BACKGROUND_FRAC * min(gray.shape)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


def flatten(gray: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """Divide out the lighting. Ink keeps its contrast; the ramp disappears."""
    paper = background(gray, kernel_size).astype(np.float32)
    ratio = gray.astype(np.float32) / np.maximum(paper, 1.0)
    return np.clip(ratio * 255.0, 0, 255).astype(np.uint8)


def clahe(
    gray: np.ndarray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid: int = DEFAULT_TILE_GRID,
) -> np.ndarray:
    engine = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return engine.apply(gray)


def correct(
    image: np.ndarray,
    *,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid: int = DEFAULT_TILE_GRID,
    kernel_size: int | None = None,
) -> np.ndarray:
    """Flatten the lighting, then restore local contrast. Returns grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return clahe(flatten(gray, kernel_size), clip_limit, tile_grid)


def quadrant_spread(gray: np.ndarray) -> float:
    """Phase 1's shadow statistic: max minus min of the four quadrant means."""
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    halves = [small[:128, :128], small[:128, 128:], small[128:, :128], small[128:, 128:]]
    means = [float(q.mean()) for q in halves]
    return max(means) - min(means)


def evaluate(condition: str = "shadow") -> dict:
    """Measure the correction on the corpus images Phase 1 measured as shadowed."""
    provenance = ROOT / "data" / "raw" / "chaos" / "_sourced_provenance.json"
    if not provenance.is_file():
        return {"images": 0}
    records = json.loads(provenance.read_text(encoding="utf-8"))
    before: list[float] = []
    after: list[float] = []
    for record in records:
        if record.get("condition") != condition:
            continue
        path = ROOT / record["output"]
        if not path.is_file():
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        before.append(quadrant_spread(gray))
        after.append(quadrant_spread(correct(gray)))
    if not before:
        return {"images": 0}
    return {
        "images": len(before),
        "condition": condition,
        "spread_before_median": round(float(np.median(before)), 1),
        "spread_after_median": round(float(np.median(after)), 1),
        "spread_before_p90": round(float(np.percentile(before, 90)), 1),
        "spread_after_p90": round(float(np.percentile(after, 90)), 1),
        "still_above_shadow_threshold": int(sum(v > SHADOW_GRADIENT for v in after)),
        "were_above_shadow_threshold": int(sum(v > SHADOW_GRADIENT for v in before)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", default="shadow")
    args = ap.parse_args(argv)

    result = evaluate(args.condition)
    if not result["images"]:
        print("no chaos corpus images to measure", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))

    checks = {
        "spread_reduced": result["spread_after_median"] < result["spread_before_median"],
        "most_shadows_cleared": (
            result["still_above_shadow_threshold"]
            <= 0.25 * max(result["were_above_shadow_threshold"], 1)
        ),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

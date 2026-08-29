"""Phase 3.1.5 / 3.3 - an evaluation set with exact stroke ground truth.

Measuring a binarizer needs to know which pixels are ink, and nobody in this project has traced
30 photographs pixel by pixel. There is a better source: the FA database records the pen
*trajectory*. Rasterising those polylines gives a stroke mask that is exactly right by
construction - not an estimate, not a careful threshold that would make the comparison
circular.

So each evaluation item is built like this:

    real human pen strokes  ->  clean render        (this is the ground-truth mask)
                            ->  photometric damage  (this is what the binarizer sees)

The damage comes from the Phase 1.3.6 augmentation policy - shadow, glare, brightness, blur,
JPEG, paper texture, stain - each of which was chosen there to stand for a named capture risk.

**What this does and does not measure.** It measures whether a binarizer recovers known ink
through known degradation, and it does so on real handwriting. It does *not* measure real
photographs: a rendered stroke has clean edges, and paper texture drawn by a program is not
paper. A number from this harness is an upper bound on real-photo performance, and the failure
gallery in 3.3.3 is where actual photographs are looked at.

Geometric transforms are deliberately excluded. Rotating the image would require rotating the
mask too, and any interpolation mismatch between the two would show up as a fake IoU loss that
has nothing to do with binarization. 3.3.2 sweeps rotation separately, warping both together.

    python -m src.preprocess.evalset --count 30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ingest import augment
from src.ir.convert.fa import CANVAS, MARGIN, _traces
from src.utils.config import ROOT
from src.utils.parallel import pmap

FA_DIR = ROOT / "data" / "raw" / "fa_bresler" / "FA_1.0"
OUT = ROOT / "data" / "interim" / "preproc_eval"

#: Rendered stroke width in pixels. Wide enough to survive JPEG and blur, narrow enough that
#: two nearby strokes do not merge - which is what a real pen does on this size of page.
STROKE_WIDTH = 3

#: Photometric damage only; see the module docstring for why geometry is excluded here.
DAMAGE = ["shadow", "glare", "brightness", "blur", "jpeg", "paper_texture", "stain"]


@dataclass
class Item:
    name: str
    photo: np.ndarray  # grayscale, damaged
    mask: np.ndarray  # bool, ground-truth ink
    damage: list[str]


def render_strokes(path: Path, size: int = CANVAS, width: int = STROKE_WIDTH):
    """(clean grayscale render, boolean ink mask) from one InkML file.

    The mask is drawn from the same polylines as the render, with the same width and without
    antialiasing, so it is the render's ink by construction rather than a threshold of it.
    """
    import xml.etree.ElementTree as ET

    try:
        traces = _traces(ET.parse(path).getroot())
    except ET.ParseError:
        return None
    if not traces:
        return None

    points = np.vstack(list(traces.values()))
    lo = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lo, 1e-6)

    render = np.full((size, size), 245, np.uint8)
    mask = np.zeros((size, size), np.uint8)
    for polyline in traces.values():
        normed = (polyline - lo) / span
        xy = ((normed * (1 - 2 * MARGIN) + MARGIN) * size).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(render, [xy], False, 35, width, cv2.LINE_AA)
        cv2.polylines(mask, [xy], False, 255, width, cv2.LINE_8)
    return render, mask > 0


def damage(gray: np.ndarray, rng: np.random.Generator, how: list[str]) -> np.ndarray:
    """Apply named photometric transforms from the Phase 1.3.6 policy, in order."""
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for name in how:
        image = augment.TRANSFORMS[name](image, rng)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def build(count: int = 30, seed: int = 42) -> list[Item]:
    """A deterministic evaluation set: `count` automata, each with its own damage recipe."""
    files = sorted(FA_DIR.glob("*.inkml"))
    if not files:
        return []
    rng = np.random.default_rng(seed)
    # Spread across writers rather than taking the first N files, which would be one hand.
    stride = max(1, len(files) // max(count, 1))
    chosen = files[::stride][: count * 2]

    # Draw every recipe up front from the single seeded generator, so the damage each file
    # receives does not depend on which worker picked it up or in what order.
    recipes = [
        (
            path,
            list(rng.choice(DAMAGE, size=int(rng.integers(1, 4)), replace=False)),
            int(rng.integers(0, 2**31)),
        )
        for path in chosen
    ]

    def _one(recipe):
        path, how, item_seed = recipe
        rendered = render_strokes(path)
        if rendered is None:
            return None
        clean, mask = rendered
        return Item(path.stem, damage(clean, np.random.default_rng(item_seed), how), mask, how)

    built = pmap(_one, recipes, prefer="threads")
    return [item for item in built if item is not None][:count]


def f1(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Pixel precision, recall, F1 and IoU of a predicted ink mask."""
    p = predicted.astype(bool)
    t = truth.astype(bool)
    tp = float(np.count_nonzero(p & t))
    fp = float(np.count_nonzero(p & ~t))
    fn = float(np.count_nonzero(~p & t))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = precision + recall
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / denominator if denominator else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--save", action="store_true", help="write the items to data/interim")
    args = ap.parse_args(argv)

    items = build(args.count)
    if not items:
        print(f"no InkML under {FA_DIR}", file=sys.stderr)
        return 1

    ink = [float(i.mask.mean()) for i in items]
    print(f"{len(items)} items, mean ink coverage {np.mean(ink):.3%}")
    from collections import Counter

    counts = Counter(d for i in items for d in i.damage)
    for name, count in counts.most_common():
        print(f"  {count:3d}  {name}")

    if args.save:
        OUT.mkdir(parents=True, exist_ok=True)
        for item in items:
            cv2.imwrite(str(OUT / f"{item.name}_photo.png"), item.photo)
            cv2.imwrite(str(OUT / f"{item.name}_mask.png"), item.mask.astype(np.uint8) * 255)
        print(f"wrote {2 * len(items)} files to {OUT.relative_to(ROOT)}")

    checks = {
        "items_built": len(items) >= min(args.count, 20),
        "every_item_has_ink": all(i.mask.any() for i in items),
        "every_item_is_damaged": all(i.damage for i in items),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 9.2 - the shape-crop corpus the scratch CNN is trained on.

    python -m src.detect.crops                  # build data/processed/shape_crops
    python -m src.detect.crops --summary

9.1 detects boxes on a page. 9.2 asks a different and much smaller question - **given a crop
that is already known to contain one shape, which shape is it** - because that is the question
a hand-built conv/pool/FC stack can actually be exercised on, and because the layer arithmetic
of 9.2.2 and 9.2.3 needs a fixed input size to be arithmetic about.

## The crop, and the two decisions in it

**64x64 greyscale.** Small enough that a scratch network trains in minutes on 40,000 crops and
that its receptive field can plausibly cover the whole input by the last layer - which is the
property 9.2.3 exists to check - and large enough that 7.4.1's descriptors are still visible in
the pixels. The size is a constant here rather than a flag, because every parameter count in
`docs/cnn_math.md` is computed against it.

**A 12% context margin.** A box cropped exactly to its annotation loses the one cue that
separates some of these classes: a `double-circle` is an outer ring plus an inner ring, and a
tight crop of the inner ring is a `circle`. The margin is padding, not a different box - the
label still refers to the annotated shape - and it is applied before the resize so the aspect
distortion is the same for every class.

## What is deliberately not done

**No aspect-preserving letterbox.** The crop is resized to a square, which distorts a 3.26-
aspect BPMN task into something squarer, and that is the choice 7.4.1 would have argued against
since `rect_aspect` was its rectangle detector. It is taken anyway and recorded, because a
letterbox pads with a constant that the first conv layer will happily learn to find, and a
network that classifies shapes by counting grey border pixels is worse than one that sees a
squashed rectangle. 9.2.5's ablations sit on top of this decision, not underneath it.

**`arrowhead` is excluded.** It is 9.1.2's derived class, its box is a size convention, and at
64x64 a resized 11-pixel head is an interpolation artefact rather than a shape. 9.2 is about
shape classification and the head is not a shape.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.detect.classes import boxes_for, pages
from src.detect.dataset import SPLITS, fa_splits, image_for, manifest_splits
from src.utils.config import ROOT

OUT = ROOT / "data" / "processed" / "shape_crops"

#: Fixed, not a flag: docs/cnn_math.md is arithmetic about this number.
SIZE = 64

#: Fraction of the box's own width and height added on each side before the resize.
MARGIN = 0.12

#: 9.1.2's eight minus the derived one. Frozen here so 9.2's confusion matrix and 9.1's mAP
#: table cannot silently come to be about different label sets.
SHAPE_CLASSES: tuple[str, ...] = (
    "rectangle",
    "rounded-rect",
    "diamond",
    "circle",
    "double-circle",
    "parallelogram",
    "freeform",
)

SHAPE_INDEX = {name: i for i, name in enumerate(SHAPE_CLASSES)}

#: Smallest box worth cropping, in native pixels on its own page. Under this the crop is more
#: interpolation than ink and the label is a claim about pixels that are not there.
MIN_SIDE = 12


def crop_one(image: np.ndarray, bbox, size: int = SIZE, margin: float = MARGIN):
    """One padded, squared, greyscale crop - or None when the box has no usable pixels."""
    import cv2

    x, y, w, h = bbox
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    height, width = image.shape[:2]
    px, py = w * margin, h * margin
    x0, y0 = int(max(0, round(x - px))), int(max(0, round(y - py)))
    x1, y1 = int(min(width, round(x + w + px))), int(min(height, round(y + h + py)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def build(out: Path = OUT, sources=("hdbpmn", "fa_bresler", "flowchartseg"), limit=None) -> dict:
    """Write `images/<split>/<class>/<id>.png`, split-consistent with 9.1.2's export.

    The splits are inherited from the same place 9.1.2 inherited them, so **a page whose boxes
    trained the detector cannot supply a crop that validates the classifier** - without that,
    9.2.7's transfer-versus-scratch comparison would be measuring memorisation.
    """
    import cv2

    inherited = manifest_splits()
    if out.exists():
        shutil.rmtree(out)

    counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    dropped = Counter()
    written = 0
    for source in sources:
        loaded = [json.loads(p.read_text(encoding="utf-8")) for p in pages(source)]
        assigned = fa_splits(loaded) if source == "fa_bresler" else {}
        for diagram in loaded:
            key = f"{source}:{diagram['id']}"
            split = assigned.get(key, inherited.get(key, "train"))
            path = image_for(diagram)
            if path is None:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            for index, box in enumerate(boxes_for(diagram, arrowheads=False)):
                name = box["cls"]
                if name not in SHAPE_INDEX:
                    dropped[name] += 1
                    continue
                patch = crop_one(image, box["bbox"])
                if patch is None:
                    dropped["too_small"] += 1
                    continue
                folder = out / "images" / split / name
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{source}__{diagram['id']}__{index}.png"), patch)
                counts[split][name] += 1
                written += 1
                if limit and written >= limit:
                    break
            if limit and written >= limit:
                break

    meta = {
        "size": SIZE,
        "margin": MARGIN,
        "classes": list(SHAPE_CLASSES),
        "per_split": {s: dict(c) for s, c in counts.items()},
        "totals": dict(sum(counts.values(), Counter())),
        "crops": written,
        "dropped": dict(dropped),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def load_split(
    split: str, out: Path = OUT, sources: tuple[str, ...] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """`(N, 1, 64, 64)` float32 in [0, 1] and `(N,)` int64 labels.

    `sources` filters on the crop's originating dataset, which every filename carries as its
    first `__`-delimited field. That filter is what lets 9.2.1 quote a hand-drawn-only figure:
    the pooled validation split is **half computer-rendered flowchartseg crops**, whose
    rectangles have exact boundaries, so a pooled accuracy is not comparable with 7.4.8's
    hdbpmn-only descriptor ceiling and would flatter the network besides.
    """
    import cv2

    xs, ys = [], []
    root = out / "images" / split
    for name in SHAPE_CLASSES:
        folder = root / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.png")):
            if sources and path.name.split("__", 1)[0] not in sources:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            xs.append(image)
            ys.append(SHAPE_INDEX[name])
    if not xs:
        return np.zeros((0, 1, SIZE, SIZE), np.float32), np.zeros((0,), np.int64)
    array = np.stack(xs).astype(np.float32) / 255.0
    return array[:, None, :, :], np.asarray(ys, dtype=np.int64)


def summary(out: Path = OUT) -> dict:
    meta = out / "meta.json"
    if not meta.is_file():
        raise SystemExit(f"not built: {meta} (run `python -m src.detect.crops`)")
    return json.loads(meta.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    result = summary(args.out) if args.summary else build(args.out, limit=args.limit)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

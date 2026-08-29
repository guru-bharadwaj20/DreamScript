"""Phase 2.2.2 - the flowchart segmentation dataset to DreamScript IR.

This source publishes an image and a node mask. That is all: no edges, no labels, no element
types. So this converter produces the weakest IR of the four, and the interesting question is
how to be useful without overclaiming.

**These images are computer-rendered, not hand-drawn.** Phase 1 recorded them as hand-drawn
on the strength of the dataset card; inspecting them here showed sixteen random samples to be
typeset flowcharts with lorem-ipsum labels, without exception. `docs/data_cards/flowchart_fc.md`
carries the correction. It matters for how the output is used: these are clean synthetic pages,
so a detector trained on them alone will not transfer to a photograph of paper.

What it does: every connected component of the mask becomes a node with a real bounding box
(that part is ground truth), a shape *guessed* from the component's contour by
`convert.geometry` (confidence well below 1.0, `shape_basis: "geometry"`), a role of `unknown`,
and no text. No edges are emitted at all, because the mask does not record any and inventing
them from proximity is exactly the Phase 10 problem this data would then be used to evaluate.

The value of these 1,319 diagrams is therefore detection training data - where are the nodes -
and nothing else. `meta.notes` says so on every file, so no later phase can mistake an empty
`edges` array for a diagram that genuinely has no connections.

    python -m src.ir.convert.flowchartseg --limit 50
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.ir.convert.geometry import classify_mask
from src.ir.model import SUFFIX, Diagram, Node
from src.utils.config import ROOT

SOURCE = "flowchartseg"
RAW = ROOT / "data" / "raw" / "flowchartseg" / "data"
OUT = ROOT / "data" / "processed" / "ir" / SOURCE
IMAGES = ROOT / "data" / "processed" / "flowchartseg_images"

#: Components smaller than this share of the page are mask speckle rather than drawn boxes.
MIN_AREA_FRAC = 2e-4


def _decode(cell: dict) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(cell["bytes"])))


def convert(row: pd.Series, index: int, *, write_image: bool = True) -> Diagram:
    image = _decode(row["image"])
    mask_rgb = _decode(row["annotation"])
    height, width = mask_rgb.shape[:2]

    mask = mask_rgb if mask_rgb.ndim == 2 else mask_rgb[..., 0]
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    diagram_id = f"fcseg_{index:05d}"
    min_area = MIN_AREA_FRAC * width * height
    nodes: list[Node] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        shape, confidence = classify_mask(labels[y : y + h, x : x + w] == label)
        nodes.append(
            Node(
                id=f"n{label}",
                shape=shape,
                bbox=[float(x), float(y), float(w), float(h)],
                text="",
                semantic_role="unknown",
                confidence=float(confidence),
                source_id=str(label),
                attrs={"shape_basis": "geometry", "mask_area": int(area)},
            )
        )

    image_path = ""
    if write_image:
        IMAGES.mkdir(parents=True, exist_ok=True)
        out = IMAGES / f"{diagram_id}.png"
        if not out.exists():
            Image.fromarray(image).save(out)
        image_path = str(out.relative_to(ROOT)).replace("\\", "/")

    return Diagram(
        id=diagram_id,
        diagram_type="flowchart",
        nodes=nodes,
        edges=[],
        meta={
            "source": SOURCE,
            "geometry": "annotated",
            "image": image_path,
            "image_size": [int(width), int(height)],
            "producer": "src.ir.convert.flowchartseg",
            "notes": (
                "Computer-rendered page, not a hand-drawn one. "
                "Boxes are ground truth from the published node mask. Shapes are guessed from "
                "each component's contour and carry that guess's confidence. Roles are unknown "
                "and there is no text. The empty `edges` array means the source records no "
                "connections, NOT that this diagram has none - use these files for detection "
                "only, never to evaluate graph assembly."
            ),
        },
    )


def frames() -> list[Path]:
    return sorted(RAW.glob("*.parquet"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-images", action="store_true", help="skip extracting the page images")
    args = ap.parse_args(argv)

    paths = frames()
    if not paths:
        print(f"no parquet under {RAW}", file=sys.stderr)
        return 1

    written = 0
    stats: dict[str, object] = {"nodes": 0, "shapes": {}, "mean_shape_confidence": 0.0}
    confidences: list[float] = []
    for path in paths:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            if written >= args.limit:
                break
            d = convert(row, written, write_image=not args.no_images)
            d.save(args.out / f"{d.id}{SUFFIX}")
            written += 1
            stats["nodes"] += len(d.nodes)
            for n in d.nodes:
                stats["shapes"][n.shape] = stats["shapes"].get(n.shape, 0) + 1
                confidences.append(n.confidence)
        if written >= args.limit:
            break

    if confidences:
        stats["mean_shape_confidence"] = round(sum(confidences) / len(confidences), 3)
    print(f"wrote {written} IR files to {args.out.relative_to(ROOT)}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())

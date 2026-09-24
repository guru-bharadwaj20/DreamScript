"""Find the container titles that CRAFT cannot see, by looking at the strip the way it is written.

    python -m src.ocr.vertical --build

A BPMN pool or lane writes its name along its leading edge, turned a quarter turn. CRAFT is a
text detector, and like most of them it is trained on writing that runs left to right; on this
corpus it misses those titles outright. The evidence is direct rather than inferred: over 40 val
pages, the ink fraction inside a container's header strip is **0.1103 where CRAFT found nothing
and 0.1005 where it found something**. The writing is equally present in both cases. Only the
detector's opinion differs.

That is why 42.6% of Participants and 28.4% of pools had no candidate line in their own strip,
fell through to cropping the bare strip, and scored 0.5978 and 0.4019 - the worst two classes in
S3 by a wide margin, and the only classes that got *worse* while everything around them improved.

## What this does

For each container, rotate its header strip a quarter turn clockwise - which is the orientation
the title was written in - run the same CRAFT detector on that, and map what it finds back into
page coordinates. A title that comes back is a tall narrow box, so `ownership.with_rotation`
already flags it for de-rotation and the rest of the pipeline needs no changes: these are simply
extra candidate lines, competing on the same terms as every other.

Nothing here reads a label or uses one. It is a detector pass over a region defined by geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.ocr import ownership as ow
from src.utils.config import ROOT

SOURCE = ROOT / "experiments" / "ocr" / "craft" / "boxes2.json"
OUT = ROOT / "experiments" / "ocr" / "craft" / "boxes3.json"

#: The strip is widened before rotating, because a title is written *against* the edge and a
#: tight strip clips its ascenders. This is candidate generation, so a margin costs nothing.
WIDEN = 1.6


def strip_of(element: dict) -> list[int]:
    x, y, w, h = (float(v) for v in element["bbox"])
    sx, sy, sw, sh = ow.header_strip([x, y, w, h])
    if w >= h:
        return [int(sx), int(sy), int(min(sw * WIDEN, w)), int(sh)]
    return [int(sx), int(sy), int(sw), int(min(sh * WIDEN, h))]


def unrotate(box: list[int], strip: list[int]) -> list[int]:
    """Map a box found in the clockwise-rotated strip back into page coordinates.

    `cv2.ROTATE_90_CLOCKWISE` sends `S[r][c]` to `R[c][H-1-r]`, so a box at `(u, v, bw, bh)` in
    `R` covers rows `v .. v+bh` and columns `H-1-u-bw .. H-1-u` of `S`.
    """
    sx, sy, sw, sh = strip
    u, v, bw, bh = box
    return [int(sx + v), int(sy + max(0, sh - 1 - u - bw)), int(bh), int(bw)]


def detect_rotated(image: np.ndarray, strip: list[int]) -> list[list[int]]:
    """CRAFT on one header strip, turned the way the title runs."""
    from src.ocr.labelcrops import reader

    sx, sy, sw, sh = strip
    patch = image[max(0, sy) : sy + sh, max(0, sx) : sx + sw]
    if patch.size == 0 or min(patch.shape[:2]) < 12:
        return []
    turned = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    rgb = cv2.cvtColor(turned, cv2.COLOR_GRAY2RGB)
    horizontal, free = reader().detect(
        rgb, text_threshold=0.3, low_text=0.2, link_threshold=0.2, canvas_size=2560, mag_ratio=2.0
    )
    found = []
    for b in horizontal[0] if horizontal else []:
        x0, x1, y0, y1 = (int(v) for v in b)
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            found.append([x0, y0, x1 - x0, y1 - y0])
    for poly in free[0] if free else []:
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            found.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return [unrotate(b, [sx, sy, patch.shape[1], patch.shape[0]]) for b in found]


def build(limit: int | None = None) -> dict:
    """Add rotated-strip detections to every page's candidate lines."""
    from src.detect.dataset import image_for
    from src.ocr.labelcrops import elements_of
    from src.parse.sequences import load_ir
    from src.preprocess.exif import load

    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    started, pages, added = time.perf_counter(), 0, 0
    for diagram in load_ir(["hdbpmn"], limit=limit):
        if diagram["id"] in cache or diagram["id"] not in base:
            continue
        path = image_for(diagram)
        if not path or not Path(path).is_file():
            continue
        image = load(Path(path), grayscale=True)
        if image is None:
            continue
        boxes = list(base[diagram["id"]]["boxes"])
        have = {tuple(b) for b in boxes}
        for element in elements_of(diagram):
            if ow.kind_of(element) != "container":
                continue
            for box in detect_rotated(image, strip_of(element)):
                if box[2] >= 4 and box[3] >= 4 and tuple(box) not in have:
                    have.add(tuple(box))
                    boxes.append(box)
                    added += 1
        cache[diagram["id"]] = {"shape": list(image.shape), "boxes": boxes}
        pages += 1
        if pages % 50 == 0:
            print(f"[vertical] {pages} pages, {added} new boxes", flush=True)
            OUT.write_text(json.dumps(cache) + "\n", encoding="utf-8")
    OUT.write_text(json.dumps(cache) + "\n", encoding="utf-8")
    return {
        "pages": pages,
        "new_boxes": added,
        "seconds": round(time.perf_counter() - started, 1),
    }


def coverage() -> dict:
    """How many container strips now hold a candidate line, before and after."""
    from src.ocr.labelcrops import elements_of, load_index
    from src.parse.sequences import load_ir

    before = json.loads(SOURCE.read_text(encoding="utf-8"))
    after = json.loads(OUT.read_text(encoding="utf-8"))
    frame = load_index()
    pages = set(frame.loc[frame["split"] == "val", "page"])
    counts = {"before": 0, "after": 0, "total": 0}
    for diagram in load_ir(["hdbpmn"], limit=None):
        if diagram["id"] not in pages:
            continue
        for element in elements_of(diagram):
            if ow.kind_of(element) != "container" or not str(element["text"]).strip():
                continue
            strip = ow.header_strip([float(v) for v in element["bbox"]])
            counts["total"] += 1
            for tag, cache in (("before", before), ("after", after)):
                lines = cache.get(diagram["id"], {}).get("boxes", [])
                if any(ow.overlap_fraction(b, strip) >= 0.6 for b in lines):
                    counts[tag] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.build:
        print(json.dumps(build(args.limit), indent=2))
        return 0
    if args.coverage:
        print(json.dumps(coverage(), indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

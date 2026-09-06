"""Crop each label from where it was actually written, not from the element that owns it.

    python -m src.ocr.labelcrops --build
    python -m src.ocr.labelcrops --summary

9.3.1 built every crop from an element's geometry: a node label is the node's box inset 10%, an
edge label is a rectangle guessed beside the polyline. That is sound for an activity, whose text
is written inside its box, and wrong for everything else - and "everything else" is a fifth of
the corpus.

## The measurement that forced this module

S3's first run scored 0.4768 CER. Split by BPMN element type it is not one number at all:

    element                crops   char%      CER    exact
    Activity                 548   16.7%   0.0706    0.743      text IS in the box
    task                     190    5.3%   0.1023    0.732      text IS in the box
    subProcessCollapsed        7    0.2%   0.0000    1.000      text IS in the box
    Event (circles)          300    8.3%   0.8210    0.023      text is OUTSIDE
    Flow (edge labels)       422    8.7%   1.0379    0.019      text is OUTSIDE
    DataObjectReference      112    2.3%   0.9773    0.071      text is OUTSIDE
    Lane / Participant/pool  316    5.5%  ~0.52-0.60            title bar, outside

**The recogniser already beats S3's 0.15 target wherever the crop contains its label.** It fails
at ~1.0 exactly where the crop cannot contain it. A BPMN event is a small circle with its label
written *underneath*; 9.3.1 crops the circle and asserts the label is "application received". No
model can read that, and thirteen of them agreeing at 0.68 in 9.3.6 is what a corpus-wide floor
looks like from the inside - not, as that row assumed, thirteen architectures sharing a
bottleneck.

## What this does instead

Find the writing first, then decide which element owns it.

    ink            adaptive threshold
    strip_rules    remove ruled paper, box outlines and connector shafts - long straight runs,
                   plus stroke components that are long and thin enough to be a connector
    erase_shapes   blank the drawn glyph of every element whose label is external, because the
                   circle is ink and would otherwise be stacked on top of the label as its
                   first "character" - measured, and it was
    raw_blobs      dilate to merge characters into words and words into a line
    stack          merge vertically stacked lines into a label block, bounded to three lines:
                   an unbounded chain walks a column of the page into one 554x527 "label"
    assign         greedy nearest-block, each block claimed once, preferring a block inside the
                   shape and then one directly below it

Coverage over 60 val pages is 88.1% of labelled elements, and the classes it exists for move
from unusable to found: Event 79.2%, DataObjectReference 77.4%, Lane 100%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.ocr.textcrops import HEIGHT, MAX_WIDTH
from src.utils.config import ROOT

OUT = ROOT / "data" / "processed" / "label_crops"
INDEX = OUT / "index.parquet"

#: Element classes whose label is written outside the drawn shape.
EXTERNAL_PREFIXES = (
    "Event",
    "startEvent",
    "endEvent",
    "messageStartEvent",
    "messageEndEvent",
    "messageIntermediateCatchEvent",
    "messageIntermediateThrowEvent",
    "timerStartEvent",
    "timerIntermediateEvent",
    "DataObjectReference",
    "dataObjectReference",
    "DataStoreReference",
    "dataStoreReference",
    "Gateway",
    "exclusiveGateway",
)
PAD = 0.12


def ink_mask(image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )


def strip_rules(mask: np.ndarray, page_w: int) -> np.ndarray:
    """Remove ruled paper, drawn outlines and connector shafts, keeping the writing."""
    long = max(40, page_w // 30)
    rect = cv2.getStructuringElement
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (long, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (1, long)))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8), 1)
    out = cv2.bitwise_and(mask, cv2.bitwise_not(lines))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    drop = np.zeros(count, bool)
    for i in range(1, count):
        _, _, w, h, area = stats[i]
        span = max(w, h)
        # Effective stroke thickness, not bounding-box fill. A long cursive word is wide, thin
        # and sparse - `area < 0.10 * w * h` called it a connector and deleted the writing,
        # which is how 'receive order by website' became a 20px crop. A real connector is one
        # stroke: its area over its length is a stroke width, a few pixels. A word is many
        # strokes and scores several times higher whatever its span.
        if span > page_w // 12 and area / span < 8.0:
            drop[i] = True
    if drop.any():
        out[drop[labels]] = 0
    return out


_READER = None


def reader():
    """CRAFT text detection, loaded once.

    The hand-rolled morphology this replaces went through four rounds of fixes and each one
    exposed the next failure: the connector filter deleted long cursive words, the stack merge
    walked a column into one box, the drawn circle became a label's first character. A detector
    trained on text finds text; a stack of thresholds finds whatever the thresholds admit. Only
    `detect` is used - the recogniser is `src.ocr.s3`.
    """
    global _READER
    if _READER is None:
        import easyocr
        import torch

        _READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _READER


def text_boxes(image: np.ndarray) -> list[list[int]]:
    """Every region of writing on the page, as `[x, y, w, h]`, per line."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    horizontal, free = reader().detect(rgb, text_threshold=0.5, low_text=0.3, link_threshold=0.3)
    out = []
    for box in horizontal[0] if horizontal else []:
        x0, x1, y0, y1 = (int(v) for v in box)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            out.append([x0, y0, x1 - x0, y1 - y0])
    for poly in free[0] if free else []:
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            out.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return out


def is_external(element_id: str) -> bool:
    return str(element_id).startswith(EXTERNAL_PREFIXES)


def erase_shapes(mask: np.ndarray, elements: list[dict]) -> np.ndarray:
    """Blank the drawn glyph of every element that carries its label outside itself."""
    out = mask.copy()
    for element in elements:
        if not is_external(element["id"]):
            continue
        x, y, w, h = (int(v) for v in element["bbox"])
        margin = int(0.12 * max(w, h))
        out[max(0, y - margin) : y + h + margin, max(0, x - margin) : x + w + margin] = 0
    return out


def raw_blobs(mask: np.ndarray) -> list[list[int]]:
    height, width = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, width // 220), 3))
    merged = cv2.morphologyEx(
        cv2.dilate(mask, kernel, 1), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 180 or w < 12 or h < 10 or h > height * 0.35 or w > width * 0.6:
            continue
        if mask[y : y + h, x : x + w].mean() / 255.0 < 0.05:
            continue
        out.append([int(x), int(y), int(w), int(h)])
    return out


def stack(boxes: list[list[int]], gap: float = 0.6, max_lines: int = 3) -> list[list[int]]:
    """Merge stacked lines into a label block, bounded so a chain cannot eat a column."""
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    out: list[list[int]] = []
    lines: list[int] = []
    unit: list[int] = []
    for box in boxes:
        placed = False
        for i, other in enumerate(out):
            ox, oy, ow, oh = other
            x, y, w, h = box
            overlap = min(ox + ow, x + w) - max(ox, x)
            vgap = y - (oy + oh)
            tall = max(unit[i], h)
            if (
                lines[i] < max_lines
                and overlap > 0.5 * min(ow, w)
                and -0.4 * h <= vgap <= gap * tall
                and (max(oy + oh, y + h) - min(oy, y)) <= (max_lines + 0.4) * tall
            ):
                nx, ny = min(ox, x), min(oy, y)
                out[i] = [nx, ny, max(ox + ow, x + w) - nx, max(oy + oh, y + h) - ny]
                lines[i] += 1
                unit[i] = tall
                placed = True
                break
        if not placed:
            out.append(list(box))
            lines.append(1)
            unit.append(box[3])
    return out


def _gap(box, blob) -> float:
    x, y, w, h = box
    bx, by, bw, bh = blob
    return float(np.hypot(max(x - (bx + bw), bx - (x + w), 0), max(y - (by + bh), by - (y + h), 0)))


def inside(box: list[int], block: list[int]) -> bool:
    x, y, w, h = box
    bx, by, bw, bh = block
    return bx >= x - 2 and by >= y - 2 and bx + bw <= x + w + 2 and by + bh <= y + h + 2


def assign(elements: list[dict], blocks: list[list[int]], page_w: int) -> dict[str, list[int]]:
    """Globally optimal element-to-block matching.

    Greedy nearest-block loses on dense pages and the failure is not subtle: a block is claimed
    by whichever element is closest to it, and the element that actually owns it is left with
    nothing inside its reach. Coverage fell from 95.6% on the sparsest hundred pages to 45.8% on
    the densest, while blocks outnumbered labels 1.8-3.7x throughout - so the candidates were
    always there and greedy was taking them in the wrong order. 10.2.5 hit the same problem
    matching predicted nodes to true ones and reached for the same tool.
    """
    from scipy.optimize import linear_sum_assignment

    if not elements or not blocks:
        return {}
    big = 1e6
    cost = np.full((len(elements), len(blocks)), big)
    for i, element in enumerate(elements):
        x, y, w, h = element["bbox"]
        reach = max(3.0 * max(w, h), page_w * 0.12)
        for j, block in enumerate(blocks):
            distance = _gap(element["bbox"], block)
            if distance > reach:
                continue
            bx, by, bw, bh = block
            centred = abs((bx + bw / 2) - (x + w / 2))
            below = by >= y + h * 0.6 and centred < max(w, bw)
            cost[i, j] = (
                distance
                - (10_000 if inside(element["bbox"], block) else 0)
                - (max(w, h) * 0.5 if below else 0)
                + 0.15 * centred
            )
    rows, cols = linear_sum_assignment(cost)
    return {
        elements[i]["id"]: blocks[j] for i, j in zip(rows, cols, strict=True) if cost[i, j] < big
    }


def union(boxes: list[list[int]]) -> list[int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def polyline_gap(points: np.ndarray, box: list[int]) -> float:
    """Distance from a text box's centre to the nearest point on a polyline.

    An edge label is written beside the connector, not beside its midpoint. Measuring from a
    stub at the midpoint put every label on a long or curved arrow out of reach and held edge
    coverage at 61%, which is 6.7% of the corpus's characters - most of the remaining budget.
    """
    x, y, w, h = box
    centre = np.array([x + w / 2.0, y + h / 2.0])
    best = float("inf")
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ab = b - a
        length = float(ab @ ab)
        t = 0.0 if length == 0 else float(np.clip((centre - a) @ ab / length, 0.0, 1.0))
        best = min(best, float(np.hypot(*(centre - (a + t * ab)))))
    return best


def own_lines(
    elements: list[dict], lines: list[list[int]], page_w: int, max_lines: int = 3
) -> dict[str, list[list[int]]]:
    """Each text line picks the element that most likely wrote it; an element may win several.

    Many-to-one on purpose: a label is one to three stacked lines, so the element has to be
    able to claim all of them, and two elements standing close together must not be forced to
    share one merged block.
    """
    won: dict[str, list[list[int]]] = {}
    for line in lines:
        best, best_cost = None, float("inf")
        for element in elements:
            x, y, w, h = element["bbox"]
            poly = element.get("polyline")
            if poly is not None:
                reach = page_w * 0.06
                distance = polyline_gap(poly, line)
            else:
                reach = max(3.0 * max(w, h), page_w * 0.12)
                distance = _gap(element["bbox"], line)
            if distance > reach:
                continue
            if poly is not None:
                cost = distance
            else:
                bx, by, bw, bh = line
                centred = abs((bx + bw / 2) - (x + w / 2))
                below = by >= y + h * 0.6 and centred < max(w, bw)
                cost = (
                    distance
                    - (10_000 if inside([int(v) for v in element["bbox"]], line) else 0)
                    - (max(w, h) * 0.5 if below else 0)
                    + 0.15 * centred
                )
            if cost < best_cost:
                best, best_cost = element["id"], cost
        if best is not None:
            won.setdefault(best, []).append(line)
    # keep the closest few lines, in reading order, so a runaway column cannot become a label
    for element_id, boxes in won.items():
        boxes.sort(key=lambda b: b[1])
        won[element_id] = boxes[:max_lines]
    return won


def inset_box(element: dict, inset: float = 0.10) -> list[int]:
    """9.3.1's node crop: the box, inset to drop the drawn outline."""
    x, y, w, h = element["bbox"]
    return [
        int(round(x + inset * w)),
        int(round(y + inset * h)),
        int(round(w * (1 - 2 * inset))),
        int(round(h * (1 - 2 * inset))),
    ]


def page_labels(image: np.ndarray, elements: list[dict]) -> dict[str, list[int]]:
    """Label block per element id.

    Two rules, because the corpus has two kinds of element and one rule cannot serve both.
    An activity's label is written *inside* its box, and 9.3.1's inset box already reads at
    0.0706 CER there - so that is kept unchanged. Blob-finding is applied only where the box
    provably cannot contain the label, which is what this module exists for. Running blobs over
    activities as well fragmented a quarter of them (`'receive order by website'` in a 20px
    crop) and made the class six times worse, so the split is by measurement, not neatness.
    """
    internal = [e for e in elements if e["kind"] == "node" and not is_external(e["id"])]
    external_ = [e for e in elements if e not in internal]

    out = {e["id"]: inset_box(e) for e in internal}

    # Assign individual text LINES, then merge the lines each element won. Stacking first and
    # assigning the blocks loses on dense pages: two elements' labels sit close enough to be
    # merged into one block, and a block can only be given to one of them - which is what held
    # coverage near 55% on the densest hundred pages while candidates outnumbered labels 2:1.
    lines = text_boxes(image)
    # Only a shape small enough to *be* a text box may claim the lines inside it. A Lane, Pool
    # or Participant is a container spanning the whole diagram: letting one claim its interior
    # swallowed 24 of 26 detected lines on a lane-heavy page and left 8 of its 9 external labels
    # with nothing. Container labels live in the header strip, which `inset_box` still covers.
    page_area = image.shape[0] * image.shape[1]
    claimed = [
        [int(v) for v in e["bbox"]]
        for e in internal
        if e["bbox"][2] * e["bbox"][3] < 0.20 * page_area
    ]
    free = [b for b in lines if not any(inside(c, b) for c in claimed)]
    # One pass over nodes and edges together. Running nodes first and letting edges take the
    # remainder was tried and is worse: nodes grab lines that belong to a connector, and edge
    # coverage falls 68.8% -> 40.9% for a 4-point gain on events.
    owner = own_lines(external_, free, image.shape[1])
    for element_id, boxes in owner.items():
        out[element_id] = union(boxes)
    return out


def crop(image: np.ndarray, block: list[int]) -> np.ndarray | None:
    """One label block, padded and normalised to the shared line height."""
    x, y, w, h = block
    pad = int(PAD * h)
    patch = image[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
    if patch.size == 0 or patch.shape[0] < 10:
        return None
    scale = HEIGHT / patch.shape[0]
    width = min(MAX_WIDTH, max(8, int(round(patch.shape[1] * scale))))
    return cv2.resize(patch, (width, HEIGHT), interpolation=cv2.INTER_AREA)


def elements_of(diagram: dict) -> list[dict]:
    """Every node with a box, plus each edge as a box around its polyline midpoint."""
    out = [
        {"id": n["id"], "bbox": n["bbox"], "kind": "node", "text": n.get("text", "")}
        for n in diagram.get("nodes", [])
        if n.get("bbox")
    ]
    for edge in diagram.get("edges", []):
        points = edge.get("polyline")
        label = edge.get("label") or edge.get("text")
        if not points or not label:
            continue
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        mid = pts[len(pts) // 2]
        out.append(
            {
                "id": edge.get("id", f"edge_{len(out)}"),
                "bbox": [float(mid[0]) - 6, float(mid[1]) - 6, 12.0, 12.0],
                "kind": "edge",
                "polyline": pts,
                "text": label,
            }
        )
    return out


def build(limit: int | None = None, out: Path = OUT) -> dict:
    """Write one crop per labelled element that the page gives a label block to."""
    import pandas as pd

    from src.detect.dataset import image_for
    from src.ocr.metrics import normalise
    from src.ocr.textcrops import load_index
    from src.parse.sequences import load_ir
    from src.preprocess.exif import load

    splits = load_index()[["page", "split", "scribe", "source"]].drop_duplicates("page")
    assignment = {r["page"]: (r["split"], r["scribe"], r["source"]) for _, r in splits.iterrows()}
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, pages, missing = [], 0, 0
    for diagram in load_ir(["hdbpmn"], limit=limit):
        meta = assignment.get(diagram["id"])
        if meta is None:
            continue
        path = image_for(diagram)
        if not path or not Path(path).is_file():
            continue
        image = load(Path(path), grayscale=True)
        if image is None:
            continue
        elements = elements_of(diagram)
        blocks = page_labels(image, elements)
        for element in elements:
            text = normalise(element["text"])
            if not text:
                continue
            block = blocks.get(element["id"])
            if block is None:
                missing += 1
                continue
            patch = crop(image, block)
            if patch is None:
                missing += 1
                continue
            name = f"hdbpmn__{diagram['id']}__{element['kind'][0]}_{element['id']}.png"
            cv2.imwrite(str(out / "images" / name), patch)
            rows.append(
                {
                    "file": name,
                    "page": diagram["id"],
                    "source": meta[2],
                    "split": meta[0],
                    "scribe": meta[1],
                    "kind": element["kind"],
                    "element": str(element["id"]).split("_")[0],
                    "provenance": "blob",
                    "text": element["text"],
                }
            )
        pages += 1
        if pages % 100 == 0:
            print(f"[labelcrops] {pages} pages, {len(rows)} crops", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_parquet(INDEX)
    summary = {
        "pages": pages,
        "crops": len(frame),
        "unassigned": missing,
        "coverage": round(len(frame) / max(1, len(frame) + missing), 4),
        "by_split": frame["split"].value_counts().to_dict(),
        "by_kind": frame["kind"].value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_index(path: Path = INDEX):
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"no label-crop index at {path}; run --build")
    return pd.read_parquet(path)


def split(name: str):
    """`(files, texts, frame)` for one split, matching `crnn.diagram_split`'s shape."""
    frame = load_index()
    frame = frame[frame["split"] == name]
    return [str(OUT / "images" / f) for f in frame["file"]], frame["text"].tolist(), frame


def unlocated(name: str) -> list[str]:
    """Labels in this split that no crop was found for.

    These have to be scored, not dropped. A label whose writing the pipeline could not locate is
    a label the pipeline failed to read, and reporting CER over only the crops that were found
    would measure the recogniser on a set chosen by the thing being measured. `src.ocr.s3` scores
    them as empty predictions.
    """
    from src.ocr.metrics import normalise
    from src.parse.sequences import load_ir

    frame = load_index()
    pages = set(frame.loc[frame["split"] == name, "page"])
    have = set(frame.loc[frame["split"] == name, "file"])
    out: list[str] = []
    for diagram in load_ir(["hdbpmn"], limit=None):
        if diagram["id"] not in pages:
            continue
        for element in elements_of(diagram):
            text = normalise(element["text"])
            if not text:
                continue
            name_ = f"hdbpmn__{diagram['id']}__{element['kind'][0]}_{element['id']}.png"
            if name_ not in have:
                out.append(element["text"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)
    if args.build:
        print(json.dumps(build(args.limit), indent=2))
        return 0
    if args.summary:
        print((OUT / "summary.json").read_text(encoding="utf-8"))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

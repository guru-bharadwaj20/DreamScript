"""Phase 9.3.1 - the text-crop corpus, cropped from detections rather than from annotations.

    python -m src.ocr.textcrops --build          # annotated boxes, both sources, nodes + edges
    python -m src.ocr.textcrops --build --detect # add the arm cropped from 9.1's detector
    python -m src.ocr.textcrops --summary

8.7 already cached 9,807 hdbpmn node crops and fine-tuned TrOCR on them. This is not that corpus
rebuilt; it is the same idea taken to the three places 8.7 did not go, each of which is a
measurement rather than a volume increase.

    edge labels        8.7 cropped nodes only. 2,441 hdbpmn edges and 2,447 fa_bresler edges
                       carry a label, which is a third of all the text in the corpus, and it is
                       the third that Phase 12 turns into branch conditions.
    fa_bresler         a second hand-drawn source, and a very different one: single alphabet
                       symbols rather than English phrases.
    detected boxes     the deployment case. 8.7's crops came from ground-truth boxes, which no
                       page at inference has.

## The finding this task exists to record

**There is no text-region annotation anywhere in this project.** 9.1.2 established it from the
detection side - `text-block` exists in hdbpmn as 175 margin notes with empty strings, and the
label text is annotated as a *string attached to a node or an edge*, never as its own box. So
every crop here is **derived**, and the two derivations are not equally sound:

    a node label    lives inside the node's box. Insetting 10% drops the drawn border and what
                    remains is the writing plus whatever else is in the box. The box is
                    annotated, so only the inset is a guess.
    an edge label   has no box at all. The only geometry an edge carries is its polyline, and
                    the label was written *beside* it. The crop is a rectangle at the polyline's
                    midpoint, offset perpendicular, sized from the page's median glyph height.

The second is a rule invented here, and a CER measured on those crops is a score for the rule as
much as for the recogniser. `ink_check()` exists so that it is not taken on faith: it reports
what fraction of derived edge boxes actually contain ink, which is the cheapest available test of
whether the rule is pointing at the writing at all.

## Cropping from detections

`provenance="detected"` runs 9.1.3's final detector over the page, matches each prediction to the
annotated node it overlaps most, and crops the *predicted* box. A prediction matched at IoU 0.62
is a real crop with a real transcription and a box that is a few percent wrong in a way no
ground-truth crop ever is. Holding the recogniser and the text fixed and moving only the box is
what separates "the OCR is weak" from "the boxes are". Predictions below `MATCH_IOU` get no
transcription and are dropped rather than guessed at - an unmatched detection is a crop with no
ground truth, and inventing one would put the detector's errors into the OCR's target.

## What it measured

**25,814 crops over 993 pages and 130 writers** - 21,912 hdbpmn and 3,902 fa_bresler, split
16,974 / 4,530 / 4,310 under 1.3.3's writer-disjoint assignment. 5,218 distinct strings, median
length 14 characters, longest 90, alphabet 85 characters.

    provenance    crops     what the box is
    annotated    11,261     a human-drawn node box, inset 10%
    derived       4,864     an edge label, at a rule's guess of where it was written
    detected      9,689     9.1.3's detector's own box, matched to a node at IoU >= 0.5

**The derived edge box misses the writing one time in five, and that is the number this task
exists to produce.** 80.59% of edge-label crops contain ink that is not the connector they were
centred on, against **99.72%** for the annotated node crops - a gap of **0.1913**. So roughly
**944 of the 4,864 edge crops are a picture of a line with no label in it**, carrying a
transcription that no recogniser can ever earn, and they set a floor under every CER 9.3.2-9.3.6
report on edges that has nothing to do with recognition.

**The first version of that check was worthless and the number is why it was replaced.** Asking
"does this crop contain any ink" scored the edge boxes at **0.9979** - higher than the annotated
node boxes - which is exactly what a box centred on a polyline should score, because it contains
the polyline by construction. The measurement said the rule had found the line, which was never
in question. Rasterising the polyline, dilating it by a stroke width and removing it turns the
same test into one the rule can fail, and it then fails it 19% of the time. A dilation kernel of
even side left one row of connector standing and passed the floor, which is why the radius is
forced odd; that bug alone was worth 1.4 points of the rate.

**The detector is a better cropper than expected: median IoU 0.9042** against the node it
matched, and it recovers **9,689 of the 11,261 text-bearing nodes, 86.0%**. So the deployment
gap 9.3.6 will price splits cleanly into two parts that can be reported separately - **14% of
labels never get a crop at all**, and the 86% that do get one whose box is 0.90 IoU rather than
1.00. The second is a much smaller problem than the first, which was not obvious before the
numbers: a detector that finds a box at all finds it well.

**Edge labels are a third of the corpus and were entirely absent from 8.7.** 4,864 of 16,125
annotated-plus-derived crops are edge labels, and their text is not like the nodes' at all - the
fa_bresler side is single alphabet symbols (`a`, `b`, `0`, `1` are the four most frequent strings
in the whole corpus) while hdbpmn's are message names. That distribution is the reason 9.3.4
refuses to snap words below three characters: on 40% of the edge corpus every candidate is one
edit from every other.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.detect.classes import pages
from src.detect.dataset import SPLITS, fa_splits, image_for, manifest_splits
from src.utils.config import ROOT

OUT = ROOT / "data" / "processed" / "text_crops"
INDEX = OUT / "index.parquet"

#: Line height every crop is normalised to. 64 rather than 32 because TrOCR's processor
#: upsamples to 384 anyway and 9.3.2's CRNN pools height away; matching 8.7 keeps the two
#: corpora comparable.
HEIGHT = 64
MAX_WIDTH = 512

#: Fraction of a node box removed on each side before cropping, to drop the drawn outline.
INSET = 0.10

#: Smallest crop worth writing, in native page pixels.
MIN_CROP = (20, 12)

#: An edge-label box is this many median-glyph-heights tall and this many wide, centred on the
#: polyline midpoint and pushed perpendicular to the local direction.
EDGE_BOX_HEIGHTS = 1.6
EDGE_BOX_WIDTHS = 3.2
EDGE_OFFSET = 0.9

#: A detected box must overlap an annotated node this much to inherit its transcription.
MATCH_IOU = 0.5

SOURCES = ("hdbpmn", "fa_bresler")


# ------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------


def iou(a, b) -> float:
    """`(x, y, w, h)` overlap; 0.0 when either box is degenerate."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def glyph_height(diagram: dict) -> float:
    """A page-relative unit for the edge-label box: the median annotated node height / 6.

    Nothing in any source records how large anyone wrote, so the box has to be sized from
    something on the same page. A node is a few text lines tall in both corpora, so a sixth of
    the median node is a defensible line height and, more importantly, it scales with the
    photograph instead of being a pixel constant that is right for one camera.
    """
    heights = [n["bbox"][3] for n in diagram["nodes"] if n.get("bbox")]
    if not heights:
        width, height = diagram["meta"].get("image_size", (1000, 1000))
        return max(8.0, min(width, height) / 60.0)
    return max(6.0, float(np.median(heights)) / 6.0)


def edge_label_box(polyline, unit: float, page_size) -> tuple[float, float, float, float] | None:
    """A rectangle where an edge's label was probably written - a rule, not an observation.

    The midpoint of a polyline by arc length is where a person writing on a connector tends to
    write, and the label sits beside the line rather than on it, so the box is pushed
    perpendicular. Which side is unknowable, so the box is made tall enough to straddle the line
    and is centred on it - a wrong side would be a guaranteed miss, whereas straddling costs a
    little connector ink inside the crop.
    """
    points = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or len(points) < 2:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(steps.sum())
    if total <= 1e-6:
        return None
    walked = np.concatenate([[0.0], np.cumsum(steps)])
    index = int(np.searchsorted(walked, total / 2.0, side="right")) - 1
    index = min(max(index, 0), len(steps) - 1)
    span = steps[index] if steps[index] > 1e-6 else 1.0
    t = (total / 2.0 - walked[index]) / span
    mid = points[index] + t * (points[index + 1] - points[index])

    width = EDGE_BOX_WIDTHS * unit
    height = (EDGE_BOX_HEIGHTS + 2 * EDGE_OFFSET) * unit
    x = mid[0] - width / 2.0
    y = mid[1] - height / 2.0
    page_w, page_h = page_size
    x = float(min(max(0.0, x), max(0.0, page_w - 1)))
    y = float(min(max(0.0, y), max(0.0, page_h - 1)))
    return (x, y, float(min(width, page_w - x)), float(min(height, page_h - y)))


# ------------------------------------------------------------------------------------------
# cropping
# ------------------------------------------------------------------------------------------


def cut(image: np.ndarray, box, inset: float = 0.0):
    """One height-normalised greyscale crop, or None when the box has too few pixels."""
    import cv2

    x, y, w, h = box
    ix, iy = inset * w, inset * h
    x0, y0 = max(0, int(round(x + ix))), max(0, int(round(y + iy)))
    x1 = min(image.shape[1], int(round(x + w - ix)))
    y1 = min(image.shape[0], int(round(y + h - iy)))
    if x1 - x0 < MIN_CROP[0] or y1 - y0 < MIN_CROP[1]:
        return None
    patch = image[y0:y1, x0:x1]
    scale = HEIGHT / patch.shape[0]
    width = min(MAX_WIDTH, max(8, int(round(patch.shape[1] * scale))))
    return cv2.resize(patch, (width, HEIGHT), interpolation=cv2.INTER_AREA)


def off_line_ink(image: np.ndarray, box, polyline, unit: float, floor: float = 0.004) -> bool:
    """Is there ink in the edge-label box that is *not* the connector itself?

    `has_ink` on an edge crop is close to useless and the first run proved it: the box is
    centred on the polyline, so it contains the connector by construction and scored 0.9979.
    That number says the rule found the line, which was never in doubt. The question is whether
    it found the *writing*, so the polyline is rasterised, dilated by a stroke width, and
    removed, and what is left is measured. A box that is only connector comes back False.
    """
    import cv2

    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + w), min(image.shape[0], y + h)
    if x1 - x0 < MIN_CROP[0] or y1 - y0 < MIN_CROP[1]:
        return False
    patch = image[y0:y1, x0:x1]
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    points = np.asarray(polyline, dtype=float) - np.array([x0, y0])
    mask = np.zeros_like(binary)
    cv2.polylines(mask, [points.round().astype(np.int32).reshape(-1, 1, 2)], False, 255, 1)
    # Odd, so the structuring element has a true centre: an even-sided kernel dilates
    # asymmetrically and leaves one row of the connector standing, which is enough dark pixels
    # to pass the floor and call a connector-only box "written".
    radius = max(5, int(round(unit / 3)) | 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
    remaining = cv2.bitwise_and(binary, cv2.bitwise_not(mask))
    return float((remaining > 0).mean()) >= floor


def has_ink(patch: np.ndarray, floor: float = 0.01) -> bool:
    """Does this crop contain writing, or is it blank paper?

    Otsu on the crop, dark side counted. A crop that is all paper thresholds into noise, so the
    test is a *share* floor rather than "any dark pixel": 1% of a 64-row crop is a few hundred
    pixels, which is a glyph and not a speck of grain.
    """
    import cv2

    if patch is None or patch.size == 0:
        return False
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((binary > 0).mean()) >= floor


def page_rows(ir_path: Path, source: str, split: str) -> tuple[list[dict], list[np.ndarray]]:
    """Every annotated node and edge label on one page, as crops plus index rows."""
    from src.preprocess.exif import load

    diagram = json.loads(ir_path.read_text(encoding="utf-8"))
    image_path = image_for(diagram)
    if image_path is None or not Path(image_path).is_file():
        return [], []
    image = load(Path(image_path), grayscale=True)
    page = diagram["id"]
    unit = glyph_height(diagram)
    size = (image.shape[1], image.shape[0])

    rows: list[dict] = []
    patches: list[np.ndarray] = []
    for node in diagram["nodes"]:
        text = (node.get("text") or "").strip()
        if not text or not node.get("bbox"):
            continue
        patch = cut(image, node["bbox"], INSET)
        if patch is None:
            continue
        rows.append(
            {
                "file": f"{source}__{page}__n_{node['id']}.png",
                "page": page,
                "source": source,
                "split": split,
                "scribe": str(diagram["meta"].get("scribe_id")),
                "kind": "node",
                "provenance": "annotated",
                "iou": 1.0,
                "has_ink": bool(has_ink(patch)),
                "text": text,
            }
        )
        patches.append(patch)

    for edge in diagram["edges"]:
        text = (edge.get("label") or "").strip()
        if not text or not edge.get("polyline"):
            continue
        box = edge_label_box(edge["polyline"], unit, size)
        if box is None:
            continue
        patch = cut(image, box)
        if patch is None:
            continue
        rows.append(
            {
                "file": f"{source}__{page}__e_{edge['id']}.png",
                "page": page,
                "source": source,
                "split": split,
                "scribe": str(diagram["meta"].get("scribe_id")),
                "kind": "edge",
                "provenance": "derived",
                "iou": float("nan"),
                "has_ink": bool(off_line_ink(image, box, edge["polyline"], unit)),
                "text": text,
            }
        )
        patches.append(patch)
    return rows, patches


def build(out: Path = OUT, sources=SOURCES, limit: int | None = None) -> dict:
    """Write `<out>/images/*.png` plus `index.parquet`, splits inherited from 1.3.3."""
    import cv2
    import pandas as pd

    images = out / "images"
    if out.exists():
        shutil.rmtree(out)
    images.mkdir(parents=True, exist_ok=True)

    inherited = manifest_splits()
    rows: list[dict] = []
    for source in sources:
        paths = pages(source)[:limit] if limit else pages(source)
        assigned: dict[str, str] = {}
        if source == "fa_bresler":
            loaded = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
            assigned = fa_splits(loaded)
        for path in paths:
            key = f"{source}:{json.loads(path.read_text(encoding='utf-8'))['id']}"
            split = assigned.get(key) or inherited.get(key, "train")
            page, patches = page_rows(path, source, split)
            for row, patch in zip(page, patches, strict=True):
                cv2.imwrite(str(images / row["file"]), patch)
            rows.extend(page)

    frame = pd.DataFrame(rows)
    frame.to_parquet(INDEX if out == OUT else out / "index.parquet", index=False)
    return summary(out)


# ------------------------------------------------------------------------------------------
# the detected arm
# ------------------------------------------------------------------------------------------


def detected_rows(ir_path: Path, source: str, split: str, model, conf: float = 0.25):
    """Crops cut from the detector's boxes, transcribed from the node each one matched."""
    from src.preprocess.exif import load

    diagram = json.loads(ir_path.read_text(encoding="utf-8"))
    image_path = image_for(diagram)
    if image_path is None or not Path(image_path).is_file():
        return [], []
    image = load(Path(image_path), grayscale=True)
    truths = [
        (n["bbox"], (n.get("text") or "").strip(), n["id"])
        for n in diagram["nodes"]
        if n.get("bbox") and (n.get("text") or "").strip()
    ]
    if not truths:
        return [], []

    predictions = model.predict(str(image_path), conf=conf, verbose=False)[0]
    boxes = predictions.boxes.xyxy.cpu().numpy() if predictions.boxes is not None else []
    scores = predictions.boxes.conf.cpu().numpy() if len(boxes) else []

    rows: list[dict] = []
    patches: list[np.ndarray] = []
    taken: set[str] = set()
    order = np.argsort(-np.asarray(scores)) if len(boxes) else []
    for i in order:
        x0, y0, x1, y1 = (float(v) for v in boxes[i])
        box = (x0, y0, x1 - x0, y1 - y0)
        best, overlap = None, 0.0
        for truth_box, text, node_id in truths:
            value = iou(box, truth_box)
            if value > overlap and node_id not in taken:
                best, overlap = (text, node_id), value
        if best is None or overlap < MATCH_IOU:
            continue
        patch = cut(image, box, INSET)
        if patch is None:
            continue
        taken.add(best[1])
        rows.append(
            {
                "file": f"{source}__{diagram['id']}__d_{best[1]}.png",
                "page": diagram["id"],
                "source": source,
                "split": split,
                "scribe": str(diagram["meta"].get("scribe_id")),
                "kind": "node",
                "provenance": "detected",
                "iou": round(overlap, 4),
                "has_ink": bool(has_ink(patch)),
                "text": best[0],
            }
        )
        patches.append(patch)
    return rows, patches


def add_detected(out: Path = OUT, source: str = "hdbpmn", limit: int | None = None) -> dict:
    """Append the detected arm to an already-built corpus.

    Restricted to hdbpmn by default because that is the source 9.1.3's detector was trained and
    validated on as photographs; fa_bresler pages are renders and a detection on one would be
    measuring the renderer.
    """
    import cv2
    import pandas as pd
    from ultralytics import YOLO

    from src.detect.train import WEIGHTS

    index_path = out / "index.parquet"
    frame = pd.read_parquet(index_path)
    frame = frame[frame["provenance"] != "detected"]

    model = YOLO(str(WEIGHTS))
    inherited = manifest_splits()
    rows: list[dict] = []
    paths = pages(source)[:limit] if limit else pages(source)
    for path in paths:
        key = f"{source}:{json.loads(path.read_text(encoding='utf-8'))['id']}"
        split = inherited.get(key, "train")
        page, patches = detected_rows(path, source, split, model)
        for row, patch in zip(page, patches, strict=True):
            cv2.imwrite(str(out / "images" / row["file"]), patch)
        rows.extend(page)

    import pandas as pd  # noqa: F811  (local re-import keeps the optional dep at call site)

    combined = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    combined.to_parquet(index_path, index=False)
    return summary(out)


# ------------------------------------------------------------------------------------------
# reporting
# ------------------------------------------------------------------------------------------


def load_index(out: Path = OUT):
    import pandas as pd

    return pd.read_parquet(out / "index.parquet")


def ink_check(frame=None) -> dict:
    """Does the derived edge-label box point at any writing?

    The node crops are the reference: their boxes are annotated, so their ink rate is what "a
    crop that contains its label" looks like on this corpus. The edge rate is the derivation's
    score, and the gap between the two is the cost of having no annotation to crop from. The two
    are not the same test - a node box is scored on containing any ink, an edge box on containing
    ink that is not the connector it was centred on - because the weaker test is trivially passed
    by an edge box and says nothing.
    """
    frame = load_index() if frame is None else frame
    out = {}
    for kind, group in frame.groupby("kind"):
        out[str(kind)] = {
            "crops": int(len(group)),
            "ink_rate": round(float(group["has_ink"].mean()), 4),
        }
    if "node" in out and "edge" in out:
        out["gap"] = round(out["node"]["ink_rate"] - out["edge"]["ink_rate"], 4)
    return out


def summary(out: Path = OUT) -> dict:
    frame = load_index(out)
    by_split = {s: int((frame["split"] == s).sum()) for s in SPLITS}
    lengths = frame["text"].str.len()
    charset = sorted({c for t in frame["text"] for c in t})
    return {
        "crops": int(len(frame)),
        "pages": int(frame["page"].nunique()),
        "scribes": int(frame["scribe"].nunique()),
        "by_split": by_split,
        "by_source": {k: int(v) for k, v in Counter(frame["source"]).items()},
        "by_kind": {k: int(v) for k, v in Counter(frame["kind"]).items()},
        "by_provenance": {k: int(v) for k, v in Counter(frame["provenance"]).items()},
        "unique_strings": int(frame["text"].nunique()),
        "median_length": float(lengths.median()),
        "max_length": int(lengths.max()),
        "charset_size": len(charset),
        "ink": ink_check(frame),
        "detected_median_iou": (
            round(float(frame.loc[frame["provenance"] == "detected", "iou"].median()), 4)
            if (frame["provenance"] == "detected").any()
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--detect", action="store_true", help="add the detected-box arm")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    if args.build:
        print(json.dumps(build(args.out, limit=args.limit), indent=2))
    if args.detect:
        print(json.dumps(add_detected(args.out, limit=args.limit), indent=2))
    if args.summary or not (args.build or args.detect):
        print(json.dumps(summary(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

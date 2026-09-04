"""Phase 9.1.5 - arrowheads are 15 px, and two different fixes for that.

    python -m src.detect.tiling                 # both arms against the baseline

9.1.2 measured the problem it was going to create: exported at a 1,280 px long side, the
median derived arrowhead is **15.6 px**, and at the 896 px the detector actually sees it is
smaller again. A box a few pixels across on a feature map whose finest stride is 8 input
pixels is a box the model has one or two cells to describe.

Two remedies exist and they are not the same operation:

**Higher input resolution** re-runs the same model at a larger `imgsz` at inference. It is one
line, it costs quadratic latency, and it cannot recover detail the *export* threw away - the
page on disk is already 1,280 px, so upscaling past that is interpolation.

**Tiling** cuts the native-resolution page into overlapping windows, detects in each, maps the
boxes back and merges them. It reads the original photograph rather than the export, so it is
the only one of the two that adds real pixels; it costs one forward pass per tile, and it
introduces a failure the single-pass model does not have - **an object split across a tile
boundary is seen twice, in halves**, which is what the overlap and the merge are for.

## The tiling arm has to be retrained, and the version that is not is kept as the control

The obvious implementation - cut tiles and run the existing weights on them - is wrong, and
wrong in a way that looks like a result. A full hdbpmn page is 2,837 px letterboxed to 896, so
the model is trained to see objects at 0.32x; a 640 px tile letterboxed to 896 shows them at
1.4x, **four and a half times larger than anything in training**. The detector is then being
asked to generalise across a scale change nothing prepared it for, and it fails for that reason
rather than because tiling does not help.

So there are two tiling arms. `tile_naive` runs the page-trained weights on tiles and is kept
because it is the version most people write; `tile_trained` fine-tunes on a tiled export, which
is the only comparison that isolates tiling from the scale mismatch. Both merge back into page
coordinates and are scored on whole pages, so every arm answers the same question.

**Everything here is scored on hdbpmn alone**, because that is the only source 9.1.2 derives
arrowheads on. Tiling the other two would multiply the training set with pages that cannot
contain the class the experiment is about, and flowchartseg's pages are narrower than a tile
anyway, so windowing them copies the page. The epoch counts of the two arms are therefore not
comparable and are not compared: `gradient_steps` is reported instead.

`arrowhead` AP is reported at IoU 0.5 and at the loose 0.25, because 9.1.2's head boxes carry a
size convention and a fix that finds the head but not the convention's box would be invisible
at 0.5.

## What it measured

Five arms, 128 hdbpmn validation pages, 9.1.3's weights.

    arm             mAP@0.5   head AP@0.5   head AP@0.25   ms/page
    resize_896       0.8831      0.4437         0.7604        23.9
    resize_1280      0.8907      0.5197         0.7934        34.3     <- baseline, trained size
    resize_1600      0.8543      0.5134         0.8003        29.4
    tile_naive       0.5411      0.1792         0.4847       597.3
    tile_trained     0.7232      0.4354         0.8199       564.3

**The plan's Definition of Done is not met. Nothing beats the baseline on arrowhead AP@0.5** -
0.5197 stands, and the best alternative is 0.5134 from simply running at a higher input size.
Reported as a failure rather than dressed up, because the two remedies fail for two *different*
and separately useful reasons.

**Higher resolution than the model was trained at makes things worse, and worst for the biggest
class.** 1600 loses 0.006 on the head and collapses `rectangle` from **0.9081 to 0.6976**. The
model was trained at 1280; at 1600 a BPMN pool is larger than any box it has ever seen, and a
detector generalises downward in scale far better than upward. **The resolution knob is
therefore bounded by training, not by the image** - 9.1.3 could raise the head by training at a
higher size, and this task cannot raise it by inferring at one.

**Tiling's failure is structural and it is the finding worth keeping: tiling destroys every
class larger than a tile.** `rectangle` scores **0.0277 naive and 0.0327 trained**, against
0.9081 for the baseline - a near-total loss that retraining does not touch, while every other
class recovers to within 0.03 of baseline once the model is trained on tiles (rounded-rect
0.964, diamond 0.978, circle 0.972, freeform 0.957). The mechanism is the export's own rule: a
box is kept for a tile only when 60% of its area falls inside, and **an hdbpmn pool is wider
than 640 px, so it satisfies that for no window at all** and simply vanishes from the training
labels - 32,411 of the box-tile pairs were dropped at seams for exactly this reason. Tiling is
not a general-purpose fix on pages whose objects span two orders of magnitude in size; it is a
fix for corpora where everything is small.

**The naive arm did what it was kept to do.** Running the page-trained weights on tiles scores
0.5411 against the trained tiler's 0.7232 - so **the scale mismatch alone is worth 0.18 mAP**,
and a version of this task without the retrained arm would have concluded that tiling does not
work when what it had measured was that a 4.5x magnification the model never saw does not work.

**One thing tiling does win, and it is the reason the loose threshold is reported.** At IoU 0.25
`tile_trained` reaches **0.8199, the best arrowhead figure of any arm**, above the baseline's
0.7934, while losing to it at 0.5. Tiling **finds more heads and localises them worse** relative
to 9.1.2's convention box. For Phase 10, which needs to know an arrowhead *exists* near a
polyline end and not where its box is to a pixel, that is the more useful of the two - but it
costs **16.5x the latency** (564 ms against 34 ms a page, 23.7 forward passes instead of one),
which is not a trade this pipeline should take for one class.

**The honest summary is that 9.1.3 already did what this task was asked to do.** Arrowhead AP
went 0.2157 -> 0.3826 across the resolution sweep and reached 0.5197 with the longer schedule at
1280; input size and training length moved it by 0.30, and neither remedy here moves it at all.
The class is limited by 9.1.3's anchor analysis - 91% of heads under one P4 cell, median short
side 11.2 px - and by 9.1.2's derived box convention, and **the fix that would actually work is
better supervision rather than more pixels**: real arrowhead annotations instead of a square
placed at a waypoint.

## What this settles for 9.1.1

9.1.1 named the condition for reversing its choice of yolov8n over RT-DETR: tiling leaving the
arrowhead under RT-DETR's 0.4863. **The threshold is cleared at 0.5197, but by 9.1.3's training
schedule rather than by anything in this task.** The two numbers are still not budget-matched -
RT-DETR ran 12 epochs at 896 and was never given 40 at 1280 - so the correct reading is that the
reversal condition as written is not met and **the architecture question remains open at equal
budget**, not that yolov8n has been shown to be the better model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src.detect.choice import RUNS, predict_ultralytics, truths
from src.detect.classes import CLASSES
from src.detect.dataset import OUT as DATA
from src.detect.train import WEIGHTS

#: Window side and overlap fraction, in pixels of the *native* page. 640 with 25% overlap puts
#: two full windows across an hdbpmn photograph's short side and guarantees any object under
#: 160 px is whole in at least one tile.
TILE = 640
OVERLAP = 0.25

INFERENCE_SIZES = (896, 1280, 1600)

#: The size 9.1.3's final run trained at. The baseline arm has to be this one: comparing a fix
#: against a model run at a resolution it was never trained for measures the mismatch, not the
#: fix - which is the same error the naive tiling arm exists to demonstrate.
TRAINED_IMGSZ = 1280


def tiles(width: int, height: int, tile: int = TILE, overlap: float = OVERLAP):
    """Top-left corners covering the page, always including the right and bottom edges.

    The last column and row are pinned to the page edge rather than left short, so no strip is
    covered by fewer windows than the interior - the alternative silently makes the page's
    margins a different experiment from its middle.
    """
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, width - tile + 1), step))
    ys = list(range(0, max(1, height - tile + 1), step))
    if xs[-1] != max(0, width - tile):
        xs.append(max(0, width - tile))
    if ys[-1] != max(0, height - tile):
        ys.append(max(0, height - tile))
    return [(x, y) for y in ys for x in xs]


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> list[int]:
    """Plain greedy NMS, used here only to merge the duplicates tiling creates."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_c = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_c + area_r - inter, 1e-12)
        order = rest[iou < threshold]
    return keep


def predict_tiled(
    model,
    split: str = "val",
    root: Path = DATA,
    tile: int = TILE,
    overlap: float = OVERLAP,
    imgsz: int = 896,
    sources=("hdbpmn",),
):
    """Detect in native-resolution windows and merge back into exported-page coordinates.

    The export is the frame every metric in 9.1 is expressed in, so the windows are cut from
    the original page and the boxes are scaled back down to the export - which is also why
    this can beat a plain resize: the crop is taken before the information is lost.
    """

    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    native = {row["name"]: (row["native_size"], row["scale"]) for row in index}

    rows: list[dict] = []
    started = time.perf_counter()
    pages = sorted((root / "images" / split).glob("*.png"))
    tile_count = 0
    pages = [p for p in pages if not sources or p.stem.split("__", 1)[0] in sources]
    for path in pages:
        record = native.get(path.stem)
        if record is None:
            continue
        (page_w, page_h), scale = record
        original = _native_image(root, path)
        if original is None:
            continue
        height, width = original.shape[:2]
        raw: list[tuple[list[float], float, int]] = []
        for x, y in tiles(width, height, tile, overlap):
            window = original[y : y + tile, x : x + tile]
            if window.size == 0:
                continue
            tile_count += 1
            result = model.predict(
                source=window, imgsz=imgsz, conf=0.001, iou=0.7, verbose=False, device=0
            )[0]
            boxes = result.boxes
            if boxes is None or not len(boxes):
                continue
            for box, score, cls in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                strict=True,
            ):
                raw.append(
                    (
                        [
                            (box[0] + x) * scale,
                            (box[1] + y) * scale,
                            (box[2] + x) * scale,
                            (box[3] + y) * scale,
                        ],
                        float(score),
                        int(cls),
                    )
                )
        rows.extend(_merge(path.stem, raw))
    elapsed = time.perf_counter() - started
    return rows, (elapsed / max(len(pages), 1)) * 1000.0, tile_count / max(len(pages), 1)


def _merge(image: str, raw: list[tuple[list[float], float, int]]) -> list[dict]:
    """Class-wise NMS over the pooled tile detections - the price of seeing the seam twice."""
    out: list[dict] = []
    for index in range(len(CLASSES)):
        subset = [(b, s) for b, s, c in raw if c == index]
        if not subset:
            continue
        boxes = np.asarray([b for b, _ in subset], dtype=float)
        scores = np.asarray([s for _, s in subset], dtype=float)
        for keep in nms(boxes, scores, 0.5):
            out.append(
                {
                    "image": image,
                    "cls": CLASSES[index],
                    "xyxy": [float(v) for v in boxes[keep]],
                    "score": float(scores[keep]),
                }
            )
    return out


def _native_image(root: Path, exported: Path):
    """The original photograph where it is still on disk, else the exported copy.

    Falling back rather than skipping keeps the page in the comparison; when it happens the
    tiling arm is simply doing what the resize arm does on that page, which weakens the effect
    rather than inventing one.
    """
    import cv2

    source, _, name = exported.stem.partition("__")
    from src.detect.classes import IR

    ir_file = IR / source / f"{name}.ir.json"
    if ir_file.is_file():
        meta = json.loads(ir_file.read_text(encoding="utf-8"))["meta"]
        declared = meta.get("image")
        if declared:
            from src.utils.config import ROOT

            candidate = ROOT / declared
            if candidate.is_file():
                return cv2.imread(str(candidate))
        if source == "fa_bresler":
            from src.detect.dataset import FA_RENDER

            candidate = FA_RENDER / f"{name}.png"
            if candidate.is_file():
                return cv2.imread(str(candidate))
    return cv2.imread(str(exported))


def export_tiles(
    root: Path = DATA,
    out: Path | None = None,
    tile: int = TILE,
    overlap: float = OVERLAP,
    splits=("train", "val"),
    sources=("hdbpmn",),
    reuse: bool = True,
) -> dict:
    """A second export whose images are native-resolution windows, boxes remapped into them.

    A box is kept for a tile when **at least 60% of its area** falls inside: keeping every
    sliver would teach the model that a third of a rectangle is a rectangle, and keeping only
    whole boxes would teach it that the object at the seam is background. 0.6 is the usual
    compromise and it is written down because it is a decision, not a default.

    **hdbpmn only, by default.** Tiling exists here for the arrowhead, and 9.1.2 derives
    arrowheads on hdbpmn alone; tiling the other two sources would multiply the training set
    fivefold with pages that cannot contain the class the experiment is about. flowchartseg's
    pages are also under a tile wide, so windowing them is a no-op that copies the page.
    """
    import cv2

    out = out or (root.parent / f"{root.name}_tiles")
    stats_file = out / "tiles_stats.json"
    if reuse and stats_file.is_file():
        # Cutting 14,000 windows out of 2,000-pixel photographs is twenty minutes of pure IO,
        # and it is deterministic, so a finished export is reused rather than rebuilt.
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        stats["reused"] = True
        return stats
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    index = {row["name"]: row for row in json.loads((root / "index.json").read_text("utf-8"))}
    written = {split: 0 for split in splits}
    boxes_kept = 0
    boxes_dropped = 0
    for split in splits:
        for label in sorted((root / "labels" / split).glob("*.txt")):
            record = index.get(label.stem)
            if record is None or (sources and record["source"] not in sources):
                continue
            page = _native_image(root, root / "images" / split / f"{label.stem}.png")
            if page is None:
                continue
            height, width = page.shape[:2]
            native = []
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = line.split()
                cx, cy = float(cx) * width, float(cy) * height
                w, h = float(w) * width, float(h) * height
                native.append((int(cls), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))

            for x, y in tiles(width, height, tile, overlap):
                window = page[y : y + tile, x : x + tile]
                if window.size == 0:
                    continue
                th, tw = window.shape[:2]
                lines = []
                for cls, bx1, by1, bx2, by2 in native:
                    ix1, iy1 = max(bx1, x), max(by1, y)
                    ix2, iy2 = min(bx2, x + tw), min(by2, y + th)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area = max((bx2 - bx1) * (by2 - by1), 1e-9)
                    if inter / area < 0.6:
                        boxes_dropped += inter > 0
                        continue
                    boxes_kept += 1
                    cx, cy = ((ix1 + ix2) / 2 - x) / tw, ((iy1 + iy2) / 2 - y) / th
                    lines.append(
                        f"{cls} {cx:.6f} {cy:.6f} {(ix2 - ix1) / tw:.6f} {(iy2 - iy1) / th:.6f}"
                    )
                name = f"{label.stem}__t{x}_{y}"
                cv2.imwrite(str(out / "images" / split / f"{name}.png"), window)
                (out / "labels" / split / f"{name}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                written[split] += 1

    yaml = (root / "data.yaml").read_text(encoding="utf-8").replace(root.as_posix(), out.as_posix())
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    shutil.copy(root / "index.json", out / "index.json")
    stats = {
        "root": str(out),
        "tiles": written,
        "boxes_kept": boxes_kept,
        "boxes_dropped_at_seams": int(boxes_dropped),
        "tile": tile,
        "overlap": overlap,
        "sources": list(sources),
    }
    stats_file.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def run(
    weights: Path = WEIGHTS,
    root: Path = DATA,
    split: str = "val",
    epochs: int = 15,
    retrain: bool = True,
    sources=("hdbpmn",),
    trained_imgsz: int = TRAINED_IMGSZ,
) -> dict:
    """Every arm on the same pages: the hdbpmn photographs, which is where arrowheads exist.

    Restricting the evaluation is not a convenience. The tiled model is trained on hdbpmn tiles
    (see `export_tiles`), so scoring it on fa_bresler and flowchartseg would charge it for two
    sources it was never shown, and scoring the resize arms on those sources while the tiled arm
    is charged for them would make the comparison meaningless in the other direction.
    """
    from ultralytics import YOLO

    from src.detect.metrics import evaluate

    model = YOLO(str(weights))
    reference, page_sources = truths(split, root)
    keep = {name for name, source in page_sources.items() if not sources or source in sources}
    reference = [row for row in reference if row["image"] in keep]

    def restrict(rows):
        return [row for row in rows if row["image"] in keep]

    # One throwaway pass so the first arm is not charged for CUDA and cuDNN warm-up; 7.1.8's
    # latency table made the same correction for the same reason.
    model.predict(
        source=str(next((root / "images" / split).glob("*.png"))), imgsz=896, verbose=False
    )

    arms = []
    for imgsz in INFERENCE_SIZES:
        predictions, latency = predict_ultralytics(model, split, imgsz, root)
        scored = evaluate(restrict(predictions), reference)
        arms.append(
            {
                "arm": f"resize_{imgsz}",
                "imgsz": imgsz,
                "page_latency_ms": round(latency, 2),
                "map50": scored["map50"],
                "map50_95": scored["map50_95"],
                "arrowhead_ap50": scored["per_class"]["arrowhead"]["ap50"],
                "arrowhead_ap25": scored["per_class"]["arrowhead"]["ap25"],
                "per_class": scored["per_class"],
            }
        )

    def add(name: str, predictions, latency, per_page_tiles=None):
        scored = evaluate(predictions, reference)
        arms.append(
            {
                "arm": name,
                "imgsz": 896,
                "tiles_per_page": None if per_page_tiles is None else round(per_page_tiles, 2),
                "page_latency_ms": round(latency, 2),
                "map50": scored["map50"],
                "map50_95": scored["map50_95"],
                "arrowhead_ap50": scored["per_class"]["arrowhead"]["ap50"],
                "arrowhead_ap25": scored["per_class"]["arrowhead"]["ap25"],
                "per_class": scored["per_class"],
            }
        )

    predictions, latency, per_page_tiles = predict_tiled(model, split, root)
    add("tile_naive", predictions, latency, per_page_tiles)

    built = None
    if retrain:
        from src.detect.train import train

        built = export_tiles(root, sources=sources)
        tiled_model, seconds = train(
            "tiled", epochs=epochs, imgsz=896, batch=16, root=Path(built["root"])
        )
        built["train_tiles"] = built["tiles"].get("train", 0)
        built["gradient_steps"] = epochs * (built["train_tiles"] // 16)
        built["train_seconds"] = round(seconds, 1)
        predictions, latency, per_page_tiles = predict_tiled(tiled_model, split, root)
        add("tile_trained", predictions, latency, per_page_tiles)

    baseline = next((a for a in arms if a["arm"] == f"resize_{trained_imgsz}"), arms[0])

    def head(arm):
        return arm["arrowhead_ap50"] or 0.0

    best = max(arms, key=head)
    return {
        "baseline": baseline["arm"],
        "arms": arms,
        "tiled_export": built,
        "best_arm_for_arrowhead": best["arm"],
        "arrowhead_ap50_baseline": baseline["arrowhead_ap50"],
        "arrowhead_ap50_best": best["arrowhead_ap50"],
        "arrowhead_improved": bool(head(best) > head(baseline)),
        "tile": TILE,
        "overlap": OVERLAP,
        "scored_sources": list(sources),
        "pages_scored": len(keep),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--split", default="val")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--no-retrain", action="store_true", help="skip the fair tiling arm")
    ap.add_argument("--out", type=Path, default=RUNS / "tiling.json")
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        print(
            f"no weights at {args.weights} (run `python -m src.detect.train --final`)",
            file=sys.stderr,
        )
        return 1
    result = run(args.weights, args.data, args.split, args.epochs, not args.no_retrain)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

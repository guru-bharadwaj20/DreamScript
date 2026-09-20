"""Phase 10.1.3 replacement - detect an arrow as an object with a tail and a head.

    python -m src.detect.arrows --export          # write the YOLO-pose dataset
    python -m src.detect.arrows --export --limit 40

## Why the tracer has to be replaced rather than tuned

`src.assemble.tracing` reconstructs an edge by following ink from a shape until it reaches
another one. `reports/s5_graph_ged_test.json` records what that costs: with the tracer's own
measured ceilings - **edge recall 0.7239, precision 0.6386** - hdbpmn's median GED floor is
**12.63**, and 2.47% of pages could reach S5's target of 3. The bar is below the ceiling, so no
amount of parameter fitting inside that approach reaches it.

The published answer is to stop following ink. Arrow R-CNN (Schäfer et al., IJDAR 2021) detects
each arrow as an *object* and regresses its endpoints, and reports the rate of fully recognised
diagrams on scanned flowcharts rising from **37.7% to 78.6%** - where the 37.7% is the
heuristic generation of methods this project's tracer belongs to. The same group published the
hdBPMN corpus used here, and needed a purpose-built network (Sketch2Process) for it.

## What this module does, and why it is small

It is small because nothing new has to be invented or installed:

* **the training signal already exists.** Every annotated edge carries a `polyline`, so
  `polyline[0]` is the tail and `polyline[-1]` is the head, and `src`/`dst` name the nodes the
  arrow joins. Measured over 40 hdbpmn pages: **489 of 489 edges have a polyline**, median 3
  points. Nothing needs annotating.
* **the model already exists.** ultralytics 8.4.155 ships `PoseTrainer`, so an arrow is one
  object class with `kpt_shape: [2, 3]` - two keypoints, each `(x, y, visible)`. This is the
  same training path 9.1 already uses for shapes, not a new framework.
* **the images already exist.** 9.1.2 exported them per split; this writes labels beside them
  and records the scale it used rather than resizing anything again.

So the work is a label export and an inference rule: detect arrows, snap each keypoint to the
nearest node box, emit `(source -> target)` with direction taken from which end the head is.

## The box for an arrow is a convention, and it is stated here

An arrow has no natural bounding box - it is a stroke. This uses the axis-aligned box enclosing
the polyline, padded by `PAD` of its own diagonal so that a perfectly horizontal or vertical
arrow still has area, since a zero-height box is not trainable. **Precision against that box is
not the quantity of interest** - `reports/detection.md` already records what happens when it is,
rating the existing `arrowhead` class at 0.4318 mAP@0.5 while it finds 0.7550 of the objects at a
loose threshold, and notes that its boxes "are a size convention rather than a measurement". What
matters downstream is whether the two keypoints land near the right shapes, which is why the
evaluation this feeds is edge recall in `src.assemble.s5`, not mAP here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import ROOT

OUT = ROOT / "data" / "processed" / "arrows"

#: Corpora whose edges are annotated with a polyline. flowchartseg is absent: 10.1.5 records
#: that its truth carries no edges at all on the pages that matter.
SOURCES = ("hdbpmn", "fa_bresler")

SPLITS = ("train", "val", "test")

#: One class. An arrow is an arrow; what kind of flow it is belongs to the IR, not the detector.
CLASSES = ("arrow",)

#: Padding on the polyline's own box, as a share of its diagonal. A straight horizontal arrow
#: has zero height otherwise, and a zero-area box is not trainable.
PAD = 0.04

#: Smallest box side in normalised units, after padding. Below this ultralytics drops the label.
MIN_SIDE = 0.004

#: A node covering at least this share of the page is a container - a BPMN pool or lane - and is
#: **excluded from endpoint snapping**. This one constant is the difference between the approach
#: working and not working, and it was found by measuring the rule's ceiling with ground-truth
#: keypoints before training anything. Distance-to-box is zero anywhere *inside* a box, so a pool
#: spanning the page is the nearest node to every point on it and wins every arrow:
#:
#:     containers kept              hdbpmn exact (src, dst)  0.2128
#:     drop nodes >= 0.30 of page                            0.4758
#:     drop nodes >= 0.12 of page                            0.9512
#:     drop nodes >= 0.05 of page                            0.9985   (94 fewer edges scored)
#:
#: 0.12 keeps 2,071 of 2,086 val edges and recovers 95.1% of them exactly; 0.05 reaches 99.9% by
#: also discarding the edges that legitimately attach to a container. fa_bresler is unaffected -
#: it has no containers - and scores 0.8717 either way.
#:
#: **So the ceiling of this inference rule is 0.9512 against the tracer's measured 0.7239 edge
#: recall**, which is what makes replacing the tracer worth the training run rather than a guess.
CONTAINER_AREA = 0.12


def _edge_rows(page, diagram: dict[str, Any]) -> list[dict[str, Any]]:
    """`{box, tail, head}` in *image* pixels for every annotated edge with a polyline."""
    scale = float(getattr(page, "scale", 1.0) or 1.0)
    width, height = page.size
    rows = []
    for edge in diagram.get("edges", []):
        polyline = edge.get("polyline")
        if not polyline or len(polyline) < 2:
            continue
        points = np.asarray(polyline, dtype=float) * scale
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        pad = PAD * float(np.hypot(x1 - x0, y1 - y0))
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append(
            {
                "box": (x0, y0, x1, y1),
                "tail": tuple(points[0]),
                "head": tuple(points[-1]),
                "src": edge.get("src"),
                "dst": edge.get("dst"),
                "directed": bool(edge.get("directed", True)),
            }
        )
    return rows


def pose_lines(page, diagram: dict[str, Any]) -> list[str]:
    """YOLO-pose labels: `0 cx cy w h  tx ty 2  hx hy 2`, all normalised.

    Visibility is 2 - "labelled and visible" - for both keypoints, because an annotated polyline
    records where the stroke went: an endpoint is never guessed or occluded in this corpus.
    """
    width, height = page.size
    lines = []
    for row in _edge_rows(page, diagram):
        x0, y0, x1, y1 = row["box"]
        cx, cy = (x0 + x1) / 2.0 / width, (y0 + y1) / 2.0 / height
        bw, bh = (x1 - x0) / width, (y1 - y0) / height
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        tx, ty = row["tail"][0] / width, row["tail"][1] / height
        hx, hy = row["head"][0] / width, row["head"][1] / height
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, tx, ty, hx, hy)):
            continue
        lines.append(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            f"{tx:.6f} {ty:.6f} 2 {hx:.6f} {hy:.6f} 2"
        )
    return lines


def snap_targets(diagram: dict[str, Any], page_size, scale: float = 1.0) -> dict[str, list[float]]:
    """`{node id: box}` for the nodes an arrow endpoint may attach to, containers excluded."""
    width, height = page_size
    area = float(width) * float(height)
    out = {}
    for node in diagram.get("nodes", []):
        box = node.get("bbox")
        if not box:
            continue
        box = [float(v) * scale for v in box]
        if area > 0 and (box[2] * box[3]) / area >= CONTAINER_AREA:
            continue
        out[str(node["id"])] = box
    return out


def _distance(point, box) -> float:
    """Squared edge-to-edge distance from a point to a box; 0 when the point is inside it."""
    x, y, w, h = box
    dx = max(x - point[0], point[0] - (x + w), 0.0)
    dy = max(y - point[1], point[1] - (y + h), 0.0)
    return float(dx * dx + dy * dy)


def link(tail, head, targets: dict[str, list[float]]) -> tuple[str, str] | None:
    """`(source, target)` for one detected arrow, by snapping each endpoint to a node.

    The head decides direction, which is the whole reason to regress two keypoints rather than
    detect a box: an undirected trace has to infer orientation afterwards, and 10.1.5 measures
    that as worth only +1 page on S5.
    """
    if not targets:
        return None
    source = min(targets, key=lambda k: _distance(tail, targets[k]))
    target = min(targets, key=lambda k: _distance(head, targets[k]))
    if source == target:
        # A self-loop is real in an automaton and vanishingly rare in BPMN; the caller decides,
        # so it is returned rather than dropped here.
        return (source, target)
    return (source, target)


def export(out: Path = OUT, sources=SOURCES, limit: int | None = None) -> dict:
    """Write the pose dataset beside 9.1.2's already-exported images."""
    from src.assemble.corpus import pages

    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    index, counts, arrows = [], {s: 0 for s in SPLITS}, 0
    no_edges = missing_image = 0
    # `pages()` defaults to the held-out splits, because that is what S5 scores. A *detector*
    # needs the training pages too, and asking for them is the whole difference between a dataset
    # with 338 pages and one with 1,006.
    for page in pages(SPLITS):
        if page.source not in sources:
            continue
        if limit is not None and len(index) >= limit:
            break
        image = Path(page.image)
        if not image.is_file():
            missing_image += 1
            continue
        try:
            diagram = json.loads(Path(page.ir_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        lines = pose_lines(page, diagram)
        if not lines:
            # A page with no annotated arrow teaches the detector nothing about arrows and would
            # only contribute background; 9.1.2's negatives already cover that role.
            no_edges += 1
            continue
        split = page.split if page.split in SPLITS else "train"
        name = f"{page.source}__{page.id}"
        shutil.copyfile(image, out / "images" / split / f"{name}.png")
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        counts[split] += 1
        arrows += len(lines)
        index.append(
            {"name": name, "source": page.source, "split": split, "arrows": len(lines)}
        )

    yaml = "\n".join(
        [
            "# Phase 10.1.3 - generated by python -m src.detect.arrows. Do not hand-edit.",
            f"path: {out.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "kpt_shape: [2, 3]",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASSES)],
            "",
        ]
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

    by_source: dict[str, dict[str, int]] = {}
    for record in index:
        by_source.setdefault(record["source"], {s: 0 for s in SPLITS})[record["split"]] += 1
    result = {
        "root": str(out.relative_to(ROOT)),
        "pages": len(index),
        "arrows": arrows,
        "arrows_per_page": round(arrows / max(1, len(index)), 2),
        "per_split": counts,
        "per_source": by_source,
        "pages_without_annotated_arrows": no_edges,
        "pages_with_no_image": missing_image,
    }
    print(json.dumps(result, indent=2))
    return result


#: Where the pose run's weights land. Separate from 9.1's `RUNS/final`, because this is a
#: different task on a different label set and must not overwrite the shape detector.
RUNS = ROOT / "experiments" / "detect" / "arrows"
BEST = RUNS / "pose" / "weights" / "best.pt"

#: 1280, matching 9.1's chosen size. An arrow's keypoints are a few pixels of ink at the end of a
#: long thin object, so resolution is the one thing this task cannot trade away.
IMGSZ = 1280

#: Batch 4 rather than 9.1's 8: at 1280 with a pose head the activations are larger, and
#: `src.utils.gpu` caps this process at 20 GiB of a shared card.
BATCH = 4
EPOCHS = 40
SEED = 42


def train(
    epochs: int = EPOCHS,
    imgsz: int = IMGSZ,
    batch: int = BATCH,
    weights: str = "yolov8n-pose.pt",
    root: Path = OUT,
    name: str = "pose",
) -> dict:
    """Fit the arrow detector. `yolov8n-pose.pt` because the pose head is the point."""
    import time

    from ultralytics import YOLO

    from src.utils import gpu

    gpu.cap()
    if not (root / "data.yaml").is_file():
        raise FileNotFoundError(f"no dataset at {root}; run --export first")

    model = YOLO(weights)
    started = time.perf_counter()
    model.train(
        data=str(root / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=SEED,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        deterministic=True,
        plots=False,
        verbose=True,
    )
    seconds = time.perf_counter() - started
    metrics = model.val(data=str(root / "data.yaml"), imgsz=imgsz, split="val", verbose=False)
    result = {
        "weights_from": weights,
        "name": name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seconds": round(seconds, 1),
        "box_map50": round(float(metrics.box.map50), 4),
        "pose_map50": round(float(metrics.pose.map50), 4),
        "weights": str(RUNS / name / "weights" / "best.pt"),
    }
    (RUNS / f"train_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


#: Detector confidence floor for an arrow. Lower than 9.1's 0.25 for shapes on purpose: a missed
#: arrow is an `edge_delete` in S5 and a spurious one is an `edge_insert`, and the test split
#: carries **1,251 deletes against 325 inserts** - so recall is worth roughly four times
#: precision here, and the threshold is set from that asymmetry rather than from habit.
CONF = 0.15


@lru_cache(maxsize=1)
def _detector(weights: str = ""):
    from ultralytics import YOLO

    from src.utils import gpu

    gpu.cap()
    path = Path(weights) if weights else BEST
    if not path.is_file():
        raise FileNotFoundError(f"no arrow detector at {path}; run --train first")
    return YOLO(str(path))


def detect(image_path, conf: float = CONF, imgsz: int = IMGSZ, weights: str = "") -> list[dict]:
    """Every arrow the detector finds, with its two keypoints in image pixels."""
    prediction = _detector(weights).predict(str(image_path), verbose=False, imgsz=imgsz, conf=conf)[0]
    keypoints = prediction.keypoints
    out = []
    if keypoints is None:
        return out
    points = keypoints.xy.cpu().numpy()
    for index, box in enumerate(prediction.boxes):
        if index >= len(points) or len(points[index]) < 2:
            continue
        tail, head = points[index][0], points[index][1]
        out.append(
            {
                "id": f"a{index:03d}",
                "tail": (float(tail[0]), float(tail[1])),
                "head": (float(head[0]), float(head[1])),
                "score": float(box.conf.item()),
            }
        )
    return out


def edges_for(page, diagram: dict[str, Any], conf: float = CONF, weights: str = "") -> list[dict]:
    """Arrow-derived edges for a predicted page, in the IR's own edge shape.

    Replaces 10.1.3's ink-following trace rather than supplementing it. Duplicate `(src, dst)`
    pairs are collapsed keeping the most confident, because two detections of one drawn arrow are
    one edge and S5 counts edges as a set of endpoint pairs.
    """
    targets = snap_targets(diagram, page.size)
    if not targets:
        return []
    best: dict[tuple[str, str], dict] = {}
    for arrow in detect(page.image, conf=conf, weights=weights):
        pair = link(arrow["tail"], arrow["head"], targets)
        if pair is None:
            continue
        keep = best.get(pair)
        if keep is None or arrow["score"] > keep["confidence"]:
            best[pair] = {
                "id": arrow["id"],
                "src": pair[0],
                "dst": pair[1],
                "directed": True,
                "label": "",
                "polyline": [list(arrow["tail"]), list(arrow["head"])],
                "confidence": round(arrow["score"], 4),
                "attrs": {"self_loop": pair[0] == pair[1], "from": "arrows"},
            }
    return list(best.values())


def apply(page, diagram: Any, conf: float = CONF, weights: str = "") -> int:
    """Replace `diagram`'s edges with arrow-derived ones, in place. Returns how many."""
    from src.ir.model import Edge

    plain = diagram.to_dict() if hasattr(diagram, "to_dict") else diagram
    rows = edges_for(page, plain, conf=conf, weights=weights)
    if hasattr(diagram, "edges"):
        diagram.edges = [
            Edge(
                id=r["id"],
                src=r["src"],
                dst=r["dst"],
                directed=r["directed"],
                label=r["label"],
                polyline=r["polyline"],
                confidence=r["confidence"],
                attrs=r["attrs"],
            )
            for r in rows
        ]
    else:
        diagram["edges"] = rows
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--weights", default="yolov8n-pose.pt")
    parser.add_argument("--name", default="pose")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not (args.export or args.train):
        parser.error("nothing to do: pass --export or --train")
    if args.export:
        export(args.out, SOURCES, args.limit)
    if args.train:
        train(
            epochs=args.epochs,
            batch=args.batch,
            weights=args.weights,
            root=args.out,
            name=args.name,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 9.1.3 - the training run, and the knob the plan names that this architecture does not have.

    python -m src.detect.train --sweep          # imgsz sweep at a short budget
    python -m src.detect.train --final          # the long run at the chosen size
    python -m src.detect.train --anchors        # the anchor analysis, no training

The plan's detail line is "anchors / imgsz / epochs; mixed precision on RTX 4500". Three of
those four are real knobs here and one is not, which is the first thing to record:

## Anchors

**YOLOv8 is anchor-free.** There is no anchor set to tune - the head predicts a distribution
over distances from each grid cell to the four box sides (DFL), and the only size prior in the
model is the stride of the three detection levels, 8/16/32 px of input. So the plan's knob
cannot be turned, and the honest substitute is to ask what it *would* have been set to and
whether the strides can express it. `anchor_analysis` runs k-means over the training boxes'
widths and heights - the classical anchor derivation - and compares the clusters to the
receptive sizes the three levels cover.

That is not a formality. It is where the arrowhead problem becomes concrete: the export's
smallest cluster is a class of boxes only a few cells across at the P3 stride, and a box
smaller than about `stride` is one the level below it cannot represent at all. 9.1.5 acts on
that; this task measures it.

## imgsz, epochs, mixed precision

Swept directly. AMP is timed with and without rather than assumed free, because bf16 on an
RTX 4500 Ada is a hardware claim and 0.1.2 recorded it as supported rather than measured under
load.

## What it measured

**Anchors.** 41,893 training boxes. The k-means anchor set an anchor-based head would have been
given runs from **[16.4, 15.2] to [790.8, 417.5]** at 896 px - a 50x range of scales in one
corpus - and the smallest cluster is the arrowhead cluster. Against v8's strides: only 0.84% of
all boxes are under one P3 cell, so the finest level can represent nearly everything, but
**35.2% are under one P4 cell** and per class that number is **91.25% for `arrowhead`** against
0.11% for `rounded-rect` and 0.00% for `diamond` and `double-circle`. The arrowhead's median
short side is **11.2 px** - one and a half P3 cells - where a rounded-rect's is 48.2 and a
double-circle's 105.9. So the head is not merely a small class, it is a class that lives on a
single feature level while every other class is described by all three, and that is the
mechanism behind every arrowhead number in 9.1.

**imgsz is the biggest knob in the task, and it is almost entirely the small class.** 10 epochs
each: 640 -> 0.7972, 896 -> 0.8683, 1280 -> 0.8729. The first step is worth **+0.0711** and the
second only **+0.0046**, so on the pooled metric resolution saturates by 896. Per class it does
not: `arrowhead` runs **0.2157 -> 0.3085 -> 0.3826**, a monotone **+0.167** across the range and
still climbing where the pooled curve has flattened. Resolution buys pixels for the class that
has none, and averaging that into a mAP hides it - the same shape as 7.1.4's argument for
reporting macro F1 next to accuracy.

**Mixed precision is not free, and it is the one result here that contradicts the usual claim.**
At 896 with everything else fixed, AMP trains **1.166x faster** and scores **0.005 lower**
mAP@0.5 (0.8683 against 0.8733). Both directions are small, and the honest reading is that a
17% speedup costs half a point - worth taking for a sweep, worth reconsidering for a final run,
and worth *measuring* rather than assuming, which is why 0.1.2's "bf16 supported" was recorded
as a capability and this is the first time it has been priced under load.

**The long run.** 40 epochs at 1280: **mAP@0.5 0.9210**, mAP@0.5:0.95 0.7813, hand-drawn slice
0.8907, and **`arrowhead` 0.5197**. The epoch axis at that size is 10 -> 40 for **+0.0481**, so
the last thirty epochs are worth ten times what the last resolution step was.

**That last number settles part of 9.1.1's open question in an unexpected direction.** 9.1.1
chose yolov8n over RT-DETR and named the condition for reversing it: RT-DETR's arrowhead AP of
**0.4863**. The same yolov8n, given 40 epochs at 1280 instead of 12 at 896, reaches **0.5197** -
past RT-DETR without tiling, without 10.6x the parameters, and for a fifth of RT-DETR's training
time even at this longer schedule. This is **not** a claim that yolov8n beats RT-DETR: RT-DETR
at the same budget was never run and would presumably also improve. What it does establish is
that **a large part of 9.1.1's architecture gap was a budget artefact rather than an
architectural one** - which is worth knowing before anyone reads a 12-epoch bake-off as a
statement about model families.

One measurement in the sweep is not usable and is flagged rather than quoted: **per-page latency
across the sweep cells is contaminated by cold-start**. Each cell pays CUDA and cuDNN warm-up on
its first page, and the 1280 final run reads 19.4 ms against the 1280 sweep cell's 35.4 ms for
the same input size. 9.1.5 adds a throwaway forward pass before timing; these numbers should be
read as an upper bound and not compared against each other.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.detect.choice import RUNS, predict_ultralytics, truths
from src.detect.classes import CLASSES
from src.detect.dataset import OUT as DATA
from src.detect.pretrained import weights as pretrained_weights
from src.utils.config import ROOT

SEED = 42
WEIGHTS = RUNS / "final" / "weights" / "best.pt"

#: The short budget used to choose a size. Long enough for the ordering to be stable, short
#: enough that three sizes fit in an afternoon.
SWEEP_EPOCHS = 10
SWEEP_SIZES = (640, 896, 1280)
FINAL_EPOCHS = 40

#: YOLOv8's three detection levels, as input pixels per output cell.
STRIDES = (8, 16, 32)


# ------------------------------------------------------------------------------------------
# anchors - the knob that is not there
# ------------------------------------------------------------------------------------------


def training_boxes(root: Path = DATA, split: str = "train") -> np.ndarray:
    """Every training box as `(width, height)` in pixels of the exported page."""
    from PIL import Image

    sizes = []
    for label in sorted((root / "labels" / split).glob("*.txt")):
        page = root / "images" / split / f"{label.stem}.png"
        if not page.is_file():
            continue
        with Image.open(page) as handle:
            width, height = handle.size
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, _, _, w, h = line.split()
            sizes.append([float(w) * width, float(h) * height])
    return np.asarray(sizes, dtype=float)


def anchor_analysis(root: Path = DATA, k: int = 9, imgsz: int = 896) -> dict:
    """The anchor set a v5-era model would have used, and whether v8's strides reach it.

    k = 9 is the classical three-per-level choice. The comparison that matters is the smallest
    cluster against the finest stride: a box whose short side is under one P3 cell cannot be
    assigned a positive sample at that level whatever the loss says.
    """
    from sklearn.cluster import KMeans

    sizes = training_boxes(root)
    if not len(sizes):
        return {"boxes": 0}
    # Boxes are measured on exported pages of varying size; the model sees them letterboxed to
    # `imgsz`, so the comparison against a stride has to be in that frame.
    scale = imgsz / 1280.0
    scaled = sizes * scale

    fitted = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(scaled)
    centres = np.sort(fitted.cluster_centers_, axis=0)
    short_side = scaled.min(axis=1)
    per_class = {}
    labels_by_class = _boxes_by_class(root, imgsz)
    for name, values in labels_by_class.items():
        if len(values):
            per_class[name] = {
                "median_short_side_px": round(float(np.median(values.min(axis=1))), 1),
                "under_p3_cell": round(float((values.min(axis=1) < STRIDES[0]).mean()), 4),
                "under_p4_cell": round(float((values.min(axis=1) < STRIDES[1]).mean()), 4),
            }
    return {
        "boxes": int(len(sizes)),
        "imgsz": imgsz,
        "kmeans_anchors": [[round(float(w), 1), round(float(h), 1)] for w, h in centres],
        "smallest_anchor_short_side": round(float(centres[0].min()), 1),
        "strides": list(STRIDES),
        "share_under_p3_cell": round(float((short_side < STRIDES[0]).mean()), 4),
        "share_under_p4_cell": round(float((short_side < STRIDES[1]).mean()), 4),
        "per_class": per_class,
        "note": (
            "YOLOv8 is anchor-free; these clusters are what an anchor-based head would have "
            "been given, reported to show what the box-size distribution asks for."
        ),
    }


def _boxes_by_class(root: Path, imgsz: int) -> dict[str, np.ndarray]:
    from PIL import Image

    buckets: dict[str, list[list[float]]] = {name: [] for name in CLASSES}
    for label in sorted((root / "labels" / "train").glob("*.txt")):
        page = root / "images" / "train" / f"{label.stem}.png"
        if not page.is_file():
            continue
        with Image.open(page) as handle:
            width, height = handle.size
        scale = imgsz / max(height, width)
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            index, _, _, w, h = line.split()
            buckets[CLASSES[int(index)]].append(
                [float(w) * width * scale, float(h) * height * scale]
            )
    return {name: np.asarray(values, dtype=float) for name, values in buckets.items()}


# ------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------


def train(
    name: str,
    weights: str = "yolov8n.pt",
    epochs: int = SWEEP_EPOCHS,
    imgsz: int = 896,
    batch: int = 8,
    amp: bool = True,
    root: Path = DATA,
):
    from ultralytics import YOLO

    # `pretrained.weights` so a bare `yolov8n.pt` lands in `models/pretrained/` rather than in
    # whatever directory this was run from - which for every command in this repo is the repo root.
    model = YOLO(pretrained_weights(weights))
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
        val=False,
        plots=False,
        verbose=False,
        amp=amp,
    )
    return model, time.perf_counter() - started


def score(model, imgsz: int, root: Path = DATA, split: str = "val") -> dict:
    from src.detect.metrics import by_source, evaluate

    reference, sources = truths(split, root)
    predictions, latency = predict_ultralytics(model, split, imgsz, root)
    result = evaluate(predictions, reference)
    result["page_latency_ms"] = round(latency, 2)
    result["by_source"] = by_source(predictions, reference, sources)
    return result


def sweep(root: Path = DATA, sizes=SWEEP_SIZES, epochs: int = SWEEP_EPOCHS) -> dict:
    """imgsz at a fixed epoch budget, plus one AMP-off control at the middle size."""
    rows = []
    for imgsz in sizes:
        batch = 16 if imgsz <= 640 else (8 if imgsz <= 896 else 4)
        model, seconds = train(f"sweep_{imgsz}", epochs=epochs, imgsz=imgsz, batch=batch, root=root)
        rows.append(
            {
                "imgsz": imgsz,
                "amp": True,
                "train_seconds": round(seconds, 1),
                "batch": batch,
                **score(model, imgsz, root),
            }
        )
    model, seconds = train(
        "sweep_896_noamp", epochs=epochs, imgsz=896, batch=8, amp=False, root=root
    )
    rows.append(
        {
            "imgsz": 896,
            "amp": False,
            "train_seconds": round(seconds, 1),
            "batch": 8,
            **score(model, 896, root),
        }
    )
    best = max((r for r in rows if r["amp"]), key=lambda r: r["map50"])
    amp_on = next(r for r in rows if r["imgsz"] == 896 and r["amp"])
    amp_off = next(r for r in rows if not r["amp"])
    return {
        "epochs": epochs,
        "rows": rows,
        "chosen_imgsz": best["imgsz"],
        "amp_speedup": round(amp_off["train_seconds"] / max(amp_on["train_seconds"], 1e-9), 3),
        "amp_map50_delta": round(amp_on["map50"] - amp_off["map50"], 4),
    }


def final(imgsz: int, epochs: int = FINAL_EPOCHS, root: Path = DATA) -> dict:
    batch = 16 if imgsz <= 640 else (8 if imgsz <= 896 else 4)
    model, seconds = train("final", epochs=epochs, imgsz=imgsz, batch=batch, root=root)
    result = {"imgsz": imgsz, "epochs": epochs, "batch": batch, "train_seconds": round(seconds, 1)}
    result.update(score(model, imgsz, root))
    result["weights"] = str(WEIGHTS.relative_to(ROOT)) if WEIGHTS.is_file() else None
    return result


def _epochs_effect(result: dict, imgsz: int) -> dict:
    """The epoch axis, assembled from runs the phase already had to pay for.

    The plan asks for epochs as a knob and a dedicated sweep over it would double the GPU bill
    for a curve whose shape is not in doubt. Instead the three points that exist at the same
    input size are collected: the 10-epoch sweep cell, 9.1.1's 12-epoch bake-off arm, and this
    task's long run. Not a controlled sweep - the runs differ in batch size where the size
    differs - and labelled as such, but it prices the last thirty epochs, which is the decision
    Phase 15's retraining cadence actually needs.
    """
    points = []
    for row in result.get("sweep", {}).get("rows", []):
        if row["imgsz"] == imgsz and row["amp"]:
            points.append(
                {
                    "epochs": result["sweep"]["epochs"],
                    "map50": row["map50"],
                    "train_seconds": row["train_seconds"],
                    "from": "9.1.3 sweep",
                }
            )
    bakeoff = RUNS / "choice.json"
    if bakeoff.is_file():
        for row in json.loads(bakeoff.read_text(encoding="utf-8")):
            if row.get("arm") == "yolov8n" and row.get("imgsz") == imgsz:
                points.append(
                    {
                        "epochs": row["epochs"],
                        "map50": row["map50"],
                        "train_seconds": row["train_seconds"],
                        "from": "9.1.1 bake-off",
                    }
                )
    final_row = result["final"]
    points.append(
        {
            "epochs": final_row["epochs"],
            "map50": final_row["map50"],
            "train_seconds": final_row["train_seconds"],
            "from": "9.1.3 final",
        }
    )
    points.sort(key=lambda p: p["epochs"])
    gain = round(points[-1]["map50"] - points[0]["map50"], 4) if len(points) > 1 else None
    return {
        "imgsz": imgsz,
        "points": points,
        "map50_gain_over_range": gain,
        "note": "assembled from separate runs at one input size, not a controlled sweep",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--imgsz", type=int, default=None, help="skip the sweep and train at this size")
    ap.add_argument("--epochs", type=int, default=FINAL_EPOCHS)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=RUNS / "train.json")
    args = ap.parse_args(argv)

    if not (args.data / "data.yaml").is_file():
        print(f"no dataset at {args.data} (run `python -m src.detect.dataset`)", file=sys.stderr)
        return 1

    result: dict = {}
    if args.anchors or not (args.sweep or args.final):
        result["anchors"] = anchor_analysis(args.data)
    if args.sweep:
        result["sweep"] = sweep(args.data, epochs=SWEEP_EPOCHS)
    if args.final:
        imgsz = args.imgsz or result.get("sweep", {}).get("chosen_imgsz") or 896
        result["final"] = final(imgsz, args.epochs, args.data)
        result["epochs_effect"] = _epochs_effect(result, imgsz)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

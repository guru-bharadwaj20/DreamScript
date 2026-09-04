"""Phase 9.1.6 - the IoU threshold, and the case where class-agnostic suppression is wrong.

    python -m src.detect.nms

Two knobs, swept on the validation split with one set of weights:

**IoU threshold.** Higher keeps more overlapping boxes (more recall, more duplicates); lower
merges harder (fewer duplicates, and eventually it deletes real objects that genuinely
overlap). This corpus has a specific reason to care: a BPMN **pool** is a `rectangle` that
*contains* every task in its lane, so a container and its contents overlap by construction,
and a `double-circle` is one ring drawn inside another. Suppression tuned on COCO's assumption
that overlapping boxes of the same class are usually duplicates is tuned on a premise this
corpus breaks.

**Class-agnostic against per-class.** Per-class is the default and only suppresses within a
class; class-agnostic suppresses across classes, which is right when the failure mode is one
object claimed by two labels and wrong when two different objects are nested. The prediction
going in was that this corpus breaks the premise, because a BPMN **pool** is a `rectangle` that
contains every task in its lane, so the sweep also counts per setting how many surviving boxes
are fully contained in a surviving box of another class. **That prediction is wrong and the
counter is what showed it** - see below. The counter is kept, because a number that refutes the
argument it was built to support is the most useful kind.

## What it measured

86,430 candidate boxes from 9.1.3's weights over 308 validation pages, suppressed twelve ways.

**Recommended: IoU 0.6, per-class.** Two of the three criteria choose it; mAP@0.5:0.95 dissents
for 0.7, and the spread of that criterion across the entire grid is **0.0045**, which is not a
preference. The default ultralytics threshold of 0.7 costs 0.0023 mAP@0.5.

**Suppression is an arrowhead knob and almost nothing else.** Across the grid the pooled figure
moves **0.0230** (0.9011 at IoU 0.9 to 0.9241 at 0.6) while `arrowhead` moves **0.1760** (0.3627
to 0.5387) - eight times as far - and `rectangle` moves 0.0194, all of it between 0.3 and 0.45.
The peak of 0.5387 is above 9.1.4's 0.5197 at the default threshold, so **setting one number
correctly bought more arrowhead AP than 9.1.5's tiling did at 16.5x the latency.** The reason is
that heads cluster densely along a polyline and are the only class where several true instances
sit within an IoU of each other, so they are the only class a threshold can destroy.

## The prediction this task was built on is wrong, and the counter that was added to check it is
## what proves it

The docstring above argued that this corpus breaks NMS's premise because a BPMN **pool** is a
`rectangle` containing every task in its lane, so class-agnostic suppression should eat the
containment structure Phase 10 depends on. **It does not. `rectangle` costs exactly 0.0000 under
class-agnostic suppression, and only 5 nested pairs are lost out of 2,744.**

The error is a confusion between *containment* and *IoU*, and it is worth stating plainly
because it is easy to make: a task inside a pool has an intersection equal to the task and a
union equal to the pool, so its IoU with its container is `task_area / pool_area` - a small
number. **Nesting produces low IoU, not high IoU, so NMS never sees it.** The nesting counter
was added to quantify the damage and instead demonstrated there is none, which is the more
useful outcome and the reason to compute a number rather than argue for it.

**Class-agnostic still loses, uniformly - 0.0181 mAP@0.5 at every threshold in the grid - and
the per-class breakdown shows the loss is not where the pooled figure implies.** Decomposed:

    parallelogram   -0.1155      but 7 instances; this is one box
    double-circle   -0.0198
    circle          -0.0094
    everything else  0.0000 +/- 0.0004

Those three account for the whole 0.0181 (0.1155 + 0.0198 + 0.0094, over eight classes). So
**four fifths of class-agnostic's apparent cost is a single box in a 7-instance class**, and the
defensible part is the remaining 0.0292 spread across `double-circle` and `circle` - which *is*
the predicted mechanism, just not the predicted class: **a double-circle is a ring drawn inside
a ring of a different label, two boxes with genuinely high IoU, and agnostic suppression deletes
one of them.** `circle` losing 0.0094 is the same event seen from the other side.

The recommendation is therefore per-class suppression, and the honest reason is not "it protects
BPMN pools" - nothing threatened those - but that **it protects concentric shapes**, which is a
small effect on a 74-instance class and would be a large one on a corpus of state machines,
where `double-circle` is how an accepting state is drawn and 2.1.4 froze it as its own shape for
exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.detect.choice import RUNS, truths
from src.detect.classes import CLASSES
from src.detect.dataset import OUT as DATA
from src.detect.train import WEIGHTS

IOU_GRID = (0.3, 0.45, 0.6, 0.7, 0.8, 0.9)


def raw_predictions(model, split: str = "val", imgsz: int = 896, root: Path = DATA) -> list[dict]:
    """Detections with suppression effectively off, so the sweep re-runs it rather than the model.

    `iou=0.95` leaves all but exact duplicates, and the boxes are kept per page so every
    setting below is applied to the identical candidate set - which is the only way the sweep
    measures suppression instead of measuring inference noise.
    """
    rows: list[dict] = []
    for path in sorted((root / "images" / split).glob("*.png")):
        result = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=0.001,
            iou=0.95,
            max_det=1000,
            verbose=False,
            device=0,
        )[0]
        boxes = result.boxes
        if boxes is None or not len(boxes):
            continue
        for box, score, cls in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), strict=True
        ):
            rows.append(
                {
                    "image": path.stem,
                    "cls": CLASSES[int(cls)],
                    "xyxy": [float(v) for v in box],
                    "score": float(score),
                }
            )
    return rows


def suppress(rows: list[dict], threshold: float, agnostic: bool) -> list[dict]:
    """Greedy NMS over an existing candidate set, per image and optionally per class."""
    from src.detect.tiling import nms

    by_image: dict[str, list[dict]] = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)

    kept: list[dict] = []
    for candidates in by_image.values():
        groups: dict[object, list[dict]] = {}
        for row in candidates:
            groups.setdefault(None if agnostic else row["cls"], []).append(row)
        for group in groups.values():
            boxes = np.asarray([r["xyxy"] for r in group], dtype=float)
            scores = np.asarray([r["score"] for r in group], dtype=float)
            kept.extend(group[i] for i in nms(boxes, scores, threshold))
    return kept


def nesting(rows: list[dict], min_score: float = 0.25) -> dict:
    """How many confident boxes sit wholly inside a confident box of a different class.

    The pools-contain-tasks structure Phase 10 depends on, counted rather than assumed - and
    the quantity class-agnostic suppression destroys.
    """
    confident = [r for r in rows if r["score"] >= min_score]
    by_image: dict[str, list[dict]] = {}
    for row in confident:
        by_image.setdefault(row["image"], []).append(row)
    nested = 0
    for candidates in by_image.values():
        for inner in candidates:
            ix1, iy1, ix2, iy2 = inner["xyxy"]
            for outer in candidates:
                if outer is inner or outer["cls"] == inner["cls"]:
                    continue
                ox1, oy1, ox2, oy2 = outer["xyxy"]
                if ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2:
                    nested += 1
                    break
    return {"confident_boxes": len(confident), "nested_pairs": nested}


def run(weights: Path = WEIGHTS, root: Path = DATA, split: str = "val", imgsz: int = 896) -> dict:
    from ultralytics import YOLO

    from src.detect.metrics import evaluate

    model = YOLO(str(weights))
    reference, _ = truths(split, root)
    candidates = raw_predictions(model, split, imgsz, root)

    rows = []
    for agnostic in (False, True):
        for threshold in IOU_GRID:
            kept = suppress(candidates, threshold, agnostic)
            scored = evaluate(kept, reference)
            rows.append(
                {
                    "agnostic": agnostic,
                    "iou": threshold,
                    "kept": len(kept),
                    "map50": scored["map50"],
                    "map50_95": scored["map50_95"],
                    "arrowhead_ap50": scored["per_class"]["arrowhead"]["ap50"],
                    "rectangle_ap50": scored["per_class"]["rectangle"]["ap50"],
                    # Kept in full because the first run of this sweep found class-agnostic
                    # suppression costing 0.018 mAP while destroying only 5 nested pairs and
                    # leaving `arrowhead` and `rectangle` untouched - which means the cost is
                    # in a class the two summary columns cannot see, and a metrics table that
                    # cannot answer "which one" invites a guess.
                    "per_class": {k: v["ap50"] for k, v in scored["per_class"].items()},
                    **nesting(kept),
                }
            )

    per_class = [r for r in rows if not r["agnostic"]]
    by_map50_95 = max(per_class, key=lambda r: r["map50_95"])["iou"]
    by_map50 = max(per_class, key=lambda r: r["map50"])["iou"]
    by_head = max(per_class, key=lambda r: r["arrowhead_ap50"] or 0)["iou"]
    # Three criteria over one grid, and they do not have to agree. The recommendation is the
    # threshold that wins the most of them, which is a rule stated in advance rather than a
    # column picked after seeing the table - and where mAP@0.5:0.95 dissents it is worth
    # noting how little it dissents by, since a 0.004 spread is not a preference.
    votes = {iou: [by_map50_95, by_map50, by_head].count(iou) for iou in IOU_GRID}
    recommended = max(IOU_GRID, key=lambda iou: (votes[iou], -abs(iou - 0.6)))

    def at(iou: float, agnostic: bool) -> dict:
        return next(r for r in rows if r["iou"] == iou and r["agnostic"] is agnostic)

    paired = (at(recommended, False), at(recommended, True))
    agnostic_by_class = {
        name: round(paired[0]["per_class"][name] - paired[1]["per_class"][name], 4)
        for name in paired[0]["per_class"]
        if paired[0]["per_class"][name] is not None and paired[1]["per_class"][name] is not None
    }
    return {
        "agnostic_cost_by_class": dict(sorted(agnostic_by_class.items(), key=lambda kv: -kv[1])),
        "candidates": len(candidates),
        "grid": rows,
        "chosen": {"iou": recommended, "agnostic": False},
        "chosen_by": {"map50_95": by_map50_95, "map50": by_map50, "arrowhead_ap50": by_head},
        "map50_95_spread": round(
            max(r["map50_95"] for r in per_class) - min(r["map50_95"] for r in per_class), 4
        ),
        "agnostic_cost_map50": round(paired[0]["map50"] - paired[1]["map50"], 4),
        "agnostic_cost_map50_95": round(paired[0]["map50_95"] - paired[1]["map50_95"], 4),
        "nested_pairs_lost_to_agnostic": paired[0]["nested_pairs"] - paired[1]["nested_pairs"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=RUNS / "nms.json")
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        print(f"no weights at {args.weights}", file=sys.stderr)
        return 1
    result = run(args.weights, args.data, args.split, args.imgsz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

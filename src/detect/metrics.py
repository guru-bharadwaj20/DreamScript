"""Phase 9.1 - one detection metric, so three frameworks can be compared at all.

Ultralytics reports its own mAP, torchvision reports none, and the two do not agree even on
what "mAP@0.5" means: ultralytics matches predictions to ground truth greedily by IoU within a
class after its own confidence filter, while a COCO evaluation ranks *all* predictions of a
class across the whole split before matching. Quoting a YOLO number beside a Faster R-CNN
number computed some other way is comparing the frameworks' reporting conventions rather than
their detectors, so every arm in 9.1.1-9.1.7 is scored here instead.

## What is implemented

COCO's average precision, with its conventions stated because they change the number:

  * per class, predictions from the entire split are pooled and sorted by confidence;
  * a prediction matches the highest-IoU unmatched ground truth of its class on its own image;
  * precision is made monotone (`precision[i] = max(precision[i:])`) and integrated at 101
    recall points, which is what makes it "COCO AP" rather than the older VOC-11 or the exact
    area under the raw curve;
  * a class with no ground truth in the split contributes `nan` and is left out of the mean -
    never scored as 0, which would silently punish a model for a class the split lacks.

`mAP@0.5` is the mean over classes at IoU 0.5; `mAP@0.5:0.95` averages that over the ten
thresholds 0.50, 0.55 ... 0.95. Both are computed from one matching pass per threshold.

## Why `basis` is carried through

9.1.2 derives the arrowhead boxes from a waypoint and a size convention, and an IoU-based
metric cannot tell a detector that found the head badly from a convention that put the box in
the wrong place. `evaluate` therefore also reports each class's AP at IoU 0.25 alongside 0.5,
because a localisation-tolerant threshold is the honest way to ask whether the head was found
at all - and the gap between the two is the size of the convention's contribution.
"""

from __future__ import annotations

import numpy as np

from src.detect.classes import CLASSES

#: COCO's threshold sweep, plus a loose one for the derived class. 0.25 is reported separately
#: and never folded into mAP@0.5:0.95.
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
LOOSE_IOU = 0.25


def iou_matrix(predictions: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two `(n, 4)` arrays of `xyxy` boxes."""
    if not len(predictions) or not len(truths):
        return np.zeros((len(predictions), len(truths)), dtype=float)
    px1, py1, px2, py2 = (predictions[:, i][:, None] for i in range(4))
    tx1, ty1, tx2, ty2 = (truths[:, i][None, :] for i in range(4))
    inter_w = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
    inter_h = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
    inter = inter_w * inter_h
    area_p = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    area_t = np.clip(tx2 - tx1, 0, None) * np.clip(ty2 - ty1, 0, None)
    union = area_p + area_t - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(matched: np.ndarray, scores: np.ndarray, positives: int) -> float:
    """101-point interpolated AP from a boolean match vector already sorted by score."""
    if positives == 0:
        return float("nan")
    if not len(matched):
        return 0.0
    order = np.argsort(-scores, kind="stable")
    hits = matched[order].astype(float)
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1e-12)
    # Monotone envelope, right to left: COCO integrates the best precision still available at
    # or beyond each recall, not the jagged curve.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, points, side="left")
    sampled = np.where(
        indices < len(precision), precision[np.clip(indices, 0, len(precision) - 1)], 0.0
    )
    return float(sampled.mean())


def _match(
    preds: list[dict], truths: list[dict], threshold: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy highest-IoU matching per image, at one threshold, for one class."""
    by_image_truth: dict[str, list[list[float]]] = {}
    for truth in truths:
        by_image_truth.setdefault(truth["image"], []).append(truth["xyxy"])
    positives = sum(len(v) for v in by_image_truth.values())

    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used: dict[str, np.ndarray] = {
        image: np.zeros(len(boxes), dtype=bool) for image, boxes in by_image_truth.items()
    }
    matched = np.zeros(len(order), dtype=bool)
    scores = np.array([preds[i]["score"] for i in order], dtype=float)
    for rank, index in enumerate(order):
        prediction = preds[index]
        candidates = by_image_truth.get(prediction["image"])
        if not candidates:
            continue
        ious = iou_matrix(
            np.asarray([prediction["xyxy"]], dtype=float), np.asarray(candidates, dtype=float)
        )[0]
        ious = np.where(used[prediction["image"]], -1.0, ious)
        best = int(np.argmax(ious)) if len(ious) else -1
        if best >= 0 and ious[best] >= threshold:
            used[prediction["image"]][best] = True
            matched[rank] = True
    return matched, scores, positives


def evaluate(
    predictions: list[dict], truths: list[dict], classes=CLASSES, thresholds=IOU_THRESHOLDS
) -> dict:
    """Per-class AP and the two mAPs.

    Both arguments are flat lists of `{image, cls, xyxy}` with `score` on the predictions -
    a format both ultralytics and torchvision can be flattened into without either one's
    conventions coming along.
    """
    per_class: dict[str, dict] = {}
    for name in classes:
        class_preds = [p for p in predictions if p["cls"] == name]
        class_truths = [t for t in truths if t["cls"] == name]
        aps = {}
        for threshold in thresholds:
            matched, scores, positives = _match(class_preds, class_truths, threshold)
            aps[threshold] = average_precision(matched, scores, positives)
        matched, scores, positives = _match(class_preds, class_truths, LOOSE_IOU)
        per_class[name] = {
            "instances": len(class_truths),
            "predictions": len(class_preds),
            "ap50": aps[0.5],
            "ap50_95": _nanmean(list(aps.values())),
            "ap25": average_precision(matched, scores, positives),
        }

    def mean(key: str) -> float:
        values = [v[key] for v in per_class.values() if not np.isnan(v[key])]
        return float(np.mean(values)) if values else float("nan")

    return {
        "per_class": {k: {m: _round(x) for m, x in v.items()} for k, v in per_class.items()},
        "map50": round(mean("ap50"), 4),
        "map50_95": round(mean("ap50_95"), 4),
        "map25": round(mean("ap25"), 4),
        "classes_scored": int(sum(1 for v in per_class.values() if v["instances"] > 0)),
        "instances": int(sum(v["instances"] for v in per_class.values())),
    }


def _nanmean(values: list[float]) -> float:
    """`np.nanmean` of an all-nan list is a warning and a nan; here it is just a nan."""
    kept = [v for v in values if not np.isnan(v)]
    return float(np.mean(kept)) if kept else float("nan")


def _round(x):
    if isinstance(x, float):
        return None if np.isnan(x) else round(x, 4)
    return x


def by_source(predictions: list[dict], truths: list[dict], sources: dict[str, str]) -> dict:
    """The same evaluation restricted to each source's pages.

    9.1.2 mixed 693 photographs with 1,619 rendered pages, so a pooled mAP is three-quarters a
    statement about rendered pages. This is how the two get reported apart.
    """
    out = {}
    for source in sorted(set(sources.values())):
        images = {image for image, s in sources.items() if s == source}
        out[source] = evaluate(
            [p for p in predictions if p["image"] in images],
            [t for t in truths if t["image"] in images],
        )
    return out

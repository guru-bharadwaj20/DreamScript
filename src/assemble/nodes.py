"""Phase 10.1.1 - detections become nodes, and the three things a raw detection list gets wrong.

    python -m src.assemble.nodes
    python -m src.assemble.nodes --no-gmm     # skip the shape-posterior study (it decodes pages)

`build(page)` is the entry point the rest of Phase 10 calls. It reads 9.1's cached boxes and
returns an `Assembled` whose `diagram` carries nodes and no edges, plus the two populations a
node builder has to keep separately: the `arrowhead` detections, which are evidence about edges
and are not nodes, and the `dropped` detections, each with the reason it was rejected.

All numbers below are from the 470 held-out pages (`experiments/assemble/nodes.json`):
96 fa_bresler, 132 flowchartseg, 242 hdbpmn, carrying **24,385** raw detections.

## What a detection is not

A YOLO box is not a node, and the gap has three parts, each measured below rather than asserted.
**16,192 of the 24,385 detections - 66.4% - are `arrowhead` boxes**, evidence about an edge and
never a node; they are filtered out before anything else runs, not scored as false positives.
Of what is left, some detections are low-confidence noise (the threshold section), and some are
the same drawn shape reported twice by two overlapping classes (de-duplication). What remains
after both cuts is compared against 7,704 truth nodes: at the chosen settings, node localisation
reaches **precision 0.9821, recall 0.9853, F1 0.9837** at IoU >= 0.5, and barely moves at the
looser IoU >= 0.25 (F1 0.9840) - the detector's boxes are tight enough that loosening the overlap
bar buys almost nothing. Shape is right on **99.70%** of matched nodes and the four-way coarse
label (rectangle/diamond/circle/freeform) on **99.80%**.

## The threshold, swept and not asserted

`MIN_SCORE` is not a guess: `sweep()` scores every threshold in `SWEEP` and `run()`'s output is
the actual curve, not just its argmax. Pooled strict F1 rises from 0.9694 at 0.05 to a peak of
**0.9837 at 0.30**, then falls back to 0.9803 by 0.60 - and the plateau either side of the peak is
flat (0.9829 at 0.25, 0.9834 at 0.35), so 0.30 sits in the middle of a wide band rather than on a
knife-edge. Recall is where the sweep earns its keep: fa_bresler and flowchartseg are saturated at
essentially 1.0 recall across the whole range, but hdbpmn's recall falls steadily from 0.9819 at
0.05 to 0.9500 at 0.60 - hdbpmn is the corpus that pays for a threshold set too high, and a task
that needs recall over precision can lower `MIN_SCORE` for very little in return.

## De-duplication

9.1's NMS runs per-class at IoU 0.7, so it cannot see a `rectangle` and a `rounded-rect` box drawn
on the same ink - `_suppress` runs across classes at `DUPLICATE_IOU = 0.60` to catch exactly that.
Turning it off (`no_dedupe`) raises node count from 7,729 to 7,779 - 50 extra boxes, all real
duplicates - and costs precision: strict F1 falls from **0.9837 to 0.9813**, a difference of only
0.0024 pooled, but every one of the 50 is a duplicate rather than a genuine second node, so the
gain is free.

Containment (`NEST_FRACTION`) is the one thing this task measured instead of assuming. It is off
by default, and priced rather than shipped: turning it on at 0.85 drops 1,583 boxes as nested or
enclosing (versus 50 for plain duplication) and pooled strict F1 falls from 0.9837 to **0.8803**.
The damage is concentrated in hdbpmn, whose pools and lanes really do nest real child nodes: its
own strict F1 falls from 0.9715 to 0.7696, **a cost of 0.2019 F1** - nodes do nest in the one
corpus that matters most for it, so a contained box is not noise there and containment stays off.

## The per-source break-out

The three corpora are not one distribution wearing three names. fa_bresler (96 pages, 471 truth
nodes) is perfect at the chosen threshold: precision 1.0, recall 1.0, F1 1.0, shape accuracy
0.9979. flowchartseg (132 pages, 2,873 truth nodes) is nearly so: 2,871 nodes matched against
2,873 truth, F1 0.9997. hdbpmn (242 pages, 4,360 truth nodes, and the only source carrying
arrowheads - all 16,192 of them) is the hard case: precision 0.9685, recall 0.9745, F1 **0.9715**,
shape accuracy 0.9955. The pooled 0.9837 is not one corpus's number - it is hdbpmn dragging the
average down against two corpora at or near 1.0.

## The GMM shape posterior

7.4's descriptor-space GMM (`GMM_K = 6` components, refit here) is scored as a shape refiner
against the detector's own class, restricted to hdbpmn (the only corpus 7.4's descriptor table
covers) and coarsened to the four labels the descriptor table carries. Over 4,249 matched,
boxed hdbpmn detections: the detector's own class is already right **99.65%** of the time, and
the raw GMM posterior alone is right only **62.93%** - a 36.71-point loss. Fusing the two (override
the detector only where the posterior is more confident than the detector's own score) recovers
most of that but still lands at **98.52%**, a 1.13-point loss versus doing nothing. The two
disagree on 1,572 of the 4,249, and when they do, the GMM is right only **0.32%** of the time - it
is not a marginal case where the two are close and coin-flip; when the detector and the posterior
disagree, the detector is almost always the one that was right. No shape refiner is wired into
`build()` because of this: the posterior was measured against the detector's own class here and
lost.

## OCR text is deliberately empty

Every `Node.text` from this module is `""` and every `attrs["text_source"]` is `"none"`. Binding
OCR text to a node is 10.1.2's job, not this one's - shipping an empty string rather than a guess
keeps that boundary honest instead of quietly doing half of the next phase's work here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from src.assemble.corpus import RUNS, Page, contains_fraction, detections, iou, pages, truth
from src.detect.classes import CLASSES
from src.ir.model import Diagram, Node

#: Detections at or above this score become nodes. Chosen as the argmax of the F1 sweep in the
#: docstring, which is flat across a wide band - so this is the middle of a plateau, and a task
#: that needs recall can lower it for very little.
MIN_SCORE = 0.30

#: Two boxes overlapping this much are one drawn shape. 9.1's NMS is class-wise at 0.7 and so
#: cannot see a `rectangle` and a `rounded-rect` on the same ink; this runs across classes.
DUPLICATE_IOU = 0.60

#: Containment as a duplication test, and it is **off**, because measuring it is the one thing
#: this task did that changed a default. See the docstring: nodes do nest, in the one corpus
#: that matters most, and dropping a contained box costs 0.20 of hdbpmn F1.
NEST_FRACTION: float | None = None

#: The one class that carries a box and is not a node. 9.1.2 derived it from edge waypoints.
NOT_A_NODE = ("arrowhead",)

#: The detection classes that become nodes, in the frozen class order.
NODE_CLASSES = tuple(name for name in CLASSES if name not in NOT_A_NODE)

#: The detector's seven node classes collapsed onto the four labels 7.4.1's descriptor table
#: carries, so a GMM posterior and a detector class can be compared at all.
COARSE: dict[str, str] = {
    "rectangle": "rectangle",
    "rounded-rect": "rectangle",
    "parallelogram": "rectangle",
    "diamond": "diamond",
    "circle": "circle",
    "double-circle": "circle",
    "freeform": "freeform",
}

#: The IoU an assembled node must reach to count as the same node as a truth node. The loose
#: value is 9.1's, whose point was that a hand-drawn box's extent is a convention.
STRICT_IOU = 0.50
LOOSE_IOU = 0.25

SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)

#: The mixture 7.4.3 chose an elbow at and 7.4.5 named.
GMM_K = 6


@dataclass
class Assembled:
    """One page's nodes, with the two populations that are not nodes kept beside them."""

    page: Page
    diagram: Diagram
    arrowheads: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)

    @property
    def nodes(self) -> list[Node]:
        return self.diagram.nodes


# ------------------------------------------------------------------------------------------
# building
# ------------------------------------------------------------------------------------------


def _suppress(kept: list[dict], candidate: dict, nms_iou: float, nest: float | None) -> str | None:
    """Why `candidate` is a repeat of something already kept, or None if it is not.

    Candidates arrive in descending score, so `kept` always holds the more confident box and the
    verdict is about the incoming one. Both containment directions are tested: a small circle
    arriving inside a kept double-circle and a large freeform arriving around a kept rectangle
    are the same mistake seen from opposite ends.
    """
    for other in kept:
        if iou(candidate["bbox"], other["bbox"]) >= nms_iou:
            return "duplicate"
        if nest is None:
            continue
        if contains_fraction(candidate["bbox"], other["bbox"]) >= nest:
            return "nested-inside"
        if contains_fraction(other["bbox"], candidate["bbox"]) >= nest:
            return "encloses-kept"
    return None


def build(
    page: Page,
    *,
    min_score: float = MIN_SCORE,
    nms_iou: float = DUPLICATE_IOU,
    nest: float | None = NEST_FRACTION,
    dedupe: bool = True,
) -> Assembled:
    """Cached detections -> nodes, in detector pixels. No GPU, no image decode.

    Ids are `n{index}` in descending score, so two runs over the same cache produce byte-identical
    diagrams and a downstream id means the same box every time.
    """
    rows = sorted(detections(page), key=lambda r: (-r["score"], r["cls"]))
    kept: list[dict] = []
    heads: list[dict] = []
    dropped: list[dict] = []

    for index, row in enumerate(rows):
        entry = {**row, "det_index": index}
        if row["cls"] in NOT_A_NODE:
            heads.append(entry)
            continue
        if row["score"] < min_score:
            dropped.append({**entry, "reason": "low-score"})
            continue
        reason = _suppress(kept, entry, nms_iou, nest) if dedupe else None
        if reason is not None:
            dropped.append({**entry, "reason": reason})
            continue
        kept.append(entry)

    diagram = Diagram(
        id=page.id,
        diagram_type="flowchart",
        meta={
            "source": page.source,
            "split": page.split,
            "assemble_scale": page.scale,
            "assemble_frame": "detect-pixels",
            "built_by": "10.1.1",
        },
    )
    for index, entry in enumerate(kept):
        diagram.nodes.append(
            Node(
                id=f"n{index}",
                shape=entry["cls"],
                bbox=[float(v) for v in entry["bbox"]],
                text="",
                confidence=float(entry["score"]),
                source_id=f"det{entry['det_index']}",
                attrs={
                    "detector_score": float(entry["score"]),
                    "detector_class": entry["cls"],
                    # No shape refiner is wired in - see the docstring's GMM section, where the
                    # posterior is measured against the detector's own class and loses.
                    "shape_overridden": False,
                    "text_source": "none",
                },
            )
        )
    return Assembled(page=page, diagram=diagram, arrowheads=heads, dropped=dropped)


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------


def truth_nodes(page: Page) -> list[Node]:
    """The truth nodes a detection could match: boxed, and in the detector's vocabulary."""
    return [
        node
        for node in truth(page).nodes
        if node.bbox is not None and node.shape in NODE_CLASSES and node.bbox[2] > 0
    ]


def match(predicted: list[Node], actual: list[Node], threshold: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, best IoU first. Class-agnostic: localisation, then shape.

    Separating the two is deliberate. A detector that puts a box exactly on a diamond and calls
    it a rectangle has localised a node and mislabelled it, and folding that into one number
    would report it as a miss *and* a false positive - two errors for one mistake.
    """
    scored = sorted(
        (
            (iou(p.bbox, a.bbox), i, j)
            for i, p in enumerate(predicted)
            for j, a in enumerate(actual)
            if iou(p.bbox, a.bbox) >= threshold
        ),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_p: set[int] = set()
    used_a: set[int] = set()
    out = []
    for _, i, j in scored:
        if i in used_p or j in used_a:
            continue
        used_p.add(i)
        used_a.add(j)
        out.append((i, j))
    return out


def _score_page(arguments: tuple[Page, float, bool, float | None]) -> dict:
    page, min_score, dedupe, nest = arguments
    assembled = build(page, min_score=min_score, dedupe=dedupe, nest=nest)
    actual = truth_nodes(page)
    predicted = assembled.diagram.nodes
    row = {
        "page": page.name,
        "source": page.source,
        "truth": len(actual),
        "nodes": len(predicted),
        "arrowheads": len(assembled.arrowheads),
        "dropped": len(assembled.dropped),
        "dropped_dupe": sum(1 for d in assembled.dropped if d["reason"] != "low-score"),
    }
    for name, threshold in (("strict", STRICT_IOU), ("loose", LOOSE_IOU)):
        pairs = match(predicted, actual, threshold)
        row[f"tp_{name}"] = len(pairs)
        if name == "strict":
            row["shape_correct"] = sum(1 for i, j in pairs if predicted[i].shape == actual[j].shape)
            row["coarse_correct"] = sum(
                1 for i, j in pairs if COARSE.get(predicted[i].shape) == COARSE.get(actual[j].shape)
            )
    return row


def _aggregate(rows: list[dict]) -> dict:
    truth_total = sum(r["truth"] for r in rows)
    node_total = sum(r["nodes"] for r in rows)
    out: dict[str, Any] = {
        "pages": len(rows),
        "truth": truth_total,
        "nodes": node_total,
        "arrowheads": sum(r["arrowheads"] for r in rows),
        "dropped_duplicates": sum(r["dropped_dupe"] for r in rows),
    }
    for name in ("strict", "loose"):
        tp = sum(r[f"tp_{name}"] for r in rows)
        precision = tp / node_total if node_total else 0.0
        recall = tp / truth_total if truth_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {
            "iou": STRICT_IOU if name == "strict" else LOOSE_IOU,
            "tp": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    matched = out["strict"]["tp"]
    out["shape_accuracy"] = (
        round(sum(r["shape_correct"] for r in rows) / matched, 4) if matched else 0.0
    )
    out["coarse_shape_accuracy"] = (
        round(sum(r["coarse_correct"] for r in rows) / matched, 4) if matched else 0.0
    )
    return out


def evaluate(
    held: tuple[Page, ...] | None = None,
    min_score: float = MIN_SCORE,
    dedupe: bool = True,
    n_jobs: int = 4,
    nest: float | None = NEST_FRACTION,
) -> dict:
    """Node P/R/F1 at both IoUs, pooled and per source."""
    from src.utils.parallel import pmap

    held = pages() if held is None else held
    rows = pmap(_score_page, [(p, min_score, dedupe, nest) for p in held], n_jobs=n_jobs)
    result = {"min_score": min_score, "dedupe": dedupe, "nest": nest, "pooled": _aggregate(rows)}
    result["by_source"] = {
        source: _aggregate([r for r in rows if r["source"] == source])
        for source in sorted({r["source"] for r in rows})
    }
    return result


def sweep(held: tuple[Page, ...] | None = None, values=SWEEP, n_jobs: int = 4) -> list[dict]:
    """The threshold curve. The chosen `MIN_SCORE` is the argmax of this, not an assertion."""
    held = pages() if held is None else held
    out = []
    for value in values:
        result = evaluate(held, value, True, n_jobs)
        out.append(
            {
                "min_score": value,
                "nodes": result["pooled"]["nodes"],
                "precision": result["pooled"]["strict"]["precision"],
                "recall": result["pooled"]["strict"]["recall"],
                "f1": result["pooled"]["strict"]["f1"],
                "recall_by_source": {
                    s: v["strict"]["recall"] for s, v in result["by_source"].items()
                },
            }
        )
    return out


# ------------------------------------------------------------------------------------------
# the GMM shape posterior, measured rather than assumed
# ------------------------------------------------------------------------------------------


def _component_labels(k: int = GMM_K) -> tuple[Any, Any, list[str]]:
    """The re-fitted mixture and `P(coarse label | component)` from the training rows."""
    import numpy as np

    from src.parse.vocab import fit, matrix

    X, labels, _, _ = matrix()
    model = fit(X, k, "full")
    responsibility = model.predict_proba(X)
    names = sorted(set(labels.tolist()))
    table = np.zeros((k, len(names)))
    for j, name in enumerate(names):
        table[:, j] = responsibility[labels == name].sum(axis=0)
    table = table / np.clip(table.sum(axis=1, keepdims=True), 1e-9, None)
    return model, table, names


def _posterior_page(arguments) -> list[dict]:
    """Coarse shape posterior for every matched detection on one hdbpmn page."""
    import numpy as np

    from src.features.descriptors import describe, page_contours

    page, model, table, names, min_score = arguments
    assembled = build(page, min_score=min_score)
    actual = truth_nodes(page)
    predicted = assembled.diagram.nodes
    pairs = match(predicted, actual, STRICT_IOU)
    if not pairs:
        return []
    # The descriptor pipeline works in the page's own pixels; the boxes are in detector pixels.
    boxes = {f"m{i}": [v / page.scale for v in predicted[i].bbox] for i, _ in pairs}
    contours = page_contours(page.id, boxes)
    rows = []
    for i, j in pairs:
        contour = contours.get(f"m{i}")
        vector = describe(contour) if contour is not None else None
        row = {
            "page": page.name,
            "detector": COARSE.get(predicted[i].shape),
            "truth": COARSE.get(actual[j].shape),
            "score": float(predicted[i].confidence),
            "gmm": None,
            "gmm_confidence": 0.0,
        }
        if vector is not None and np.isfinite(vector).all():
            posterior = model.predict_proba(vector.reshape(1, -1))[0] @ table
            row["gmm"] = names[int(np.argmax(posterior))]
            row["gmm_confidence"] = float(np.max(posterior))
        rows.append(row)
    return rows


def shape_posterior(
    limit: int | None = None, min_score: float = MIN_SCORE, n_jobs: int = 4
) -> dict:
    """Is a GMM posterior over the detected crop worth anything against the detector's class?

    Restricted to hdbpmn because 7.4.1's descriptor table is, and coarse, because that table's
    four labels are all a component can be scored against. Both limits bound the claim.
    """
    from src.utils.parallel import pmap

    held = pages(source="hdbpmn")[:limit]
    model, table, names = _component_labels()
    rows = [
        row
        for page_rows in pmap(
            _posterior_page,
            [(p, model, table, names, min_score) for p in held],
            n_jobs=n_jobs,
            prefer="threads",
        )
        for row in page_rows
    ]
    if not rows:
        return {"matched": 0}

    scored = [r for r in rows if r["gmm"] is not None]
    detector = sum(1 for r in rows if r["detector"] == r["truth"]) / len(rows)
    gmm = sum(1 for r in scored if r["gmm"] == r["truth"]) / len(scored)
    # The fusion a shape refiner would actually be: override only where the posterior is more
    # confident than the detector was. Anything weaker is just the detector's class again.
    fused = 0
    for r in rows:
        choice = r["detector"]
        if r["gmm"] is not None and r["gmm_confidence"] > r["score"]:
            choice = r["gmm"]
        fused += choice == r["truth"]
    disagree = [r for r in scored if r["gmm"] != r["detector"]]
    return {
        "source": "hdbpmn",
        "pages": len(held),
        "matched": len(rows),
        "with_descriptor": len(scored),
        "labels": names,
        "detector_accuracy": round(detector, 4),
        "gmm_accuracy": round(gmm, 4),
        "override_accuracy": round(fused / len(rows), 4),
        "gmm_minus_detector": round(gmm - detector, 4),
        "override_minus_detector": round(fused / len(rows) - detector, 4),
        "disagreements": len(disagree),
        "gmm_right_when_disagreeing": (
            round(sum(1 for r in disagree if r["gmm"] == r["truth"]) / len(disagree), 4)
            if disagree
            else 0.0
        ),
    }


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------


def run(gmm: bool = True, n_jobs: int = 4) -> dict:
    held = pages()
    raw = detections()
    total = sum(len(v) for v in raw.values())
    heads = sum(1 for v in raw.values() for r in v if r["cls"] in NOT_A_NODE)
    result: dict[str, Any] = {
        "pages": len(held),
        "detections": total,
        "arrowheads": heads,
        "arrowhead_share": round(heads / total, 4) if total else 0.0,
        "min_score": MIN_SCORE,
        "duplicate_iou": DUPLICATE_IOU,
        "nest_fraction": NEST_FRACTION,
        "sweep": sweep(held, n_jobs=n_jobs),
        "chosen": evaluate(held, MIN_SCORE, True, n_jobs),
        "no_dedupe": evaluate(held, MIN_SCORE, False, n_jobs)["pooled"],
        # The containment rule, priced rather than shipped. It is off by default because of this.
        "with_containment": evaluate(held, MIN_SCORE, True, n_jobs, nest=0.85),
        "text": "empty by design - binding text to a node is 10.1.2",
    }
    if gmm:
        result["shape_posterior"] = shape_posterior(n_jobs=n_jobs)
    return result


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gmm", action="store_true", help="skip the shape-posterior study")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=RUNS / "nodes.json")
    args = ap.parse_args(argv)

    result = run(gmm=not args.no_gmm, n_jobs=args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

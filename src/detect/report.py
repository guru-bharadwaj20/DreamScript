"""Phase 9.1.4 - the detection metrics, and the four ways the target can be met or missed.

    python -m src.detect.report                 # reports/detection.md + the figure

The plan's Definition of Done is a single number, `mAP@0.5 >= 0.80`, and 9.1.2 already
established that this corpus can answer it four different ways depending on what is averaged:

  * **pooled** over all eight classes and all three sources, which is the number a
    one-line evaluation prints and is three-quarters a statement about computer-rendered pages;
  * **hand-drawn only**, restricted to the 693 hdbpmn photographs, which is the deployment
    question and the only one Phase 13 should quote;
  * **annotated classes only**, dropping `arrowhead` - whose boxes are 9.1.2's size convention
    rather than a measurement, so its AP prices the convention as much as the detector;
  * **at IoU 0.25**, where a box that found the object but not the convention still counts.

All four are reported together with the verdict against 0.80 for each, because picking one and
calling it "the" mAP is exactly the move 7.1.2's OOB-metric trap and 8.8's validity indices
were about.

Per-class AP, per-source AP, the precision-recall curves and the confidence-vs-precision
calibration go in `reports/detection.md` and `reports/figures/p9_detection.png`.

## What it measured

**The target is met on all four readings**, which is worth stating plainly before the caveats:
pooled **0.9210**, hand-drawn only **0.8907**, annotated classes only **0.9784**, loose IoU
**0.9553**, against a bar of 0.80. 9.1.3's weights, 308 validation pages, 39,548 predictions,
30.7 ms a page.

**The four views disagree by 0.088, and every one of the disagreements is informative.**

  * Dropping `arrowhead` moves the pooled figure from 0.9210 to **0.9784** - so **the derived
    class costs 0.057 of the headline mAP on its own**, and a reader given only the pooled
    number would not know that seven of eight classes are effectively solved.
  * Restricting to the hand-drawn photographs costs **0.030** at IoU 0.5 and **0.158** at
    0.5:0.95 (0.7813 -> 0.6230). The strict metric is where the rendered pages flatter the
    model most, because a computer-rendered rectangle has an exact boundary and a photographed
    pen stroke does not. **Any single mAP quoted for this detector is three-quarters a
    statement about pages nobody drew.**
  * The loose threshold rescues `arrowhead` from **0.5197 to 0.7934** - a gap of **0.274**,
    the largest of any class by a factor of nine. That gap is 9.1.2's size convention being
    priced: four heads in five are *found*, and only half of them land inside an IoU of 0.5 of
    a box whose size was chosen by a rule rather than measured.

**`arrowhead` at mAP@0.5:0.95 is 0.1534** and that is close to the arithmetic floor rather than
a model failure. 9.1.3 measured the median head at 11.2 px; an IoU of 0.75 on an 11-pixel box
requires the prediction to be correct to about one pixel on every side, and IoU 0.95 is not
achievable at all at that scale by anything that quantises coordinates. Averaging ten thresholds
up to 0.95 for a class this small measures resolution, not detection, and this is the number to
cite when Phase 10 asks how much to trust an arrowhead's *position* as opposed to its existence.

**The classes rank differently under the two thresholds, and the ordering under the strict one
is a legibility ranking.** At IoU 0.5 seven classes sit between 0.927 and 1.000. At 0.5:0.95
they spread from `double-circle` 0.9974 and `rectangle` 0.9572 down to `freeform` 0.6959 and
`circle` 0.7846 - which is not how well each class is recognised but **how well its boundary is
defined**: a double circle and a ruled rectangle have unambiguous edges, and a hand-drawn circle
and a BPMN data object do not. 7.4.1 measured the same thing from the other side when it found
drawn circles at circularity 0.520 against an ideal 1.0.

`parallelogram`'s 0.9274 rests on **7 instances** and is not a measurement of anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.detect.choice import RUNS, predict_ultralytics, truths
from src.detect.classes import CLASSES
from src.detect.dataset import OUT as DATA
from src.detect.train import WEIGHTS
from src.utils.config import ROOT

REPORT = ROOT / "reports" / "detection.md"
FIGURE = ROOT / "reports" / "figures" / "p9_detection.png"

TARGET = 0.80
#: 9.1.2's derived class. Excluded from one of the four views, never silently.
DERIVED = "arrowhead"


def views(predictions: list[dict], reference: list[dict], sources: dict[str, str]) -> dict:
    """The four readings of the same predictions, each with its own verdict."""
    from src.detect.metrics import evaluate

    pooled = evaluate(predictions, reference)
    annotated = evaluate(
        [p for p in predictions if p["cls"] != DERIVED],
        [t for t in reference if t["cls"] != DERIVED],
        classes=tuple(c for c in CLASSES if c != DERIVED),
    )
    hand = {name for name, source in sources.items() if source == "hdbpmn"}
    hand_drawn = evaluate(
        [p for p in predictions if p["image"] in hand],
        [t for t in reference if t["image"] in hand],
    )
    return {
        "pooled": {
            "map50": pooled["map50"],
            "map50_95": pooled["map50_95"],
            "meets": pooled["map50"] >= TARGET,
        },
        "hand_drawn_only": {
            "map50": hand_drawn["map50"],
            "map50_95": hand_drawn["map50_95"],
            "meets": hand_drawn["map50"] >= TARGET,
            "pages": len(hand),
        },
        "annotated_classes_only": {
            "map50": annotated["map50"],
            "map50_95": annotated["map50_95"],
            "meets": annotated["map50"] >= TARGET,
        },
        "loose_iou_0_25": {"map25": pooled["map25"], "meets": pooled["map25"] >= TARGET},
        "target": TARGET,
        "_pooled": pooled,
        "_hand": hand_drawn,
    }


def pr_curve(predictions: list[dict], reference: list[dict], cls: str, threshold: float = 0.5):
    """Recall and precision arrays for one class, for the figure."""
    from src.detect.metrics import _match

    preds = [p for p in predictions if p["cls"] == cls]
    truths_ = [t for t in reference if t["cls"] == cls]
    matched, scores, positives = _match(preds, truths_, threshold)
    if positives == 0 or not len(matched):
        return np.array([]), np.array([])
    order = np.argsort(-scores, kind="stable")
    hits = matched[order].astype(float)
    tp, fp = np.cumsum(hits), np.cumsum(1 - hits)
    return tp / positives, tp / np.maximum(tp + fp, 1e-12)


def figure(predictions, reference, per_class, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colours = plt.get_cmap("tab10")
    for index, name in enumerate(CLASSES):
        recall, precision = pr_curve(predictions, reference, name)
        if not len(recall):
            continue
        axes[0].plot(recall, precision, lw=1.6, color=colours(index % 10), label=name)
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-recall at IoU 0.5", fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, fontsize=7)

    names = [n for n in CLASSES if per_class[n]["ap50"] is not None]
    values = [per_class[n]["ap50"] for n in names]
    loose = [per_class[n]["ap25"] for n in names]
    y = np.arange(len(names))
    axes[1].barh(y, loose, color="#cfd8dc", label="AP@0.25")
    axes[1].barh(y, values, height=0.55, color="#37474f", label="AP@0.5")
    axes[1].axvline(TARGET, color="#c62828", lw=1.2, ls="--", label=f"target {TARGET}")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("average precision")
    axes[1].set_title("Per class, tight and loose", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Phase 9.1.4 - detection metrics on the validation split", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(result: dict, path: Path = REPORT) -> Path:
    lines = [
        "# Detection metrics",
        "",
        "Phase 9.1.4. Generated by `python -m src.detect.report`.",
        "",
        "## The target, four ways",
        "",
        "| view | mAP | meets 0.80 |",
        "| :--- | ---: | :--- |",
    ]
    for name in ("pooled", "hand_drawn_only", "annotated_classes_only", "loose_iou_0_25"):
        entry = result["views"][name]
        value = entry.get("map50", entry.get("map25"))
        lines.append(
            f"| {name.replace('_', ' ')} | {value:.4f} | {'yes' if entry['meets'] else 'NO'} |"
        )
    lines += [
        "",
        "## Per class",
        "",
        "| class | instances | AP@0.5 | AP@0.5:0.95 | AP@0.25 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASSES:
        entry = result["per_class"][name]

        def fmt(x):
            return "-" if x is None else f"{x:.4f}"

        lines.append(
            f"| {name} | {entry['instances']} | {fmt(entry['ap50'])} | "
            f"{fmt(entry['ap50_95'])} | {fmt(entry['ap25'])} |"
        )
    lines += [
        "",
        "## By source",
        "",
        "| source | pages | mAP@0.5 | mAP@0.5:0.95 |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for source, entry in sorted(result["by_source"].items()):
        lines.append(
            f"| {source} | {result['pages_per_source'][source]} | "
            f"{entry['map50']:.4f} | {entry['map50_95']:.4f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "The pooled row is three-quarters computer-rendered pages (9.1.2). `hand_drawn_only`",
        "is the deployment question. `annotated_classes_only` drops `arrowhead`, whose boxes",
        "are a size convention rather than a measurement, and `loose_iou_0_25` asks whether",
        "the object was found at all rather than whether the convention's box was reproduced.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(weights: Path = WEIGHTS, root: Path = DATA, split: str = "val", imgsz: int = 896) -> dict:
    from ultralytics import YOLO

    from src.detect.metrics import by_source, evaluate

    model = YOLO(str(weights))
    reference, sources = truths(split, root)
    predictions, latency = predict_ultralytics(model, split, imgsz, root)

    pooled = evaluate(predictions, reference)
    result = {
        "split": split,
        "imgsz": imgsz,
        "pages": len(sources),
        "page_latency_ms": round(latency, 2),
        "predictions": len(predictions),
        "map50": pooled["map50"],
        "map50_95": pooled["map50_95"],
        "map25": pooled["map25"],
        "per_class": pooled["per_class"],
        "by_source": by_source(predictions, reference, sources),
        "pages_per_source": {
            source: sum(1 for s in sources.values() if s == source)
            for source in set(sources.values())
        },
    }
    result["views"] = {
        k: v for k, v in views(predictions, reference, sources).items() if not k.startswith("_")
    }
    result["report"] = str(write_report(result).relative_to(ROOT))
    result["figure"] = str(figure(predictions, reference, pooled["per_class"]).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=RUNS / "report.json")
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        print(f"no weights at {args.weights}", file=sys.stderr)
        return 1
    result = run(args.weights, args.data, args.split, args.imgsz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

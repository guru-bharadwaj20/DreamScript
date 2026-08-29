"""Phase 3.3.1 - stroke IoU of the whole preprocessing pipeline.

3.1.5 measured binarizers against each other. This measures the **pipeline** - illumination
correction, median filter, binarization, ruled-line suppression, small-component removal - as
one thing, against known ink, and reports the number plan.md 3.3.1 asks for.

The plan says "hand-traced GT on 30 images". Nothing here is hand-traced, and the substitute is
better rather than worse: the FA database records the pen trajectory, so rasterising it gives a
mask that is right by construction instead of by a careful person's judgment. The cost is
stated in `src/preprocess/evalset.py` and repeated here because it matters when reading the
number - a rendered stroke has cleaner edges than a photographed one, so **this is an upper
bound on real-photo performance**, and 3.3.3 goes looking at real photographs precisely because
this number cannot speak for them.

IoU rather than F1, because the plan asks for IoU and because it is the harsher of the two on
exactly the failure that matters here: a threshold that thickens every stroke by a pixel keeps
F1 respectable and loses IoU.

## Result

    30 items, mean IoU 0.821, median 0.868, worst 0.558   (the bar is 0.80)
    mean precision 0.823, mean recall 0.998

**The bar is met, and the two component numbers say something the IoU alone hides.** Recall is
essentially perfect - the pipeline loses almost no true ink - and every point of the loss is
precision. What comes out is a *fatter* stroke than what went in: blur and JPEG spread a
three-pixel line into a five-pixel gradient, and a threshold that keeps the whole stroke keeps
the halo with it. That is the right direction to fail in for what follows, since thinning
(3.1.9) reduces a stroke to its centreline anyway and does not care how wide it arrived, while
ink that never made it into the mask is gone for good.

Nine of the thirty items sit below the bar individually. The per-damage breakdown says which:

    jpeg 0.726   blur 0.737   glare 0.790   paper_texture 0.806
    stain 0.836   shadow 0.848   brightness 0.850

Compression and blur - the two that soften edges - are the expensive ones. Shadow and
brightness are near the top, which is the illumination correction of 3.1.4 doing exactly the job
it was built for.

    python -m src.preprocess.stroke_iou --count 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.config import ROOT

REPORT = ROOT / "reports" / "stroke_iou.md"

#: The bar plan.md 3.3.1 sets.
TARGET_IOU = 0.80


def pipeline(gray: np.ndarray) -> np.ndarray:
    """The Phase 3.1 photometric chain, in the order the config file declares it.

    Geometric stages - page detection, rectification, deskew - are deliberately not here. They
    move pixels, and the ground-truth mask would have to move with them; 3.3.2 sweeps rotation
    with both warped together, which is the honest way to include them.
    """
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.illumination import correct
    from src.preprocess.rules import suppress

    return denoise(suppress(binarize(median(correct(gray)))))


def measure(count: int = 30) -> list[dict]:
    from src.preprocess.evalset import build, f1
    from src.utils.parallel import pmap

    items = build(count)

    def one(item) -> dict:
        predicted = pipeline(item.photo)
        scores = f1(predicted, item.mask)
        return {
            "name": item.name,
            "damage": item.damage,
            **{k: round(v, 4) for k, v in scores.items()},
        }

    return pmap(one, items, prefer="threads")


def by_damage(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name in row["damage"]:
            grouped[name].append(row["iou"])
    return {
        name: {"items": len(values), "mean_iou": round(float(np.mean(values)), 4)}
        for name, values in sorted(grouped.items(), key=lambda kv: float(np.mean(kv[1])))
    }


def summarise(rows: list[dict]) -> dict:
    ious = np.array([r["iou"] for r in rows])
    return {
        "items": len(rows),
        "mean_iou": round(float(ious.mean()), 4),
        "median_iou": round(float(np.median(ious)), 4),
        "worst_iou": round(float(ious.min()), 4),
        "items_below_target": int((ious < TARGET_IOU).sum()),
        "mean_precision": round(float(np.mean([r["precision"] for r in rows])), 4),
        "mean_recall": round(float(np.mean([r["recall"] for r in rows])), 4),
    }


def report(rows: list[dict]) -> Path:
    summary = summarise(rows)
    damage = by_damage(rows)
    lines = [
        "# Phase 3.3.1 — stroke IoU",
        "",
        f"{summary['items']} items with exact stroke ground truth, put through the Phase 3.1 ",
        "photometric pipeline. The ground truth is the rasterised pen trajectory from the FA ",
        "database rather than a hand tracing — see `src/preprocess/evalset.py` for why, and for ",
        "why this is an **upper bound** on real photographs.",
        "",
        "| | |",
        "| :--- | ---: |",
        f"| mean IoU | **{summary['mean_iou']:.3f}** |",
        f"| median IoU | {summary['median_iou']:.3f} |",
        f"| worst IoU | {summary['worst_iou']:.3f} |",
        f"| items below the {TARGET_IOU:.2f} bar | {summary['items_below_target']} |",
        f"| mean precision | {summary['mean_precision']:.3f} |",
        f"| mean recall | {summary['mean_recall']:.3f} |",
        "",
        "## By degradation",
        "",
        "Each item carries one to three of the Phase 1.3.6 transforms, so an item counts under ",
        "each of them. Worst first.",
        "",
        "| damage | items | mean IoU |",
        "| :--- | ---: | ---: |",
    ]
    lines += [
        f"| {name} | {row['items']} | {row['mean_iou']:.3f} |" for name, row in damage.items()
    ]
    lines += [
        "",
        "## Worst ten items",
        "",
        "| item | damage | IoU | precision | recall |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: r["iou"])[:10]:
        lines.append(
            f"| `{row['name']}` | {', '.join(row['damage'])} | {row['iou']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=30)
    args = ap.parse_args(argv)

    rows = measure(args.count)
    if not rows:
        print("no evaluation items; check data/raw/fa_bresler", file=sys.stderr)
        return 1
    summary = summarise(rows)
    path = report(rows)
    print(json.dumps({**summary, "report": str(path.relative_to(ROOT))}, indent=2))
    print(json.dumps(by_damage(rows), indent=2))

    checks = {
        f"mean_iou_at_least_{TARGET_IOU:.2f}": summary["mean_iou"] >= TARGET_IOU,
        "thirty_items_measured": summary["items"] >= 30,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name} ({summary['mean_iou']:.3f})")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

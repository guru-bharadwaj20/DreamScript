"""Phase 3.1.8 - suppressing ruled and squared paper.

Most of the hdBPMN corpus is drawn on squared exercise paper, and the grid survives
binarization: it is ink, as far as a threshold is concerned. Left in, it fuses every shape into
one connected component, gives the line detector of 3.2.4 thousands of perfect lines to find,
and buries the drawing.

The method is directional morphology. Opening the mask with a long, one-pixel-tall structuring
element leaves only runs that are horizontal for that whole length - a ruled line survives, a
box edge survives, a handwritten stroke does not. The same with a tall, one-pixel-wide element
finds the verticals.

**The hard part is not finding the lines, it is not deleting the drawing with them.** A box
edge is also a long straight run. Two rules keep it:

1. **Span.** A ruled line runs most of the way across the page; a box edge does not. Only
   components spanning at least `MIN_SPAN_FRAC` of the image are candidates.
2. **Family.** Ruling comes in dozens of parallel lines. A single page-spanning line is a long
   stroke, so nothing is removed unless at least `MIN_RULE_COUNT` candidates share a direction.
3. **Repair.** Where a rule crosses a stroke, deleting the rule takes a bite out of the stroke.
   After removal the mask is closed along the *perpendicular* direction, which rejoins a stroke
   broken by a horizontal cut without rejoining the horizontal line itself.

Measured on 24 evaluation items with a squared grid drawn over known ink:

    grid pixels removed        99.98%
    precision                  0.337  ->  0.991
    true-ink recall            1.000  ->  0.967
    F1                         0.504  ->  0.979

The 3.3% of true ink lost is the price of the removal, paid where a stroke runs *along* a rule
and is indistinguishable from it. It is a real cost and it is worth it: without suppression two
ink pixels in three are grid.

    python -m src.preprocess.rules
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from src.utils.parallel import pmap

#: Length of the directional structuring element, as a fraction of the image's matching side.
KERNEL_FRAC = 0.06

#: A component must span at least this fraction of the page to count as a rule rather than a
#: box edge. Squared-paper rules run edge to edge; a drawn box is a fraction of the width.
MIN_SPAN_FRAC = 0.55

#: Closing kernel used to repair strokes cut by a removed rule.
REPAIR = 3

#: How many page-spanning lines must be found in one direction before any of them is treated
#: as ruling. Ruled and squared paper carries dozens; a diagram carries one or two long strokes,
#: and a single stroke that happens to cross the page is content, not paper. Without this the
#: suppressor deletes a lone full-height line - which is exactly what it did the first time it
#: was tested on a vertical stroke spanning 80% of the image.
MIN_RULE_COUNT = 4


def _directional(mask: np.ndarray, horizontal: bool) -> np.ndarray:
    height, width = mask.shape
    length = max(15, int(KERNEL_FRAC * (width if horizontal else height)))
    shape = (length, 1) if horizontal else (1, length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)


def find_rules(mask: np.ndarray, min_span_frac: float = MIN_SPAN_FRAC) -> np.ndarray:
    """Pixels belonging to page rules: long runs that also span most of the page."""
    height, width = mask.shape
    rules = np.zeros(mask.shape, np.uint8)
    for horizontal in (True, False):
        candidate = _directional(mask, horizontal)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        span_limit = min_span_frac * (width if horizontal else height)
        spanning = [
            label
            for label in range(1, count)
            if stats[label, cv2.CC_STAT_WIDTH if horizontal else cv2.CC_STAT_HEIGHT] >= span_limit
        ]
        if len(spanning) < MIN_RULE_COUNT:
            continue  # one or two long lines are the drawing, not the paper
        for label in spanning:
            rules[labels == label] = 1
    return rules > 0


def suppress(mask: np.ndarray, min_span_frac: float = MIN_SPAN_FRAC) -> np.ndarray:
    """Remove page rules and repair the strokes they cut."""
    rules = find_rules(mask, min_span_frac)
    if not rules.any():
        return mask.astype(bool)
    cleaned = np.logical_and(mask, ~rules).astype(np.uint8)
    # Close along both axes with a small kernel: a stroke cut by a horizontal rule is rejoined
    # vertically, and vice versa. The kernel is deliberately smaller than the rule spacing, so
    # this cannot bridge two separate strokes into one.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REPAIR, REPAIR))
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel).astype(bool)


def draw_grid(shape: tuple[int, int], spacing: int = 32, thickness: int = 1) -> np.ndarray:
    """A squared-paper grid, for measuring the suppression against known ink."""
    grid = np.zeros(shape, np.uint8)
    for y in range(spacing, shape[0], spacing):
        cv2.line(grid, (0, y), (shape[1] - 1, y), 255, thickness)
    for x in range(spacing, shape[1], spacing):
        cv2.line(grid, (x, 0), (x, shape[0] - 1), 255, thickness)
    return grid > 0


def evaluate(count: int = 24) -> dict:
    """Draw a grid over known ink, then measure what suppression recovers."""
    from src.preprocess.evalset import build, f1

    items = build(count)
    if not items:
        return {"items": 0}

    def score(item):
        grid = draw_grid(item.mask.shape)
        ruled = np.logical_or(item.mask, grid)
        cleaned = suppress(ruled)
        return {
            "before": f1(ruled, item.mask),
            "after": f1(cleaned, item.mask),
            "grid_pixels_removed": 1.0
            - float(np.logical_and(cleaned, grid & ~item.mask).sum())
            / max(float((grid & ~item.mask).sum()), 1.0),
        }

    scores = pmap(score, items, prefer="threads")
    mean = lambda key, when: round(  # noqa: E731 - a local reducer, not an API
        float(np.mean([s[when][key] for s in scores])), 4
    )
    return {
        "items": len(items),
        "f1_before": mean("f1", "before"),
        "f1_after": mean("f1", "after"),
        "precision_before": mean("precision", "before"),
        "precision_after": mean("precision", "after"),
        "true_ink_recall_before": mean("recall", "before"),
        "true_ink_recall_after": mean("recall", "after"),
        "grid_pixels_removed": round(float(np.mean([s["grid_pixels_removed"] for s in scores])), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=24)
    args = ap.parse_args(argv)

    result = evaluate(args.count)
    if not result["items"]:
        print("no evaluation items", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))

    checks = {
        "most_of_the_grid_is_removed": result["grid_pixels_removed"] >= 0.90,
        "f1_improves": result["f1_after"] > result["f1_before"],
        "true_ink_mostly_survives": result["true_ink_recall_after"] >= 0.90,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 3.1.5 - binarization, chosen by measurement rather than by reputation.

Three methods are implemented and compared on the same images with the same ground truth:

* **Otsu** - one global threshold from the intensity histogram. Fast, and correct only when the
  page is lit evenly.
* **Adaptive Gaussian** - a local mean threshold. Handles ramps, but its window size has to
  suit the stroke width or it hollows out thick strokes.
* **Sauvola** - a local threshold that also uses the local *standard deviation*, so flat paper
  stays paper even when it is dark. This is the usual choice for document images, and the
  point of this module is to check that rather than assume it.

The comparison runs on `src.preprocess.evalset`, where the ink is known exactly because it was
rasterised from the pen trajectory. Every method sees the identical damaged image.

Sauvola is implemented here rather than imported: it is two box filters and some arithmetic,
and adding scikit-image as a dependency for it would be the larger cost.

**The measured answer was not the expected one.** Sauvola is the textbook choice and it does
win on the raw images (F1 0.907 against Otsu's 0.766). But run 3.1.4's illumination correction
first and the ordering inverts - Otsu 0.912, Sauvola 0.897 - because flattening the page has
already removed the uneven lighting that local thresholding exists to survive, and what is left
is a global decision that Otsu makes better. So the default is `otsu` *with correction on*,
Sauvola remains selectable for its slightly better worst case, and the caveat stands that this
was measured on rendered strokes rather than photographs.

    python -m src.preprocess.binarize
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from src.preprocess import illumination
from src.preprocess.evalset import build, f1
from src.utils.parallel import pmap

DEFAULT_WINDOW = 25
DEFAULT_K = 0.2

#: Sauvola's dynamic range of the standard deviation. 128 for 8-bit images, by the paper.
SAUVOLA_R = 128.0


def otsu(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask > 0


def adaptive(gray: np.ndarray, window: int = DEFAULT_WINDOW, c: int = 10) -> np.ndarray:
    window = max(3, window | 1)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, window, c
    )
    return mask > 0


def sauvola(gray: np.ndarray, window: int = DEFAULT_WINDOW, k: float = DEFAULT_K) -> np.ndarray:
    """t = mean * (1 + k * (std / R - 1)); ink is anything below t.

    On flat paper the standard deviation is near zero, so the threshold drops well below the
    local mean and the paper is not mistaken for ink. That is the whole idea, and it is why
    Sauvola beats a plain local mean on a page that is dark but blank.
    """
    window = max(3, window | 1)
    image = gray.astype(np.float32)
    mean = cv2.boxFilter(image, cv2.CV_32F, (window, window), normalize=True)
    mean_square = cv2.boxFilter(image * image, cv2.CV_32F, (window, window), normalize=True)
    std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    threshold = mean * (1.0 + k * (std / SAUVOLA_R - 1.0))
    return image < threshold


METHODS = {"otsu": otsu, "adaptive": adaptive, "sauvola": sauvola}


def binarize(
    image: np.ndarray,
    method: str = "otsu",
    *,
    window: int = DEFAULT_WINDOW,
    k: float = DEFAULT_K,
    correct_illumination: bool = True,
) -> np.ndarray:
    """The project's entry point: returns a boolean ink mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if correct_illumination:
        gray = illumination.correct(gray)
    if method == "sauvola":
        return sauvola(gray, window, k)
    if method == "adaptive":
        return adaptive(gray, window)
    if method == "otsu":
        return otsu(gray)
    raise ValueError(f"unknown binarization method {method!r}; expected one of {sorted(METHODS)}")


def compare(count: int = 30, *, correct_illumination: bool = True) -> dict:
    items = build(count)
    results: dict[str, dict] = {}
    for name in METHODS:
        scores = pmap(
            lambda item, m=name: f1(
                binarize(item.photo, m, correct_illumination=correct_illumination), item.mask
            ),
            items,
            prefer="threads",
        )
        results[name] = {
            key: round(float(np.mean([s[key] for s in scores])), 4)
            for key in ("precision", "recall", "f1", "iou")
        }
        results[name]["worst_f1"] = round(float(np.min([s["f1"] for s in scores])), 4)
    return {"items": len(items), "methods": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=30)
    args = ap.parse_args(argv)

    with_correction = compare(args.count, correct_illumination=True)
    if not with_correction["items"]:
        print("no evaluation items; is the FA database present?", file=sys.stderr)
        return 1
    without = compare(args.count, correct_illumination=False)

    print(json.dumps({"illumination_corrected": with_correction}, indent=2))
    print(
        json.dumps(
            {
                "raw_no_illumination_correction": {
                    name: {"f1": scores["f1"]} for name, scores in without["methods"].items()
                }
            },
            indent=2,
        )
    )

    ranked = sorted(with_correction["methods"].items(), key=lambda kv: -kv[1]["f1"])
    best, best_scores = ranked[0]
    print(f"\n  chosen: {best}  (F1 {best_scores['f1']:.3f}, IoU {best_scores['iou']:.3f})")
    for name, scores in ranked[1:]:
        print(f"    {name:9s} F1 {scores['f1']:.3f}")

    checks = {
        "all_methods_scored": len(with_correction["methods"]) == 3,
        "chosen_method_beats_otsu": best_scores["f1"] >= with_correction["methods"]["otsu"]["f1"],
        "illumination_correction_helps": (
            with_correction["methods"][best]["f1"] > without["methods"][best]["f1"]
        ),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

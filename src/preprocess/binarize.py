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

## The caveat came due: `PHOTO`

The caveat was not decoration. On real hdBPMN photographs the default admits far too much of
the paper: **mean ink fraction 0.0954** where a pen drawing is 2-5% ink, and the skeleton of
that mask shatters into hundreds of fragments that 10.1.3's tracer cannot re-chain. Re-scored
end to end by **tracing F1** on 76 *train* hdBPMN pages (every 6th page, all eleven exercises,
ground-truth node boxes, containers excluded - the held-out pages are where 10.1.3 reports and
were not looked at):

    illumination   threshold             ink     traced   recall   precision      F1
    correct        otsu       (default) 0.0954    13,115   0.5708      0.0535   0.0977
    correct        sauvola k0.3         0.0711     9,132   0.5228      0.0703   0.1239
    flat           otsu                 0.0605     6,695   0.4625      0.0848   0.1434
    flat           sauvola w25 k0.25    0.0405     3,125   0.3697      0.1453   0.2086
    flat           + area 2e-4          0.0314     2,493   0.3738      0.1841   0.2467  <- PHOTO

**CLAHE is the single largest contributor**, and that is the part the rendered-stroke
comparison could not see: on synthetic ink there is no paper texture for it to amplify, so the
step that restores bite on flat scans is the step that promotes photograph grain to ink.
Dropping it and keeping only the flat-field division (3.1.4's step 1) takes otsu's F1 from
0.0977 to 0.1434 on its own. Sauvola past the k=0.2 the evalset chose adds the rest.

`PHOTO` is that setting, named rather than made the default: the evalset measurement above is
still the right answer for rendered strokes, and Phases 3, 4 and 9 are calibrated on the
default. Photograph consumers ask for it by name - `binarize(image, **PHOTO)`.

**Not the whole story.** The same sweep's perfect-ink control - the ground-truth polylines
rasterised, so the mask is exactly right - scores recall 0.7239 at precision 0.6386. `PHOTO`
closes about a third of the gap to that; the rest is the walk and the geometry, not the
threshold.

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

#: Binarisation for photographs of paper, measured by tracing F1 rather than by evalset F1 -
#: see the module docstring. Ink fraction drops 0.0954 -> 0.0314 and 10.1.3's tracing F1 rises
#: 0.0977 -> 0.2467 on 76 train hdBPMN pages. `area` is not this module's - it is the
#: connected-component floor the caller should apply afterwards (`denoise.remove_small_components`),
#: and it is quoted here because the three settings were chosen together and only mean their
#: numbers together.
PHOTO = {"method": "sauvola", "window": 25, "k": 0.25, "illumination_mode": "flat"}
PHOTO_MIN_AREA_FRAC = 2e-4

#: Sauvola's dynamic range of the standard deviation. 128 for 8-bit images, by the paper.
SAUVOLA_R = 128.0


def otsu(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask > 0


def adaptive(gray: np.ndarray, window: int = DEFAULT_WINDOW, c: int = 10) -> np.ndarray:
    """`cv2.adaptiveThreshold` with a Gaussian neighbourhood; `c` is subtracted from the mean."""
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


#: name -> the function, and name -> which of `binarize`'s knobs that function takes.
#:
#: `METHODS` existed and `binarize` did not dispatch through it: it re-implemented the same
#: routing as an if/elif chain, which is how `adaptive`'s `c` came to be dropped at the call site
#: - the chain called `adaptive(gray, window)` and there was no `c` on `binarize` to pass. Two
#: tables that must agree, one of which nothing checked.
#:
#: The signatures differ (otsu takes none, adaptive takes `c`, sauvola takes `k`), so the
#: argument list is declared rather than guessed, and a test asserts each entry matches the
#: function's real signature.
METHODS = {"otsu": otsu, "adaptive": adaptive, "sauvola": sauvola}
METHOD_ARGS: dict[str, tuple[str, ...]] = {
    "otsu": (),
    "adaptive": ("window", "c"),
    "sauvola": ("window", "k"),
}

#: `cv2.adaptiveThreshold`'s constant subtracted from the local mean. 10 is `adaptive`'s own
#: default and stays it, so passing it through changes nothing and makes it reachable.
DEFAULT_C = 10


def binarize(
    image: np.ndarray,
    method: str = "otsu",
    *,
    window: int = DEFAULT_WINDOW,
    k: float = DEFAULT_K,
    c: int = DEFAULT_C,
    correct_illumination: bool = True,
    illumination_mode: str | None = None,
) -> np.ndarray:
    """The project's entry point: returns a boolean ink mask.

    `illumination_mode` selects how much of 3.1.4 runs before the threshold:

        "correct"   flat-field division then CLAHE - 3.1.4 whole, and the default
        "flat"      the division only, no CLAHE; what `PHOTO` uses, and why
        "none"      the raw grayscale

    It defaults to `"correct"` or `"none"` according to `correct_illumination`, so every
    existing caller keeps the behaviour it had.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if illumination_mode is None:
        illumination_mode = "correct" if correct_illumination else "none"
    if illumination_mode == "correct":
        gray = illumination.correct(gray)
    elif illumination_mode == "flat":
        gray = illumination.flatten(gray)
    elif illumination_mode != "none":
        raise ValueError(
            f"unknown illumination_mode {illumination_mode!r}; expected correct, flat or none"
        )
    if method not in METHODS:
        raise ValueError(
            f"unknown binarization method {method!r}; expected one of {sorted(METHODS)}"
        )
    options = {"window": window, "k": k, "c": c}
    return METHODS[method](gray, **{name: options[name] for name in METHOD_ARGS[method]})


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

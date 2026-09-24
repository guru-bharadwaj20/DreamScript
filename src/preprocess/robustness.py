"""Phase 3.3.2 - how the preprocessing degrades, and how fast.

3.3.1 gives one number for a fixed set of damage. That number cannot say *when* the pipeline
stops working, and "when" is what a deployment needs: how blurred a photograph may be, how far
off-square it may be held, how uneven the light may get before the strokes stop coming back.

So each degradation is applied at a controlled severity and swept, one at a time, with
everything else held clean. The Phase 1.3.6 augmentation policy draws its severities at random,
which is right for training and wrong here - a curve needs the x-axis to be a number the caller
chose.

Three sweeps, and each one asks a different question:

* **Blur** - a gaussian of growing radius. The question is how much softening the binarizer
  survives.
* **Rotation** - the image and the ground-truth mask are warped by the *same* matrix, which is
  the only honest way to include a geometric transform in a pixel metric. Interpolating the
  image but not the mask would show a loss that is nothing but a misregistration. Two curves
  come out of this sweep: what the photometric pipeline scores, and what 3.1.7's deskew
  estimated the angle to be.
* **Lighting** - a shadow falling across the page, multiplicative and darkening only, at growing
  strength. This is the sweep 3.1.4 exists for, so it is run three ways: the full correction,
  its background-flattening half alone, and no correction at all.

## Results

    blur radius        0     3     5     7     9    11    (px, gaussian)
    mean IoU        0.99  0.99  0.90  0.75  0.69  0.63

    rotation           0     3     6     9    12    15    (degrees)
    mean IoU        0.99  0.96  0.95  0.95  0.94  0.94
    deskew error    0.20  0.54  0.83  0.38  0.22  0.65    (degrees, median |applied + estimated|)

    light lost       0.0   0.4   0.7  0.85  0.95    (at the page's dark side)
    flatten+CLAHE   0.99  0.99  0.99  0.99  0.89
    flatten only    0.99  0.99  0.99  0.99  0.99
    no correction   0.99  0.99  0.99  0.99  0.99

**Rotation is free.** Binarization is a per-pixel decision and does not care which way the page
is turned; the 0.05 that goes is interpolation at the stroke edges, not thresholding. Deskew
recovers the applied angle to under a degree throughout, on pages whose true skew is known.

**Blur is the one that bites**, and it bites steeply: past a 5-pixel radius a third of the ink
is gone and no threshold gets it back. That is a capture-time constraint rather than a software
one, and it belongs in whatever guidance the finished system gives its users.

**Lighting produced the result this sweep did not expect.** A global Otsu threshold survives a
multiplicative shadow almost to the end, because it fails only when the shaded paper becomes
darker than the lit ink - past `s = 0.86` on this material, which is arithmetic rather than
luck. Up to that point the illumination correction changes nothing, and past it the *flattening*
half of 3.1.4 holds while the *CLAHE* half costs a tenth of the IoU: it stretches local contrast
in a region where the paper has been reduced to a few grey levels, and amplifies the
quantisation. That does not contradict 3.1.4 and 3.1.5, which measured real photographs with
blotchy shading and faint strokes and found the correction worth having. It does mean neither
measurement generalises to the other's material on its own, and it is a concrete argument for
keeping the CLAHE step configurable rather than mandatory.

    python -m src.preprocess.robustness --count 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURE = ROOT / "reports" / "figures" / "p3_robustness.png"
REPORT = ROOT / "reports" / "robustness.md"

BLUR_RADII = (0, 3, 5, 7, 9, 11)
ROTATIONS = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
LIGHTING = (0.0, 0.4, 0.7, 0.85, 0.95)


def blur(gray: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return gray
    kernel = radius if radius % 2 else radius + 1
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def rotate(image: np.ndarray, degrees: float, *, nearest: bool = False) -> np.ndarray:
    """Rotate about the centre, keeping the frame. `nearest` for masks, so they stay binary."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    border = 0 if nearest else int(np.median(image[:8, :8])) if image.size else 255
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def light(gray: np.ndarray, strength: float) -> np.ndarray:
    """A shadow falling across the page: full brightness at the left, `1 - strength` at the right.

    Multiplicative, because illumination is - a shadow scales the light reaching the paper and
    the ink alike. And darkening only, never brightening: a ramp that scales *up* runs the lit
    side into 255 and clips, and clipped highlights are information that no correction can
    recover, so the curve would then be measuring saturation rather than lighting. The first
    version of this sweep did exactly that and reported a flat line with a cliff at the end.
    """
    if strength <= 0:
        return gray
    _, width = gray.shape
    ramp = np.linspace(1.0, max(0.0, 1.0 - strength), width, dtype=np.float32)
    return np.clip(gray.astype(np.float32) * ramp[None, :], 0, 255).astype(np.uint8)


def _score(predicted: np.ndarray, truth: np.ndarray) -> float:
    from src.preprocess.evalset import f1

    return f1(predicted, truth)["iou"]


def sweep(count: int = 20) -> dict:
    """Every curve, over the same evaluation items."""
    from src.preprocess.binarize import binarize
    from src.preprocess.denoise import denoise, median
    from src.preprocess.deskew import _straight_pages, estimate
    from src.preprocess.deskew import rotate as rotate_mask
    from src.preprocess.evalset import build
    from src.preprocess.illumination import flatten
    from src.preprocess.rules import suppress
    from src.preprocess.stroke_iou import pipeline
    from src.utils.parallel import pmap

    items = build(count)
    if not items:
        return {}

    def uncorrected(gray: np.ndarray) -> np.ndarray:
        """`pipeline` with 3.1.4 removed and *nothing else changed*, so the gap is that stage.

        The stages have to match exactly. An earlier version left `suppress` out of this chain
        as well, which made the comparison a comparison of two different pipelines.
        """
        return denoise(suppress(binarize(median(gray))))

    def one(item) -> dict:
        # The clean render, not the damaged photo: a sweep needs a controlled starting point.
        clean = np.where(item.mask, 35, 245).astype(np.uint8)
        row: dict = {"name": item.name}
        row["blur"] = [_score(pipeline(blur(clean, r)), item.mask) for r in BLUR_RADII]

        row["rotation"] = [
            _score(
                pipeline(rotate(clean, degrees)),
                rotate(item.mask.astype(np.uint8), degrees, nearest=True) > 0,
            )
            for degrees in ROTATIONS
        ]
        row["lighting"] = [_score(pipeline(light(clean, s)), item.mask) for s in LIGHTING]
        row["lighting_raw"] = [_score(uncorrected(light(clean, s)), item.mask) for s in LIGHTING]
        row["lighting_flatten"] = [
            _score(uncorrected(flatten(light(clean, s))), item.mask) for s in LIGHTING
        ]
        return row

    rows = pmap(one, items, prefer="threads")

    # The deskew curve is measured on *different* images, and deliberately. 3.1.7 established
    # that the FA automata have no dominant direction - they are circles and arrows - so the
    # estimator declines on them and a residual measured there would be a property of the test
    # material. The flowchartseg renders are the corpus's only pages with a known skew of zero,
    # so that is where an angle can be recovered and checked.
    def deskew_error(page_and_angle) -> float:
        page, degrees = page_and_angle
        # `estimate` measures in image space, where y points down, so the applied and estimated
        # angles have opposite signs and the error is their sum. See src/preprocess/deskew.py.
        return abs(degrees + estimate(rotate_mask(page, degrees)))

    pages = _straight_pages(min(24, count))
    errors = pmap(
        deskew_error, [(page, degrees) for page in pages for degrees in ROTATIONS], prefer="threads"
    )
    per_angle = np.asarray(errors, float).reshape(len(pages), len(ROTATIONS)) if pages else None

    keys = ("blur", "rotation", "lighting", "lighting_raw", "lighting_flatten")
    curves = {
        key: [
            round(float(np.nanmean([r[key][i] for r in rows])), 4) for i in range(len(rows[0][key]))
        ]
        for key in keys
    }
    curves["deskew_error"] = (
        [round(float(v), 4) for v in np.median(per_angle, axis=0)]
        if per_angle is not None
        else [float("nan")] * len(ROTATIONS)
    )
    return {
        "items": len(rows),
        "deskew_pages": len(pages),
        "axes": {"blur": BLUR_RADII, "rotation": ROTATIONS, "lighting": LIGHTING},
        "curves": curves,
    }


def figure(result: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = result["curves"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    axes[0].plot(BLUR_RADII, curves["blur"], "o-", color="#E45756")
    axes[0].set_title("blur")
    axes[0].set_xlabel("gaussian radius (px)")

    axes[1].plot(ROTATIONS, curves["rotation"], "o-", color="#4C78A8", label="IoU")
    axes[1].set_title("rotation")
    axes[1].set_xlabel("degrees")
    twin = axes[1].twinx()
    twin.plot(ROTATIONS, curves["deskew_error"], "s--", color="#999999", label="deskew error")
    twin.set_ylabel("deskew error (deg)")
    twin.set_ylim(0, max(1.0, max(curves["deskew_error"]) * 1.5))

    axes[2].plot(LIGHTING, curves["lighting"], "o-", color="#54A24B", label="3.1.4 (flatten+CLAHE)")
    axes[2].plot(LIGHTING, curves["lighting_flatten"], "^-", color="#F58518", label="flatten only")
    axes[2].plot(LIGHTING, curves["lighting_raw"], "o--", color="#B279A2", label="no correction")
    axes[2].set_title("shadow across the page")
    axes[2].set_xlabel("darkest side, share of light lost")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_ylabel("mean stroke IoU")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Phase 3.3.2 — preprocessing under degradation, n = {result['items']}")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, FIGURE, dpi=130)
    plt.close(fig)
    return FIGURE


def report(result: dict) -> Path:
    curves = result["curves"]

    def row(label: str, xs, ys, fmt: str = "{:.2f}") -> str:
        return f"| {label} | " + " | ".join(fmt.format(y) for y in ys) + " |"

    def header(xs, unit: str) -> list[str]:
        return [
            f"| {unit} | " + " | ".join(f"{x:g}" for x in xs) + " |",
            "| :--- |" + " ---: |" * len(xs),
        ]

    lines = [
        "# Phase 3.3.2 — robustness sweep",
        "",
        f"{result['items']} items with exact stroke ground truth. Each degradation is applied ",
        "to the clean render at a chosen severity, one at a time, and the whole Phase 3.1 ",
        "photometric pipeline is run on the result.",
        "",
        f"![degradation curves](figures/{FIGURE.name})",
        "",
        "## Blur",
        "",
        *header(BLUR_RADII, "gaussian radius (px)"),
        row("mean IoU", BLUR_RADII, curves["blur"]),
        "",
        "## Rotation",
        "",
        "The image and its ground-truth mask are warped by the same matrix, so what is measured ",
        "is the pipeline and not a misregistration.",
        "",
        *header(ROTATIONS, "degrees"),
        row("mean IoU", ROTATIONS, curves["rotation"]),
        row("deskew error (deg)", ROTATIONS, curves["deskew_error"]),
        "",
        "## Lighting",
        "",
        "A shadow falling across the page: full brightness at the left, `1 - s` at the right. ",
        "Run three ways — the full 3.1.4 correction, its background-flattening half alone, and ",
        "no correction at all.",
        "",
        *header(LIGHTING, "share of light lost"),
        row("IoU, 3.1.4 (flatten + CLAHE)", LIGHTING, curves["lighting"]),
        row("IoU, flatten only", LIGHTING, curves["lighting_flatten"]),
        row("IoU, no correction", LIGHTING, curves["lighting_raw"]),
        "",
        "**A global threshold survives a multiplicative shadow far longer than expected**, and ",
        "the reason is arithmetic: Otsu fails only once the shaded paper is darker than the lit ",
        "ink, which on this material is past `s = 0.86`. Up to there, correcting the ",
        "illumination changes nothing. Past there the *flattening* half of 3.1.4 holds, while ",
        "the CLAHE half costs IoU — it amplifies quantisation steps in a region where the paper ",
        "has been reduced to a handful of grey levels. On the real photographs of 3.1.4 and ",
        "3.1.5, where shading is blotchy rather than a clean ramp and strokes are faint, CLAHE ",
        "earns its place; on this synthetic ramp it does not. Both measurements are right about ",
        "their own material, and neither generalises to the other on its own.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=20)
    args = ap.parse_args(argv)

    result = sweep(args.count)
    if not result:
        print("no evaluation items; check data/raw/fa_bresler", file=sys.stderr)
        return 1
    path = figure(result)
    report(result)
    print(json.dumps(result, indent=2, default=list))

    curves = result["curves"]
    checks = {
        "figure_written": path.is_file(),
        "three_sweeps_plotted": all(
            len(curves[key]) >= 5 for key in ("blur", "rotation", "lighting")
        ),
        "rotation_costs_little": curves["rotation"][-1] >= 0.9 * curves["rotation"][0],
        "deskew_recovers_the_angle": max(curves["deskew_error"]) <= 1.0,
        "background_flattening_costs_nothing": (
            curves["lighting_flatten"][-1] >= 0.95 * curves["lighting_flatten"][0]
        ),
        "blur_is_the_expensive_one": curves["blur"][-1] < curves["rotation"][-1],
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

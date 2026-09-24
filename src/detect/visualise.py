"""Phase 9.2.4 / 9.2.6 - the first-layer filters, the feature maps, and where the class evidence is.

    python -m src.detect.visualise      # p9_filters.png, p9_featuremaps.png, p9_gradcam.png

Three figures on the trained network from 9.2.1, all on **real crops from the validation split**
rather than a synthetic square, because the question is what the network learned from
photographed pen strokes and a clean rendering would not answer it.

## 9.2.4 - filters and feature maps

The 16 first-layer 3x3 kernels, and the 16 activations they produce on one real sketch. A 3x3
filter is nine numbers and there is a limit to what can be read off it; what *can* be read is
whether the layer has differentiated at all - whether the kernels are distinguishable from each
other and from their random initialisation - and the figure reports the standard deviation
across kernels beside them so that is a number rather than an impression.

## 9.2.6 - Grad-CAM

The gradient of a class logit with respect to the last convolutional feature map, global-average
-pooled into per-channel weights, applied to that map and ReLU'd:

    w_k = mean_ij (d y_c / d A^k_ij)          alpha for channel k
    L   = ReLU( sum_k w_k A^k )

At the last conv layer the map is **8x8** before the final pool, so a Grad-CAM heatmap here has
64 cells upsampled to 64x64 - each heatmap pixel is an 8x8 block, and the resolution of any
claim made from it is 8 pixels, not 1. That is stated on the figure because a smooth bicubic
upsample makes a coarse map look precise.

The layer is chosen deliberately: 9.2.3 measured the final receptive field at 52 of 64 pixels,
so this is the deepest place where a spatial claim is still meaningful. The fully connected
layer that follows holds 70% of the parameters and has no spatial extent at all, so **Grad-CAM
cannot see the component doing most of the work** - a limit worth knowing before reading the
heatmaps as an explanation of the network.

## What it measured

**9.2.4 - the filters did differentiate, and the number says so where the picture cannot.**
The 16 first-layer kernels have weight standard deviation 0.2028 and norms spanning 0.4823 to
0.7342, and **the mean absolute cosine similarity between distinct kernels is 0.2852** with a
maximum of 0.8233. Sixteen 3x3 kernels rendered as small red-blue squares are close to
unreadable and every such figure ever published looks equally convincing, so the figure carries
that cosine on its title: near 1.0 would mean the layer learned one filter sixteen times, which
is the failure a filter grid is supposed to reveal and almost never does. At 0.285 the layer is
genuinely diverse; the one pair at 0.82 is the redundancy that remains.

**9.2.6 - the class evidence is mostly not on the ink, and which classes are the exception is
the finding.** Measured over 290 validation crops rather than the seven on the figure, as
*enrichment* - the share of Grad-CAM mass falling on ink, divided by the share of pixels that
are ink - so that a crop being mostly paper cannot masquerade as a result:

    class            median enrichment   crops above 1.0
    freeform                   1.117           60%
    double-circle              1.104           74%
    rounded-rect               0.753           36%
    diamond                    0.410            9%
    parallelogram              0.031            0%
    rectangle                  0.000           20%
    circle                     0.000           22%

**Median across classes 0.41: the heat sits on paper at less than half of chance.** And the two
classes above 1.0 are exactly the two whose identity *is* a local ink feature - a
`double-circle` is defined by its second ring and a `freeform` by an irregular boundary, and
both are things you must look at a stroke to see. The classes at zero are the ones whose
identity is the **shape of the enclosed region**: what makes a rectangle is where its area
extends, not what its outline is made of, and at the last conv layer that is represented by
which of the 8x8 cells are active rather than by whether they sit on a line.

Read with 9.2.3 this is coherent rather than surprising. The final receptive field is 52 of 64
pixels and the fully connected layer holds 70% of the parameters, so **the network's global
shape reasoning happens after the last convolution, in a layer with no spatial extent that
Grad-CAM cannot see at all.** A heatmap here shows where the *local* evidence was gathered, and
for four of seven classes the honest answer is that the discriminative local evidence is not on
the strokes.

**Three limits are recorded rather than smoothed over.** The map is **8x8 upsampled to 64x64**,
so every claim has a resolution of 8 pixels and the figure says so. **17 of 307 crops have an
empty ink mask at a 0.5 grey threshold** - faint pencil - and are excluded rather than scored
zero, since that is a limit of the mask and not evidence about the model. And `parallelogram`'s
row rests on **7 crops**, which is the same 7 instances that make its F1 unreadable everywhere
else in 9.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.detect.crops import OUT as CROPS
from src.detect.crops import SHAPE_CLASSES, load_split
from src.detect.scratchcnn import WEIGHTS, load
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURES = ROOT / "reports" / "figures"
FILTERS = FIGURES / "p9_filters.png"
MAPS = FIGURES / "p9_featuremaps.png"
GRADCAM = FIGURES / "p9_gradcam.png"


def first_conv(model):
    import torch.nn as nn

    return next(m for m in model.modules() if isinstance(m, nn.Conv2d))


def last_conv_index(model) -> int:
    import torch.nn as nn

    return max(i for i, m in enumerate(model) if isinstance(m, nn.Conv2d))


def filter_stats(model) -> dict:
    """Whether the first layer differentiated, as numbers rather than an impression."""
    weights = first_conv(model).weight.detach().cpu().numpy()
    flat = weights.reshape(len(weights), -1)
    norms = np.linalg.norm(flat, axis=1)
    # Pairwise cosine similarity between kernels: near 1 everywhere means they all learned the
    # same thing, which is the failure mode a filter grid is supposed to reveal and usually does
    # not, because sixteen small grey squares all look alike.
    unit = flat / np.maximum(norms[:, None], 1e-12)
    similarity = unit @ unit.T
    off_diagonal = similarity[~np.eye(len(unit), dtype=bool)]
    return {
        "kernels": int(len(weights)),
        "weight_std": round(float(weights.std()), 4),
        "norm_min": round(float(norms.min()), 4),
        "norm_max": round(float(norms.max()), 4),
        "mean_abs_cosine_between_kernels": round(float(np.abs(off_diagonal).mean()), 4),
        "max_abs_cosine_between_kernels": round(float(np.abs(off_diagonal).max()), 4),
    }


def gradcam(model, x: np.ndarray, target: int) -> np.ndarray:
    """A `(H, W)` map in [0, 1] for one crop and one class."""
    import torch
    import torch.nn.functional as functional

    model.eval()
    index = last_conv_index(model)
    tensor = torch.from_numpy(x[None]).float()
    activations = tensor
    for layer in list(model)[: index + 1]:
        activations = layer(activations)
    activations.retain_grad()
    out = activations
    for layer in list(model)[index + 1 :]:
        out = layer(out)
    score = out[0, target]
    model.zero_grad()
    score.backward()

    weights = activations.grad.mean(dim=(2, 3), keepdim=True)
    cam = functional.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    span = cam.max() - cam.min()
    return (cam - cam.min()) / span if span > 1e-12 else np.zeros_like(cam)


def _examples(root: Path = CROPS, per_class: int = 1) -> list[tuple[np.ndarray, int]]:
    """One correctly-labelled validation crop per class that has any."""
    x, y = load_split("val", root)
    picked = []
    for index in range(len(SHAPE_CLASSES)):
        where = np.flatnonzero(y == index)
        for row in where[:per_class]:
            picked.append((x[row], index))
    return picked


def figure_filters(model, path: Path = FILTERS) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = first_conv(model).weight.detach().cpu().numpy()[:, 0]
    stats = filter_stats(model)
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for ax, kernel in zip(axes.ravel(), weights, strict=False):
        ax.imshow(kernel, cmap="RdBu_r", vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Phase 9.2.4 - the 16 first-layer 3x3 kernels  |  mean |cos| between kernels "
        f"{stats['mean_abs_cosine_between_kernels']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def figure_maps(model, root: Path = CROPS, path: Path = MAPS) -> Path:
    import matplotlib
    import torch

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = _examples(root)
    crop, label = examples[0]
    with torch.no_grad():
        activations = model[0](torch.from_numpy(crop[None]).float())
        activations = activations[0].cpu().numpy()

    fig, axes = plt.subplots(2, 9, figsize=(12, 3.2))
    axes[0, 0].imshow(crop[0], cmap="gray")
    axes[0, 0].set_title(SHAPE_CLASSES[label], fontsize=8)
    axes[1, 0].axis("off")
    for ax in (axes[0, 0],):
        ax.set_xticks([])
        ax.set_yticks([])
    for position, ax in enumerate(list(axes[0, 1:]) + list(axes[1, 1:])):
        if position < len(activations):
            ax.imshow(activations[position], cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Phase 9.2.4 - first-layer activations on a real crop", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def figure_gradcam(model, root: Path = CROPS, path: Path = GRADCAM) -> tuple[Path, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = _examples(root)
    fig, axes = plt.subplots(2, len(examples), figsize=(1.7 * len(examples), 4))
    axes = np.atleast_2d(axes)
    mass = {}
    for column, (crop, label) in enumerate(examples):
        cam = gradcam(model, crop, label)
        axes[0, column].imshow(crop[0], cmap="gray")
        axes[0, column].set_title(SHAPE_CLASSES[label], fontsize=8)
        axes[1, column].imshow(crop[0], cmap="gray")
        axes[1, column].imshow(cam, cmap="inferno", alpha=0.55)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        # How much of the heat sits on ink rather than on paper: the one quantitative check a
        # heatmap figure can carry, since "it looks like it is on the shape" is not a finding.
        #
        # Reported **against the share of pixels that are ink at all**, because a crop is mostly
        # paper: a rectangle's outline is a few per cent of its 4,096 pixels, so a perfectly
        # uniform heatmap already puts only a few per cent of its mass on ink. Without that
        # denominator the raw share reads as "the network ignores the strokes" when it may only
        # mean "there are not many strokes".
        ink = crop[0] < 0.5
        share = float(cam[ink].sum() / max(cam.sum(), 1e-9)) if ink.any() else 0.0
        pixels = float(ink.mean())
        mass[SHAPE_CLASSES[label]] = {
            "cam_share_on_ink": round(share, 4),
            "ink_pixel_share": round(pixels, 4),
            "enrichment": round(share / pixels, 3) if pixels > 1e-9 else None,
        }
    fig.suptitle(
        "Phase 9.2.6 - Grad-CAM at the last conv layer (8x8 map upsampled; resolution is 8 px)",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path, mass


def cam_statistics(model, root: Path = CROPS, per_class: int = 50, threshold: float = 0.5) -> dict:
    """Ink enrichment over many crops, so the figure's claim is a statistic and not an anecdote.

    The seven crops on the figure are one example each; this repeats the same measurement over
    up to `per_class` validation crops and reports the median enrichment per class. Crops whose
    ink mask is empty at this threshold are counted and excluded rather than scored as zero -
    a faint pencil stroke above 0.5 grey is a limit of the mask, not evidence about the model.
    """
    x, y = load_split("val", root)
    out: dict[str, dict] = {}
    empty = 0
    for index, name in enumerate(SHAPE_CLASSES):
        rows = np.flatnonzero(y == index)[:per_class]
        values = []
        for row in rows:
            crop = x[row]
            ink = crop[0] < threshold
            if not ink.any():
                empty += 1
                continue
            cam = gradcam(model, crop, index)
            share = float(cam[ink].sum() / max(cam.sum(), 1e-9))
            values.append(share / float(ink.mean()))
        if values:
            out[name] = {
                "crops": len(values),
                "median_enrichment": round(float(np.median(values)), 3),
                "share_above_one": round(float(np.mean(np.asarray(values) > 1.0)), 3),
            }
    pooled = [v["median_enrichment"] for v in out.values()]
    return {
        "per_class": out,
        "median_enrichment": round(float(np.median(pooled)), 3) if pooled else None,
        "crops_with_empty_ink_mask": empty,
        "ink_threshold": threshold,
    }


def run(weights: Path = WEIGHTS, root: Path = CROPS) -> dict:
    model, _ = load(weights)
    stats = filter_stats(model)
    filters = figure_filters(model)
    maps = figure_maps(model, root)
    cam, mass = figure_gradcam(model, root)
    enrichments = [v["enrichment"] for v in mass.values() if v["enrichment"] is not None]
    return {
        "filter_stats": stats,
        "gradcam_ink": dict(mass),
        # Above 1.0 the heat concentrates on strokes; at 1.0 it is indifferent to them; below,
        # it sits on paper. This is the number, not the raw share.
        "mean_ink_enrichment": round(float(np.mean(enrichments)), 3) if enrichments else None,
        "cam_statistics": cam_statistics(model, root),
        "figures": [str(p.relative_to(ROOT)) for p in (filters, maps, cam)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--root", type=Path, default=CROPS)
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        print(
            f"no weights at {args.weights} (run `python -m src.detect.scratchcnn`)", file=sys.stderr
        )
        return 1
    print(json.dumps(run(args.weights, args.root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

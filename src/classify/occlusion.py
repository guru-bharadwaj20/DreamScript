"""Phase 7.1.6 - coffee stains, torn corners, a finger over the lens, and where the model breaks.

    python -m src.classify.occlusion      # writes reports/figures/p7_occlusion.png

3.3 measured robustness to blur, rotation and illumination - perturbations that degrade the whole
page uniformly. This task measures the other kind: **occlusion, where part of the page is
destroyed and the rest is untouched.** The plan names four, and they are four genuinely different
damage models rather than one with different masks:

    coffee_stain    a dark translucent blob somewhere in the middle. Ink survives underneath but
                    the local contrast collapses, so 3.1's illumination correction has something
                    to react to and 3.2's binarization may lose the strokes it covers.
    torn_corner     a corner replaced by page-white. Removes content and *changes the bounding
                    box*, which is the mechanism that should hurt the handcrafted geometry
                    features specifically - `global_aspect` and `global_bbox_fill` are computed
                    from exactly that.
    finger          an opaque out-of-focus wedge intruding from one edge, the way a thumb does.
                    Occludes and adds a large smooth region the binarizer must not call ink.
    crop            the page cut to a sub-rectangle and rescaled. Loses content at the edges and
                    changes the scale of everything that remains.

## Where the occlusion is applied, and why it matters

**To the raw photograph, before 3.1's illumination correction** - not to the 224x224 normalised
array. Those are different experiments. A real coffee stain is something the correction step sees
and tries to compensate for, sometimes making things worse elsewhere on the page; occluding after
correction would measure a model's robustness to a perturbation the pipeline never had a chance
to react to, which is the easier and less interesting question.

The cost of doing it properly is that every severity level re-runs the full preprocessing chain.
The page is decoded from disk once and held in memory, so what is paid per variant is the
correction and the threshold, not the JPEG.

## What is scored

6.3.7's winning configuration - an RBF SVM on 6.1.3's PCA of a frozen CLIP embedding - fitted
**once on clean pages** and then evaluated on damaged ones. Fitting on clean data and testing on
damaged data is the deployment question; fitting on damaged data would be augmentation, which is
Phase 14's subject.

## What it measured

300 stratified real pages, four damage models, six severities, 7,200 damaged pages embedded and
scored. **Nothing halves the score. `half_life` is `None` for all four models**, and the reason
the column exists is that reporting "the curve did not reach the threshold" is a result and
inventing a ceiling is not.

**Half the page can be destroyed and the classifier still works.** Clean macro F1 is 0.9475 -
which is the guard passing, since it lands within 0.024 of 6.3.7's 0.9718 on the full corpus and
the first version of this task scored 0.0274 on clean pages through a broken projection. At
severity 0.5 the four end at 0.9297 / 0.8926 / 0.8824 / 0.7990. **A coffee stain over half the
page costs 0.0178 macro F1**, which is within the noise of this sample, and the ordering of harm
is `finger` (-0.1485) > `torn_corner` (-0.0651) > `crop` (-0.0549) > `coffee_stain` (-0.0178).

That ordering is the finding, and it is about *what each model removes* rather than how much:

  * a **stain** is multiplicative, so the ink survives underneath and 3.1's illumination
    correction has exactly the kind of low-frequency intensity change it was built to flatten -
    the strokes are still there and the page still reads;
  * a **finger** is opaque, and it is the only model that both deletes content and adds a large
    smooth dark region that was never on the paper. It costs eight times what the stain does;
  * a **torn corner** and a **crop** delete content without adding anything, and land between.

The general result is that **diagram type is a redundant, global property**: a CLIP page
embedding of half a flowchart still says flowchart, because the evidence - box-and-arrow
texture, layout regularity, stroke density - is repeated everywhere on the page. This is the
complement of 3.3's finding rather than a contradiction: perturbations that degrade the *whole*
page uniformly change every part of the evidence at once, while occlusion leaves most of it
untouched.

**Accuracy and macro F1 disagree about how bad it is, and macro F1 is right.** Across the worst
curve accuracy falls 0.9867 -> 0.9600 while macro F1 falls 0.9475 -> 0.7990: the damage lands
almost entirely on the small classes, so a reader watching accuracy would see a 2.7-point dent
where the minority classes lost 15. That is 7.1.4's LightGBM collapse in a different setting and
the same argument for the metric.

Two things are recorded as *not* established. The curves are **not monotone** - the stain scores
0.9092 at severity 0.2 and 0.9297 at 0.35 and 0.5 - and with 300 pages over five classes a
single page moving class is worth about 0.01 macro F1, so the wiggles are sampling noise and
only the endpoints should be read. And the docstring's prediction that a torn corner would hurt
**specifically** because it changes the bounding box, moving `global_aspect` and
`global_bbox_fill`, is **untested here**: 6.3.7's winner has no such features - it resizes to
224x224 and embeds - so that hypothesis needs 4.2's handcrafted table to be run through the same
damage, which is a separate experiment and is not claimed by this one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p7_occlusion.png"

SEED = 42

KINDS = ("coffee_stain", "torn_corner", "finger", "crop")

#: 6.1.3 selected 128 columns; the projection is refitted here so that the training and
#: evaluation sides are guaranteed to share one basis.
PCA_COMPONENTS = 128

#: Fraction of the page area affected. 0.0 is the clean control and is always scored, because a
#: robustness curve without its own baseline cannot be read.
SEVERITIES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)


def occlude(gray: np.ndarray, kind: str, severity: float, seed: int = SEED) -> np.ndarray:
    """Apply one damage model to a raw greyscale page. `severity` is the affected area fraction."""
    import cv2

    if severity <= 0:
        return gray
    rng = np.random.default_rng(seed)
    height, width = gray.shape
    out = gray.copy()

    if kind == "coffee_stain":
        # A filled ellipse darkened multiplicatively rather than painted flat: a stain leaves the
        # ink visible underneath, which is what makes it different from the finger.
        radius = np.sqrt(severity * height * width / np.pi)
        centre = (
            int(rng.uniform(radius, max(radius + 1, width - radius))),
            int(rng.uniform(radius, max(radius + 1, height - radius))),
        )
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.ellipse(
            mask,
            centre,
            (int(radius * 1.2), int(radius * 0.85)),
            float(rng.uniform(0, 180)),
            0,
            360,
            255,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (0, 0), radius * 0.15)
        alpha = (mask.astype(np.float32) / 255.0) * 0.65
        out = (out.astype(np.float32) * (1 - alpha) + 60 * alpha).astype(np.uint8)

    elif kind == "torn_corner":
        # Page-white, not black: a torn corner shows the sheet behind, and filling it dark would
        # add ink rather than remove it.
        side = int(np.sqrt(severity * 2) * min(height, width))
        corner = int(rng.integers(0, 4))
        triangle = {
            0: [(0, 0), (side, 0), (0, side)],
            1: [(width, 0), (width - side, 0), (width, side)],
            2: [(0, height), (side, height), (0, height - side)],
            3: [(width, height), (width - side, height), (width, height - side)],
        }[corner]
        cv2.fillPoly(out, [np.array(triangle, dtype=np.int32)], int(np.percentile(gray, 95)))

    elif kind == "finger":
        # Opaque, blurred at the boundary, intruding from one edge - dark and featureless.
        depth = int(severity * min(height, width) * 1.6)
        edge = int(rng.integers(0, 4))
        mask = np.zeros_like(gray, dtype=np.uint8)
        if edge == 0:
            cv2.ellipse(
                mask,
                (int(width * rng.uniform(0.2, 0.8)), 0),
                (int(depth * 0.8), depth),
                0,
                0,
                360,
                255,
                -1,
            )
        elif edge == 1:
            cv2.ellipse(
                mask,
                (int(width * rng.uniform(0.2, 0.8)), height),
                (int(depth * 0.8), depth),
                0,
                0,
                360,
                255,
                -1,
            )
        elif edge == 2:
            cv2.ellipse(
                mask,
                (0, int(height * rng.uniform(0.2, 0.8))),
                (depth, int(depth * 0.8)),
                0,
                0,
                360,
                255,
                -1,
            )
        else:
            cv2.ellipse(
                mask,
                (width, int(height * rng.uniform(0.2, 0.8))),
                (depth, int(depth * 0.8)),
                0,
                0,
                360,
                255,
                -1,
            )
        mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, depth * 0.08))
        alpha = mask.astype(np.float32) / 255.0
        out = (out.astype(np.float32) * (1 - alpha) + 45 * alpha).astype(np.uint8)

    elif kind == "crop":
        keep = np.sqrt(max(1e-6, 1.0 - severity))
        new_h, new_w = int(height * keep), int(width * keep)
        top = int(rng.integers(0, max(1, height - new_h + 1)))
        left = int(rng.integers(0, max(1, width - new_w + 1)))
        out = gray[top : top + new_h, left : left + new_w]

    else:
        raise ValueError(f"kind must be one of {list(KINDS)}; got {kind!r}")

    return out


def page_array_from(gray: np.ndarray, mode: str = "gray") -> np.ndarray:
    """6.1.1's `page_array`, but starting from an in-memory image rather than a path.

    Factored this way so one decode serves every occlusion variant of a page, while each variant
    still pays for the full 3.1 correction and 3.2 threshold - which is the whole point of
    occluding before preprocessing rather than after.
    """
    import cv2

    from src.embed.backbone import SIZE
    from src.preprocess import binarize as binarization
    from src.preprocess import illumination

    corrected = illumination.correct(gray)
    if mode == "binary":
        ink = binarization.binarize(corrected, "otsu", correct_illumination=False)
        canvas = np.where(ink, 0.0, 1.0).astype(np.float32)
    elif mode == "gray":
        canvas = corrected.astype(np.float32) / 255.0
    else:
        raise ValueError(f"input mode must be binary or gray; got {mode!r}")
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def sample_pages(data, limit: int = 300, seed: int = SEED):
    """A stratified subset, so the 40-row class is not lost and the run is affordable."""
    from sklearn.model_selection import train_test_split

    if limit >= len(data.y):
        return np.arange(len(data.y))
    index, _ = train_test_split(
        np.arange(len(data.y)), train_size=limit, random_state=seed, stratify=data.y
    )
    return np.sort(index)


def embed_variants(
    paths, kinds=KINDS, severities=SEVERITIES, mode: str = "gray", batch_size: int = 64
) -> dict:
    """Every (kind, severity) variant of every page, embedded with 6.1.1's backbone.

    Returns `(kind, severity) -> (rows, 512)`. The clean control is embedded once and shared
    across kinds, because at severity 0 the four damage models are the same image.
    """
    import cv2

    from src.embed.backbone import batch_tensor, device, load

    model, width, statistics = load("clip_vit_b32")
    import torch

    raw = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        raw.append(image)

    def forward(arrays: list[np.ndarray]) -> np.ndarray:
        out = np.full((len(arrays), width), np.nan, dtype=np.float32)
        for start in range(0, len(arrays), batch_size):
            chunk = arrays[start : start + batch_size]
            keep = [start + i for i, a in enumerate(chunk) if a is not None]
            usable = [a for a in chunk if a is not None]
            if not usable:
                continue
            tensor = batch_tensor(usable, statistics).to(device())
            with torch.no_grad():
                vectors = model(pixel_values=tensor).image_embeds
            out[keep] = vectors.float().cpu().numpy()
        return out

    results = {}
    clean = forward([page_array_from(g, mode) if g is not None else None for g in raw])
    for kind in kinds:
        for severity in severities:
            if severity == 0.0:
                results[(kind, severity)] = clean
                continue
            arrays = [
                (
                    page_array_from(occlude(g, kind, severity, SEED + i), mode)
                    if g is not None
                    else None
                )
                for i, g in enumerate(raw)
            ]
            results[(kind, severity)] = forward(arrays)
    return results


def curve(data, index, variants: dict, kinds=KINDS, severities=SEVERITIES) -> dict:
    """Accuracy-vs-occlusion, from a model fitted once on clean pages.

    The model is 6.3.7's winner and it is fitted on the **clean** embeddings of the rows outside
    the evaluation sample, so no damaged page and no evaluated page is ever in its training set.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC

    from src.embed.cache import load as load_cache

    _, frame, _ = load_cache()
    position = dict(zip(frame["id"].tolist(), frame["row"].tolist(), strict=True))

    # One projection, fitted here and used for *both* sides.
    #
    # The first version of this task fitted the SVM on 6.1.3's saved PCA matrix and then projected
    # the damaged 512-d vectors through a freshly fitted PCA. Those are two different bases - a
    # refit is free to flip the sign of any component and to order near-equal ones differently -
    # so the model was being asked to classify points expressed in axes it had never seen. It
    # scored 0.0274 macro F1 on *undamaged* pages, against Phase 6's 0.9742, and the flat
    # occlusion curves that produced were an artefact of a broken projection rather than a
    # robustness result. The clean-page score is now the guard: it is reported, and it has to
    # land near 6.3.7's number before any damaged score means anything.
    full = np.load(ROOT / "data" / "features" / "embeddings.npy")
    projector = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(full)
    matrix = projector.transform(full)

    held_out = np.array([i for i in range(len(data.y)) if i not in set(index.tolist())])
    train_rows = np.array([position[data.ids[i]] for i in held_out])
    fitted = SVC(kernel="rbf", gamma=1e-5, C=1000.0, random_state=SEED).fit(
        matrix[train_rows], data.y[held_out]
    )

    truth = data.y[index]
    rows = {}
    for kind in kinds:
        series = []
        for severity in severities:
            vectors = variants[(kind, severity)]
            usable = ~np.isnan(vectors).any(axis=1)
            projected = projector.transform(vectors[usable])
            predicted = fitted.predict(projected)
            series.append(
                {
                    "severity": severity,
                    "macro_f1": round(
                        float(f1_score(truth[usable], predicted, average="macro", zero_division=0)),
                        4,
                    ),
                    "accuracy": round(float((predicted == truth[usable]).mean()), 4),
                    "scored": int(usable.sum()),
                }
            )
        rows[kind] = series
    return rows


def half_life(series: list[dict], baseline: float) -> float | None:
    """The severity at which macro F1 first falls below half its clean value.

    A single comparable number per damage model, so the four curves can be ranked without
    eyeballing them - and None when the model never degrades that far, which is itself a result.
    """
    for point in series:
        if point["severity"] > 0 and point["macro_f1"] < baseline / 2:
            return point["severity"]
    return None


def figure(rows: dict, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colours = plt.get_cmap("tab10")
    for index, (kind, series) in enumerate(sorted(rows.items())):
        ax.plot(
            [p["severity"] for p in series],
            [p["macro_f1"] for p in series],
            marker="o",
            lw=1.8,
            color=colours(index),
            label=kind,
        )
    ax.set_xlabel("fraction of the page occluded")
    ax.set_ylabel("macro F1")
    ax.set_title(
        "Phase 7.1.6 - occlusion robustness, RBF-SVM on CLIP embeddings fitted on clean pages",
        fontsize=9,
    )
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(
    corpus: str = "real", limit: int = 300, kinds=KINDS, severities=SEVERITIES, write: bool = True
) -> dict:
    from src.classify.supportvectors import image_paths
    from src.embed.hybrid import dataset

    data = dataset("embedding", corpus)
    index = sample_pages(data, limit)
    paths = image_paths(data.ids[index])
    keep = np.array([i for i in index if data.ids[i] in paths])
    if not len(keep):
        raise FileNotFoundError("no page images on disk; pull the corpus with dvc first")

    variants = embed_variants([paths[data.ids[i]] for i in keep], kinds, severities)
    rows = curve(data, keep, variants, kinds, severities)

    baselines = {kind: series[0]["macro_f1"] for kind, series in rows.items()}
    return {
        "corpus": corpus,
        "pages_scored": int(len(keep)),
        "kinds": list(kinds),
        "severities": list(severities),
        "curves": rows,
        "clean_macro_f1": baselines[kinds[0]],
        "at_half_page": {kind: series[-1]["macro_f1"] for kind, series in rows.items()},
        "drop_at_half_page": {
            kind: round(baselines[kind] - series[-1]["macro_f1"], 4)
            for kind, series in rows.items()
        },
        "half_life": {kind: half_life(series, baselines[kind]) for kind, series in rows.items()},
        "figure": str(figure(rows).relative_to(ROOT)) if write else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.corpus, args.limit, tuple(args.kinds), SEVERITIES, not args.no_figure)
    except (FileNotFoundError, KeyError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

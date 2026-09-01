"""Phase 6.1.1 - choosing a frozen backbone, and asking whether binarizing helps or hurts.

    python -m src.embed.backbone --limit 400        # the selection sweep
    python -m src.embed.backbone --backbone resnet18 --input gray

Phase 4 spent nine feature families describing a diagram by hand and Phase 5 got 0.7898 macro F1
out of them. This is the other approach: hand the page to a network trained on something else
entirely, take the layer before its classifier, and see whether the label is in there. Nothing
is fine-tuned - the backbone is a fixed function, which is what makes 6.1.2's cache legitimate
and what keeps this affordable.

    resnet18       512-d, ImageNet-1k
    resnet34       512-d, ImageNet-1k, twice the depth for the same width
    clip_vit_b32   512-d, CLIP's image tower - trained on captioned web images, not on labels

CLIP is in the sweep because its training objective is the one most likely to survive the domain
gap: ImageNet is photographs of objects, and a photograph of a whiteboard is not one.

## The question the plan's wording hides

The task says "on 224x224 **binarized** images", and that instruction deserves a measurement
rather than obedience. Binarizing is what Phase 3 exists for and it is the right input for the
geometry of Phase 4 - but these backbones were trained on natural photographs with texture and
shading, and a two-tone line drawing is far outside that distribution. Throwing away the
greyscale might be discarding exactly the surface statistics the early convolutions know how to
read. So both are run:

    binary   3.1's illumination correction and Otsu, ink black on white
    gray     the corrected greyscale, unthresholded

## How a backbone is scored

By the only thing that matters here: a logistic regression on the embedding, cross-validated the
way Phase 5 cross-validates, so **the number is directly comparable with 5.1.1's 0.7898**. Six
combinations of backbone and input, each embedded once and scored once. That comparison is the
whole point of 6.1 and it is set up so it cannot be dodged later.

## What it measured

All 1,340 real photographs - the rows of 4.2.2's table, not the 1,695 the disk offers - six
combinations, on a CUDA device with 32 cores decoding:

    backbone       input    macro F1   accuracy    forward
    clip_vit_b32   gray      0.9720     0.9925     1,162 pages/s
    clip_vit_b32   binary    0.9669     0.9918       895 pages/s
    resnet34       gray      0.9352     0.9836     1,713 pages/s
    resnet34       binary    0.9222     0.9791     1,721 pages/s
    resnet18       binary    0.9173     0.9813     1,789 pages/s
    resnet18       gray      0.9125     0.9784     1,439 pages/s

    5.1.1, 33 handcrafted features       0.7898

**A frozen CLIP embedding beats nine families of hand-built geometry by 0.18 macro F1**, having
been told nothing about diagrams. That is the headline of Phase 6 and it arrives before a single
parameter has been trained.

**Grey beats binary on the two backbones that matter** (CLIP +0.0051, ResNet-34 +0.0130), and
ResNet-18 is the lone exception (-0.0048). The plan's wording said "binarized" and the
measurement says otherwise, though not by much: the honest summary is that thresholding costs a
little and buys nothing here, so 6.1.2 caches grey. Phase 3's binarization remains right for
Phase 4's geometry - it is the input to *these* networks it does not suit, which is what the
domain-gap argument predicted.

## The result this corpus could have faked, and the test that says it did not

Source and label are nearly the same variable here: hdBPMN is 600 of 600 flowcharts,
sketch2code 600 of 600 wireframes, and only the 140-page chaos corpus mixes three classes under
one capture protocol. **A representation that merely recognised the dataset would score 0.9328
accuracy** - and an embedding of a photograph encodes paper, lighting and camera far more
readily than 33 geometric ratios do. 4.1.5 caught that leak in one feature; here it could have
been the entire representation, and a 0.97 would have meant nothing.

It is not. Both halves were measured:

    source predicted from the embedding        0.9776 - 0.9918 accuracy
    within chaos alone, 140 rows, 3 classes    clip 0.9925 macro F1
                                               resnet34 0.9850, resnet18 0.9849
    within chaos, 4.2.2's handcrafted features       0.9135

Provenance **is** plainly readable in the embedding - that much of the suspicion was right. But
with provenance held constant, on one source and one protocol, the embedding still beats the
handcrafted features by 0.079. The advantage survives the only control this corpus can apply, so
it is a fact about the representation rather than about the cameras. Phase 14's ablation still
has to re-ask this on a corpus where the sources are crossed with the classes; nothing here can
settle it for good.

## Where the time goes, and why 6.1.2 exists

    decoding, 1,340 pages, 32 threads      144 s per input mode
    forward pass, all 1,340 pages          0.7 - 1.5 s

**The GPU is idle for 99% of this task.** Decoding, illumination-correcting and resizing the
photographs costs two hundred times what embedding them does, and the arrays depend only on the
input mode, so the sweep decodes twice and embeds six times rather than decoding six times.
That ratio is the entire argument for 6.1.2: cache the embedding and the next task is free;
cache nothing and every consumer pays 144 seconds to look at the same pixels again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

#: 224 is what all three backbones were trained at. Resizing the page rather than cropping it,
#: because a diagram cropped to its centre is a different diagram.
SIZE = 224

#: One entry per backbone: how to build it, and the embedding width it returns.
BACKBONES = ("resnet18", "resnet34", "clip_vit_b32")

INPUTS = ("binary", "gray")

#: ImageNet statistics for the torchvision models; CLIP ships its own and they differ enough to
#: matter (its mean is ~0.48/0.46/0.41 against ImageNet's 0.485/0.456/0.406 - close - but its
#: standard deviations are ~0.27 against 0.225, which is a 20% difference in contrast).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load(name: str):
    """A frozen backbone in eval mode, its output width, and its normalisation statistics."""
    import torch

    if name in ("resnet18", "resnet34"):
        import torchvision.models as models

        builder, weights = (
            (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1)
            if name == "resnet18"
            else (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1)
        )
        model = builder(weights=weights)
        # Replace the 1000-way classifier with the identity: the penultimate pooled vector is
        # the representation, and the ImageNet logits are about cats.
        model.fc = torch.nn.Identity()
        model.eval().to(device())
        return model, 512, (IMAGENET_MEAN, IMAGENET_STD)

    if name == "clip_vit_b32":
        from transformers import CLIPVisionModelWithProjection

        model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        model.eval().to(device())
        return model, int(model.config.projection_dim), (CLIP_MEAN, CLIP_STD)

    raise ValueError(f"unknown backbone {name!r}; known: {list(BACKBONES)}")


def page_array(path, mode: str = "binary") -> np.ndarray:
    """One page as a 224x224 single-channel float array in [0, 1], ink dark.

    Both modes go through 3.1's illumination correction first, so the only difference between
    them is the threshold - which is what makes the binary-vs-grey comparison a comparison of
    one decision rather than of two pipelines.
    """
    import cv2

    from src.preprocess import binarize as binarization
    from src.preprocess import illumination

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read {path}")
    corrected = illumination.correct(image)

    if mode == "binary":
        # `binarize` returns an ink mask - True where the pen is - so it is inverted back into
        # a picture: ink dark on white, the way the page looked.
        ink = binarization.binarize(corrected, "otsu", correct_illumination=False)
        canvas = np.where(ink, 0.0, 1.0).astype(np.float32)
    elif mode == "gray":
        canvas = corrected.astype(np.float32) / 255.0
    else:
        raise ValueError(f"input mode must be binary or gray; got {mode!r}")

    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def batch_tensor(arrays: list[np.ndarray], statistics) -> object:
    """A stack of single-channel pages as a normalised 3-channel batch.

    The grey plane is repeated across R, G and B rather than left in one channel: the first
    convolution of every one of these backbones expects three, and a page has no colour to lose.
    """
    import torch

    mean, std = statistics
    stack = torch.from_numpy(np.stack(arrays)).unsqueeze(1).repeat(1, 3, 1, 1)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return (stack - mean_t) / std_t


def _safe_page_array(path, mode: str):
    """`page_array`, returning None instead of raising - a dead page must not kill a batch."""
    try:
        return page_array(path, mode)
    except (OSError, ValueError):
        return None


def prepare(paths, mode: str = "binary", n_jobs: int | None = None) -> list:
    """Every page decoded, corrected, thresholded and resized once, across all the cores.

    Separated from `embed` because the 224x224 array is a function of the **input mode alone** -
    three backbones handed the same mode want byte-identical pixels. Doing this inside `embed`
    made the selection sweep decode all 1,340 photographs six times to produce two distinct sets
    of arrays, and decoding is the whole cost: the measured rate was 2.8 pages a second against
    a GPU that finishes a batch of 64 in milliseconds.

    Threads rather than processes, because the work is inside OpenCV, which releases the GIL and
    would otherwise pay to pickle a 16-megapixel array to a worker and a small one back.
    """
    from src.utils.parallel import pmap

    return pmap(
        lambda path: _safe_page_array(path, mode),
        list(paths),
        n_jobs=n_jobs if n_jobs is not None else os.cpu_count(),
        prefer="threads",
    )


def embed(
    paths,
    name: str = "resnet18",
    mode: str = "binary",
    batch_size: int = 64,
    n_jobs: int | None = None,
    prepared: list | None = None,
) -> dict:
    """Embed a list of image paths with one backbone. Unreadable pages become rows of nan.

    `prepared` accepts the output of `prepare` so a caller running several backbones over the
    same input mode pays for the decoding once. The reported rate then describes the GPU rather
    than the disk, and both numbers are worth having separately.
    """
    import torch

    model, width, statistics = load(name)
    paths = list(paths)
    out = np.full((len(paths), width), np.nan, dtype=np.float32)

    started = time.perf_counter()
    loaded_all = prepare(paths, mode, n_jobs) if prepared is None else list(prepared)
    if len(loaded_all) != len(paths):
        raise ValueError(
            f"prepared has {len(loaded_all)} entries for {len(paths)} paths; they must align"
        )
    decoded = time.perf_counter() - started

    for start in range(0, len(paths), batch_size):
        loaded = loaded_all[start : start + batch_size]
        arrays = [array for array in loaded if array is not None]
        keep = [start + offset for offset, array in enumerate(loaded) if array is not None]
        if not arrays:
            continue
        tensor = batch_tensor(arrays, statistics).to(device())
        with torch.no_grad():
            if name == "clip_vit_b32":
                vectors = model(pixel_values=tensor).image_embeds
            else:
                vectors = model(tensor)
        out[keep] = vectors.float().cpu().numpy()
    elapsed = time.perf_counter() - started

    readable = ~np.isnan(out).any(axis=1)
    forward = max(elapsed - decoded, 1e-9)
    return {
        "backbone": name,
        "input": mode,
        "device": device(),
        "dimension": width,
        "rows": len(paths),
        "readable": int(readable.sum()),
        "seconds": round(elapsed, 1),
        "decode_seconds": round(decoded, 1),
        "forward_seconds": round(forward, 1),
        "pages_per_second": round(len(paths) / elapsed, 1) if elapsed else 0.0,
        # The rate the device actually sustains, once the disk is out of the way. The gap
        # between this and the line above is the argument for `prepare` existing.
        "forward_pages_per_second": round(len(paths) / forward, 1),
        "embeddings": out,
    }


def probe(embeddings: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 42) -> dict:
    """Macro F1 of a logistic regression on the embedding, scored the way Phase 5 scores.

    Same estimator family, same fold count, same metric, so this number can be put next to
    5.1.1's 0.7898 without an argument about protocol. The scaler is refit inside every fold,
    exactly as 4.2.4 requires.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    usable = ~np.isnan(embeddings).any(axis=1)
    X, y = embeddings[usable], labels[usable]
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    predicted = cross_val_predict(
        pipeline,
        X,
        y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    return {
        "rows": int(len(y)),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_f1": round(float(f1_score(y, predicted, average="macro", zero_division=0)), 4),
    }


def sources_for(ids) -> np.ndarray:
    """The source dataset of each id, from 4.2.2's table."""
    import pandas as pd

    from src.classify.data import TABLE

    frame = pd.read_parquet(TABLE)
    lookup = dict(zip(frame["id"].tolist(), frame["source"].tolist(), strict=True))
    return np.array([lookup[identifier] for identifier in ids], dtype=object)


def provenance(embeddings: np.ndarray, labels: np.ndarray, sources: np.ndarray) -> dict:
    """Is the probe reading the diagram, or the camera that photographed it?

    This corpus cannot tell those apart on its own, and the crosstab says why: **hdBPMN is
    600 of 600 flowcharts and sketch2code is 600 of 600 wireframes**, so source determines class
    for 1,200 of the 1,340 rows. Any representation that identifies the *dataset* a page came
    from scores 0.896 accuracy without having looked at a single arrow, and an embedding of a
    photograph encodes paper, lighting, resolution and camera far more readily than 33
    hand-built geometric ratios do. 4.1.5 caught the same problem in one feature; here it could
    be the whole representation.

    Two numbers separate the explanations:

    `source_accuracy`   how well the embedding predicts the *source* dataset. Near 1.0 means
                        provenance is in there, plainly readable.
    `within_chaos`      macro F1 over the chaos corpus alone - 140 pages, three classes, one
                        source, one capture protocol. With provenance held constant this is
                        diagram-type discrimination and nothing else, and it is the only part
                        of this corpus that can ask the question.
    """
    result = {}
    usable = ~np.isnan(embeddings).any(axis=1)

    # A single-source corpus has no provenance to predict - the probe would be asked to
    # separate one class from itself. That is the `--corpus synthetic` case, not an error.
    if len(set(sources[usable].tolist())) > 1:
        scored = probe(embeddings[usable], sources[usable])
        result["source_accuracy"] = scored["accuracy"]
        result["source_macro_f1"] = scored["macro_f1"]
    result["label_determined_by_source"] = round(
        float(
            sum(
                max((labels[sources == name] == kind).sum() for kind in set(labels.tolist()))
                for name in set(sources.tolist())
            )
            / len(labels)
        ),
        4,
    )

    inside = usable & (sources == "chaos")
    if inside.sum() > 20 and len(set(labels[inside].tolist())) > 1:
        within = probe(embeddings[inside], labels[inside])
        result["within_chaos"] = {
            "rows": within["rows"],
            "classes": sorted({str(name) for name in labels[inside]}),
            "accuracy": within["accuracy"],
            "macro_f1": within["macro_f1"],
        }
    return result


def handcrafted_within_chaos(ids, labels) -> dict:
    """The same within-chaos probe on 4.2.2's features, so the two representations can be
    compared where provenance is held constant rather than only where it is not."""
    import pandas as pd

    from src.classify.data import TABLE, feature_columns
    from src.features.extractor import FEATURE_NAMES
    from src.features.scaling import feature_scaler

    frame = pd.read_parquet(TABLE).set_index("id")
    sources = sources_for(ids)
    inside = sources == "chaos"
    if inside.sum() <= 20:
        return {}
    names = feature_columns(list(FEATURE_NAMES))
    matrix = frame.loc[list(np.asarray(ids)[inside]), names].to_numpy(float)
    # The handcrafted table has holes by design (4.2.3), so it is imputed before the probe -
    # the embedding has none and needs no equivalent step.
    return probe(feature_scaler().fit_transform(matrix), labels[inside])


def corpus(limit: int | None = None, corpus_name: str = "real"):
    """The image paths, labels and ids of **exactly the rows in 4.2.2's handcrafted table**.

    Returns `(paths, labels, ids)`.

    Walking the images on disk is the obvious implementation and it is wrong. 4.2.2 built its
    table from a stratified sample and the manifest holds more pages than it kept: on this
    machine the disk offers **1,695 real photographs (764 flowcharts, 791 wireframes)** where
    the feature table holds **1,340 (600/600/50/50/40)**. A probe scored on the larger set is
    scored on a different, easier problem, and putting that number beside 5.1.1's 0.7898 would
    be a comparison between two corpora dressed up as a comparison between two representations.

    So the table is the authority: its ids select the rows, in its own order. That is also what
    makes 6.1.2's cache joinable and 6.1.4's concatenation row-for-row honest.

    A `limit` still breaks comparability - the sample is stratified per type, which flattens
    600/600/50/50/40 into roughly equal classes - so it is a debugging convenience only, and
    `run` records which kind of corpus produced its numbers.
    """
    import pandas as pd

    from src.classify.data import TABLE
    from src.features.build import collect

    if not Path(TABLE).is_file():
        raise FileNotFoundError(f"no feature table at {TABLE}; run python -m src.features.build")
    frame = pd.read_parquet(TABLE)
    if corpus_name == "real":
        frame = frame[~frame["synthetic"].astype(bool)]
    elif corpus_name == "synthetic":
        frame = frame[frame["synthetic"].astype(bool)]
    frame = frame.reset_index(drop=True)

    available = {row["id"]: row for row in collect(None)}
    missing = [identifier for identifier in frame["id"] if identifier not in available]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(frame)} pages in the feature table are not on disk "
            f"(first: {missing[0]!r})"
        )

    rows = [available[identifier] for identifier in frame["id"]]
    if limit is not None and limit < len(rows):
        types = sorted({row["diagram_type"] for row in rows})
        per_type = max(1, limit // max(1, len(types)))
        chosen: list[dict] = []
        for kind in types:
            subset = [row for row in rows if row["diagram_type"] == kind]
            step = max(1, len(subset) // per_type)
            chosen += subset[::step][:per_type]
        rows = chosen
    return (
        [row["path"] for row in rows],
        np.array([row["diagram_type"] for row in rows], dtype=object),
        np.array([row["id"] for row in rows], dtype=object),
    )


def run(
    limit: int | None = None,
    backbones=BACKBONES,
    inputs=INPUTS,
    batch_size: int = 64,
    n_jobs: int | None = None,
) -> dict:
    paths, labels, ids = corpus(limit)
    sources = sources_for(ids)
    results = []
    decode_seconds = {}
    # Outer loop over the input mode, not the backbone: the decoded arrays depend only on the
    # mode, so this decodes the corpus twice instead of six times.
    for mode in inputs:
        started = time.perf_counter()
        decoded = prepare(paths, mode, n_jobs)
        # Recorded here because `embed` cannot see it: handed `prepared`, it reports a decode
        # cost of zero, which is true of the call and false of the work.
        decode_seconds[mode] = round(time.perf_counter() - started, 1)
        for name in backbones:
            embedded = embed(paths, name, mode, batch_size, n_jobs, prepared=decoded)
            scored = probe(embedded["embeddings"], labels)
            results.append(
                {
                    **{k: v for k, v in embedded.items() if k != "embeddings"},
                    **{f"probe_{k}": v for k, v in scored.items()},
                    **provenance(embedded["embeddings"], labels, sources),
                }
            )

    best = max(results, key=lambda row: row["probe_macro_f1"])
    counts = {str(name): int((labels == name).sum()) for name in sorted(set(labels.tolist()))}
    # 5.1.1's 0.7898 is on the full imbalanced real corpus. A subsample is stratified per type,
    # which rebalances the classes and makes macro F1 mean something else, so the flag travels
    # with the numbers rather than being left to whoever reads them.
    comparable = limit is None and min(counts.values()) != max(counts.values())
    return {
        "rows": len(paths),
        "comparable_with_5_1_1": comparable,
        "classes": counts,
        "decode_seconds": decode_seconds,
        "device": device(),
        "workers": os.cpu_count(),
        "results": results,
        "best": {
            "backbone": best["backbone"],
            "input": best["input"],
            "macro_f1": best["probe_macro_f1"],
        },
        "handcrafted_reference": {
            "task": "5.1.1",
            "macro_f1": 0.7898,
            "within_chaos": handcrafted_within_chaos(ids, labels),
        },
    }


#: Selected by the sweep below, and read by 6.1.2 so the cache is built from one decision.
BEST_BACKBONE = "clip_vit_b32"
BEST_INPUT = "gray"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES))
    ap.add_argument("--inputs", nargs="*", default=list(INPUTS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        result = run(
            args.limit, tuple(args.backbones), tuple(args.inputs), args.batch_size, args.jobs
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

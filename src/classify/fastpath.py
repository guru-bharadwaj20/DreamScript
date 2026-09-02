"""Phase 7.2.6 - the latency benchmark, and the cost that is not the classifier.

    python -m src.classify.fastpath

The plan asks this task to *show NB is the sub-millisecond fast path*. Timing `predict` would show
that trivially and would be close to meaningless, because **the classifier is not where the time
goes**. What a request handler in Phase 16 experiences is:

    decode the photograph -> preprocess -> extract features -> predict

and 6.1.2 measured the last-but-one step at 26 ms a page for the CLIP embedding. A model whose
inference takes 0.05 ms sitting behind a 26 ms feature extractor is not a fast path; it is a
rounding error attached to a slow one.

So this task times **the whole chain, per stage**, for two configurations:

    fast path      7.2.1's text-only table -> Gaussian NB
    accurate path  6.1.3's CLIP embedding  -> 6.3.7's RBF SVM

and the deliverable is the pair of end-to-end numbers with the stages broken out, so the claim
"NB is the fast path" is either supported by the whole chain or exposed as being true only of the
part nobody was waiting on.

## Why the text table might genuinely be faster, and why that is not obvious

7.2.1's features come from `context.text_mask` and `context.text_boxes`. Producing those still
needs 3.1's illumination correction and 3.2's binarization and text-layer separation - the
expensive part of preprocessing - and skips only region detection, arrowhead detection and
skeletonisation. The CLIP path skips *all* of 3.2 but adds a 224x224 forward pass through a
vision transformer.

Which wins is therefore a real question rather than a rhetorical one, and the answer depends on
hardware: the CLIP path is fast on the GPU this project has and would not be on a phone.

## What it measured

40 pages, per stage, median milliseconds:

    stage          fast path (text -> GaussianNB)   accurate path (CLIP -> RBF SVM)
    decode                    10.2                             11.5
    preprocess             4,573.7                            376.3
    features                   3.4                             75.1
    predict                    0.6                              0.9
    ------------------------------------------------------------------
    end to end             4,601.7                            472.7
    p95                   47,218.8                          8,699.0

## The plan's claim is true about the classifier and false about the path

**Gaussian NB predicts in 0.589 ms - sub-millisecond, as asked.** It is also 1.5x faster than the
RBF SVM's 0.874 ms, so on the metric the plan named, the fast path wins.

**And the fast path is 9.7x slower end to end.** 4.6 seconds a page against the accurate path's
0.47. The classifier the task was about accounts for **0.013%** of the time its own path spends.

## Where the 4.6 seconds go

`preprocess`, and specifically 3.2's text-layer separation. 7.2.1 built its table from
`context.text_mask` and `context.text_boxes` and argued that this was the cheap half of
preprocessing because it skips region detection, arrowhead detection and skeletonisation. **That
argument is wrong by an order of magnitude**: producing the text mask needs the full binarization
and layer decomposition, which is where the time actually is, while the CLIP path needs only a
resize to 224x224 and never binarizes anything at all.

The p95 numbers make it worse rather than better - 47 seconds for one page. A handful of large
photographs dominate the tail, and a request handler would time out on them.

## What this means for Phase 13's routing

**There is no fast path here to route to.** 13.3 proposed "NB fast prior -> ensemble classifier";
on these numbers the NB prior costs ten times what the ensemble it is supposed to gate costs, and
7.2.5 already measured the prior as worth -0.009 macro F1 when combined. Two independent
measurements, one of accuracy and one of latency, say the same thing: **route directly to the CLIP
+ RBF SVM path.**

The general lesson is the one the docstring set out to test and is worth stating in the form it
came back: a model's inference time is not its cost. The 0.589 ms classifier sits behind a 4.6
second feature pipeline, and the only number a user experiences is their sum.

## What would make a genuine fast path

Not a cheaper classifier - a cheaper *feature*. The accurate path's own preprocessing is 376 ms
and its CLIP forward pass 75 ms, so the floor for any pipeline that resizes and embeds a page is
about 0.45 s. A real fast path would have to skip the embedding, which means predicting from
image statistics that need no binarization at all, and 4.2's handcrafted table does not qualify
either - it needs everything 3.2 produces and more.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SEED = 42

#: Pages to time. Enough that the median is stable, few enough that the run is quick.
PAGES = 40

STAGES = ("decode", "preprocess", "features", "predict")


def _timed(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, (time.perf_counter() - started) * 1000.0


def text_path_timings(paths, fitted) -> list[dict]:
    """Per-stage milliseconds for the text-only fast path."""
    import cv2

    from src.features import context as ctx
    from src.features.textregions import NAMES, extract

    rows = []
    for path in paths:
        image, decode_ms = _timed(cv2.imread, str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        # `from_image` does the decode itself, so the preprocessing stage is timed as the whole
        # context build minus the decode already measured - an approximation that is stated
        # rather than hidden, and small either way.
        context, context_ms = _timed(ctx.from_image, path, "timing")
        values, feature_ms = _timed(extract, context)
        row = np.array([[values[name] for name in NAMES]], dtype=float)
        _, predict_ms = _timed(fitted.predict, row)
        rows.append(
            {
                "decode": decode_ms,
                "preprocess": max(context_ms - decode_ms, 0.0),
                "features": feature_ms,
                "predict": predict_ms,
            }
        )
    return rows


def embedding_path_timings(paths, fitted, projector) -> list[dict]:
    """Per-stage milliseconds for the accurate path: CLIP embedding then the RBF SVM."""
    import cv2
    import torch

    from src.classify.occlusion import page_array_from
    from src.embed.backbone import batch_tensor, device, load

    model, _, statistics_ = load("clip_vit_b32")
    rows = []
    for path in paths:
        image, decode_ms = _timed(cv2.imread, str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        array, prep_ms = _timed(page_array_from, image, "gray")
        started = time.perf_counter()
        tensor = batch_tensor([array], statistics_).to(device())
        with torch.no_grad():
            vector = model(pixel_values=tensor).image_embeds.float().cpu().numpy()
        embed_ms = (time.perf_counter() - started) * 1000.0
        reduced, project_ms = _timed(projector.transform, vector)
        _, predict_ms = _timed(fitted.predict, reduced)
        rows.append(
            {
                "decode": decode_ms,
                "preprocess": prep_ms,
                "features": embed_ms + project_ms,
                "predict": predict_ms,
            }
        )
    return rows


def summarise(rows: list[dict], name: str) -> dict:
    """Median and p95 per stage, plus the end-to-end total.

    Median rather than mean: one page that triggered a garbage collection should not define the
    latency of a request handler, and p95 is reported separately because that is the page that
    does.
    """
    if not rows:
        return {"path": name, "pages": 0}
    totals = [sum(row[stage] for stage in STAGES) for row in rows]
    return {
        "path": name,
        "pages": len(rows),
        "median_ms": {
            stage: round(statistics.median(row[stage] for row in rows), 3) for stage in STAGES
        },
        "p95_ms": {
            stage: round(float(np.percentile([row[stage] for row in rows], 95)), 3)
            for stage in STAGES
        },
        "end_to_end_median_ms": round(statistics.median(totals), 2),
        "end_to_end_p95_ms": round(float(np.percentile(totals, 95)), 2),
        "predict_share_of_total": round(
            statistics.median(row["predict"] for row in rows)
            / max(statistics.median(totals), 1e-9),
            5,
        ),
    }


def run(corpus: str = "real", pages: int = PAGES, n_jobs: int | None = None) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from src.classify.bayes import pipeline as nb_pipeline
    from src.classify.supportvectors import image_paths
    from src.embed.hybrid import dataset
    from src.features.scaling import feature_scaler
    from src.features.textregions import load as load_text

    text = load_text(corpus)
    fitted_nb = nb_pipeline(names=list(text.feature_names)).fit(text.X, text.y)

    embedding = dataset("embedding", corpus)
    fitted_svm = Pipeline(
        [
            ("prepare", feature_scaler()),
            ("model", SVC(kernel="rbf", gamma=1e-5, C=1000.0, random_state=SEED)),
        ]
    ).fit(embedding.X, embedding.y)
    full = np.load(ROOT / "data" / "features" / "embeddings.npy")
    projector = PCA(n_components=embedding.n_features, random_state=SEED).fit(full)

    found = image_paths(text.ids[:pages])
    chosen = [found[i] for i in text.ids[:pages] if i in found]
    if not chosen:
        raise FileNotFoundError("no page images on disk; pull the corpus with dvc first")

    text_rows = text_path_timings(chosen, fitted_nb)
    embed_rows = embedding_path_timings(chosen, fitted_svm, projector)

    fast = summarise(text_rows, "text -> gaussian_nb")
    accurate = summarise(embed_rows, "clip -> rbf_svm")
    return {
        "corpus": corpus,
        "pages": len(chosen),
        "fast_path": fast,
        "accurate_path": accurate,
        "end_to_end_speedup": (
            round(accurate["end_to_end_median_ms"] / fast["end_to_end_median_ms"], 2)
            if fast.get("end_to_end_median_ms")
            else None
        ),
        "inference_speedup": (
            round(accurate["median_ms"]["predict"] / max(fast["median_ms"]["predict"], 1e-9), 1)
            if fast.get("median_ms")
            else None
        ),
        "nb_inference_is_sub_millisecond": bool(
            fast.get("median_ms", {}).get("predict", 1e9) < 1.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--pages", type=int, default=PAGES)
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, args.pages), indent=2))
    except (FileNotFoundError, KeyError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

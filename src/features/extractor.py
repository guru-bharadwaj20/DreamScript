"""Phase 4.2.1 - `FeatureExtractor`: the nine families behind one sklearn transformer.

    from src.features.extractor import FeatureExtractor

    pipeline = Pipeline([("features", FeatureExtractor()), ("model", DecisionTreeClassifier())])
    pipeline.fit(train_paths, train_labels)

`transform` takes image paths and returns an `(n_images, 34)` float array in a fixed column
order. That is the whole interface, and it is the interface every model from Phase 5 to Phase 8
consumes, so three properties matter more than anything clever:

**One page is opened once.** The families all read a `PageContext`, and building one is where the
seconds go - 3.1's whole chain, then the split, then the regions. Extracting nine families from
one context costs a few milliseconds on top. A transformer that called nine `extract` functions
each of which loaded the image would be nine times slower for the same numbers.

**The column order is frozen by construction.** `FEATURE_NAMES` is built by walking `FAMILIES` in
order and taking each family's own `NAMES`, so a family cannot silently reorder its outputs, and
a column index means the same thing in a model trained today and one loaded next month.
`get_feature_names_out` returns exactly that list, which is what makes the importances in 4.2.6
and the pruning in 4.2.5 refer to features by name rather than by position.

**`transform` is stateless.** There are no fitted parameters here - nothing is learned from the
data, so nothing can leak from test into train. `fit` exists to satisfy the sklearn contract and
to record `n_features_out_`; scaling and imputation, which *do* learn, are separate transformers
in 4.2.3 and 4.2.4 precisely so that the leakage question has one place to be answered.

## Parallelism

`n_jobs` defaults to every core. Extraction is per-page and independent, so 3's `pmap` spreads
it across processes; the worker is handed a *path* and returns 34 floats, so there is nothing
expensive to pickle in either direction. Results come back in submission order, never completion
order, so a parallel `transform` and a serial one are identical arrays - pinned by a test,
because a feature table whose row order depends on scheduling would be an unreproducible corpus.

Measured over 300 synthetic pages on 32 cores: **95.5 s serial, 20.3 s parallel - 4.7x**, with
the two arrays identical. Threads were tried first, on the reasoning that OpenCV releases the
GIL, and reached only **2.3x**: a meaningful share of a small page is spent in pure Python -
region building, the containment walk, the per-family dictionaries - and that part does not
release anything. The speed-up is well short of 32 either way because process start-up is a
fixed few seconds; on the 6,000-page build in 4.2.2 the same pool amortises much better.

## A page that fails does not fail the batch

An unreadable file, or a page where every family returns `nan`, produces a row of `nan` rather
than an exception. 3.3.3 found photographs where 84% of the "ink" was the desk; those pages
exist in the corpus and a transformer that raises on them cannot build a feature table at all.
4.2.3 is where the `nan` becomes an imputed value plus a missingness indicator, which keeps
"this page was hard" as information rather than dropping it.

    python -m src.features.extractor <image> [<image> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.features import (
    connectivity,
    containment,
    direction,
    geometry,
    layout,
    ratios,
    shapes,
    structural,
    textstats,
)
from src.features.context import PageContext
from src.features.context import from_image as build_context

#: The nine families of 4.1, in the order their columns appear. Adding a family here is the
#: only edit needed to widen the feature vector.
FAMILIES = (
    structural,
    ratios,
    shapes,
    layout,
    geometry,
    textstats,
    connectivity,
    direction,
    containment,
)

#: Every feature name, in column order. Frozen by construction rather than by hand.
FEATURE_NAMES: tuple[str, ...] = tuple(name for family in FAMILIES for name in family.NAMES)


def from_context(context: PageContext) -> dict[str, float]:
    """Every family's features for one already-built page context."""
    values: dict[str, float] = {}
    for family in FAMILIES:
        values.update(family.extract(context))
    return values


def from_image(path: str | Path) -> dict[str, float]:
    """Every family's features for one image, opening and preprocessing it once."""
    return from_context(build_context(Path(path)))


def _row(path: str | Path) -> np.ndarray:
    try:
        values = from_image(path)
    except (OSError, ValueError, cv2.error):
        # A page that cannot be read is a row of nan, not a dead batch. See the docstring.
        return np.full(len(FEATURE_NAMES), np.nan)
    return np.array([values.get(name, np.nan) for name in FEATURE_NAMES], float)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Images in, a fixed-width float matrix out. Stateless, parallel, order-preserving.

    Parameters
    ----------
    n_jobs
        Workers for the parallel map. `None` means every core; 1 forces serial.
    """

    def __init__(self, n_jobs: int | None = None):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):  # sklearn's parameter name
        """Nothing is learned. Present for the sklearn contract, and it records the width."""
        self.n_features_out_ = len(FEATURE_NAMES)
        self.feature_names_out_ = list(FEATURE_NAMES)
        return self

    def transform(self, X) -> np.ndarray:
        from src.utils.parallel import pmap

        paths = list(X)
        if not paths:
            return np.zeros((0, len(FEATURE_NAMES)))
        jobs = self.n_jobs if self.n_jobs is not None else os.cpu_count()
        rows = pmap(_row, paths, n_jobs=jobs, prefer="processes")
        return np.vstack(rows)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)

    def __sklearn_tags__(self):
        # The input is a list of paths, not a numeric array, so sklearn's array validation
        # does not apply and would reject a perfectly valid call; and `nan` is a legitimate
        # output here rather than a fault - see the docstring.
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.requires_fit = False
        tags.input_tags.allow_nan = True
        return tags


def to_frame(paths, n_jobs: int | None = None):
    """The same matrix as a DataFrame with named columns. What 4.2.2 writes to parquet."""
    import pandas as pd

    matrix = FeatureExtractor(n_jobs=n_jobs).fit_transform(list(paths))
    return pd.DataFrame(matrix, columns=list(FEATURE_NAMES))


def benchmark(limit: int = 200, n_jobs: int | None = None) -> dict:
    """Serial against parallel: the speed-up, and that both give the identical array."""
    import time

    from src.utils.config import ROOT

    items = structural._synthetic_pages(limit)
    paths = [ROOT / item["file"] for item in items if (ROOT / item["file"]).is_file()]
    if not paths:
        return {"pages": 0}

    started = time.perf_counter()
    serial = FeatureExtractor(n_jobs=1).fit_transform(paths)
    serial_seconds = time.perf_counter() - started

    started = time.perf_counter()
    parallel = FeatureExtractor(n_jobs=n_jobs).fit_transform(paths)
    parallel_seconds = time.perf_counter() - started

    return {
        "pages": len(paths),
        "features": len(FEATURE_NAMES),
        "workers": n_jobs or os.cpu_count(),
        "serial_seconds": round(serial_seconds, 1),
        "parallel_seconds": round(parallel_seconds, 1),
        "speedup": round(serial_seconds / parallel_seconds, 1) if parallel_seconds else 0.0,
        "identical": bool(np.array_equal(serial, parallel, equal_nan=True)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=int, metavar="PAGES", help="serial vs parallel timing")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker count")
    ap.add_argument("images", nargs="*", type=Path)
    args = ap.parse_args(argv)

    if args.benchmark:
        result = benchmark(args.benchmark, args.jobs)
        if not result["pages"]:
            print("no synthetic corpus; run the 1.3.7 generator first", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0 if result["identical"] else 1

    if not args.images:
        print(json.dumps({"features": len(FEATURE_NAMES), "names": list(FEATURE_NAMES)}, indent=2))
        return 0

    matrix = FeatureExtractor(n_jobs=args.jobs).fit_transform(args.images)
    for path, row in zip(args.images, matrix, strict=True):
        print(path.name)
        print(
            json.dumps(dict(zip(FEATURE_NAMES, [round(v, 4) for v in row], strict=True)), indent=2)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

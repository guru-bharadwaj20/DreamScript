"""Phase 6.1.3 - PCA to 128 dimensions, and what the discarded directions were worth.

    python -m src.embed.reduce          # retained variance, and the probe at each width

The backbone returns 512 numbers per page and the real corpus has 1,340 rows. That ratio is the
problem this task exists for: **a linear model on 512 features and 1,072 training rows has
almost as many parameters per class as it has examples**, and every model in 6.2 and 6.3 pays
for the width again - an RBF kernel over 512 dimensions is a distance computation in a space
where distances concentrate, and 5.3.1 already showed what dimension does to a neighbourhood.

The plan asks for 128 and a retained-variance report. Both are provided, and so is the thing
that makes the report worth reading: **the probe score at every width**, because retained
variance measures how much of the *embedding* survives and not how much of the *label* does.
Those are different questions and they routinely have different answers - a direction can carry
1% of the variance and all of the class separation.

## The projection has to be fitted inside the fold

PCA is unsupervised, which makes it tempting to fit once on everything and cross-validate
afterwards. That is a leak: the components are a function of the test rows' positions even
though they never saw a label, and 4.2.4 measured the analogous scaler leak at 0.205 standard
deviations. Every score below refits the PCA on the training side of each fold. The stored
128-column table is fitted on everything, because it is an artefact for later phases rather than
a score - and the difference between those two uses is exactly what this docstring has to make
unambiguous.

## What it measured

On the same 1,340 rows Phase 5 scored, CLIP-ViT-B/32 greyscale embeddings from 6.1.2's cache:

    width    retained variance    probe macro F1    accuracy
     512     1.0000  (unreduced)      0.9713         0.9925
     256     0.9765                   0.9726         0.9925
     128     0.9154                   0.9623         0.9903
      64     0.8285                   0.9659         0.9903
      32     0.7267                   0.9624         0.9903
      16     0.6099                   0.9375         0.9828

    5.1.1, 33 handcrafted features     0.7898

**Retained variance is a bad guide to what the label needs, and this table is the proof.** 128
components keep 91.5% of the embedding and score 0.9623; 64 components keep 82.9% - eight and a
half points less of the matrix - and score **higher**, at 0.9659. The curve is not monotone in
width at all, and it could not be if variance were what mattered. A direction can carry a
percent of the variance and all of the class separation, which is exactly why the plan's
"retained-variance report" is reported here beside a probe score rather than instead of one.

**The plan asks for 128 and 128 is defensible, but it is not the best width and nothing here is
sensitive to the choice.** Everything from 32 to 512 lands between 0.962 and 0.973 - a spread of
0.011, smaller than the fold-to-fold spread Phase 5 measured for its own models. 256 scores
highest (0.9726, fractionally above the unreduced 0.9713, since dropping the last 2% of the
variance removes more noise than signal). 128 is kept as the stored artefact because it is what
the plan specifies, it costs 0.009 against unreduced, and it makes 6.1.4's hybrid table four
times narrower - which matters more there than 0.009 does here.

**The finding that outruns the task: 16 numbers beat 33.** A 16-dimensional projection retaining
only 61% of the embedding scores 0.9375 macro F1, against 0.7898 for the entire handcrafted
feature table. Nine feature families, four sub-phases of Phase 4, and a frozen backbone needs
sixteen principal components to beat all of it by 0.148. 6.1.4 is where that stops being a
remark and becomes a decision about what Phase 6's models are fitted on.

## A note on 0.9713 against 6.1.1's 0.9720

The same estimator on the same rows, differing in the seventh row of the corpus only because
6.1.1 embedded in batches of 64 and 6.1.2's cache used 256. Batched matrix multiplication on a
GPU is not associative in floating point, so the embeddings differ in the last bits and one page
falls the other way. It is worth stating rather than rounding away: **the difference is 0.0007
and it is arithmetic, not method.**
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.utils.config import ROOT

REDUCED = ROOT / "data" / "features" / "embeddings_pca128.npy"

#: The width the plan asks for, plus the neighbours that say whether it is the right one.
WIDTHS = (16, 32, 64, 128, 256)

TARGET = 128


def _relative(path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def variance_curve(embeddings: np.ndarray, widths=WIDTHS, seed: int = 42) -> dict:
    """Cumulative explained variance at each width, from one fit on everything.

    This is a description of the embedding matrix, not a score, so fitting it on all the rows
    is the correct thing to do here and would not be below.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    usable = ~np.isnan(embeddings).any(axis=1)
    scaled = StandardScaler().fit_transform(embeddings[usable])
    full = PCA(n_components=min(scaled.shape), random_state=seed).fit(scaled)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    return {
        "rows": int(usable.sum()),
        "input_dimension": int(embeddings.shape[1]),
        "retained": {
            str(width): round(float(cumulative[min(width, len(cumulative)) - 1]), 4)
            for width in widths
        },
        "components_for_90_percent": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_for_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
        "components_for_99_percent": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def probe_at(embeddings: np.ndarray, labels: np.ndarray, width: int | None, seed: int = 42):
    """Macro F1 of a logistic regression at one width, PCA refitted inside every fold.

    `width=None` is the unreduced embedding, so the table always carries its own baseline.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    usable = ~np.isnan(embeddings).any(axis=1)
    X, y = embeddings[usable], labels[usable]

    steps = [("scale", StandardScaler())]
    if width is not None:
        steps.append(("pca", PCA(n_components=width, random_state=seed)))
    steps.append(("model", LogisticRegression(max_iter=2000, random_state=seed)))

    started = time.perf_counter()
    predicted = cross_val_predict(
        Pipeline(steps),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    return {
        "width": width if width is not None else int(X.shape[1]),
        "reduced": width is not None,
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_f1": round(float(f1_score(y, predicted, average="macro", zero_division=0)), 4),
        "seconds": round(time.perf_counter() - started, 1),
    }


def project(embeddings: np.ndarray, width: int = TARGET, seed: int = 42):
    """The stored reduction: one PCA fitted on every usable row. An artefact, not a score."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    usable = ~np.isnan(embeddings).any(axis=1)
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("pca", PCA(n_components=width, random_state=seed))]
    ).fit(embeddings[usable])

    out = np.full((len(embeddings), width), np.nan, dtype=np.float32)
    out[usable] = pipeline.transform(embeddings[usable]).astype(np.float32)
    return out, pipeline


def write(reduced: np.ndarray, path: Path = REDUCED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, reduced)
    temporary.replace(path)
    return path


def run(
    widths=WIDTHS,
    target: int = TARGET,
    corpus: str = "real",
    store: bool = True,
    phase5_rows: bool = True,
) -> dict:
    from src.embed.cache import load

    embeddings, frame, header = load()
    if corpus == "real":
        keep = ~frame["synthetic"].astype(bool).to_numpy()
    elif corpus == "synthetic":
        keep = frame["synthetic"].astype(bool).to_numpy()
    else:
        keep = np.ones(len(frame), bool)

    if phase5_rows:
        # The cache holds every page on disk; 4.2.2's table holds the 1,340 real rows Phase 5
        # actually scored. Scoring the wider set here would make every number below
        # incomparable with 6.1.1's 0.9720 and 5.1.1's 0.7898 - the same trap 6.1.1's `corpus`
        # exists to avoid, and it changes the answer: 1,695 rows score 0.9602 where 1,340
        # score 0.9720, a gap that is entirely the corpus and nothing to do with PCA.
        import pandas as pd

        from src.classify.data import TABLE

        wanted = set(pd.read_parquet(TABLE)["id"].tolist())
        keep = keep & frame["id"].isin(wanted).to_numpy()

    subset = embeddings[keep]
    labels = frame["diagram_type"].to_numpy(dtype=object)[keep]

    curve = variance_curve(subset, widths)
    scores = [probe_at(subset, labels, None)] + [
        probe_at(subset, labels, width) for width in widths
    ]
    best = max(scores, key=lambda row: row["macro_f1"])

    result = {
        "backbone": header.get("backbone"),
        "input": header.get("input"),
        "corpus": corpus,
        "rows_scored": int(keep.sum()),
        "comparable_with_6_1_1": bool(phase5_rows and corpus == "real"),
        "variance": curve,
        "probe": scores,
        "best_width": best["width"],
        "target": target,
        "target_score": next((row["macro_f1"] for row in scores if row["width"] == target), None),
        "unreduced_score": scores[0]["macro_f1"],
    }
    if store:
        # The stored table covers the whole cache, both corpora, so 6.1.4 can join it to the
        # handcrafted table row for row.
        reduced, _ = project(embeddings, target)
        result["path"] = _relative(write(reduced))
        result["stored_shape"] = list(reduced.shape)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", nargs="*", type=int, default=list(WIDTHS))
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--no-store", action="store_true")
    ap.add_argument(
        "--all-rows",
        action="store_true",
        help="score every cached row rather than the 1,340 Phase 5 scored (not comparable)",
    )
    args = ap.parse_args(argv)

    try:
        result = run(
            tuple(args.widths),
            args.target,
            args.corpus,
            not args.no_store,
            not args.all_rows,
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

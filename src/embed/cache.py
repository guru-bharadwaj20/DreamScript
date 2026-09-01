"""Phase 6.1.2 - the embedding table, built once and joined by id.

    python -m src.embed.cache --build          # writes data/features/embeddings.npy + index
    from src.embed.cache import load, aligned

6.1.1 chose a backbone by measuring six of them; this builds its output for the whole corpus and
stores it beside 4.2.2's handcrafted table. Everything after this - 6.1.3's PCA, 6.1.4's hybrid
table, the MLP of 6.2 and the SVMs of 6.3 - reads this file rather than re-running a backbone,
which is the difference between a sweep that costs seconds and one that costs an hour.

## A matrix and an index, never a matrix alone

`embeddings.npy` is `(rows, dimension)` float32 and says nothing about which row is which page.
The index parquet beside it carries `id`, `source`, `diagram_type`, `split`, `scribe_id`,
`synthetic` **and `row`** - the position in the matrix - so a caller joins on `id` rather than
trusting that two files were written in the same order. 4.2.2 made the opposite choice for a
reason it stated (a labels file that will eventually be joined wrong), and the reason a matrix
cannot follow it is that `.npy` has nowhere to put a string column.

`aligned()` is the function that matters: given the handcrafted table, it returns the embedding
rows **in that table's order**, and raises rather than guessing if an id is missing. 6.1.4's
concatenation is only meaningful if the two tables are row-for-row the same pages.

## What is stored is a decision, and it is recorded

The header of the index records the backbone, the input mode and the embedding width, because an
embedding matrix whose provenance is not written down is a matrix nobody can reproduce. Rebuild
with a different backbone and the file says so.

## What it measured

    backbone      clip_vit_b32, greyscale       6.1.1's selection
    rows          6,695                         1,695 real photographs + 5,000 synthetic pages
    dimension     512                           float32
    size          13.7 MB
    build         174.7 s on a CUDA device      38.3 pages/s
    unreadable    0
    duplicate ids 0

**Every page in the corpus embedded, none of them lost.** The 38.3 pages a second is a decoding
rate and not a GPU one - 6.1.1 measured the forward pass at over a thousand pages a second on
the same device, and the three minutes here are almost entirely OpenCV reading JPEGs off a disk
and correcting their illumination.

That is the number that justifies this file existing. **Every later task now pays 13.7 MB and a
`np.load` instead of three minutes**, and there are at least six of them: 6.1.3's PCA, 6.1.4's
hybrid table, 6.2's architecture and optimizer sweeps, 6.3's kernel sweeps. The optimizer study
of 6.2.4 alone refits five optimizers across many epochs; had each of those runs re-embedded the
corpus, the sweep would have spent hours doing arithmetic it had already done.

The cache deliberately covers **more rows than Phase 5 scores**. 4.2.2's table holds 1,340 of
the 1,695 real pages, and the synthetic corpus is not in Phase 5 at all - but 6.1.4 joins by id
and takes what it needs, and a cache that has to be rebuilt the first time a task wants the
synthetic rows is a cache that failed at its one job. Building the superset costs 13.7 MB.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.embed.backbone import BEST_BACKBONE, BEST_INPUT, embed
from src.utils.config import ROOT

MATRIX = ROOT / "data" / "features" / "embeddings.npy"
INDEX = ROOT / "data" / "features" / "embeddings_index.parquet"
META = ROOT / "data" / "features" / "embeddings_meta.json"

#: Carried from 4.2.2 so a row here can be joined to a row there without a second lookup.
IDENTITY = ("id", "source", "diagram_type", "split", "scribe_id", "adverse", "synthetic")


def _relative(path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build(
    backbone: str = BEST_BACKBONE,
    mode: str = BEST_INPUT,
    limit: int | None = None,
    batch_size: int = 64,
    n_jobs: int | None = None,
) -> dict:
    """Embed the whole corpus - both the real photographs and 1.3.7's synthetic pages."""
    import pandas as pd

    from src.features.build import collect

    rows = collect(limit)
    if not rows:
        raise FileNotFoundError(
            "no images found; the manifest and the synthetic index must both be present"
        )

    started = time.perf_counter()
    result = embed([row["path"] for row in rows], backbone, mode, batch_size, n_jobs)
    elapsed = time.perf_counter() - started

    frame = pd.DataFrame({column: [row[column] for row in rows] for column in IDENTITY})
    frame.insert(0, "row", np.arange(len(rows)))
    return {
        "matrix": result["embeddings"],
        "index": frame,
        "meta": {
            "backbone": backbone,
            "input": mode,
            "dimension": result["dimension"],
            "rows": len(rows),
            "readable": result["readable"],
            "unreadable": len(rows) - result["readable"],
            "device": result["device"],
            "seconds": round(elapsed, 1),
            "pages_per_second": round(len(rows) / elapsed, 1) if elapsed else 0.0,
        },
    }


def write(built: dict, matrix: Path = MATRIX, index: Path = INDEX, meta: Path = META) -> dict:
    """Atomically, the way 4.2.2 writes its table: a kill leaves the previous cache intact."""
    matrix.parent.mkdir(parents=True, exist_ok=True)

    def save_matrix(path: Path) -> None:
        # Handed an open file rather than a name: `np.save` appends `.npy` to any path that
        # does not already end in it, so saving to `embeddings.npy.tmp` silently writes
        # `embeddings.npy.tmp.npy` and the atomic rename below then has nothing to rename.
        with open(path, "wb") as handle:
            np.save(handle, built["matrix"])

    # All three staged, then all three renamed. Writing and renaming them one at a time would
    # make each file atomic and the *set* of them not: a failure while writing the index would
    # leave a new matrix beside a stale index, which is the one corruption `load` cannot detect
    # when the lengths happen to match.
    plan = [
        (matrix, save_matrix),
        (index, lambda p: built["index"].to_parquet(p, index=False)),
        (meta, lambda p: p.write_text(json.dumps(built["meta"], indent=2), encoding="utf-8")),
    ]
    staged = []
    try:
        for path, save in plan:
            temporary = path.with_suffix(path.suffix + ".tmp")
            save(temporary)
            staged.append((temporary, path))
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    for temporary, path in staged:
        temporary.replace(path)
    return {
        "matrix": _relative(matrix),
        "index": _relative(index),
        "meta": _relative(meta),
        **built["meta"],
    }


def load(matrix: Path = MATRIX, index: Path = INDEX, meta: Path = META):
    """`(embeddings, index_frame, meta)`. Raises if the two files disagree about their length."""
    import pandas as pd

    if not Path(matrix).is_file():
        raise FileNotFoundError(
            f"no embedding cache at {matrix}; run python -m src.embed.cache --build"
        )
    embeddings = np.load(matrix)
    frame = pd.read_parquet(index)
    if len(frame) != len(embeddings):
        raise ValueError(
            f"embedding cache is inconsistent: {len(embeddings)} rows of matrix against "
            f"{len(frame)} rows of index"
        )
    header = json.loads(Path(meta).read_text(encoding="utf-8")) if Path(meta).is_file() else {}
    return embeddings, frame, header


def aligned(ids, matrix: Path = MATRIX, index: Path = INDEX, *, drop_missing: bool = False):
    """Embedding rows in the caller's id order - the only safe way to join these two tables.

    Returns `(embeddings, mask)`, where `mask` marks which of `ids` were found. Without
    `drop_missing` a missing id raises: silently returning a shorter matrix is how a feature
    table and a label vector end up off by one.
    """
    embeddings, frame, _ = load(matrix, index)
    position = dict(zip(frame["id"].tolist(), frame["row"].tolist(), strict=True))
    ids = list(ids)
    mask = np.array([identifier in position for identifier in ids])

    if not mask.all() and not drop_missing:
        missing = [identifier for identifier, found in zip(ids, mask, strict=True) if not found]
        raise KeyError(
            f"{len(missing)} of {len(ids)} ids are not in the embedding cache "
            f"(first: {missing[0]!r}); rebuild it or pass drop_missing=True"
        )
    order = [position[identifier] for identifier, found in zip(ids, mask, strict=True) if found]
    return embeddings[order], mask


def summary(embeddings: np.ndarray, frame, header: dict) -> dict:
    """What is in the cache, for a caller that wants to check before trusting it."""
    usable = ~np.isnan(embeddings).any(axis=1)
    return {
        **header,
        "rows": int(len(frame)),
        "dimension": int(embeddings.shape[1]),
        "usable_rows": int(usable.sum()),
        "megabytes": round(embeddings.nbytes / 1e6, 1),
        "by_corpus": {
            "real": int((~frame["synthetic"].astype(bool)).sum()),
            "synthetic": int(frame["synthetic"].astype(bool).sum()),
        },
        "by_type": {
            str(name): int(count)
            for name, count in frame["diagram_type"].value_counts().sort_index().items()
        },
        "duplicate_ids": int(len(frame) - frame["id"].nunique()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="rebuild the cache from the images")
    ap.add_argument("--backbone", default=BEST_BACKBONE)
    ap.add_argument("--input", dest="mode", default=BEST_INPUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        if args.build:
            built = build(args.backbone, args.mode, args.limit, args.batch_size, args.jobs)
            print(json.dumps(write(built), indent=2))
        else:
            print(json.dumps(summary(*load()), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

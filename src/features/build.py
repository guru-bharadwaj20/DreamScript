"""Phase 4.2.2 - build `data/features/handcrafted.parquet` for the corpus.

    python -m src.features.build --limit 6000        # a sample, all cores
    python -m src.features.build --all               # everything in the manifest

One row per image: the 34 columns from 4.2.1, plus the identity columns a model needs and must
never train on - `id`, `source`, `diagram_type`, `split`, `scribe_id`, `adverse`, `synthetic`.
Those are carried in the same table rather than joined back later, because a feature table whose
labels live in a second file is a table that will eventually be joined in the wrong order.

## What goes in it

Two corpora, marked by the `synthetic` column and always distinguishable:

* the **synthetic** pages of 1.3.7, which have ground-truth graphs and cover all five types
  evenly - 1,000 each;
* the **real** photographs of 1.1/1.2, which are the ones that matter and are wildly
  unbalanced: 1.3.4 measured 24,370 flowcharts against 40 circuits.

Both are written, with their splits preserved from the manifest, and Phase 5 decides what to
train on. Mixing them silently into one pool would make every number afterwards
uninterpretable; one table with a flag is what lets 14 ablate on it.

## Two sampling bugs, both found by building the table and reading the summary

Neither raised an error, and both would have quietly ruined Phase 5.

The first build asked for 3,000 rows per corpus and produced **88 real rows and no wireframes
at all**. An evenly-spaced slice of the manifest is a slice of DIDI and IAM, whose archives are
DVC-tracked and not present here, so almost every sampled path did not exist and was dropped
after sampling. The second failure was underneath it: even with the missing files removed, a
slice that ignores the label is a slice of the majority class, and the corpus is 68% flowcharts.
The same mistake had been made on the synthetic side, where 1.3.7 writes its pages grouped by
type - the first 3,000 of 5,000 are three types and none of the other two.

Both are fixed the same way: drop absent files *first*, then sample per diagram type. The
sampling is now stratified on both sides and the summary prints the class counts, so the next
version of this mistake is visible in the output rather than three phases downstream.

## The run

    rows              4,340   (3,000 synthetic, 1,340 real)
    by type           flowchart 1,200 / wireframe 1,200 / er 650 / state 650 / circuit 640
    by split          train 3,485 / validation 729 / test 126
    features             34
    workers              32
    wall clock        648 s
    rate               6.7 pages/s
    rows of all-nan       0
    missing cells      11.9%

6.7 pages a second against the 14.8 the 4.2.1 benchmark suggests, because that benchmark ran on
900px synthetic renders and half of this table is 4,000px photographs. The missing cells are
concentrated in four columns and are not damage: `layout_col_regularity` is absent on 51% of
pages because regularity is undefined below three rows of shapes, exactly as 4.1.4 documented.
**No row came back entirely `nan`**, so no page failed outright.

The table is written atomically - a temporary file moved into place - so an interrupted build
leaves the previous table intact rather than a half-written parquet that pandas will read
without complaining.

## Reproducibility

A row is a pure function of the image bytes and the tuned constants, exactly as in 3.2.9, so
the build is deterministic and re-running it overwrites with identical values. What is *not*
pinned here is the sample: `--limit` takes an evenly-spaced slice so all five types appear, and
the slice is a function of the manifest's order, which DVC pins. `--all` avoids the question.

    python -m src.features.build --limit 6000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.features.extractor import FEATURE_NAMES, FeatureExtractor
from src.utils.config import ROOT

OUT = ROOT / "data" / "features" / "handcrafted.parquet"

#: Columns that identify a row. Never features - a model that trains on `id` has memorised the
#: corpus, and one that trains on `source` has learned 4.1.5's leak on purpose.
#:
#: **Defined here because this module writes the table.** It was copy-pasted into four files -
#: here, `embed/cache.py`, `mlops/drift.py` and `pipeline/fallback.py` - each with its own
#: comment saying the same thing. Add a column to the manifest, update three of the four, and one
#: stage silently trains on a leak while the other two do not; nothing would have reported it,
#: because each copy is internally consistent. The other three import this.
IDENTITY = ("id", "source", "diagram_type", "split", "scribe_id", "adverse", "synthetic")


def _real_rows(limit: int | None) -> list[dict]:
    """Real photographs from the manifest, stratified by diagram type.

    Two corrections that a naive slice gets wrong, both found by building the table and looking
    at what came out. Files that are not on disk are dropped **before** sampling: most of the
    manifest's 35,674 rows are DIDI and IAM, whose archives are DVC-tracked and not pulled here,
    and an evenly-spaced slice over the full list returned 88 usable rows out of 3,000. And the
    sample is taken per diagram type rather than over the whole frame: 1.3.4 measured the corpus
    at 24,370 flowcharts against 40 circuits, so any slice that ignores the label is a slice of
    flowcharts, and the first build produced **no wireframes at all**.
    """
    import pandas as pd

    manifest = ROOT / "data" / "processed" / "manifest.parquet"
    if not manifest.is_file():
        return []
    frame = pd.read_parquet(manifest)
    frame = frame[frame["diagram_type"] != "text"]  # IAM lines are handwriting, not diagrams

    present = [r for r in frame.to_dict("records") if (ROOT / r["path"]).is_file()]
    if limit is not None:
        types = sorted({r["diagram_type"] for r in present})
        per_type = max(1, limit // max(1, len(types)))
        chosen: list[dict] = []
        for kind in types:
            subset = [r for r in present if r["diagram_type"] == kind]
            step = max(1, len(subset) // per_type)
            chosen += subset[::step][:per_type]
        present = chosen

    return [
        {
            "id": r["id"],
            "source": r["source"],
            "diagram_type": r["diagram_type"],
            "split": r["split"],
            "scribe_id": r.get("scribe_id"),
            "adverse": bool(r.get("adverse", False)),
            "synthetic": False,
            "path": ROOT / r["path"],
        }
        for r in present
    ]


def _synthetic_rows(limit: int | None) -> list[dict]:
    index = ROOT / "data" / "processed" / "synthetic" / "index.json"
    if not index.is_file():
        return []
    items = json.loads(index.read_text(encoding="utf-8"))
    if limit is not None and limit < len(items):
        # Per type, for the same reason `_real_rows` does it: 1.3.7 writes its 5,000 pages
        # grouped by diagram type, so an evenly-spaced slice of the first 3,000 is three types
        # and none of the other two. The first build made exactly that mistake.
        types = sorted({item["diagram_type"] for item in items})
        per_type = max(1, limit // max(1, len(types)))
        chosen: list[dict] = []
        for kind in types:
            subset = [item for item in items if item["diagram_type"] == kind]
            step = max(1, len(subset) // per_type)
            chosen += subset[::step][:per_type]
        items = chosen
    return [
        {
            "id": item["id"],
            "source": "synthetic",
            "diagram_type": item["diagram_type"],
            # 1.3.7's pages carry no split of their own; they are training material, and
            # saying so here is better than letting Phase 5 assume it.
            "split": "train",
            "scribe_id": None,
            "adverse": bool(item.get("augmentations")),
            "synthetic": True,
            "path": ROOT / item["file"],
        }
        for item in items
    ]


def collect(limit: int | None = None, *, real: bool = True, synthetic: bool = True) -> list[dict]:
    """The rows to build, real and synthetic, each capped at `limit` if one is given."""
    rows: list[dict] = []
    if synthetic:
        rows += _synthetic_rows(limit)
    if real:
        rows += _real_rows(limit)
    return [row for row in rows if row["path"].is_file()]


def build(rows: list[dict], n_jobs: int | None = None):
    """Extract every row's features and return the table as a DataFrame."""
    import pandas as pd

    if not rows:
        return pd.DataFrame(columns=[*IDENTITY, *FEATURE_NAMES])

    matrix = FeatureExtractor(n_jobs=n_jobs).fit_transform([row["path"] for row in rows])
    table = pd.DataFrame(matrix, columns=list(FEATURE_NAMES))
    for column in IDENTITY:
        table.insert(IDENTITY.index(column), column, [row[column] for row in rows])
    return table


def write(table, path: Path = OUT) -> Path:
    """Atomically: written beside the target and moved, so a kill leaves the old table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def summarise(table) -> dict:
    """What the table holds, and how much of it is missing."""
    features = table[list(FEATURE_NAMES)]
    return {
        "rows": int(len(table)),
        "features": len(FEATURE_NAMES),
        "by_type": {k: int(v) for k, v in table["diagram_type"].value_counts().items()},
        "by_split": {k: int(v) for k, v in table["split"].value_counts().items()},
        "synthetic_rows": int(table["synthetic"].sum()),
        "all_nan_rows": int(features.isna().all(axis=1).sum()),
        "missing_share": round(float(features.isna().to_numpy().mean()), 4),
        "worst_columns": {
            name: round(float(share), 3)
            for name, share in features.isna().mean().sort_values(ascending=False).head(5).items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1200, help="cap per corpus; --all for no cap")
    ap.add_argument("--all", action="store_true", help="every image in the manifest")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker count")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    rows = collect(
        None if args.all else args.limit,
        real=not args.no_real,
        synthetic=not args.no_synthetic,
    )
    if not rows:
        print("nothing to build; is the corpus present?", file=sys.stderr)
        return 1

    started = time.perf_counter()
    table = build(rows, args.jobs)
    elapsed = time.perf_counter() - started
    written = write(table, args.out)

    report = summarise(table)
    report["seconds"] = round(elapsed, 1)
    report["pages_per_second"] = round(len(rows) / elapsed, 1) if elapsed else 0.0
    report["workers"] = args.jobs
    report["path"] = str(written.relative_to(ROOT))
    print(json.dumps(report, indent=2))
    return 0 if report["rows"] and report["all_nan_rows"] < len(table) else 1


if __name__ == "__main__":
    sys.exit(main())

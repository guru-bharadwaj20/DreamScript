"""Phase 7.2.1 - a feature table built from the text layer alone, and what it costs to be cheap.

    python -m src.features.textregions --build     # writes data/features/textregions.parquet

7.2's Naive Bayes is the pipeline's fast path: the plan's 7.2.6 asks it to be sub-millisecond,
and 7.2.5 feeds its output into the downstream classifier and the HMM as a prior. A fast path is
only fast if its *features* are cheap, and 4.2.2's 33-column table is not - it needs 3.2's
primitive extraction, region detection, arrowhead detection and skeletonisation.

So this table is deliberately restricted to **statistics of the text layer only**. Everything
here comes from `context.text_mask` and `context.text_boxes`, which 3.2.8 produces before any
shape or connector analysis runs. Nothing here needs a region, a segment, an arrowhead or a
skeleton.

## The 14 columns, and the argument each one makes

    count / density
      text_n_blocks          how many labels. ER diagrams are label-heavy, circuits are not.
      text_blocks_per_area   the same, per unit page - a photograph of half a page is not a
                             diagram with half the labels.
      text_ink_share         share of *all* ink that is writing.
      text_coverage          share of the page area covered by text boxes.

    size and shape of the labels
      text_width_mean        mean block width in page units - one word against a sentence.
      text_width_std         do the labels vary, or are they uniform? A wireframe's field
                             labels are alike; a flowchart's process descriptions are not.
      text_height_mean       mean block height, which is roughly the pen size.
      text_aspect_mean       mean width/height. A long thin block is a sentence; a squarish one
                             is a single word or a stacked pair.

    layout of the labels
      text_x_std             horizontal spread of block centres, in page units.
      text_y_std             vertical spread.
      text_nn_distance       mean distance to the nearest other block - how clustered the
                             writing is. Wireframe labels sit in rows; ER attributes cluster
                             around their entity.
      text_row_alignment     share of blocks sharing a y-band with another block. Wireframes
                             and ER diagrams line their labels up; flowcharts scatter them.
      text_largest_share     the biggest block's share of all text area. A title dominates.
      text_edge_share        share of blocks whose centre is in the outer 15% of the page -
                             annotations and legends live at the margins.

**`text_inside_share` from 4.1.6 is deliberately absent**, even though it is the single strongest
text feature in Phase 4 (1.000 for flowcharts against 0.000 for circuits). It requires
`context.regions`, which means shape detection, which is exactly the cost this table exists to
avoid. Leaving out the best feature is the price of the fast path, and 7.2.6 is where that price
gets weighed against the latency it buys.

## What it measured

1,695 pages (1,340 real, 355 synthetic), 14 columns, built from the text layer alone.

**28 pages have no text blocks at all** and produce a defined count (0) with undefined shape and
layout statistics; `text_nn_distance` and `text_row_alignment` are additionally undefined for the
39 further pages holding exactly one block, for a total of 67 missing values in those two columns.
That is a property of the corpus, not a bug, and it is left as NaN rather than imputed here so
that 4.2.3's imputation policy stays the one place the decision is made.

Ranked by the ANOVA F against diagram type on the real corpus, the table is not flat:

    text_aspect_mean       1115.9
    text_n_blocks           419.7
    text_blocks_per_area    371.8
    text_row_alignment      305.3
    text_width_mean         296.0

`text_aspect_mean` separating three times harder than the count is the surprise. The plan's own
intuition for 7.2.3 is about *how many* labels a page has - many labels means ER - and the count
is only the second-best column here. What actually separates is the **shape of a label**: a
flowchart's process descriptions are long thin sentences, a wireframe's field captions are short
squarish words, and that difference is visible without knowing what any of them say.

The class means show the count intuition is real but not in the direction the plan guessed:

    class            n_blocks   row_alignment   ink_share   coverage   edge_share
    flowchart          37.6         0.857         0.249       0.090       0.389
    er_diagram         22.7         0.540         0.481       0.167       0.314
    circuit            22.5         0.701         0.180       0.064       0.322
    wireframe           9.5         0.400         0.227       0.037       0.451
    state_machine       8.4         0.369         0.136       0.024       0.278

**Flowcharts are the label-heavy class, not ER diagrams** - 37.6 blocks a page against 22.7 -
because every box in a flowchart carries a sentence while an ER diagram carries short attribute
names. ER wins on `text_ink_share` instead (0.481, nearly double any other class): its pages are
*mostly* writing even though they are not the pages with the most blocks. The plan's heuristic
"many labels => ER" is therefore wrong as stated on this corpus, and 7.2.3 has to learn the
correct version rather than encode the guessed one.

The two smallest classes are the two that look alike here: circuit and er_diagram sit within 0.2
blocks of each other on the headline column, and 7.2.2 is where that costs something.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.utils.config import ROOT

TABLE = ROOT / "data" / "features" / "textregions.parquet"

NAMES = (
    "text_n_blocks",
    "text_blocks_per_area",
    "text_ink_share",
    "text_coverage",
    "text_width_mean",
    "text_width_std",
    "text_height_mean",
    "text_aspect_mean",
    "text_x_std",
    "text_y_std",
    "text_nn_distance",
    "text_row_alignment",
    "text_largest_share",
    "text_edge_share",
)

#: Fraction of the page counted as margin for `text_edge_share`.
EDGE = 0.15

#: Two blocks are in the same row when their centres are within this fraction of the long side.
ROW_BAND = 0.02


def extract(context) -> dict[str, float]:
    """The 14 columns, from the text layer alone.

    A page with no text returns 0 for the two counts - "no writing" is a fact - and `nan` for
    every statistic *of* the blocks, because the mean width of no labels is not zero. 4.2.3's
    imputer handles the nans and its indicator columns record which pages had none, which is how
    4.1.6 already treats the same situation.
    """
    from src.features import textstats

    labels = textstats.blocks(context)
    long_side = context.long_side
    page_area = context.page_area
    ink = float(context.mask.sum())

    empty = {name: float("nan") for name in NAMES}
    empty["text_n_blocks"] = float(len(labels))
    empty["text_blocks_per_area"] = 0.0
    empty["text_ink_share"] = (float(context.text_mask.sum()) / ink) if ink else float("nan")
    empty["text_coverage"] = 0.0
    if not labels:
        return empty

    widths = np.array([w for _, _, w, _ in labels], float)
    heights = np.array([h for _, _, _, h in labels], float)
    areas = widths * heights
    centres = np.array([(x + w / 2, y + h / 2) for x, y, w, h in labels], float)

    # Normalised by the long side so a 4000px photograph and a 1000px render agree.
    w_norm = widths / long_side
    h_norm = heights / long_side
    c_norm = centres / long_side

    if len(labels) > 1:
        from scipy.spatial.distance import cdist

        distances = cdist(c_norm, c_norm)
        np.fill_diagonal(distances, np.inf)
        nearest = float(distances.min(axis=1).mean())
        band = ROW_BAND
        ys = c_norm[:, 1]
        aligned = float(np.mean([(np.abs(ys - y) < band).sum() > 1 for y in ys]))
    else:
        # One block has no nearest neighbour and cannot be aligned with anything. Reported as
        # nan rather than 0, which would claim the labels are maximally spread and unaligned.
        nearest = float("nan")
        aligned = float("nan")

    relative = centres / np.array([context.width, context.height], float)
    edge = float(
        np.mean(
            (relative[:, 0] < EDGE)
            | (relative[:, 0] > 1 - EDGE)
            | (relative[:, 1] < EDGE)
            | (relative[:, 1] > 1 - EDGE)
        )
    )

    return {
        "text_n_blocks": float(len(labels)),
        "text_blocks_per_area": len(labels) / (page_area / long_side**2),
        "text_ink_share": (float(context.text_mask.sum()) / ink) if ink else float("nan"),
        "text_coverage": float(areas.sum()) / page_area,
        "text_width_mean": float(w_norm.mean()),
        "text_width_std": float(w_norm.std()),
        "text_height_mean": float(h_norm.mean()),
        "text_aspect_mean": float(np.mean(widths / np.maximum(heights, 1.0))),
        "text_x_std": float(c_norm[:, 0].std()),
        "text_y_std": float(c_norm[:, 1].std()),
        "text_nn_distance": nearest,
        "text_row_alignment": aligned,
        "text_largest_share": float(areas.max() / areas.sum()) if areas.sum() else float("nan"),
        "text_edge_share": edge,
    }


def _row(item: dict) -> dict | None:
    from src.features import context as ctx

    path = ROOT / item["path"]
    if not path.is_file():
        return None
    try:
        context = ctx.from_image(path, item["id"])
    except (OSError, ValueError):
        return None
    return {"id": item["id"], "diagram_type": item["diagram_type"], **extract(context)}


def build(corpus: str = "real", limit: int | None = None, n_jobs: int | None = None) -> dict:
    """Extract the table for every page and write it beside 4.2.2's."""
    import os

    import pandas as pd

    from src.utils.parallel import pmap

    manifest = pd.read_parquet(ROOT / "data" / "processed" / "manifest.parquet")
    if corpus == "real":
        manifest = manifest[manifest["source"] != "synthetic"]
    elif corpus == "synthetic":
        manifest = manifest[manifest["source"] == "synthetic"]
    if limit:
        manifest = manifest.head(limit)

    items = manifest[["id", "path", "diagram_type"]].to_dict("records")
    rows = pmap(_row, items, n_jobs=n_jobs or os.cpu_count(), prefer="threads")
    good = [row for row in rows if row is not None]

    frame = pd.DataFrame(good)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(TABLE, index=False)
    return {
        "path": str(TABLE.relative_to(ROOT)),
        "rows": len(frame),
        "unreadable": len(rows) - len(good),
        "columns": list(NAMES),
        "missing_share": {name: round(float(frame[name].isna().mean()), 4) for name in NAMES},
    }


def load(corpus: str = "real"):
    """The text table as a Phase 5 `Dataset`, over the same rows 4.2.2's table covers."""
    import pandas as pd

    from src.classify.data import Dataset
    from src.classify.data import load as load_handcrafted

    if not TABLE.is_file():
        raise FileNotFoundError(
            f"no text-region table at {TABLE}; run python -m src.features.textregions --build"
        )
    base = load_handcrafted(corpus)
    frame = pd.read_parquet(TABLE).set_index("id")

    missing = [identifier for identifier in base.ids if identifier not in frame.index]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(base.ids)} ids are missing from the text table "
            f"(first: {missing[0]!r}); rebuild it"
        )
    matrix = frame.loc[list(base.ids), list(NAMES)].to_numpy(dtype=float)
    return Dataset(
        X=matrix,
        y=base.y,
        groups=base.groups,
        ids=base.ids,
        feature_names=list(NAMES),
        corpus=f"{corpus}:textregions",
        sources=base.sources,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    if args.build:
        print(json.dumps(build(args.corpus, args.limit, args.jobs), indent=2))
        return 0
    try:
        data = load(args.corpus)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(data.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

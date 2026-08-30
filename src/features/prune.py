"""Phase 4.2.5 - correlation pruning: drop one of every pair that says the same thing.

    keep, dropped = prune(matrix, FEATURE_NAMES)      # |r| > 0.95

Several families were built knowing they would overlap. 4.1.6 predicted `text_area_frac` would
duplicate 4.1.5's ink coverage; 4.1.3's five shape fractions sum to one, so the fifth is a
linear function of the other four; 4.1.7's component count is close to 4.1.1's node count on a
page where nothing connects. Rather than argue about which to keep, the table is measured and
one of each too-correlated pair is dropped.

## Which one of the pair goes

Pearson `|r|` above `THRESHOLD` makes a pair redundant. Given a pair, the survivor is chosen by
a rule stated once and applied without exception:

1. **the one that is missing less often** - a feature present on every page is worth more than
   an equivalent one that is `nan` on a fifth of them;
2. then **the earlier column**, which is the earlier family in 4.1's order.

Rule 1 is the one that matters and rule 2 is only a tie-break, chosen so the result does not
depend on dictionary ordering. The alternative - keeping whichever correlates less with the
rest - is a form of feature selection, and feature selection belongs in 4.2.6 with a target to
select against, not here where the label is not being looked at.

**Nothing about the label is used in this module.** Pruning by correlation between features is
safe to do on train and test together; pruning by correlation *with the target* is not, and is
deliberately absent.

## What was actually dropped

Measured on the 4,340-row table at |r| > 0.95, **one feature of 34 goes**:

    dropped                 because it duplicates    |r|    missing share
    layout_node_density     node_count              0.962   0.137 vs 0.000

and it is dropped rather than its partner because it is absent on 13.7% of pages where
`node_count` is never absent - rule 1 doing exactly the work it was written for. The duplication
is real and was avoidable in hindsight: 4.1.4 divides the node count by a normalised page area
that is almost the same number for every page in the corpus, so the "density" is the count.

**Every prediction the family docstrings made about redundancy was wrong**, and by a wide
margin:

    predicted duplicate pair                            actual |r|
    text_area_frac / global_ink_coverage (4.1.6)           0.046
    layout_row_regularity / layout_col_regularity (4.1.4)  0.030
    shape_frac_freeform / shape_frac_rectangle (4.1.3)     0.312
    conn_components / node_count (4.1.7)                   0.561

4.1.6 argued that `text_area_frac` was partly a measurement of how much drawing there is, and
that reasoning was sound - but "partly" turns out to be r = 0.046, which is nothing. The five
shape fractions sum to 1 by construction and *still* only reach 0.31 pairwise, because with five
terms no single pair is determined by the constraint. The lesson is the cheap one: **redundancy
is an empirical property of the corpus, not something to be reasoned out from the definitions**,
which is why this module measures instead of taking the docstrings' word.

The strongest surviving pairs are all below the threshold and all worth keeping an eye on in
4.2.6: `conn_mean_degree`/`conn_cycle_count` at 0.870 (the cyclomatic number is built from the
edge count), `contain_nested_share`/`contain_max_depth` at 0.844, and - the interesting one -
`arrowhead_count`/`global_ink_coverage` at 0.805, which says plainly that 3.2.6's arrowhead
detector is largely counting how much ink is on the page.

    python -m src.features.prune
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: Above this absolute Pearson correlation, two features are the same feature.
THRESHOLD = 0.95


def correlations(matrix: np.ndarray) -> np.ndarray:
    """Pairwise |r| over columns, computed pairwise-complete so `nan` does not erase a column."""
    data = np.asarray(matrix, float)
    columns = data.shape[1]
    out = np.zeros((columns, columns))
    for i in range(columns):
        for j in range(i + 1, columns):
            pair = data[:, [i, j]]
            usable = ~np.isnan(pair).any(axis=1)
            if usable.sum() < 3:
                continue
            a, b = pair[usable, 0], pair[usable, 1]
            if a.std() == 0 or b.std() == 0:
                # A constant column correlates with nothing; it is 4.2.6's job to notice that
                # it also predicts nothing.
                continue
            value = abs(float(np.corrcoef(a, b)[0, 1]))
            out[i, j] = out[j, i] = value
    return out


def prune(
    matrix: np.ndarray, names: list[str] | tuple[str, ...], threshold: float = THRESHOLD
) -> tuple[list[str], list[dict]]:
    """(kept names, dropped records). Deterministic, and independent of the label."""
    names = list(names)
    data = np.asarray(matrix, float)
    missing = np.isnan(data).mean(axis=0)
    matrix_r = correlations(data)

    kept = set(range(len(names)))
    dropped: list[dict] = []
    # Pairs in a fixed order - strongest correlation first, then column order - so the outcome
    # cannot depend on iteration order.
    pairs = [
        (matrix_r[i, j], i, j)
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if matrix_r[i, j] > threshold
    ]
    for value, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if i not in kept or j not in kept:
            continue
        loser = j if (missing[j], j) > (missing[i], i) else i
        winner = i if loser == j else j
        kept.discard(loser)
        dropped.append(
            {
                "dropped": names[loser],
                "kept": names[winner],
                "r": round(float(value), 4),
                "missing_share_dropped": round(float(missing[loser]), 4),
                "missing_share_kept": round(float(missing[winner]), 4),
            }
        )
    return [names[i] for i in sorted(kept)], dropped


def strongest_pairs(matrix: np.ndarray, names, limit: int = 10) -> list[dict]:
    """The most correlated pairs, whether or not they cross the threshold."""
    names = list(names)
    matrix_r = correlations(matrix)
    pairs = [
        {"a": names[i], "b": names[j], "r": round(float(matrix_r[i, j]), 4)}
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    return sorted(pairs, key=lambda p: -p["r"])[:limit]


def run(table=None, threshold: float = THRESHOLD) -> dict:
    import pandas as pd

    from src.features.build import OUT
    from src.features.extractor import FEATURE_NAMES

    if table is None:
        if not OUT.is_file():
            return {"rows": 0}
        table = pd.read_parquet(OUT)

    matrix = table[list(FEATURE_NAMES)].to_numpy(float)
    kept, dropped = prune(matrix, FEATURE_NAMES, threshold)
    return {
        "rows": int(len(table)),
        "threshold": threshold,
        "kept": kept,
        "kept_count": len(kept),
        "dropped": dropped,
        "strongest_pairs": strongest_pairs(matrix, FEATURE_NAMES),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args(argv)

    table = None
    if args.table:
        import pandas as pd

        table = pd.read_parquet(args.table)
    result = run(table, args.threshold)
    if not result["rows"]:
        print("no feature table; run python -m src.features.build first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

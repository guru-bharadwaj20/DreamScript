"""Phase 4.2.3 - missing values: median imputation, and a flag saying it happened.

    MedianImputer().fit(train_matrix).transform(any_matrix)

Two outputs per column that can go missing: the column with its `nan` replaced by the **training
median**, and a companion `*_missing` indicator that is 1 where the value was absent. The width
of the matrix grows from 34 to 34 + (however many columns were ever missing in training).

## Why the indicator, and not just the median

Phase 4's `nan` values are not measurement noise. Every one of them is a page where a question
did not apply, and 4.1.2 argued the case at length: `edges_per_node` is missing exactly when
there are no shapes, and a page with no shapes is a strong statement about diagram type - it is
usually a circuit. Filling the median and moving on replaces that statement with the population
average and tells the model the page was ordinary. The indicator keeps the fact.

This is measurable rather than a matter of taste. Over the 4,340-row table, **26 of the 34
columns go missing somewhere and 63.5% of rows have at least one hole** - so this is not a rare
repair, it is most of the corpus. And the holes are anything but random:

    diagram type     rows with any missing value
    circuit                    0.958
    wireframe                  0.801
    flowchart                  0.571
    er_diagram                 0.397
    state_machine              0.369

**A circuit page is missing something 96% of the time and a state machine 37%**, so the
indicator column alone carries about as much information about diagram type as several of the
features it accompanies. That is the argument for the indicator in one number: throwing it away
would discard a signal that separates the classes by a factor of 2.6.

Twenty-one columns share the same 13.7% missing rate, because they share one cause: no region
was found, so every per-node and per-shape question is unanswerable at once. Above them sit
`layout_col_regularity` at 51.4% and `layout_row_regularity` at 46.9%, undefined whenever there
are fewer than three rows of shapes, exactly as 4.1.4 said they would be.

## The median comes from training rows only

`fit` records one median per column, and it must see only training rows. A median computed over
train and test together carries information about the test distribution into every imputed
value - a small leak, but a real one, and the kind that makes a model look better than it is.
`fit` therefore takes the matrix it is given and nothing else; the caller passes the training
split, and 4.2.4's leakage test covers this transformer as well as the scaler.

A column that is missing in *every* training row has no median to take. That is recorded as
`nan` in `medians_` and filled with 0.0 at transform time, with the indicator carrying the
whole signal - which is the honest outcome: a column with no observed values anywhere is not a
measurement, it is an absence, and pretending to a central value would invent one.

## What it does not do

No iterative or model-based imputation - no `IterativeImputer`, no kNN fill. Those estimate a
missing value from the other columns, and here the other columns are missing for the same
reason: a page with no regions has *all* of the per-node ratios missing at once, so there is
nothing to estimate from. Median plus indicator is the honest floor, and it is what 5's
interpretable models can explain.

    python -m src.features.impute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#: Suffix of the companion column. Kept out of `FEATURE_NAMES` so 4.2.1's contract is unchanged.
SUFFIX = "_missing"


class MedianImputer(BaseEstimator, TransformerMixin):
    """Median fill plus a missingness indicator, fitted on training rows only.

    Parameters
    ----------
    add_indicator
        Emit the `*_missing` companion columns. On by default; the off path exists so that 14
        can ablate the indicator and measure what it was worth.
    """

    def __init__(self, add_indicator: bool = True):
        self.add_indicator = add_indicator

    def fit(self, X, y=None):  # noqa: N803
        matrix = np.asarray(X, float)
        if matrix.ndim != 2:
            raise ValueError("expected a 2-d feature matrix")
        self.n_features_in_ = matrix.shape[1]
        with np.errstate(invalid="ignore"):
            # nanmedian over an all-nan column warns and returns nan; that is the intended
            # answer here, so the warning is suppressed rather than the case avoided.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.medians_ = np.nanmedian(matrix, axis=0)
        # Only columns that were actually missing in training get an indicator. A column that
        # is always present does not need one, and 34 constant zero columns would be noise for
        # 4.2.5 to prune and for 4.2.6 to rank.
        self.missing_columns_ = np.flatnonzero(np.isnan(matrix).any(axis=0))
        return self

    def transform(self, X) -> np.ndarray:  # noqa: N803
        matrix = np.asarray(X, float).copy()
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {matrix.shape[1]}")

        missing = np.isnan(matrix)
        fill = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        matrix[missing] = np.take(fill, np.where(missing)[1])
        if not self.add_indicator or not len(self.missing_columns_):
            return matrix
        indicators = missing[:, self.missing_columns_].astype(float)
        return np.hstack([matrix, indicators])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        from src.features.extractor import FEATURE_NAMES

        names = list(input_features) if input_features is not None else list(FEATURE_NAMES)
        if not self.add_indicator:
            return np.asarray(names, dtype=object)
        return np.asarray(
            names + [f"{names[i]}{SUFFIX}" for i in self.missing_columns_], dtype=object
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def report(table=None) -> dict:
    """Which columns go missing, how often, and whether that tracks diagram type."""
    import pandas as pd

    from src.features.build import OUT
    from src.features.extractor import FEATURE_NAMES

    if table is None:
        if not OUT.is_file():
            return {"rows": 0}
        table = pd.read_parquet(OUT)

    features = table[list(FEATURE_NAMES)]
    share = features.isna().mean()
    missing_any = features.isna().any(axis=1)
    return {
        "rows": int(len(table)),
        "columns_ever_missing": int((share > 0).sum()),
        "rows_with_any_missing": round(float(missing_any.mean()), 4),
        "by_column": {
            name: round(float(value), 4) for name, value in share[share > 0].sort_values().items()
        },
        "rows_with_any_missing_by_type": {
            kind: round(float(missing_any[table["diagram_type"] == kind].mean()), 4)
            for kind in sorted(table["diagram_type"].unique())
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=None)
    args = ap.parse_args(argv)

    table = None
    if args.table:
        import pandas as pd

        table = pd.read_parquet(args.table)
    result = report(table)
    if not result["rows"]:
        print("no feature table; run python -m src.features.build first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

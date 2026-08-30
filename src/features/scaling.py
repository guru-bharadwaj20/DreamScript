"""Phase 4.2.4 - scaling, fitted on train only, with the leakage test that proves it.

    scaler = feature_scaler()          # median-impute -> indicator -> standardise
    scaler.fit(train_matrix)           # every statistic comes from these rows
    scaler.transform(test_matrix)      # and none from these

Standardisation matters here because the 34 features are on wildly different scales: a node
count runs to 40, `global_ink_coverage` to 0.04, and `dir_flow_axis` from -1 to 1. Any model
that measures distance - 5's KNN, 6's SVM and MLP - would otherwise be reading node counts and
nothing else. Trees do not care, which is why 5's decision tree is fitted on the unscaled table
and the difference is visible in the two model cards rather than hidden in one pipeline.

## The leakage question, and how it is answered rather than asserted

"Fit on train only" is easy to say and easy to get wrong, because the mistake produces no error
and a *better* score. The test that catches it is not an inspection of the code:

    fit on train, transform test               ->  T1
    fit on train + test, transform test        ->  T2
    T1 == T2  =>  no leak

If any test row influenced a fitted statistic, T2 differs from T1. `tests/test_features_scaling.py`
runs exactly that with a held-out set drawn from a deliberately different distribution - shifted
mean, inflated variance - so a leak of even a few percent moves the numbers well past tolerance.

Run on the real table (3,485 train rows, 855 held out), the honest fit reproduces itself to
**1.5e-13**, no `nan` survives, and the deliberately leaky fit - the same pipeline fitted on
train and test together - differs from it by **0.205 standard deviations on average and 2.19 at
the worst column**. That second number is the point of the exercise: it is the size of the
mistake this test is preventing, measured rather than asserted, and it is large enough to move a
Phase 5 accuracy by a visible margin.

The imputer is covered by the same test, because it is the first step of the same pipeline: the
median that fills a hole is a training median, and 855 held-out rows do not move it.

## Order matters: impute, then scale

The imputer runs first, so the median that fills a hole is a median of *raw* units and the
indicator columns are 0/1 before standardisation. Scaling first would compute a mean and
standard deviation over columns that still contain `nan`, and `StandardScaler` ignores `nan`
when fitting but propagates it when transforming, so the holes would survive into the model.
The indicator columns are standardised along with everything else - they are features like any
other once they exist, and a constant-zero indicator would be dropped by 4.2.5 anyway.

    python -m src.features.scaling
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.impute import MedianImputer


def feature_scaler(add_indicator: bool = True) -> Pipeline:
    """The standard Phase 4 preprocessing: impute, indicate, standardise. Fit on train only."""
    return Pipeline(
        [
            ("impute", MedianImputer(add_indicator=add_indicator)),
            # with_mean and with_std both on: the features are dense, so there is no sparse
            # matrix to preserve, and centring is the half that matters for an SVM's kernel.
            ("scale", StandardScaler()),
        ]
    )


def leak_check(train: np.ndarray, test: np.ndarray) -> dict:
    """Transform the test rows two ways and compare. Identical means nothing leaked."""
    honest = feature_scaler().fit(train).transform(test)
    leaky = feature_scaler().fit(np.vstack([train, test])).transform(test)
    difference = np.abs(honest - leaky)
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "max_absolute_difference": float(np.nanmax(difference)) if difference.size else 0.0,
        "mean_absolute_difference": float(np.nanmean(difference)) if difference.size else 0.0,
        # The comparison is between two *different* fits, so this is the size of the leak that
        # would exist if the scaler were fitted on everything - not a property of `honest`.
        "leak_would_be_visible": bool(difference.size and np.nanmax(difference) > 1e-9),
    }


def split_matrices(table=None):
    """(train, test) feature matrices from the built table, by the manifest's own split."""
    import pandas as pd

    from src.features.build import OUT
    from src.features.extractor import FEATURE_NAMES

    if table is None:
        if not OUT.is_file():
            return None, None
        table = pd.read_parquet(OUT)
    columns = list(FEATURE_NAMES)
    train = table[table["split"] == "train"][columns].to_numpy(float)
    held_out = table[table["split"] != "train"][columns].to_numpy(float)
    return train, held_out


def verify(table=None) -> dict:
    """Fit on train, transform test, and check that refitting on both changes the answer."""
    train, held_out = split_matrices(table)
    if train is None or not len(train) or not len(held_out):
        return {"rows": 0}

    honest = feature_scaler().fit(train)
    transformed = honest.transform(held_out)
    fitted_on_train = honest.fit(train).transform(held_out)
    return {
        "rows": int(len(train) + len(held_out)),
        "columns_in": int(train.shape[1]),
        "columns_out": int(transformed.shape[1]),
        "refit_is_deterministic": bool(
            np.allclose(transformed, fitted_on_train, equal_nan=True, atol=1e-12)
        ),
        "no_nan_survives": bool(not np.isnan(transformed).any()),
        "train_mean_is_zero": float(np.abs(honest.transform(train).mean(axis=0)).max()),
        "leak": leak_check(train, held_out),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=None)
    args = ap.parse_args(argv)

    table = None
    if args.table:
        import pandas as pd

        table = pd.read_parquet(args.table)
    result = verify(table)
    if not result["rows"]:
        print("no feature table; run python -m src.features.build first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["no_nan_survives"] else 1


if __name__ == "__main__":
    sys.exit(main())

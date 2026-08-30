"""Phase 4.2.6 - ranking the features: mutual information and the ANOVA F statistic.

    python -m src.features.importance          # ranks the built table, writes the chart

Two rankings, deliberately not one:

* **ANOVA F** asks whether a feature's *mean* differs between diagram types. It is fast, it is
  the classical answer, and it is blind to anything non-linear: a feature that is high for
  wireframes and circuits and low for the three in between has a group mean like everything
  else and scores badly.
* **Mutual information** asks how much knowing the feature reduces uncertainty about the type,
  in any shape at all. It catches the non-monotone cases F misses, and it is noisier.

Reporting both is the point. A feature high on MI and low on F is one the linear models in 5.1
cannot use and the trees can, and that difference is the argument for having both model
families - so this ranking is read again in 5.3, not just here.

Both are computed on the **training rows only**, and after 4.2.3's imputation, because both
estimators reject `nan`. The indicator columns are ranked alongside the features, which is the
honest way to find out whether "this page had no shapes" is worth more than the shape features
themselves.

## The ranking

3,485 training rows, 60 columns (34 features and 26 missingness indicators), five classes. Top
fifteen by mutual information, with each feature's rank under the F statistic beside it:

    feature                  MI      F rank    F
    global_aspect          1.284        8      534
    layout_node_density    1.210       49       76
    global_bbox_fill       1.048        3      901
    line_to_curve          0.858        1     2370
    edges_per_node         0.758       14      283
    layout_nn_distance     0.668       45       99
    shape_frac_rectangle   0.659        7      563
    conn_mean_degree       0.628       19      184
    text_label_length      0.618       40      130
    node_count             0.551       47       96
    shape_frac_diamond     0.547       41      128
    dir_axis_aligned       0.546       11      474
    text_per_node          0.504       53       52
    shape_frac_ellipse     0.463       13      379
    edge_count             0.456       20      168

**The top feature is the one 4.1.5 flagged as a leak.** `global_aspect` - the width over height
of the photograph, a property of the camera and not of the diagram - carries more mutual
information about diagram type than any geometric feature in the table. That was predicted, and
seeing it confirmed at rank 1 is the strongest argument yet that Phase 14's ablation is not
optional: a classifier trained on this table will be reading page shape first.

**Six of the top fifteen are invisible to the F statistic.** `layout_node_density` is 2nd by MI
and 49th by F; `node_count` 10th and 47th; `text_per_node` 13th and 53rd. These are the
non-monotone features - high for circuits *and* wireframes, low for the three in between - so
their class means sit on top of each other and F sees nothing. **A linear model cannot use
them and a tree can**, which is the concrete argument for running both families in Phase 5
rather than assuming the linear ones will do. `line_to_curve` is the opposite case and the only
feature that is top-five on both: it orders the types monotonically, exactly as 4.1.2 measured.

At the bottom, the three direction indicators carry essentially nothing (`dir_angle_entropy_missing`
scores **0.0000**), because they fire on 1.1% of pages. They are not dropped here - 4.2.5's
threshold is about redundancy, not about worth - and Phase 5's own selection is where they
should go. The best indicator, `layout_col_regularity_missing` at 0.284, ranks 25th of 60,
ahead of nine real features: "this page has fewer than three rows of shapes" is genuinely
informative, which is what 4.2.3 argued when it kept the indicators at all.

Two features that their own families warned about land where the warnings said they would:
`arrows_per_node` is 26th by MI and **59th of 60 by F**, and `conn_self_loops` - which 4.1.7
measured as an inverted signal - is 31st. Neither is worthless, and neither measures what its
name says.

## Reading it honestly

A high rank here is **not** evidence that a feature measures what its name says, and
`global_aspect` at rank 1 is the proof. A ranking cannot tell a signal from a leak; only 14's
ablation can, and this module's job is to say which features are worth ablating first.

    python -m src.features.importance --top 15
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p4_feature_importance.png"


def rank(matrix: np.ndarray, labels, names, random_state: int = 42) -> list[dict]:
    """Mutual information and ANOVA F for every column, as one ranked table."""
    from sklearn.feature_selection import f_classif, mutual_info_classif

    names = list(names)
    data = np.asarray(matrix, float)
    target = np.asarray(labels)

    mutual = mutual_info_classif(data, target, random_state=random_state)
    # The nearest-neighbour MI estimator returns a small positive number for a column that is
    # constant - 0.0043 measured on a column of ones - because it works from distances between
    # jittered points. A constant carries no information by definition, so it is set to zero
    # rather than left to rank above a real feature that happens to be weak.
    constant = data.std(axis=0) == 0
    mutual[constant] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # f_classif warns about the same constant columns; they are handled above.
        warnings.simplefilter("ignore", UserWarning)
        f_values, p_values = f_classif(data, target)
    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0)

    f_order = np.argsort(-f_values)
    f_rank = np.empty(len(names), int)
    f_rank[f_order] = np.arange(1, len(names) + 1)

    rows = [
        {
            "feature": names[i],
            "mutual_information": round(float(mutual[i]), 4),
            "f_statistic": round(float(f_values[i]), 2),
            "f_rank": int(f_rank[i]),
            "p_value": float(p_values[i]) if np.isfinite(p_values[i]) else 1.0,
        }
        for i in range(len(names))
    ]
    rows.sort(key=lambda row: -row["mutual_information"])
    for position, row in enumerate(rows, start=1):
        row["mi_rank"] = position
    return rows


def figure(rows: list[dict], top: int = 15, path: Path = FIGURE) -> Path:
    """A ranked bar chart of the top features by mutual information."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = rows[:top][::-1]
    names = [row["feature"] for row in best]
    values = [row["mutual_information"] for row in best]

    fig, ax = plt.subplots(figsize=(9, 0.36 * len(best) + 1.4))
    positions = np.arange(len(best))
    ax.barh(positions, values, color="#3b6ea5")
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("mutual information with diagram type (nats)")
    ax.set_title(f"Phase 4.2.6 - top {len(best)} handcrafted features")
    for position, row in zip(positions, best, strict=True):
        ax.text(
            row["mutual_information"],
            position,
            f"  F rank {row['f_rank']}",
            va="center",
            fontsize=8,
            color="#444444",
        )
    ax.margins(x=0.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(table=None, top: int = 15, write_figure: bool = True) -> dict:
    import pandas as pd

    from src.features.build import OUT
    from src.features.extractor import FEATURE_NAMES
    from src.features.impute import MedianImputer

    if table is None:
        if not OUT.is_file():
            return {"rows": 0}
        table = pd.read_parquet(OUT)

    train = table[table["split"] == "train"]
    if train.empty:
        train = table
    matrix = train[list(FEATURE_NAMES)].to_numpy(float)
    labels = train["diagram_type"].to_numpy()

    imputer = MedianImputer().fit(matrix)
    filled = imputer.transform(matrix)
    names = list(imputer.get_feature_names_out(list(FEATURE_NAMES)))

    rows = rank(filled, labels, names)
    result = {
        "rows": int(len(train)),
        "features_ranked": len(names),
        "classes": sorted(set(labels.tolist())),
        "top_by_mutual_information": rows[:top],
        "bottom_by_mutual_information": rows[-5:],
        "disagreements": [row for row in rows[:top] if row["f_rank"] > 2 * top],
    }
    if write_figure:
        result["figure"] = str(figure(rows, top).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    table = None
    if args.table:
        import pandas as pd

        table = pd.read_parquet(args.table)
    result = run(table, args.top, not args.no_figure)
    if not result["rows"]:
        print("no feature table; run python -m src.features.build first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

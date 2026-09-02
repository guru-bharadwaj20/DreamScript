"""Phase 7.1.5 - Gini, permutation and SHAP, and the three answers they give to one question.

    python -m src.classify.shapley      # writes reports/figures/p7_shap_summary.png

The plan asks for Gini, permutation and SHAP importances and a SHAP summary plot. Three methods
for one question is only worth the compute if they can disagree, so this task is built around
where they do.

    Gini (impurity)   how much each feature reduced impurity across the forest's splits. Free -
                      it falls out of training - and **biased toward high-cardinality features**,
                      because a continuous column has many candidate thresholds and a binary one
                      has one. 4.2.3 added 26 missingness indicators to this table; every one of
                      them is binary, and Gini importance systematically under-rates them.
    permutation       how much held-out macro F1 falls when one column is shuffled. Measures the
                      *fitted model's* reliance, has no cardinality bias, and **splits the credit
                      badly between correlated features** - shuffling one of two near-duplicates
                      changes little because the other still carries the signal.
    SHAP              each feature's additive contribution to each individual prediction, with
                      the game-theoretic guarantee that contributions sum to the prediction.
                      Per-row rather than per-model, so it is the only one of the three that can
                      say *which pages* a feature mattered for.

4.2.6 already ranked this table by mutual information and the ANOVA F, and found the two
disagreeing sharply - `layout_node_density` ranked 2nd by MI and 49th by F. This task adds the
three model-based rankings to that pair, and the interesting result would be a feature that ranks
high everywhere except one place.

## The leak this task has to avoid

Permutation importance computed on training rows measures memorisation, not reliance. It is
computed **on held-out folds** here, which costs a refit per fold and is the only version of the
number that means anything for generalization.

## What it measured

A 300-tree forest on the handcrafted table - 58 prepared columns (33 features and 25 missingness
indicators), 1,340 real pages, SHAP on 400 stratified rows. Top ten by each method:

    rank   gini                    permutation             shap
     1     dir_flow_axis           text_area_frac          dir_flow_axis
     2     dir_angle_entropy       dir_flow_axis           text_label_length
     3     dir_axis_aligned        text_label_length       text_block_count
     4     text_label_length       dir_angle_entropy       dir_angle_entropy
     5     text_area_frac          global_ink_coverage     text_area_frac
     6     global_ink_coverage     global_bbox_fill        line_to_curve
     7     line_to_curve           line_to_curve           dir_axis_aligned
     8     text_block_count        text_block_count        global_ink_coverage
     9     global_bbox_fill        global_aspect           global_bbox_fill
    10     shape_frac_ellipse      edge_count              arrowhead_count

## The three methods are really two

    gini vs shap           0.9843
    gini vs permutation    0.4999
    permutation vs shap    0.4688

**Gini and SHAP agree at Spearman 0.984 - they are the same ranking.** That is not a coincidence
of this corpus: both are read off the fitted trees' structure, so both answer "what did the forest
split on", and neither has any way to know whether those splits helped on data the forest has not
seen. Running both is defensible for the *per-row* explanations SHAP adds, and indefensible as two
independent checks on which features matter.

**Permutation is the one that disagrees, and it is the one measured on held-out rows.** It ranks
`text_area_frac` first where Gini ranks it fifth, and it demotes almost everything structural.

## What permutation demotes, and why that is the interesting result

    feature                gini   perm   shap
    arrowhead_count          13     53     10
    layout_nn_distance       18     55     20
    node_count               21     51     19
    shape_frac_rectangle     20     47     21
    layout_row_regularity    27     57     27

**20 of 58 columns have negative permutation importance** - shuffling them makes held-out macro F1
*better*. `arrowhead_count`, `node_count`, `conn_components` and `conn_cycle_count` are all in
that list, and all four are counts the forest split on heavily. This is the correlated-feature
failure the docstring predicted, in its strongest form: these columns are near-duplicates of each
other and of `edge_count`, so shuffling one leaves the information intact and removes only the
noise it was carrying. The forest used them; the forest did not need them.

The size of the effect is worth stating plainly - the best column is worth +0.107 macro F1 when
shuffled and the worst is worth -0.008. **Nothing in this table is individually load-bearing.**
4.2.5's pruning already found the same thing from the other direction.

## The Gini cardinality bias, measured

25 of the 58 columns are 4.2.3's binary missingness indicators. Mean rank of an indicator against
mean rank of a real feature:

    method        indicators   features   best indicator
    gini             45.6         17.3         26th
    shap             44.8         17.9         18th
    permutation      36.2         24.4         23rd

**Gini and SHAP put the indicators 28 ranks below the features; permutation puts them 12 below.**
The predicted bias is there and it is large - a binary column offers one candidate threshold
against a continuous column's hundreds, so impurity accounting under-credits it - but the
correction changes the conclusion less than the docstring implied: no indicator reaches the top
17 under any method, and `layout_col_regularity_missing`, the best of them under 4.2.6's mutual
information, is 58th of 58 by permutation. The indicators are genuinely weak here, and the bias
was making a weak column look slightly weaker.

## The feature that ranks high everywhere

`dir_flow_axis` is 1st, 2nd and 1st. 4.2.6's mutual information had it near the top as well, and
5.3.2's tree used it at the root. **Five methods spanning impurity, held-out ablation, additive
attribution, mutual information and a single tree's first split all name the same column**, which
is about as close to a stable answer as this table gets: how the arrows are oriented is what a
diagram type looks like.

The figure is `reports/figures/p7_shap_summary.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classify.activations import TABLES

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p7_shap_summary.png"

SEED = 42

METHODS = ("gini", "permutation", "shap")


def model(table: str = "handcrafted", **kwargs):
    """7.1.2's forest - the only model of the three importance methods all apply to.

    A forest rather than a booster because Gini importance is defined for it, permutation applies
    to anything, and `shap.TreeExplainer` handles it exactly rather than by sampling.
    """
    from src.classify.forest import pipeline

    settings = {"n_estimators": 300, "max_features": "sqrt", "max_depth": None, **kwargs}
    return pipeline(**settings)


def prepared_names(fitted, data) -> list[str]:
    """Column names *after* 4.2.4's preparation, which is not the same list as before it.

    `feature_scaler()` appends one `*_missing` indicator per column that had a hole in the
    training rows, so the fitted model sees more columns than the table has. Gini and SHAP are
    read off the model and have to be named with this list; permutation importance is computed
    on the pipeline's input and uses `data.feature_names`.
    """
    imputer = fitted.named_steps["prepare"].named_steps["impute"]
    names = list(data.feature_names)
    return names + [f"{names[column]}_missing" for column in imputer.missing_columns_]


def gini(data, table: str = "handcrafted") -> dict:
    """Impurity decrease, averaged over the forest. Free, and biased toward wide columns."""
    fitted = model(table).fit(data.X, data.y)
    values = fitted.named_steps["model"].feature_importances_
    return _named(values, prepared_names(fitted, data))


def permutation(
    data, table: str = "handcrafted", folds: int = 5, repeats: int = 5, n_jobs: int | None = None
) -> dict:
    """Held-out macro-F1 drop when each column is shuffled, averaged over folds.

    Computed per fold on the *test* rows: a permutation importance measured on the training set
    reports how much the model memorised a column, which is a different and much larger number.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    totals = None
    names: list[str] = []
    for train_idx, test_idx in splitter.split(data.X, data.y):
        fitted = model(table).fit(data.X[train_idx], data.y[train_idx])
        # The columns are shuffled *after* preparation, so this method scores the same column
        # space as Gini and SHAP - missingness indicators included. Shuffling before the imputer
        # would let the imputer partly undo the shuffle, and would leave the indicators unrankable.
        prepare = fitted.named_steps["prepare"]
        names = names or prepared_names(fitted, data)
        result = permutation_importance(
            fitted.named_steps["model"],
            prepare.transform(data.X[test_idx]),
            data.y[test_idx],
            scoring="f1_macro",
            n_repeats=repeats,
            random_state=SEED,
            n_jobs=n_jobs if n_jobs is not None else -1,
        )
        totals = result.importances_mean if totals is None else totals + result.importances_mean
    return _named(totals / folds, names)


def shap_values(data, table: str = "handcrafted", sample: int = 400):
    """Per-row SHAP contributions from an exact tree explainer.

    Subsampled because `TreeExplainer` on 300 trees x 1,340 rows x 5 classes is the most
    expensive thing in Phase 7; the sample is stratified so the 40-row class is not lost.
    """
    import shap
    from sklearn.model_selection import train_test_split

    fitted = model(table).fit(data.X, data.y)
    prepared = fitted.named_steps["prepare"].transform(data.X)

    if sample and sample < len(data.y):
        index, _ = train_test_split(
            np.arange(len(data.y)), train_size=sample, random_state=SEED, stratify=data.y
        )
    else:
        index = np.arange(len(data.y))

    explainer = shap.TreeExplainer(fitted.named_steps["model"])
    values = explainer.shap_values(prepared[index], check_additivity=False)
    # sklearn's forests return (rows, features, classes); older/other shapes come back as a list.
    array = np.asarray(values)
    if array.ndim == 3:
        magnitude = np.abs(array).mean(axis=2).mean(axis=0)
    else:
        magnitude = np.abs(array).mean(axis=0)
    return magnitude, array, prepared[index], index, prepared_names(fitted, data)


def shap_importance(data, table: str = "handcrafted", sample: int = 400) -> dict:
    magnitude, _, _, _, names = shap_values(data, table, sample)
    return _named(magnitude, names)


def _named(values, names) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        name: round(float(value), 6)
        for name, value in sorted(zip(names, values, strict=True), key=lambda pair: -pair[1])
    }


def ranks(scores: dict, names: list[str]) -> dict:
    """Rank of each feature under one method, 1 = most important."""
    order = sorted(names, key=lambda name: -scores.get(name, 0.0))
    return {name: position + 1 for position, name in enumerate(order)}


def agreement(rankings: dict, names: list[str]) -> dict:
    """Spearman between every pair of methods, plus the features they disagree about most.

    The disagreements are the deliverable. Three methods that agree perfectly would mean two of
    them were unnecessary, and the biases described above predict specific disagreements.
    """
    import itertools

    from scipy import stats

    ordered = {method: [rankings[method][name] for name in names] for method in rankings}
    pairs = {}
    for a, b in itertools.combinations(sorted(ordered), 2):
        pairs[f"{a} vs {b}"] = round(float(stats.spearmanr(ordered[a], ordered[b]).statistic), 4)

    spread = {
        name: max(rankings[m][name] for m in rankings) - min(rankings[m][name] for m in rankings)
        for name in names
    }
    worst = sorted(spread, key=lambda name: -spread[name])[:8]
    return {
        "spearman": pairs,
        "largest_rank_disagreements": {
            name: {method: rankings[method][name] for method in sorted(rankings)} for name in worst
        },
    }


def figure(data, table: str = "handcrafted", sample: int = 400, path: Path = FIGURE) -> Path:
    """The SHAP summary plot the plan asks for."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    _, array, prepared, _, names = shap_values(data, table, sample)
    # A beeswarm needs one 2-D matrix; for multiclass the convention is to plot the mean absolute
    # contribution across classes, which is what the bar form of the summary shows.
    per_feature = np.abs(array).mean(axis=2) if array.ndim == 3 else np.abs(array)

    plt.figure(figsize=(8, 9))
    shap.summary_plot(
        per_feature,
        features=prepared,
        feature_names=names,
        show=False,
        max_display=25,
    )
    plt.title(
        f"Phase 7.1.5 - mean |SHAP| per feature, Random Forest on the {table} table "
        f"({sample} stratified rows)",
        fontsize=9,
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close("all")
    return path


def run(
    table: str = "handcrafted",
    corpus: str = "real",
    sample: int = 400,
    n_jobs: int | None = None,
    write: bool = True,
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    # Every ranking is over the prepared column space - 4.2.3's missingness indicators are
    # features here, and the Gini bias against them is one of the results this task is looking for.
    names = prepared_names(model(table).fit(data.X, data.y), data)

    scores = {
        "gini": gini(data, table),
        "permutation": permutation(data, table, n_jobs=n_jobs),
        "shap": shap_importance(data, table, sample),
    }
    rankings = {method: ranks(values, names) for method, values in scores.items()}

    summary = {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": len(names),
        "shap_sample": sample,
        "top10": {method: list(values)[:10] for method, values in scores.items()},
        "scores": scores,
        "agreement": agreement(rankings, names),
        # Permutation importance can be negative - shuffling a column can help - and that is a
        # real statement about a feature the model would be better off without.
        "features_with_negative_permutation_importance": [
            name for name, value in scores["permutation"].items() if value < 0
        ],
    }
    if write:
        summary["figure"] = str(figure(data, table, sample).relative_to(ROOT))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="handcrafted", choices=list(TABLES))
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.table, args.corpus, args.sample, args.jobs, not args.no_figure)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

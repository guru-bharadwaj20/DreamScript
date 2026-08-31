"""Phase 5.3.3 - what the linear model actually weighs, and which way each weight points.

    python -m src.classify.coefficients      # writes reports/logreg_coefficients.md

5.1.1's logistic regression is the best model in Phase 5 and the only one whose parameters are
a table rather than a structure. Multinomial softmax gives one coefficient per (class, feature),
and each one has a reading that a tree's splits do not:

    sign        positive means the feature pushes a page *towards* this class
    magnitude   how far, per one standard deviation of the feature

The units matter and they are free here. Because 4.2.4's scaler standardises every column, the
33 coefficients of one class are already on a common scale and can be ranked directly - a
comparison that would be meaningless on raw columns where `node_count` runs to 40 and
`global_ink_coverage` to 0.04. `exp(coefficient)` is the multiplicative change in that class's
odds per standard deviation, which is the form to quote to anyone who has to act on it.

## L1 is doing feature selection, and the table should say how much

5.1.1 selected an **l1 penalty at C = 10**, so a coefficient here is not merely small when the
feature is unhelpful - it is exactly zero, and the model has dropped the column for that class.
The report therefore counts zeros per class as well as ranking magnitudes: sparsity is a result,
not a formatting detail, and a feature zeroed for all five classes is one the selected model has
discarded entirely.

## What a coefficient is not

A coefficient is a weight *in the presence of the other 32*, not a measure of the feature on its
own - which is exactly why it can disagree with 4.2.6's univariate mutual-information ranking,
and that disagreement is reported rather than smoothed over. Two correlated features will split
one effect between them, and 4.2.5 already pruned the worst of those pairs at |r| = 0.962.

## What it measured

Fitted on all 1,340 real photographs, 58 standardised columns (33 features + 25 indicators):

    class            kept   zeroed   largest |coef|   intercept
    flowchart          56     2 ( 3%)    3.497         +2.741
    er_diagram         54     4 ( 7%)    2.384         -1.560
    wireframe          54     4 ( 7%)    4.359         +1.076
    circuit            53     5 ( 9%)    1.603         +0.995
    state_machine      23    35 (60%)    1.441         -3.252

**Circuits are predicted by elimination, and the sign column is where that shows.** Six of the
eight largest weights for `circuit` are **negative** - not much curve (`line_to_curve` -1.60),
short labels (-1.50), little text per node (-1.24), few cycles (-1.12) - and its largest
coefficient of any sign, 1.603, is the smallest of the five classes. The model has almost no
positive evidence for a circuit; it has a description of what a circuit is *not*. That is the
mechanism behind three numbers already published: 5.2.3's 0.277 F1, 5.2.4's finding that the
class ranks at 0.788 AUC while only 23% are ever predicted, and 5.2.6's "over half of all
circuits are called flowcharts". A class defined by absences loses every argmax to a class
defined by presences.

**State machines are the opposite, and L1 says so by throwing away 60% of the table.** 35 of 58
weights are exactly zero for that class - every other class keeps ~54 - and the 23 that survive
are a description rather than an exclusion: round nodes (`shape_frac_ellipse` +0.95,
`shape_frac_circle` +0.64), self-loops (+0.65), high angular entropy (+0.82). The sparsest class
is the one 5.2.3 scored at 0.955 F1 and 5.3.2's tree captured in a single rule. Three models
agree that state machines are simple and circuits are not, by three unrelated measurements.

**The camera leak is in the top eight for three of the five classes, with opposite signs**:
`global_aspect` +1.25 for flowchart, +0.89 for circuit, -1.99 for wireframe. That is not a
diagram property being weighed, it is the model learning that wide pages come from sketch2code
and every sketch2code page is a wireframe - 4.1.5's finding, now visible as a parameter. 5.1.1
measured the whole leak as worth 0.0052 macro F1, which is the reassuring half of the story; the
coefficient is the unreassuring half.

**Against 4.2.6's univariate ranking: Spearman +0.484.** Half-agreement, and the disagreements
are the interesting rows. `text_area_frac` carries the fourth-largest coefficient in the model
and ranks **47th of 60** by mutual information: on its own it says nothing, and conditioned on
the other 32 columns it is one of the strongest weights flowchart has. Univariate screening
would have thrown it away. This is the same argument 4.2.6 made from the other direction when it
found six of its top fifteen invisible to ANOVA F, and it is why neither ranking should be used
to select features without a model attached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classify.data import Dataset, load
from src.classify.linear import BEST_PARAMS, best_estimator
from src.utils.config import ROOT

REPORT = ROOT / "reports" / "logreg_coefficients.md"

#: A coefficient below this is treated as zeroed. `saga` stops at a tolerance, not at exactly
#: 0.0, so testing `== 0` would under-count L1's sparsity by a handful of columns.
ZERO = 1e-6


def _relative(path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def fitted(dataset: Dataset):
    """5.1.1's selected model on the whole corpus, with the expanded column names."""
    estimator = best_estimator().fit(dataset.X, dataset.y)
    prepare = estimator.named_steps["prepare"]
    names = list(prepare.named_steps["impute"].get_feature_names_out(dataset.feature_names))
    return estimator.named_steps["model"], names


def table(model, names) -> dict:
    """One row per (class, feature): the coefficient, its sign and its odds ratio."""
    classes = [str(name) for name in model.classes_]
    coefficients = np.asarray(model.coef_, float)
    rows = {
        name: [
            {
                "feature": names[column],
                "coefficient": round(float(coefficients[index, column]), 4),
                "odds_ratio": round(float(np.exp(coefficients[index, column])), 3),
                "direction": (
                    "towards"
                    if coefficients[index, column] > ZERO
                    else "away" if coefficients[index, column] < -ZERO else "dropped"
                ),
            }
            for column in range(coefficients.shape[1])
        ]
        for index, name in enumerate(classes)
    }
    for name in rows:
        rows[name].sort(key=lambda row: -abs(row["coefficient"]))
    return {
        "classes": classes,
        "features": len(names),
        "intercepts": {
            name: round(float(model.intercept_[index]), 4) for index, name in enumerate(classes)
        },
        "by_class": rows,
    }


def sparsity(model, names) -> dict:
    """How much of the model L1 threw away, per class and overall."""
    coefficients = np.abs(np.asarray(model.coef_, float))
    classes = [str(name) for name in model.classes_]
    zeroed = coefficients <= ZERO
    dropped_everywhere = [names[column] for column in np.flatnonzero(zeroed.all(axis=0))]
    return {
        "penalty": BEST_PARAMS["penalty"],
        "C": BEST_PARAMS["C"],
        "per_class": {
            name: {
                "zeroed": int(zeroed[index].sum()),
                "kept": int((~zeroed[index]).sum()),
                "share_zeroed": round(float(zeroed[index].mean()), 4),
                "largest": round(float(coefficients[index].max()), 4),
            }
            for index, name in enumerate(classes)
        },
        "zeroed_overall": round(float(zeroed.mean()), 4),
        "dropped_for_every_class": sorted(dropped_everywhere),
    }


def agreement_with_univariate(model, names, dataset: Dataset) -> dict:
    """Does the fitted model rank features the way 4.2.6's mutual information did?

    Spearman over the largest absolute coefficient a feature gets in any class against its
    mutual information. A low correlation is not a fault in either method - one is univariate
    and one is conditional on 32 other columns - but it is the number that says how much of
    4.2.6's ranking survived being put inside a model.
    """
    from scipy import stats

    from src.features.importance import rank
    from src.features.scaling import feature_scaler

    scaled = feature_scaler().fit_transform(dataset.X)
    ranked = {row["feature"]: row for row in rank(scaled, dataset.y, names)}
    strength = np.abs(np.asarray(model.coef_, float)).max(axis=0)
    mutual = np.array([ranked[name]["mutual_information"] for name in names])
    rho, p_value = stats.spearmanr(strength, mutual)

    order = np.argsort(-strength)
    return {
        "spearman": round(float(rho), 4),
        "p_value": float(p_value),
        "top5_by_coefficient": [
            {
                "feature": names[i],
                "max_abs_coefficient": round(float(strength[i]), 4),
                "mi_rank": ranked[names[i]]["mi_rank"],
            }
            for i in order[:5]
        ],
    }


def to_markdown(built: dict, sparse: dict, versus: dict, dataset: Dataset, top: int = 8) -> str:
    lines = [
        "# Phase 5.3.3 — logistic regression coefficients",
        "",
        f"5.1.1's selected model (`{BEST_PARAMS}`) fitted on all {len(dataset.y)} real "
        f"photographs, {built['features']} standardised columns. A coefficient is the change in "
        "the log-odds of that class per **one standard deviation** of the feature, holding the "
        "others fixed; `exp(coefficient)` is the same thing as a multiplier on the odds.",
        "",
        "## What L1 kept",
        "",
        "| class | features kept | zeroed | largest \\|coefficient\\| | intercept |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for name in built["classes"]:
        cell = sparse["per_class"][name]
        lines.append(
            f"| {name} | {cell['kept']} | {cell['zeroed']} ({cell['share_zeroed']:.0%}) | "
            f"{cell['largest']:.3f} | {built['intercepts'][name]:+.3f} |"
        )
    dropped = sparse["dropped_for_every_class"]
    lines += [
        "",
        f"**{sparse['zeroed_overall']:.0%} of all (class, feature) weights are exactly zero.** "
        + (
            f"{len(dropped)} features are dropped for every class: "
            + ", ".join(f"`{name}`" for name in dropped)
            + "."
            if dropped
            else "No feature is dropped for every class."
        ),
        "",
        "## The strongest weights per class",
        "",
    ]
    for name in built["classes"]:
        lines += [
            f"### {name}",
            "",
            "| feature | coefficient | odds ratio per SD | pushes |",
            "| :--- | ---: | ---: | :--- |",
        ]
        for row in built["by_class"][name][:top]:
            lines.append(
                f"| `{row['feature']}` | {row['coefficient']:+.3f} | {row['odds_ratio']:.2f} | "
                f"{row['direction']} |"
            )
        lines.append("")

    lines += [
        "## Against 4.2.6's univariate ranking",
        "",
        f"Spearman correlation between each feature's largest absolute coefficient and its "
        f"mutual information: **{versus['spearman']:+.3f}**.",
        "",
        "| feature | max \\|coefficient\\| | rank by mutual information |",
        "| :--- | ---: | ---: |",
    ]
    for row in versus["top5_by_coefficient"]:
        lines.append(
            f"| `{row['feature']}` | {row['max_abs_coefficient']:.3f} | {row['mi_rank']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(corpus: str = "real", top: int = 8, write: bool = True) -> dict:
    dataset = load(corpus)
    model, names = fitted(dataset)
    built = table(model, names)
    sparse = sparsity(model, names)
    versus = agreement_with_univariate(model, names, dataset)

    summary = {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "sparsity": sparse,
        "versus_univariate": versus,
        "intercepts": built["intercepts"],
        "top_by_class": {name: built["by_class"][name][:top] for name in built["classes"]},
    }
    if write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(to_markdown(built, sparse, versus, dataset, top), encoding="utf-8")
        summary["report"] = _relative(REPORT)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, args.top, not args.no_write), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

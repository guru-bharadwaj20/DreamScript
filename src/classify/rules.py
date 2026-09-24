"""Phase 5.3.2 - reading the tree out as sentences, and checking the sentences are true.

    python -m src.classify.rules       # writes reports/tree_rules.md and reports/figures/p5_tree.png

The decision tree is the only model in Phase 5 whose decision can be stated in words, and this
is where that is cashed in. Every leaf of 5.1.3's selected tree is one rule - the conjunction of
the tests along its root-to-leaf path - and a rule is worth printing only with the two numbers
that say whether to believe it:

    coverage    how many training rows reach this leaf
    purity      what share of them are the class the leaf predicts

A rule with coverage 4 and purity 1.00 is a memorised page, not a finding. Rules are therefore
ranked by coverage and the tail is summarised rather than listed, and the report says how much
of the corpus the printed rules actually account for.

## The thresholds are in standardised units, and that is a problem worth fixing

The tree is fitted inside the Phase 4 pipeline, so it splits on **standardised** columns:
`node_count <= 0.42` means 0.42 standard deviations above the training mean, which is
unreadable. Every threshold is therefore mapped back through the fitted scaler into the
feature's own units before printing, so the report says `node_count <= 12.4` and a reader can
check it against a page. The standardised form is kept alongside for anyone reproducing the fit.

## What a tree cannot tell you

These rules are a description of one fitted tree, not of the problem. 5.1.3 measured the tree at
0.7539 macro F1, below both other models, and 5.2.8 established that gap is real at p = 6e-6, so
a rule here is the reasoning of the **weakest** of Phase 5's three models. It is included because
a wrong explanation that can be checked is worth more than a right one that cannot, and because
5.3.4 uses these paths to name the error classes.

## What it measured

5.1.3's pruned tree refit on all 1,340 real photographs: **54 leaves** - the same 54 the
pruning path selected - mean depth 7.0, maximum 11, median 5 rows per leaf, mean purity 0.919.
The 15 largest rules account for **86% of the corpus**, so the printed table is most of the
model rather than a sample of it.

    rank  predicts       rows   purity   rule
      1   wireframe       454    1.000   dir_angle_entropy <= 0.67 and text_label_length > 0.068
                                         and dir_flow_axis <= -0.44 and arrows_per_node <= 4.1
                                         and text_block_count <= 21.5
      2   flowchart       215    1.000   dir_angle_entropy <= 0.67 and text_label_length <= 0.068
                                         and text_block_count > 18.5 and dir_axis_aligned > 0.96
      6   state_machine    50    1.000   dir_angle_entropy > 0.67 and text_area_frac <= 0.28
                                         and arrowhead_count <= 7.5 and global_bbox_fill > 0.66

**Leaves per class is the whole finding:**

    class            leaves   rows
    flowchart          23      600
    wireframe          12      600
    circuit            11       40
    er_diagram          7       50
    state_machine       1       50

**State machines are one rule, and circuits are eleven.** A single four-condition path takes all
50 state machines at purity 1.000 - high angular entropy, little text, few arrowheads, a
well-filled bounding box, which is a readable description of a page of circles with curved
arrows between them. The 40 circuits are scattered across 11 leaves whose coverages are
14, 8, 5, 5, 4, 3, 3, 2, 2, 1, 1 and whose purities run down to **0.25**: five of those leaves
predict circuit while most of the rows in them are something else, which is what
`class_weight="balanced"` buys and what 5.2.6 saw from the other side as 75 flowcharts called
circuits. A class the tree cannot describe is a class it shatters, and the leaf count says so
more directly than any score in 5.2.

**The tree agrees with 4.2.6 about which features matter and not about the ranking.** By
appearances across all 54 paths: `text_label_length` 75, `dir_angle_entropy` 54,
`dir_flow_axis` 30, `text_block_count` 29. The root split is `dir_angle_entropy <= 0.67`, which
separates curved-arrow pages from straight-line ones before anything else is asked -
4.1.7's feature doing the work 4.1.1's counts were expected to do. `global_aspect`, the camera
leak, appears 22 times, so the tree is reading the photograph as well as the drawing.

**These rules are the reasoning of Phase 5's weakest model.** 5.2.8 established the tree is
worse than both other models at p = 6e-6, and the largest circuit rule is nine conditions deep -
a sentence no reader can check against a page. The readable rules are the ones for the classes
the model already gets right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.classify.data import Dataset, load
from src.classify.tree import BEST_PARAMS, best_estimator
from src.features.impute import SUFFIX as INDICATOR_SUFFIX
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

REPORT = ROOT / "reports" / "tree_rules.md"
FIGURE = ROOT / "reports" / "figures" / "p5_tree.png"

#: Deeper than this and the rendered tree is a grey smear; the text rules carry the rest.
RENDER_DEPTH = 3


def _relative(path) -> str:
    """Repo-relative when it can be; absolute when a caller redirected it elsewhere."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def fitted(dataset: Dataset):
    """5.1.3's selected tree, fitted on the whole corpus, with its column names and scaler."""
    estimator = best_estimator().fit(dataset.X, dataset.y)
    prepare = estimator.named_steps["prepare"]
    names = list(prepare.named_steps["impute"].get_feature_names_out(dataset.feature_names))
    return estimator, estimator.named_steps["model"], prepare, names


def unstandardise(prepare, column: int, threshold: float) -> float:
    """A split threshold back in the feature's own units.

    `StandardScaler` centres and scales, so the inverse is `mean + threshold * scale`. The
    indicator columns are 0/1 before scaling and come back as a fraction, which is why the
    report prints those splits as "present" / "missing" rather than as a number.
    """
    scale = prepare.named_steps["scale"]
    return float(scale.mean_[column] + threshold * scale.scale_[column])


def leaf_membership(estimator, dataset: Dataset) -> dict[int, np.ndarray]:
    """The true labels of the training rows that land in each leaf.

    `tree_.value` cannot be used for this. 5.1.3 selected `class_weight="balanced"`, so the
    stored distribution is *weighted* - a leaf holding 4 circuits and 4 flowcharts records 0.5
    circuit against 0.02 flowchart, and reporting that as purity 0.96 would be a fabrication.
    Pushing the corpus back through the fitted tree gives the row counts themselves.
    """
    prepare = estimator.named_steps["prepare"]
    ids = estimator.named_steps["model"].apply(prepare.transform(dataset.X))
    return {node: dataset.y[ids == node] for node in np.unique(ids)}


def extract(tree: DecisionTreeClassifier, names, classes, prepare, membership) -> list[dict]:
    """One record per leaf: its path as a list of conditions, its class, coverage and purity."""
    inner = tree.tree_
    leaves: list[dict] = []

    def walk(node: int, path: list[dict]) -> None:
        if inner.children_left[node] == inner.children_right[node]:
            predicted = str(classes[int(np.argmax(inner.value[node][0]))])
            members = membership.get(node, np.array([], dtype=object))
            total = len(members)
            purity = float((members == predicted).mean()) if total else 0.0
            leaves.append(
                {
                    "predicts": predicted,
                    "coverage": total,
                    "purity": round(purity, 4),
                    "depth": len(path),
                    "conditions": list(path),
                }
            )
            return

        column = int(inner.feature[node])
        threshold = float(inner.threshold[node])
        raw = unstandardise(prepare, column, threshold)
        for child, operator in (
            (inner.children_left[node], "<="),
            (inner.children_right[node], ">"),
        ):
            walk(
                child,
                path
                + [
                    {
                        "feature": names[column],
                        "operator": operator,
                        "threshold": round(raw, 4),
                        "standardised": round(threshold, 4),
                    }
                ],
            )

    walk(0, [])
    leaves.sort(key=lambda leaf: -leaf["coverage"])
    return leaves


def sentence(leaf: dict) -> str:
    """One leaf as a readable rule, with repeated features collapsed into a range.

    A root-to-leaf path often tests the same feature twice - `x > 3` then `x <= 9` - and
    printing both is how tree rules become unreadable. Collapsing them into `3 < x <= 9` is the
    single change that makes these legible.
    """
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    order: list[str] = []
    for condition in leaf["conditions"]:
        name = condition["feature"]
        if name not in order:
            order.append(name)
        if condition["operator"] == "<=":
            upper[name] = min(upper.get(name, np.inf), condition["threshold"])
        else:
            lower[name] = max(lower.get(name, -np.inf), condition["threshold"])

    parts = []
    for name in order:
        low, high = lower.get(name), upper.get(name)
        if name.endswith(INDICATOR_SUFFIX):
            # 4.2.3's indicators are 0/1 before scaling, so a threshold anywhere between them
            # is a test of presence. "layout_nn_distance_missing <= 0.5" is not English.
            base = name[: -len(INDICATOR_SUFFIX)]
            parts.append(f"{base} is {'present' if high is not None else 'missing'}")
        elif low is not None and high is not None:
            parts.append(f"{low:g} < {name} <= {high:g}")
        elif low is not None:
            parts.append(f"{name} > {low:g}")
        else:
            parts.append(f"{name} <= {high:g}")
    return " and ".join(parts)


def summarise(leaves: list[dict], top: int = 15) -> dict:
    """Coverage accounting, so the report can say what the printed rules leave out."""
    total = sum(leaf["coverage"] for leaf in leaves)
    printed = sum(leaf["coverage"] for leaf in leaves[:top])
    singletons = [leaf for leaf in leaves if leaf["coverage"] == 1]
    by_class: dict[str, int] = {}
    for leaf in leaves:
        by_class[leaf["predicts"]] = by_class.get(leaf["predicts"], 0) + 1
    return {
        "leaves": len(leaves),
        "rows": total,
        "printed": top,
        "coverage_of_printed": round(printed / total, 4) if total else 0.0,
        "median_coverage": int(np.median([leaf["coverage"] for leaf in leaves])),
        "singleton_leaves": len(singletons),
        "singleton_share": round(len(singletons) / len(leaves), 4) if leaves else 0.0,
        "mean_depth": round(float(np.mean([leaf["depth"] for leaf in leaves])), 2),
        "max_depth": max(leaf["depth"] for leaf in leaves) if leaves else 0,
        "leaves_per_class": dict(sorted(by_class.items())),
        "mean_purity": round(float(np.mean([leaf["purity"] for leaf in leaves])), 4),
    }


def to_markdown(leaves: list[dict], stats: dict, dataset: Dataset, top: int = 15) -> str:
    lines = [
        "# Phase 5.3.2 — decision tree rules",
        "",
        f"5.1.3's selected tree (`{BEST_PARAMS}`) refitted on all {len(dataset.y)} real "
        f"photographs: **{stats['leaves']} leaves**, mean depth {stats['mean_depth']}, maximum "
        f"{stats['max_depth']}. Thresholds are in each feature's own units, mapped back through "
        "the fitted scaler, and 4.2.3's missingness indicators are printed as "
        "*present* / *missing* rather than as a threshold on a 0/1 column.",
        "",
        "## Shape of the tree",
        "",
        "| | |",
        "| :--- | ---: |",
        f"| leaves | {stats['leaves']} |",
        f"| median rows per leaf | {stats['median_coverage']} |",
        f"| leaves holding a single row | {stats['singleton_leaves']} "
        f"({stats['singleton_share']:.1%}) |",
        f"| mean leaf purity | {stats['mean_purity']:.3f} |",
        f"| rows covered by the {top} rules below | {stats['coverage_of_printed']:.1%} |",
        "",
        "Leaves per predicted class: "
        + ", ".join(f"{name} {count}" for name, count in stats["leaves_per_class"].items())
        + ".",
        "",
        f"## The {top} rules by coverage",
        "",
        "| # | predicts | rows | purity | rule |",
        "| ---: | :--- | ---: | ---: | :--- |",
    ]
    for position, leaf in enumerate(leaves[:top], start=1):
        lines.append(
            f"| {position} | **{leaf['predicts']}** | {leaf['coverage']} | "
            f"{leaf['purity']:.3f} | `{sentence(leaf)}` |"
        )

    lines += [
        "",
        "## Rules for the minority classes",
        "",
        "The three small classes are where the tree earns its class weighting, and where its "
        "rules are least trustworthy - a leaf covering four rows is a memorised page.",
        "",
        "| predicts | rows | purity | rule |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for name in ("circuit", "er_diagram", "state_machine"):
        for leaf in [leaf for leaf in leaves if leaf["predicts"] == name][:3]:
            lines.append(
                f"| {name} | {leaf['coverage']} | {leaf['purity']:.3f} | `{sentence(leaf)}` |"
            )
    lines.append("")
    return "\n".join(lines)


def figure(dataset: Dataset, depth: int = RENDER_DEPTH, path: Path = FIGURE) -> Path:
    """The top of the tree, rendered. The whole 54-leaf tree is unreadable at any page size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    _, model, prepare, names = fitted(dataset)
    fig, ax = plt.subplots(figsize=(17, 8))
    plot_tree(
        model,
        max_depth=depth,
        feature_names=names,
        class_names=[str(name) for name in model.classes_],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(
        f"5.1.3's tree, top {depth} levels of {model.get_depth()} — thresholds standardised",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def run(corpus: str = "real", top: int = 15, write: bool = True) -> dict:
    dataset = load(corpus)
    estimator, model, prepare, names = fitted(dataset)
    leaves = extract(model, names, model.classes_, prepare, leaf_membership(estimator, dataset))
    stats = summarise(leaves, top)

    summary = {"corpus": corpus, "rows": int(len(dataset.y)), **stats}
    if write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(to_markdown(leaves, stats, dataset, top), encoding="utf-8")
        summary["report"] = _relative(REPORT)
        summary["figure"] = _relative(figure(dataset, path=FIGURE))
    summary["top_rules"] = [
        {
            "predicts": leaf["predicts"],
            "coverage": leaf["coverage"],
            "purity": leaf["purity"],
            "rule": sentence(leaf),
        }
        for leaf in leaves[:top]
    ]
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--top", type=int, default=15)
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

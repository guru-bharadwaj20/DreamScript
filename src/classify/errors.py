"""Phase 5.3.4 - a taxonomy of Phase 5's mistakes, and the prediction the plan got wrong.

    python -m src.classify.errors      # writes reports/error_taxonomy.md

5.2.6 counted the confusions. This task asks the three questions counting cannot answer: which
errors are a property of the **corpus** rather than of a model, how **confident** the models are
when they are wrong, and **what the misread pages actually look like** in feature space.

Four cuts, all from 5.2.1's out-of-fold predictions so every model is judged on the same rows:

    ranked pairs    every (true, predicted) confusion, over all three models
    agreement       how many of the three models miss each row - 3 of 3 is the hard core
    confidence      the probability the model assigned to the class it chose and got wrong
    profile         the mean standardised feature gap between the misread rows of a class and
                    the correctly read ones - which is what "why" looks like when it is a number

## The plan made a prediction here, and it is checked rather than quietly dropped

The task's own line in `plan.md` reads *"expect flowchart <-> state"*. That is a hypothesis
written before any of Phase 5 was measured, and this module reports where that pair actually
ranks instead of describing the errors that were found as though they had been expected. A plan
that records its guesses is only useful if the guesses are scored.

## What it measured

**The plan's prediction is wrong, and not narrowly.** `flowchart <-> state_machine` does not
appear anywhere in the top ten confusions, in either direction. State machines are the class the
models find easiest - 5.2.3 scored them 0.955, 5.3.2's tree took all 50 in one rule, 5.3.3's L1
kept 23 of 58 weights for them - and nothing is ever confused with them. The confusion the plan
expected is a description of what these diagram types look like to a person, not of what they
look like in 33 geometric features.

    #   true          predicted      pages   share of class   logreg/knn/tree
    1   circuit    -> flowchart        64        0.533          24 / 20 / 20
    2   flowchart  -> circuit          50        0.028          15 / 12 / 23
    3   flowchart  -> er_diagram       32        0.018           4 / 11 / 17
    4   wireframe  -> flowchart        30        0.017           5 /  4 / 21
    5   er_diagram -> flowchart        29        0.193           9 / 12 /  8

**Only 30 pages of 1,340 (2.2%) are missed by all three models - and 19 of them are circuits.**
That is 47.5% of the entire circuit class against 0.0% of state machines and 0.33% of
flowcharts. 205 rows are missed by at least one model, so most errors are model-specific noise
and the hard core is small, specific and almost entirely one class. This is the population Phase
9's detector has to attack, and it is 30 pages - which is the encouraging way to read a corpus
that scores 0.79 on macro F1.

**All three models are confident when they are wrong, in three different ways:**

    model    confidence when right   when wrong   errors   confident errors
    logreg          0.973              0.825        79       31 (39%)
    knn             1.000              1.000        97       97 (100%)
    tree            0.992              0.970       131      117 (89%)

kNN is wrong at confidence exactly 1.0 on every single one of its 97 errors, which is k = 1
having no probability to express doubt with; the tree is confident on 89% of its errors, which
is 5.2.7's nine-point over-confidence seen per row. Logistic regression is the only model that
hedges before it misses, and 5.2.7 measured it as the only calibrated one. **A confidence
threshold would catch 61% of logreg's errors and none of kNN's** - which decides how Phase 13
should route a low-confidence page to the user.

**What the misread circuits look like.** Against the circuits that all three models read
correctly, the misread ones have `contain_nested_count` **3.77 standard deviations lower** and
`conn_components` 3.07 lower - they are sparser, flatter drawings with fewer separate ink
islands and almost no nesting. The circuits the models get right are the dense, deeply nested
ones. **This profile rests on 8 correctly-read pages** and is reported as a lead rather than a
result; with 40 circuits in the corpus there is no version of this comparison that is not small.

**By source, the chaos corpus fails five times more often** - 0.319 mean error rate against
0.060 for hdBPMN and 0.036 for sketch2code. That number is confounded and must be read with the
column beside it: the chaos corpus is the only source holding circuits, ER diagrams and state
machines, so it carries all three minority classes and both large classes live elsewhere. The
honest statement is that the difficulty tracks the *class*, not the capture conditions - which
is a different conclusion from the one the error rate alone invites, and 1.3's chaos protocol is
not the thing failing here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classify.cv import REAL_MODELS, SEEDS, out_of_fold
from src.classify.data import Dataset, load
from src.features.scaling import feature_scaler
from src.utils.config import ROOT

REPORT = ROOT / "reports" / "error_taxonomy.md"

#: The pair `plan.md` predicted would dominate, written down before Phase 5 was measured.
PREDICTED_PAIR = ("flowchart", "state_machine")

#: Above this, a wrong prediction is "confident" - the model was not hedging, it was mistaken.
CONFIDENT = 0.90


def _relative(path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def predictions(dataset: Dataset, models=REAL_MODELS, seed: int = SEEDS[0]) -> dict:
    """One out-of-fold prediction vector per model, all over the same rows."""
    return {model: out_of_fold(model, dataset, seed=seed) for model in models}


def ranked_pairs(dataset: Dataset, runs: dict, top: int = 10) -> list[dict]:
    """Every (true, predicted) confusion, pooled over the models, ranked by count.

    Pooling is the right unit for a taxonomy: the question is which confusions this *approach*
    makes, not which one estimator makes. The per-model split is kept so a pair that only one
    model makes is visible as such.
    """
    tally: dict[tuple, dict] = {}
    for name, run in runs.items():
        wrong = run.y_pred != dataset.y
        for true, predicted in zip(dataset.y[wrong], run.y_pred[wrong], strict=True):
            cell = tally.setdefault(
                (str(true), str(predicted)),
                {"true": str(true), "predicted": str(predicted), "count": 0, "by_model": {}},
            )
            cell["count"] += 1
            cell["by_model"][name] = cell["by_model"].get(name, 0) + 1

    support = {name: int((dataset.y == name).sum()) for name in set(dataset.y.tolist())}
    for cell in tally.values():
        # Per model, so the share is comparable with 5.2.6's row-normalised matrix rather than
        # being three times too large.
        cell["share_of_true_class"] = round(cell["count"] / (support[cell["true"]] * len(runs)), 4)
    return sorted(tally.values(), key=lambda cell: -cell["count"])[:top]


def agreement(dataset: Dataset, runs: dict) -> dict:
    """How many of the three models miss each row - the systematic / model-specific split."""
    missed = np.vstack([run.y_pred != dataset.y for run in runs.values()]).sum(axis=0)
    counts = {int(n): int((missed == n).sum()) for n in range(len(runs) + 1)}
    hard = missed == len(runs)
    hard_by_class = {
        name: {
            "rows": int((dataset.y == name).sum()),
            "missed_by_all": int((hard & (dataset.y == name)).sum()),
        }
        for name in sorted(set(dataset.y.tolist()))
    }
    for cell in hard_by_class.values():
        cell["share"] = round(cell["missed_by_all"] / cell["rows"], 4) if cell["rows"] else 0.0
    return {
        "models": len(runs),
        "rows_missed_by_n_models": counts,
        "systematic": int(counts[len(runs)]),
        "systematic_share": round(counts[len(runs)] / len(dataset.y), 4),
        "any_error": int(len(dataset.y) - counts[0]),
        "by_class": hard_by_class,
    }


def confidence(dataset: Dataset, runs: dict) -> dict:
    """How sure each model was when it was right, and when it was wrong."""
    out = {}
    for name, run in runs.items():
        chosen = run.y_proba.max(axis=1)
        wrong = run.y_pred != dataset.y
        out[name] = {
            "mean_when_right": round(float(chosen[~wrong].mean()), 4),
            "mean_when_wrong": round(float(chosen[wrong].mean()), 4) if wrong.any() else 0.0,
            "errors": int(wrong.sum()),
            "confident_errors": int((wrong & (chosen >= CONFIDENT)).sum()),
            "confident_error_share": (
                round(float((chosen[wrong] >= CONFIDENT).mean()), 4) if wrong.any() else 0.0
            ),
        }
    return out


def profile(dataset: Dataset, runs: dict, true: str, predicted: str, top: int = 6) -> dict:
    """Why one confusion happens: the feature gap between the misread rows and the read ones.

    Both groups are the *same true class*, so the difference cannot be a class effect. Features
    are standardised first, so a gap is in standard deviations and the six largest are
    comparable across features on wildly different units.
    """
    names = list(
        feature_scaler()
        .fit(dataset.X)
        .named_steps["impute"]
        .get_feature_names_out(dataset.feature_names)
    )
    scaled = feature_scaler().fit_transform(dataset.X)

    confused = np.zeros(len(dataset.y), bool)
    for run in runs.values():
        confused |= (dataset.y == true) & (run.y_pred == predicted)
    correct = np.zeros(len(dataset.y), bool)
    for run in runs.values():
        correct |= (dataset.y == true) & (run.y_pred == true)
    correct &= ~confused

    if not confused.any() or not correct.any():
        return {"true": true, "predicted": predicted, "comparable": False}

    gap = scaled[confused].mean(axis=0) - scaled[correct].mean(axis=0)
    order = np.argsort(-np.abs(gap))
    return {
        "true": true,
        "predicted": predicted,
        "comparable": True,
        "misread_rows": int(confused.sum()),
        "read_rows": int(correct.sum()),
        "largest_gaps": [
            {
                "feature": names[i],
                "gap_in_sd": round(float(gap[i]), 3),
                "direction": "higher" if gap[i] > 0 else "lower",
            }
            for i in order[:top]
        ],
    }


def by_source(dataset: Dataset, runs: dict) -> dict:
    """Error rate per source dataset - the check that 1.2's corpora are not one problem each."""
    if not len(dataset.sources):
        return {}
    out = {}
    for source in sorted(set(dataset.sources.tolist())):
        mask = dataset.sources == source
        rates = [float((run.y_pred[mask] != dataset.y[mask]).mean()) for run in runs.values()]
        out[str(source)] = {
            "rows": int(mask.sum()),
            "error_rate": round(float(np.mean(rates)), 4),
            "classes": sorted({str(name) for name in dataset.y[mask]}),
        }
    return out


def hypothesis(pairs: list[dict], predicted_pair=PREDICTED_PAIR) -> dict:
    """Where the pair `plan.md` predicted actually ranks, in both directions."""
    a, b = predicted_pair
    found = {}
    for position, cell in enumerate(pairs, start=1):
        if {cell["true"], cell["predicted"]} == {a, b}:
            found[f"{cell['true']}->{cell['predicted']}"] = {
                "rank": position,
                "count": cell["count"],
            }
    return {
        "predicted": f"{a} <-> {b}",
        "found": found,
        "holds": bool(found) and min(row["rank"] for row in found.values()) <= 3,
        "actual_top": f"{pairs[0]['true']} -> {pairs[0]['predicted']}" if pairs else None,
    }


def to_markdown(result: dict, dataset: Dataset) -> str:
    lines = [
        "# Phase 5.3.4 — error taxonomy",
        "",
        f"All three models' out-of-fold predictions over the {len(dataset.y)} real photographs, "
        "from 5.2.1's partition at seed 42 - so every model is judged on identical rows.",
        "",
        "## The prediction `plan.md` made",
        "",
        f"The task line reads *expect {result['hypothesis']['predicted']}*. "
        + (
            "It holds."
            if result["hypothesis"]["holds"]
            else f"**It does not hold.** The largest confusion is "
            f"`{result['hypothesis']['actual_top']}`, and the predicted pair appears at "
            + (
                ", ".join(
                    f"rank {row['rank']} ({name}, {row['count']} pages)"
                    for name, row in result["hypothesis"]["found"].items()
                )
                or "no rank at all inside the top ten"
            )
            + "."
        ),
        "",
        "## Confusions, pooled over the three models",
        "",
        "| # | true | predicted | pages | share of the true class | logreg / knn / tree |",
        "| ---: | :--- | :--- | ---: | ---: | :--- |",
    ]
    for position, cell in enumerate(result["pairs"], start=1):
        split = " / ".join(str(cell["by_model"].get(name, 0)) for name in REAL_MODELS)
        lines.append(
            f"| {position} | {cell['true']} | {cell['predicted']} | {cell['count']} | "
            f"{cell['share_of_true_class']:.3f} | {split} |"
        )

    accord = result["agreement"]
    lines += [
        "",
        "## Systematic or model-specific",
        "",
        "| missed by | pages |",
        "| :--- | ---: |",
    ]
    for n, count in accord["rows_missed_by_n_models"].items():
        label = {0: "no model", accord["models"]: "all three models"}.get(int(n), f"{n} of 3")
        lines.append(f"| {label} | {count} |")
    lines += [
        "",
        f"**{accord['systematic']} pages ({accord['systematic_share']:.1%}) are missed by all "
        "three models** - a hard core that is a property of the corpus rather than of any "
        "estimator, and the population Phase 9's detector has to attack.",
        "",
        "| class | rows | missed by all three | share |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for name, cell in accord["by_class"].items():
        lines.append(f"| {name} | {cell['rows']} | {cell['missed_by_all']} | {cell['share']:.3f} |")

    lines += [
        "",
        "## Confidence when wrong",
        "",
        "| model | mean confidence when right | when wrong | errors | "
        f"confident (≥ {CONFIDENT:.2f}) errors |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for name, cell in result["confidence"].items():
        lines.append(
            f"| `{name}` | {cell['mean_when_right']:.3f} | {cell['mean_when_wrong']:.3f} | "
            f"{cell['errors']} | {cell['confident_errors']} "
            f"({cell['confident_error_share']:.0%}) |"
        )

    shape = result["profile"]
    if shape.get("comparable"):
        lines += [
            "",
            f"## What the misread pages look like — `{shape['true']} -> {shape['predicted']}`",
            "",
            f"{shape['misread_rows']} misread {shape['true']} pages against "
            f"{shape['read_rows']} read correctly, same true class, features standardised so "
            "the gap is in standard deviations.",
            "",
            "| feature | gap (SD) | the misread pages are |",
            "| :--- | ---: | :--- |",
        ]
        for row in shape["largest_gaps"]:
            lines.append(f"| `{row['feature']}` | {row['gap_in_sd']:+.3f} | {row['direction']} |")

    if result["by_source"]:
        lines += [
            "",
            "## Error rate by source corpus",
            "",
            "| source | rows | mean error rate | classes present |",
            "| :--- | ---: | ---: | :--- |",
        ]
        for name, cell in sorted(
            result["by_source"].items(), key=lambda item: -item[1]["error_rate"]
        ):
            lines.append(
                f"| {name} | {cell['rows']} | {cell['error_rate']:.3f} | "
                f"{', '.join(cell['classes'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def run(corpus: str = "real", models=REAL_MODELS, top: int = 10, write: bool = True) -> dict:
    dataset = load(corpus)
    runs = predictions(dataset, models)
    pairs = ranked_pairs(dataset, runs, top)
    result = {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "pairs": pairs,
        "hypothesis": hypothesis(pairs),
        "agreement": agreement(dataset, runs),
        "confidence": confidence(dataset, runs),
        "profile": profile(dataset, runs, pairs[0]["true"], pairs[0]["predicted"]),
        "by_source": by_source(dataset, runs),
    }
    if write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(to_markdown(result, dataset), encoding="utf-8")
        result["report"] = _relative(REPORT)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=list(REAL_MODELS))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    try:
        print(
            json.dumps(run(args.corpus, tuple(args.models), args.top, not args.no_write), indent=2)
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

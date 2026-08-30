"""Phase 5.2.2 - grouped cross-validation by scribe: is the model reading diagrams or writers?

    python -m src.classify.grouped

Stratified CV lets two pages by the same person land on both sides of a fold. If a model has
learned that *this writer's* pen width and box wobble mean "flowchart", stratified CV rewards it
and the number is a lie about how the model will behave on a page from someone new. Grouped CV
by `scribe_id` removes that: every page a writer produced is in the same fold, so a held-out
page is always from an unseen hand.

The difference between the two numbers is the size of the writer effect, and it is the single
most important measurement in 5.2, because 1.2 built the chaos corpus from 126 writers
specifically to be able to make it.

## What it measured

Same models, same three seeds, five folds, 1,340 real pages and 170 writers:

    model      stratified   grouped     drop     accuracy (strat -> grouped)
    logreg       0.7898     0.7921    -0.0023      0.9413 -> 0.9423
    knn          0.7614     0.7256    +0.0358      0.9269 -> 0.9182
    tree         0.7426     0.7227    +0.0199      0.8970 -> 0.8871

**There is no writer effect worth the name.** Logistic regression is fractionally *better* when
every held-out page comes from an unseen hand; the tree and kNN lose 0.02 and 0.04, which is
inside their own fold-to-fold spread of 0.027-0.040. On the strongest reading available from
this corpus, the handcrafted features of Phase 4 describe the diagram rather than the person who
drew it - which is what they were designed to do, and the first time it has been tested.

That kNN loses the most is the expected ordering rather than a surprise: it is the only model
whose prediction *is* a specific training page, so it is the one that can most directly answer
"I have seen this hand before".

Per source, using logistic regression and one seed:

    source        rows   writers   stratified   grouped    drop
    chaos          140      63       0.700       0.721    -0.021
    hdbpmn         600     107       0.960       0.962    -0.002
    sketch2code    600       0       0.978       0.982    -0.003

The 63-writer chaos corpus - built by 1.2 precisely for this test, and the hardest source in the
project at 0.70 accuracy either way - shows no drop either. sketch2code has no writer ids, so its
two columns are the same measurement twice and are shown only to make that visible.

## The result this run actually turned up

Per-class recall, which is not what the task set out to measure and matters more:

    class            recall
    state_machine     1.000
    wireframe         0.978
    flowchart         0.960
    er_diagram        0.780
    circuit           0.225

**Circuits are found 22.5% of the time.** 4.1.1 predicted this in the feature phase - a
circuit's components are open symbols, not closed shapes, so `node_count` sees 3.66 fewer nodes
than the graph has and most per-node features are undefined - and here is the cost in a
classifier. 40 training rows is part of it, but the 50-row state machine class scores 1.000, so
it is not only scarcity. This is 5.3.4's error taxonomy in advance, and the honest headline for
Phase 5: the pipeline classifies four diagram types well and one badly.

## Where this test does and does not apply

5's `data` module gives a row with no recorded writer its own group, so it can never be paired
with anything. That is the conservative choice, and it has a consequence that has to be stated
with the result: **600 of the 1,340 rows - every sketch2code wireframe - have no `scribe_id`**,
so for those rows grouped CV is identical to stratified CV. The writer effect measured here is
therefore a writer effect **on the hdBPMN flowcharts and the chaos corpus**, diluted by 45% of
the corpus for which the question could not be asked. The per-source table below separates them.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sklearn.metrics import recall_score

from src.classify.cv import SEEDS, evaluate, out_of_fold
from src.classify.data import Dataset, load


def compare(dataset: Dataset, model: str, seeds=SEEDS) -> dict:
    """Stratified against grouped, same model, same seeds."""
    stratified = evaluate(model, dataset, strategy="stratified", seeds=seeds)
    grouped = evaluate(model, dataset, strategy="grouped", seeds=seeds)
    return {
        "model": model,
        "stratified_macro_f1": stratified["macro_f1"],
        "grouped_macro_f1": grouped["macro_f1"],
        "drop": round(stratified["macro_f1"] - grouped["macro_f1"], 4),
        "stratified_accuracy": stratified["accuracy"],
        "grouped_accuracy": grouped["accuracy"],
        "grouped_std": grouped["macro_f1_std"],
    }


def by_source(dataset: Dataset, model: str, seed: int = SEEDS[0]) -> dict:
    """The same comparison per data source, since only some sources record a writer."""
    out = {}
    for strategy in ("stratified", "grouped"):
        predictions = out_of_fold(model, dataset, strategy=strategy, seed=seed)
        for source in sorted(set(dataset.sources.tolist())):
            mask = dataset.sources == source
            row = out.setdefault(
                source,
                {
                    "rows": int(mask.sum()),
                    "writers": int(
                        len({g for g in dataset.groups[mask] if str(g).startswith("scribe:")})
                    ),
                },
            )
            row[f"{strategy}_accuracy"] = round(
                float((predictions.y_pred[mask] == predictions.y_true[mask]).mean()), 4
            )
    for row in out.values():
        row["drop"] = round(row["stratified_accuracy"] - row["grouped_accuracy"], 4)
    return out


def per_class(dataset: Dataset, model: str, seed: int = SEEDS[0]) -> dict:
    """Recall per diagram type under both strategies - where the drop actually lands."""
    classes = sorted(set(dataset.y.tolist()))
    out = {name: {} for name in classes}
    for strategy in ("stratified", "grouped"):
        predictions = out_of_fold(model, dataset, strategy=strategy, seed=seed)
        recalls = recall_score(
            predictions.y_true,
            predictions.y_pred,
            labels=classes,
            average=None,
            zero_division=0,
        )
        for name, value in zip(classes, recalls, strict=True):
            out[name][strategy] = round(float(value), 4)
    for row in out.values():
        row["drop"] = round(row["stratified"] - row["grouped"], 4)
    return out


def run(corpus: str = "real", models=("logreg", "knn", "tree")) -> dict:
    dataset = load(corpus)
    known = np.array([str(g).startswith("scribe:") for g in dataset.groups])
    return {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "rows_with_a_known_writer": int(known.sum()),
        "writers": int(len({g for g in dataset.groups if str(g).startswith("scribe:")})),
        "models": [compare(dataset, model) for model in models],
        "by_source": by_source(dataset, "logreg"),
        "per_class_logreg": per_class(dataset, "logreg"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=["logreg", "knn", "tree"])
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, tuple(args.models)), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

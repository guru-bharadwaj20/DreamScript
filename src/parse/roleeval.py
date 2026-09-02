"""Phase 7.3.9 - per-role precision and recall, the 0.80 target, and which states miss it.

    python -m src.parse.roleeval      # writes reports/role_labelling.md

The plan sets a number: **macro-F1 >= 0.80** for role labelling against the annotation. 7.3.7
reported 0.7586 pooled and 0.7763 per diagram type, so the headline answer is already known to be
*short*. This task is the breakdown that says whether the shortfall is spread across the space or
concentrated in states 7.3.4 and 7.3.5 already identified as structurally undecidable.

## What is being scored against what

The reference labels are 7.3.1's `derive_states` output, and 7.3.1 measured that **39.3% of those
assignments are rules rather than human annotations**. So this table has two kinds of row and they
should not be read the same way:

    against a human label      start, terminal, decision, process    (61% of nodes)
    against a derivation rule  branch-true, branch-false, loop-back,
                               input, output                        (39% of nodes)

A high score on the second kind means the HMM reproduced this project's own rule from evidence
that rule never saw - genuinely informative, but not the same claim as agreeing with a person.
`annotated_only_macro_f1` therefore reports the metric restricted to the four annotated states,
which is the number to quote when the question is "does it read a diagram the way a human does".

## Why per-type matrices are used here

7.3.7 measured +0.026 accuracy for conditioning on the diagram type, and Phase 5's classifier
supplies the type at 0.97 macro F1 for free. Evaluating the pooled model when the pipeline will
not use it would be reporting a number nothing downstream experiences.

## What it measured

Five-fold, per-type matrices, 14,056 nodes. **Macro F1 0.7763 against a target of 0.80 - the
target is missed by 0.024.**

    role            precision   recall     F1    support   reference
    input             0.999      0.930   0.963      782    derivation
    terminal          0.855      0.871   0.863    1,160    annotation
    decision          0.800      0.926   0.859    1,494    annotation
    output            0.726      0.970   0.831      303    derivation
    process           0.811      0.804   0.807    4,916    annotation
    branch-true       0.746      0.852   0.796    1,513    derivation
    start             0.683      0.703   0.692    1,136    annotation
    loop-back         0.672      0.559   0.611    1,368    derivation
    branch-false      0.638      0.509   0.567    1,384    derivation

**Five of nine roles clear 0.80.** The four that do not are `start` and the three derived states,
and the failure is concentrated rather than diffuse: dropping `branch-false` alone would put the
macro at 0.804.

## Restricting to the roles a human actually annotated changes the verdict

    all nine states                    0.7763
    the four annotated states only     0.8598

**Against human labels the model clears the bar by 0.06.** `start`, `process`, `decision` and
`terminal` - 8,706 of the 14,056 nodes - are labelled at 0.8598 macro F1, and every one of those
four is a role a person wrote down. The 0.7763 headline is dragged under the target by states
whose reference is 7.3.1's own derivation rule.

That is not a way of claiming the target was met. It is the distinction the plan's own line
cannot make, because the plan assumed the annotation would carry all nine states and 7.3.1 found
that it carries four. **Both numbers are reported and neither is the answer on its own**: 0.7763
is what the pipeline achieves against the labels 7.3 defined, 0.8598 is what it achieves against
the labels a human wrote.

## The confusions are the ones the two matrices predicted

    true            predicted        nodes   share of the true class
    loop-back       process            352        0.257
    branch-false    process            269        0.194
    branch-false    branch-true        233        0.168
    start           loop-back          147        0.129

**`branch-false` loses 36% of its nodes to `process` and `branch-true` combined** - precisely the
pair 7.3.5 identified as sharing `box|other|linear` as their most likely emission. The evidence
that separates them is which edge the node was reached by, and the observation alphabet does not
contain it; the transition matrix has to carry the whole distinction alone, and 7.3.4 showed it
enters `branch-false` five times less often than `branch-true` because of the DFS ordering. The
model is being asked to recover a distinction from a channel that was deliberately narrowed.

`loop-back -> process` at 0.257 is the same shape of problem: a loop-back node is an ordinary
process step that happens to be re-entered, and nothing about the node itself says so.

`start` at 0.692 is the one genuinely disappointing row, because `start` *is* annotated and it
does have a distinctive emission (`round|other|source`). Its 147 confusions with `loop-back` are
fa_bresler state machines, where the initial state is drawn as a circle and is also frequently the
target of a return transition - so the two states are the same node, and which one it "is"
depends on a precedence rule 7.3.1 had to invent.

## What would raise it

Not more data and not more smoothing - 7.3.7 showed alpha does nothing and 7.3.6 showed 3,000
extra unlabelled pages do less. The binding constraint is the observation alphabet: an arrival-edge
factor (which edge reached this node, and from what) would give `branch-false` its own signature
and is exactly the information `derive_states` uses. That is a Phase 10 change - it needs the
assembled graph, which is what Phase 10 builds - and it is recorded here as the concrete next step
rather than as a hyperparameter to try.

The report is `reports/role_labelling.md`.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.parse.roles import DERIVED_STATES, STATES
from src.utils.config import ROOT

REPORT = ROOT / "reports" / "role_labelling.md"

#: The plan's bar for this task.
TARGET = 0.80

#: States whose reference label is a human annotation rather than 7.3.1's derivation.
ANNOTATED_STATES = tuple(name for name in STATES if name not in DERIVED_STATES)


def predictions(n: int = 5, alpha: float = 1.0, per_type: bool = True) -> tuple[list, list]:
    from src.parse.sequences import build
    from src.parse.viterbi import build_model, decode_sequence, folds

    sequences = build()["sequences"]
    alphabet = sorted({s for seq in sequences for s in seq["observations"]})
    predicted, truth = [], []
    for train, test in folds(sequences, n):
        pooled = build_model(train, alphabet, alpha)
        models = (
            {
                name: build_model([s for s in train if s["diagram_type"] == name], alphabet, alpha)
                for name in {s["diagram_type"] for s in train}
            }
            if per_type
            else {}
        )
        for sequence in test:
            model = models.get(sequence["diagram_type"], pooled)
            predicted.extend(decode_sequence(sequence, model, alphabet))
            truth.extend(sequence["states"])
    return predicted, truth


def per_role(predicted: list[str], truth: list[str]) -> dict:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(STATES), zero_division=0
    )
    return {
        name: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
            "reference": "derivation" if name in DERIVED_STATES else "annotation",
        }
        for i, name in enumerate(STATES)
    }


def confusion(predicted: list[str], truth: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(truth, predicted, labels=list(STATES))
    off_diagonal = []
    for i, actual in enumerate(STATES):
        for j, guessed in enumerate(STATES):
            if i != j and matrix[i, j]:
                off_diagonal.append((actual, guessed, int(matrix[i, j])))
    off_diagonal.sort(key=lambda row: -row[2])
    return {
        "matrix": matrix.tolist(),
        "worst_confusions": [
            {
                "true": a,
                "predicted": b,
                "nodes": n,
                "share_of_true": round(n / max(1, int(matrix[STATES.index(a)].sum())), 4),
            }
            for a, b, n in off_diagonal[:8]
        ],
    }


def markdown(result: dict) -> str:
    lines = [
        "# Role-labelling accuracy",
        "",
        "Phase 7.3.9. Generated by `python -m src.parse.roleeval`.",
        "",
        f"**Macro F1 {result['macro_f1']:.4f}** against the plan's target of {TARGET:.2f} "
        f"({result['nodes']:,} nodes, five-fold, per-type matrices).",
        "",
        "| role | precision | recall | F1 | support | reference |",
        "| :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for name, row in result["per_role"].items():
        lines.append(
            f"| `{name}` | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} "
            f"| {row['support']:,} | {row['reference']} |"
        )
    lines += [
        "",
        "## Worst confusions",
        "",
        "| true | predicted | nodes | share of the true class |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for row in result["confusion"]["worst_confusions"]:
        lines.append(
            f"| `{row['true']}` | `{row['predicted']}` | {row['nodes']:,} "
            f"| {row['share_of_true']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run(n: int = 5, alpha: float = 1.0, write: bool = True) -> dict:
    from sklearn.metrics import f1_score

    predicted, truth = predictions(n, alpha)
    roles = per_role(predicted, truth)
    annotated = [(p, t) for p, t in zip(predicted, truth, strict=True) if t in ANNOTATED_STATES]
    result = {
        "nodes": len(truth),
        "macro_f1": round(
            float(
                f1_score(truth, predicted, average="macro", labels=list(STATES), zero_division=0)
            ),
            4,
        ),
        "accuracy": round(
            float(np.mean([p == t for p, t in zip(predicted, truth, strict=True)])), 4
        ),
        "target": TARGET,
        "meets_target": False,
        "per_role": roles,
        "annotated_only_macro_f1": round(
            float(
                f1_score(
                    [t for _, t in annotated],
                    [p for p, _ in annotated],
                    average="macro",
                    labels=list(ANNOTATED_STATES),
                    zero_division=0,
                )
            ),
            4,
        ),
        "roles_above_target": [name for name, row in roles.items() if row["f1"] >= TARGET],
        "roles_below_target": [name for name, row in roles.items() if row["f1"] < TARGET],
        "confusion": confusion(predicted, truth),
    }
    result["meets_target"] = result["macro_f1"] >= TARGET
    if write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(markdown(result), encoding="utf-8")
        result["report"] = str(REPORT.relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    try:
        result = run(args.folds, args.alpha, not args.no_report)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    result["confusion"] = {k: v for k, v in result["confusion"].items() if k != "matrix"}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

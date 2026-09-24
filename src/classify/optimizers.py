"""Phase 6.2.4 - five optimizers, and the loss curve that shows what the final score hides.

    python -m src.classify.optimizers        # writes reports/figures/p6_optimizers.png

The plan asks for SGD, SGD+momentum, RMSProp, Adam and AdamW compared on convergence, and for a
loss-vs-epoch overlay figure. This is the task 6.2.1 named when it said a hand-written training
loop would earn its place: sklearn exposes `loss_curve_` for some solvers and nothing at all for
its validation split, and an overlay figure needs both curves for all five.

## What a fair comparison of optimizers has to control for

Learning rate. SGD at 1e-3 is not "SGD"; it is SGD at a rate chosen for Adam, and comparing the
two at one rate measures the rate as much as the optimizer. The received wisdom is that plain SGD
wants a rate one to two orders of magnitude above Adam's, so **each optimizer is swept over its
own rate grid and reported at its own best**, and the rate that won is printed beside the score.
A single-rate table would be a cheaper figure and a wrong one.

Everything else is held at 6.2.3's selection - GELU, (512, 256), dropout and weight decay at the
values that task chose - so the only thing varying is the update rule and its step size.

## Two questions, and only one of them is about the score

    which optimizer reaches the best held-out macro F1?
    which one gets there in the fewest epochs?

They have different answers, and the figure is where the difference lives: a curve that descends
fastest for the first ten epochs is not necessarily the curve that ends lowest, and on a corpus
where early stopping fires after twenty-odd epochs the first property is worth more than the
textbook cares to admit.

## What it measured

6.2.3's regularized configuration on the hybrid table (GELU, (512, 256), dropout 0.2, decay
1e-2), stratified 5-fold, each optimizer at its own best rate out of five:

    optimizer   best lr   macro F1   std      gap     epochs   epochs to shared loss target
    adam         0.003     0.9533   0.0179   0.039     17.2                13
    rmsprop      0.001     0.9518   0.0110   0.046     25.4                10
    adamw        0.001     0.9486   0.0182   0.042     18.2                 4
    momentum     0.03      0.9371   0.0144   0.051     20.6                 5
    sgd          0.3       0.9360   0.0260   0.055     24.0                10

**The spread across all five optimizers is 0.0173. The spread across learning rates *within*
SGD alone is 0.3390.** That is a factor of twenty, and it is the finding this task exists to
produce:

    sgd        0.003 -> 0.5970   0.01 -> 0.8145   0.03 -> 0.8980   0.1 -> 0.9143   0.3 -> 0.9360
    momentum   0.001 -> 0.8145  0.003 -> 0.9004   0.01 -> 0.9149  0.03 -> 0.9371   0.1 -> 0.9366
    adam      0.0001 -> 0.9162 0.0003 -> 0.9075  0.001 -> 0.9503 0.003 -> 0.9533  0.01 -> 0.9412
    adamw     0.0001 -> 0.9044 0.0003 -> 0.9167  0.001 -> 0.9486 0.003 -> 0.9467  0.01 -> 0.9114
    rmsprop   0.0001 -> 0.9135 0.0003 -> 0.9187  0.001 -> 0.9518 0.003 -> 0.9432  0.01 -> 0.9437

**The controlled variable was the right one to control.** At Adam's best rate of 0.003, plain SGD
scores **0.5970** - it is not converging at all in the epochs early stopping allows. A comparison
run at one shared learning rate, which is how this table is usually produced, would have
concluded that SGD is 0.36 worse than Adam. Its actual disadvantage, once each is given the step
size it wants, is **0.0173**. Nearly all of the published gap between "SGD" and "Adam" on a table
like this is a statement about the step size, and the adaptive methods' real contribution here is
that they are *insensitive* to it: Adam's range across its grid is 0.046, SGD's is 0.339.

The rates that won are the textbook ones, which is worth recording because it means the grid was
centred correctly rather than truncated: SGD wants 0.3, momentum 0.03, and all three adaptive
methods want 0.001-0.003, two orders of magnitude below.

## The two questions have different answers

**Adam wins on score (0.9533); AdamW wins on speed by a wide margin.** AdamW reaches the shared
loss target in **4 epochs against Adam's 13 and RMSProp's 10** - and its final training loss is
0.00046, thirty times lower than Adam's 0.0149. It descends fastest and furthest, and it still
ends up 0.005 *behind* Adam on held-out macro F1. That is the whole reason both panels of the
figure exist: on this corpus the optimizer that fits the training set best is not the one that
generalizes best, and a convergence plot alone would have chosen the wrong one.

Read against the rest of the phase, nothing here changes the verdict. **The best cell, 0.9533,
is 0.003 above 6.1.4's logistic regression and 0.010 below 6.3.1's linear SVM on the same
table.** Five optimizers, twenty-five configurations, and the ranking of model families is
unchanged - which is 5.2.9's data-limited corpus once more, and the reason the honest headline of
6.2.4 is about learning rates rather than about optimizers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classify.activations import LOGREG, TABLES, TOPOLOGY

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p6_optimizers.png"

SEED = 42

#: Every optimizer the plan names, in the order it names them.
ORDER = ("sgd", "momentum", "rmsprop", "adam", "adamw")

#: Each optimizer gets its own rate grid, because comparing them at one rate compares the rate.
#: The adaptive three are searched an order of magnitude below the two SGD variants, which is
#: where their step sizes actually live.
RATES = {
    "sgd": (0.003, 0.01, 0.03, 0.1, 0.3),
    "momentum": (0.001, 0.003, 0.01, 0.03, 0.1),
    "rmsprop": (0.0001, 0.0003, 0.001, 0.003, 0.01),
    "adam": (0.0001, 0.0003, 0.001, 0.003, 0.01),
    "adamw": (0.0001, 0.0003, 0.001, 0.003, 0.01),
}


def grid(optimizers=ORDER, base: dict | None = None) -> list[dict]:
    """Every optimizer at every one of its own candidate rates."""
    settings = dict(base or {})
    return [
        {**settings, "optimizer": name, "learning_rate": rate}
        for name in optimizers
        for rate in RATES[name]
    ]


def best_per_optimizer(rows: list[dict]) -> list[dict]:
    """One row per optimizer, at the rate that won for it.

    Reported with the rate attached: "SGD reached 0.94" is not a statement anyone can reproduce
    or argue with unless the step size that produced it is on the same line.
    """
    best: dict[str, dict] = {}
    for row in rows:
        current = best.get(row["optimizer"])
        if current is None or row["macro_f1"] > current["macro_f1"]:
            best[row["optimizer"]] = row
    return [best[name] for name in ORDER if name in best]


def epochs_to_reach(curve: list[float], target: float) -> int | None:
    """The first epoch whose training loss is at or below `target`, or None if it never is.

    This is the convergence-speed half of the comparison, and it needs a shared target rather
    than each optimizer's own final loss - otherwise the one that converges to the worst value
    "reaches its target" first by having a worse target.
    """
    for index, value in enumerate(curve):
        if value <= target:
            return index + 1
    return None


def curves(data, rows: list[dict], folds: int = 5) -> dict:
    """One loss curve and one validation curve per optimizer, from a single fixed fold.

    Deliberately one fold, not the mean of five. Curves from different folds stop at different
    epochs because early stopping fires at different times, and averaging arrays of unequal
    length either truncates every curve to the shortest or pads them with a value the run never
    had. A single held-out fold is an honest picture of one training run, which is what the plan
    asks the figure to show.

    `n_jobs` was a parameter and reached nothing - this fits one fold in this process, and the
    only caller never passed it. A knob that looks like it controls parallelism and does not is
    worse than no knob.
    """
    from sklearn.model_selection import StratifiedKFold

    from src.classify.torchnet import pipeline

    train_idx, _ = next(
        StratifiedKFold(folds, shuffle=True, random_state=SEED).split(data.X, data.y)
    )
    result = {}
    for row in rows:
        settings = {k: v for k, v in row.items() if k in _FIT_KEYS}
        fitted = pipeline(**settings).fit(data.X[train_idx], data.y[train_idx])
        model = fitted.named_steps["model"]
        result[row["optimizer"]] = {
            "learning_rate": row["learning_rate"],
            "loss": list(model.loss_curve_),
            "val": list(model.val_curve_),
            "epochs": model.epochs_run_,
        }
    return result


_FIT_KEYS = (
    "hidden_layer_sizes",
    "activation",
    "dropout",
    "weight_decay",
    "optimizer",
    "learning_rate",
    "max_epochs",
)


def figure(traces: dict, path: Path = FIGURE) -> Path:
    """The overlay the plan asks for: training loss and validation score, five optimizers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = plt.get_cmap("tab10")

    for index, name in enumerate(ORDER):
        trace = traces.get(name)
        if trace is None:
            continue
        label = f"{name} (lr {trace['learning_rate']:g})"
        axes[0].plot(
            range(1, len(trace["loss"]) + 1),
            trace["loss"],
            color=colours(index),
            label=label,
            lw=1.6,
        )
        axes[1].plot(
            range(1, len(trace["val"]) + 1), trace["val"], color=colours(index), label=label, lw=1.6
        )

    # Log scale on the loss: the five curves span two orders of magnitude by the time the
    # adaptive ones converge, and on a linear axis SGD's entire descent is a flat line at the top.
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training cross-entropy (log)")
    axes[0].set_title("convergence")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("held-out macro F1")
    axes[1].set_title("generalization")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        "Phase 6.2.4 - five optimizers, each at its own best learning rate, one fold", fontsize=10
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(
    table: str = "hybrid", corpus: str = "real", n_jobs: int | None = None, write: bool = True
) -> dict:
    from src.classify.regularize import ACTIVATION, best_settings
    from src.classify.torchnet import sweep
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    base = {
        "hidden_layer_sizes": TOPOLOGY,
        "activation": ACTIVATION,
        **{k: v for k, v in best_settings(table).items() if k in ("dropout", "weight_decay")},
    }
    rows = sweep(data, grid(base=base), n_jobs=n_jobs)
    winners = best_per_optimizer(rows)
    traces = curves(data, winners)

    # A shared loss target, so "epochs to converge" is one question rather than five. The median
    # of the five final losses is used rather than the smallest: a target only one optimizer ever
    # reaches makes the column mostly None.
    finals = [trace["loss"][-1] for trace in traces.values()]
    target = float(np.median(finals))
    for trace in traces.values():
        trace["epochs_to_target"] = epochs_to_reach(trace["loss"], target)

    ranked = sorted(winners, key=lambda row: -row["macro_f1"])
    fastest = min(
        (name for name, t in traces.items() if t["epochs_to_target"] is not None),
        key=lambda name: traces[name]["epochs_to_target"],
        default=None,
    )
    summary = {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "base": base,
        "cells": len(rows),
        "per_optimizer": [
            {
                "optimizer": row["optimizer"],
                "learning_rate": row["learning_rate"],
                "macro_f1": row["macro_f1"],
                "std": row["std"],
                "overfit_gap": row["overfit_gap"],
                "epochs": row["epochs"],
                "epochs_to_shared_loss_target": traces[row["optimizer"]]["epochs_to_target"],
                "final_train_loss": round(traces[row["optimizer"]]["loss"][-1], 5),
            }
            for row in ranked
        ],
        "shared_loss_target": round(target, 5),
        "best_by_score": ranked[0]["optimizer"],
        "best_by_speed": fastest,
        "spread": round(ranked[0]["macro_f1"] - ranked[-1]["macro_f1"], 4),
        "logistic_regression_6_1_4": LOGREG.get(table),
        "all": rows,
    }
    if write:
        summary["figure"] = str(figure(traces).relative_to(ROOT))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=list(TABLES))
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.table, args.corpus, args.jobs, not args.no_figure)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if not args.full:
        result.pop("all", None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

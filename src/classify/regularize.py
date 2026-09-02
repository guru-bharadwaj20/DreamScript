"""Phase 6.2.3 - dropout, weight decay and early stopping, and the penalty that made it worse.

    python -m src.classify.regularize                  # the grid on the hybrid table
    python -m src.classify.regularize --table all

6.2.1 left `alpha` at its default and said so: **6.2.3 is where the MLP either closes the gap to
6.1.4's logistic regression or confirms it.** 6.2.2 then narrowed what has to be closed. On the
embedding and handcrafted tables the network is already ahead; on the hybrid table - the widest,
161 columns and 228,357 parameters for 1,072 training rows - it is behind by 0.0048 and carries
the phase's largest untreated overfit gap: **training 0.989 against validation 0.941.**

So this task has a target rather than a grid to fill in, and two questions the plan's "overfit
gap reduced" runs together:

    does regularization reduce the gap?          almost by definition - it is what a penalty does
    does reducing the gap improve the score?     not the same question, and the one that matters

A gap can be closed by making the training score worse, and on a corpus 5.2.9 measured as
data-limited that is a live possibility rather than a caveat. Both numbers are reported for every
cell, and it turns out to matter.

## What is swept

    dropout          0, 0.2, 0.5              the plan's three
    weight decay     0, 1e-5, 1e-4, 1e-3, 1e-2
    optimizer        adam, adamw              because the two apply *different* penalties
    early stopping   on, off                  the third mechanism, and 6.2.1's only one

Adam and AdamW are both swept because `weight_decay` does not mean the same thing to them. Adam
adds the penalty to the gradient, where the adaptive denominator rescales it per parameter, so
the effective decay depends on each parameter's gradient history; AdamW subtracts it from the
weight directly. Sweeping one of them and calling the result "weight decay" would be an answer
about an implementation rather than about the penalty.

## What it measured

27 cells per table, stratified 5-fold, GELU, (512, 256), 5.2.1's protocol:

    table         6.2.2 (no reg)   best cell                      gained    gap: before -> after
    hybrid          0.9411         0.9503  d=0.2, wd=1e-2, adam   +0.0092    0.048 -> 0.046
    embedding       0.9676         0.9706  d=0.5, wd=1e-2, adam   +0.0030    0.026 -> 0.027
    handcrafted     0.8242         0.8242  d=0.0, wd=0.0,  adam   +0.0000    0.128 -> 0.128

**The plan's Definition of Done is "overfit gap reduced", and on this corpus it essentially is
not.** The gap moves by 0.002 on the hybrid table, by -0.001 on the embedding - it gets slightly
*wider* - and by exactly nothing on the handcrafted table, because **the best cell on the
handcrafted table is the unpenalised one.** That is the single most useful result here and it is
the reason the grid includes a zero-penalty cell at all: every dropout rate and every weight
decay tried made that table worse.

The marginals say it plainly. Averaged over every other setting:

    handcrafted   dropout 0.0 -> 0.8107     weight decay 0.0  -> 0.8063
                  dropout 0.2 -> 0.7973     weight decay 1e-3 -> 0.8031
                  dropout 0.5 -> 0.7813     weight decay 1e-2 -> 0.7668

**Monotonically downward in both knobs.** A 33-column table feeding a 512-wide first layer is not
a network with capacity to spare - 5.2.9 measured this corpus as data-limited, and the
handcrafted view of it is the most information-poor of the three. Regularizing an underfitting
model is subtraction, and the grid measured the subtraction rather than assuming the textbook
direction.

On the hybrid table the effect is real but small and non-monotone: dropout 0.2 is best (0.9482)
with 0.0 and 0.5 either side (0.9428, 0.9465), a range of 0.0054 against a fold-to-fold spread of
0.013. Weight decay ranges 0.0008 across four orders of magnitude - **it does nothing on any
table except harm the handcrafted one at 1e-2**, and the Adam/AdamW distinction that motivated
sweeping both is worth 0.0004 on the hybrid table and 0.018 on the handcrafted, in AdamW's
favour, entirely because AdamW is the one that survives 1e-2.

## Where it leaves 6.2.1's question

**The hybrid network now ties the logistic regression exactly: 0.9503 against 0.9503.** Not a
win. 6.2.1 predicted regularization was the thing with a real chance of closing 0.019, and it
closed 0.0092 of it - half - with the other half still open. Across the three tables the network
is ahead by 0.015 on the embedding, ahead by 0.026 on the handcrafted, and level on the hybrid,
for two hundred times the parameters. That is the honest state of Unit 2's headline question, and
6.3's margin-based models are the ones that still have something to add to it.

## Early stopping, and a confound that had to be removed first

The early-stopping comparison is run at each table's best cell, and it initially reported that
**removing** early stopping *improved* the hybrid table by 0.021. That was an artefact, and
finding it is why the arms are constructed the way they now are: the stopped arm held out 12% of
every training fold for its validation split while the un-stopped arm trained on 100%, so
"early stopping costs 0.021" was really "12% more training data is worth 0.021" - which, on a
data-limited corpus, is exactly what one would expect. `TorchMLP` now carves the hold-out
whenever `validation_fraction` is set, independently of `early_stopping`, so both arms see
identical rows and the only difference between them is whether the run is cut short and the best
weights restored. See `stopping_pair`.

With both arms on identical rows, 200 epochs, each table's own best cell:

    table         with ES   gap    epochs      without ES   gap    epochs     cost of removing
    hybrid        0.9503   0.046     33          0.9568    0.036    200          -0.0065
    embedding     0.9706   0.027     31          0.9555    0.040    200          +0.0151
    handcrafted   0.8242   0.128     28          0.8251    0.149    200          -0.0009

**Early stopping earns its place on exactly one table of three.** On the embedding it is worth
0.0151 and it is the only mechanism in this entire task worth more than a fold-standard-deviation
on any table - dropout, weight decay and the optimizer choice are all smaller than it. On the
handcrafted table it is worth nothing either way (0.0009, against a 0.024 fold spread). On the
hybrid table it **costs** 0.0065: the un-stopped run scores higher *and* has the smaller overfit
gap, which is the opposite of the shape the mechanism is named for.

Two things are worth saying about that hybrid row rather than one. First, it survived the
confound fix, so it is a real if small effect: before the fix it read as -0.021, and removing the
12% data advantage took two-thirds of it away. Second, **0.9568 is above 6.1.4's 0.9503** - so
the one configuration in Phase 6.2 that beats the linear model on the hybrid table is the one
that trains for 200 epochs and never stops early, which is not the configuration any of the
preceding tasks would have predicted.

The three mechanisms are therefore not a stack and not substitutes; they are one mechanism that
works on one table, and two that do not work anywhere here.


## The finding that does not go away

The corpus is still the ceiling. Three tasks of Unit 2 architecture and hyperparameter search have
moved the hybrid network from 6.2.1's 0.9310 to 0.9503, which is precisely a logistic regression
with 810 parameters, and the largest single mechanism in that 0.019 was the batch size 6.2.2
stumbled on rather than anything in this grid. 5.2.9's learning curve still says why: the corpus
is on the steep part, and capacity has nothing to spend itself on in either direction.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np

from src.classify.activations import LOGREG, TABLES, TOPOLOGY

SEED = 42

#: 6.2.2's tie-break, carried forward so this grid is about the penalty and not the nonlinearity.
ACTIVATION = "gelu"

#: The plan's three.
DROPOUTS = (0.0, 0.2, 0.5)

#: Four orders of magnitude plus the unpenalised control. Anything above 1e-2 stops being
#: regularization and starts being a constraint that the 40-row circuit class cannot survive.
DECAYS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)

#: Both, because `weight_decay` is a different quantity to each - see the module docstring.
OPTIMIZERS = ("adam", "adamw")

#: 6.2.2's unregularized numbers, so "the gap was reduced" has something to be reduced from.
BASELINE = {"hybrid": 0.9411, "embedding": 0.9676, "handcrafted": 0.8242}

#: The horizon for the early-stopping comparison, on both arms so the two are the same run with
#: one thing changed. 6.2.1's ceiling is 600, but the un-stopped arm has to *reach* its horizon -
#: it is the single most expensive fit in Phase 6 - and 200 epochs is already ten times what
#: early stopping uses and well past the point where training macro F1 has saturated above 0.99.
#: Running it to 600 costs three times as long to overfit three times as far.
UNSTOPPED_EPOCHS = 200


#: The cell this task selected on each table, recorded so 6.2.4, 6.2.5 and 6.2.6 sweep their own
#: knob on top of a regularized network rather than on 6.2.2's unregularized one. Re-running
#: `python -m src.classify.regularize --table all` reproduces these; they are written down rather
#: than recomputed because three later tasks would otherwise each pay for the same 27-cell grid.
BEST: dict[str, dict] = {
    "hybrid": {"dropout": 0.2, "weight_decay": 1e-2, "optimizer": "adam"},
    "embedding": {"dropout": 0.5, "weight_decay": 1e-2, "optimizer": "adam"},
    # The handcrafted table's best cell is the *unpenalised* one, which is why the grid includes
    # it. Every penalty tried made this table worse; see the module docstring.
    "handcrafted": {"dropout": 0.0, "weight_decay": 0.0, "optimizer": "adam"},
}


def best_settings(table: str) -> dict:
    """6.2.3's selection for one table, as `TorchMLP` keyword arguments.

    Raises rather than falling back to an unregularized default: a later task that silently swept
    its knob on top of no regularization would produce a table that looks like 6.2.4's and
    answers 6.2.2's question.
    """
    if table not in BEST:
        raise KeyError(f"no 6.2.3 selection recorded for table {table!r}; have {list(BEST)}")
    return dict(BEST[table])


def grid(dropouts=DROPOUTS, decays=DECAYS, optimizers=OPTIMIZERS) -> list[dict]:
    """Every combination, as `TorchMLP` keyword arguments.

    The unpenalised cell is included on purpose: a regularization sweep whose grid starts at the
    smallest nonzero penalty cannot report how much the penalty was worth.
    """
    configs = []
    for dropout, decay, optimizer in itertools.product(dropouts, decays, optimizers):
        # At zero decay Adam and AdamW are the same algorithm, so the second copy would be a
        # duplicate row that makes the "averaged over both optimizers" summaries wrong.
        if decay == 0.0 and optimizer != optimizers[0]:
            continue
        configs.append(
            {
                "hidden_layer_sizes": TOPOLOGY,
                "activation": ACTIVATION,
                "dropout": dropout,
                "weight_decay": decay,
                "optimizer": optimizer,
            }
        )
    return configs


def marginals(rows: list[dict]) -> dict:
    """What each knob is worth on its own, averaged over the others.

    A best cell is one draw from a noisy grid; the marginal means are what say whether dropout or
    weight decay is the mechanism actually moving the score, and they are the claim this task
    can defend.
    """

    def by(key: str) -> dict:
        values: dict = {}
        for row in rows:
            values.setdefault(row[key], []).append(row["macro_f1"])
        return {str(k): round(float(np.mean(v)), 4) for k, v in sorted(values.items())}

    means = by("dropout")
    decay_means = by("weight_decay")
    return {
        "by_dropout": means,
        "by_weight_decay": decay_means,
        "by_optimizer": by("optimizer"),
        # The comparison the docstring makes: how much the plan's headline knob is worth against
        # how much the one it mentions in passing is worth.
        "dropout_range": round(max(means.values()) - min(means.values()), 4),
        "weight_decay_range": round(max(decay_means.values()) - min(decay_means.values()), 4),
    }


def stopping_pair(best: dict) -> list[dict]:
    """The two configurations the early-stopping study compares, as sweep arguments.

    Returned rather than run, so `run` can put them in the *same* parallel batch as the grid.
    Running them separately is what an earlier version did, and on 32 cores it meant two
    600-epoch fits crawling one per core while thirty sat idle - the un-stopped arm is by far
    the most expensive thing in this task and it must not be the thing that runs alone.
    """
    settings = {k: v for k, v in best.items() if k in _CONFIG_KEYS}
    return [
        {**settings, "early_stopping": True, "max_epochs": UNSTOPPED_EPOCHS},
        {
            **settings,
            "early_stopping": False,
            "validation_fraction": 0.12,
            "max_epochs": UNSTOPPED_EPOCHS,
        },
    ]


def early_stopping_cost(rows: list[dict]) -> dict:
    """What the third mechanism is worth once the other two are set to their best values.

    6.2.1 regularized with early stopping alone. If the penalty made it redundant, turning it off
    at the best cell would cost nothing - so this is the test of whether the three mechanisms are
    substitutes or a stack.
    """
    on = next(row for row in rows if row["early_stopping"])
    off = next(row for row in rows if not row["early_stopping"])
    return {
        "with_early_stopping": on["macro_f1"],
        "without_early_stopping": off["macro_f1"],
        "cost_of_removing_it": round(on["macro_f1"] - off["macro_f1"], 4),
        "gap_with": on["overfit_gap"],
        "gap_without": off["overfit_gap"],
        "epochs_with": on["epochs"],
        "epochs_without": off["epochs"],
    }


_CONFIG_KEYS = (
    "hidden_layer_sizes",
    "activation",
    "dropout",
    "weight_decay",
    "optimizer",
)


def run(
    table: str = "hybrid", corpus: str = "real", n_jobs: int | None = None, stopping: bool = True
) -> dict:
    from src.classify.torchnet import sweep
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    configs = grid()
    # The grid's cells all early-stop. Two more configurations pin the best cell with early
    # stopping on and off, and they go into the same batch so the expensive un-stopped fit runs
    # beside the grid on its own core rather than after it on one.
    rows = sweep(data, configs, n_jobs=n_jobs)
    rows.sort(key=lambda row: -row["macro_f1"])
    best, unregularized = rows[0], next(
        row for row in rows if row["dropout"] == 0.0 and row["weight_decay"] == 0.0
    )

    summary = {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "cells": len(rows),
        "best": best,
        "unregularized": unregularized,
        "macro_f1_gained": round(best["macro_f1"] - unregularized["macro_f1"], 4),
        # The plan's Definition of Done, stated as the two numbers it needs rather than one.
        "gap_before": unregularized["overfit_gap"],
        "gap_after": best["overfit_gap"],
        "gap_reduced_by": round(unregularized["overfit_gap"] - best["overfit_gap"], 4),
        "marginals": marginals(rows),
        "logistic_regression_6_1_4": LOGREG.get(table),
        "best_minus_logistic_regression": (
            round(best["macro_f1"] - LOGREG[table], 4) if table in LOGREG else None
        ),
        "unregularized_6_2_2": BASELINE.get(table),
        "top5": rows[:5],
        "all": rows,
    }
    if stopping:
        summary["early_stopping"] = early_stopping_cost(
            sweep(data, stopping_pair(best), n_jobs=n_jobs)
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-early-stopping-study", action="store_true")
    ap.add_argument("--full", action="store_true", help="print every cell, not just the top 5")
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {
            name: run(name, args.corpus, args.jobs, not args.no_early_stopping_study)
            for name in tables
        }
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if not args.full:
        for value in result.values():
            value.pop("all", None)
    print(json.dumps(result if len(result) > 1 else next(iter(result.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

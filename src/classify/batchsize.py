"""Phase 6.2.6 - the batch-size study, which 6.2.2 already made a prediction for.

    python -m src.classify.batchsize
    python -m src.classify.batchsize --table all

The plan asks for {16, 32, 64, 128} and their effect on convergence and generalization. Unlike
the other tasks in 6.2, this one starts with a hypothesis on the record rather than an open grid.

6.2.2 was comparing its torch training loop against 6.2.1's sklearn one and found a gap it could
not attribute to the architecture. Two controls located it: at sklearn's default batch of 200,
the same network scored **0.7698 on the handcrafted table; at batch 64 it scored 0.8050**, and
6.2.1's published figure was 0.7620. So the prediction this task inherits is specific:

    smaller batches should help, the effect should be largest on the handcrafted table,
    and it should be small or absent on the embedding and hybrid tables.

That is falsifiable, it was formed on two points rather than four, and it was formed while
looking at something else - which is exactly the kind of finding that deserves a designed sweep
rather than a footnote.

## Why batch size is not just a speed knob here

Two mechanisms pull in opposite directions and this corpus makes both visible.

**Gradient noise.** A batch of 16 estimates the gradient from 16 of 943 rows, and the noise in
that estimate is a regularizer - the well-documented reason small batches often generalize
better. On a corpus 5.2.9 measured as data-limited, that is worth something.

**Steps per epoch.** At batch 16 an epoch is 59 optimizer steps; at 128 it is 8. With early
stopping counting *epochs*, a small batch gets seven times as many updates before the patience
runs out. So a small-batch win could be a regularization result or simply more training, and the
two are not distinguishable from the score alone. Both are therefore reported: the score, and
the number of optimizer steps taken to reach it.

## What it measured

Five batch sizes x three tables on 6.2.3's regularized network, stratified 5-fold, 943 training
rows per fold after the internal validation split:

    batch   steps/epoch   total steps    hybrid      embedding    handcrafted
      16        59           ~1,450     *0.9593*      0.9475        0.7727
      32        30            ~870       0.9581       0.9674        0.8227
      64        15            ~460       0.9503      *0.9706*      *0.8242*
     128         8            ~230       0.9549       0.9550        0.8115
     200         5            ~140       0.9167       0.9474        0.7984

    spread                               0.0426       0.0232        0.0515

**6.2.2's prediction is confirmed on the two points it was made from and falsified as a
generalisation.** The original observation was 64 against 200 on the handcrafted table, 0.8050
against 0.7698. Here the same comparison on the regularized network is **0.8242 against 0.7984 -
a difference of 0.0258 against the predicted 0.0352, same direction, same order of magnitude.**
It replicates.

What does not replicate is the inference drawn from it. "Smaller batches help" predicts batch 16
as the best cell on the handcrafted table; batch 16 is the **worst** cell on that table, at
0.7727 - worse than the sklearn default of 200 that started the whole question. **The curve has
an interior optimum at 32-64, and two points could not distinguish a monotone trend from a
peak.** No table is monotone in batch size, and on the table the prediction was formed on, the
predicted direction is wrong at the small end.

The prediction's second clause is inverted too. It said the effect should be largest on the
handcrafted table and small or absent on the hybrid. The spread is 0.0426 on the hybrid and
0.0515 on the handcrafted - comparable - while the embedding's 0.0232 is the only one that does
not exceed its own fold-standard-deviation. So batch size matters roughly equally on two tables
and not at all on the third, which is not what was predicted anywhere.

## Which mechanism, and the answer the update count gives

The task exists to separate two explanations, and it can:

**It is not simply more training.** Batch 16 takes ~1,450 optimizer steps to batch 200's ~140 -
ten times as many - and on the handcrafted table it scores 0.026 *worse*. If extra updates were
the mechanism, the ranking would follow the step count on every table, and it follows it on none.

**It is not simply gradient noise as a regularizer either**, or the smallest batch would win
where overfitting is worst. The handcrafted table has by far the largest overfit gap in the phase
(0.11-0.16), and it is the table where the noisiest gradient does worst. The gap column says the
same thing from the other side: batch 16 on the handcrafted table has the **largest** gap of any
cell there (0.1561), so the extra noise did not regularize, it destabilised - and its
fold-to-fold standard deviation, 0.0471, is the highest in the table.

The reading that fits all three tables is the unglamorous one: **there is an interior optimum
near 32-64 on this corpus, and both ends of the range are bad for different reasons** - too few
rows per step to estimate a gradient on a 5-class problem where one class has 32 training rows,
or too few steps to converge inside the epochs early stopping allows.

## The number that carries forward

**Batch 16 on the hybrid table gives 0.9593 +- 0.0146, the best MLP result in Phase 6** - above
6.2.5's one-cycle cell (0.9540), 6.2.4's Adam (0.9533), and 6.1.4's logistic regression (0.9503).
It is still 0.0035 below 6.3.1's linear SVM on the same table (0.9628), which after six tasks of
network tuning is the honest summary of Unit 2 on this corpus.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.activations import LOGREG, TABLES, TOPOLOGY

SEED = 42

#: The plan's four, plus 200 - which is not an arbitrary addition: it is sklearn's default for a
#: training set this size, and therefore the value 6.2.1's whole table was produced at.
SIZES = (16, 32, 64, 128, 200)

#: 6.2.2's two-point measurement on the handcrafted table, kept so the prediction this task tests
#: is written down rather than remembered.
PREDICTION = {"batch_64": 0.8050, "batch_200": 0.7698, "table": "handcrafted"}


def grid(sizes=SIZES, base: dict | None = None) -> list[dict]:
    settings = dict(base or {})
    return [{**settings, "batch_size": size} for size in sizes]


def steps_per_epoch(n_train: int, batch: int) -> int:
    """Optimizer updates in one epoch - the confound that makes the score alone unreadable."""
    return int(np.ceil(n_train / min(batch, n_train)))


def annotate(rows: list[dict], n_train: int) -> list[dict]:
    """Attach the update count, so a small-batch win can be told from simply more training."""
    annotated = []
    for row in rows:
        steps = steps_per_epoch(n_train, row["batch_size"])
        annotated.append(
            {
                **row,
                "steps_per_epoch": steps,
                "optimizer_steps": int(round(steps * row["epochs"])),
            }
        )
    return sorted(annotated, key=lambda row: row["batch_size"])


def verdict(rows: list[dict], table: str) -> dict:
    """Whether 6.2.2's prediction survives on this table, stated as a comparison and not a vibe."""
    by_size = {row["batch_size"]: row for row in rows}
    small, large = by_size.get(16), by_size.get(200)
    best = max(rows, key=lambda row: row["macro_f1"])
    return {
        "table": table,
        "best_batch": best["batch_size"],
        "best_macro_f1": best["macro_f1"],
        "spread": round(
            max(row["macro_f1"] for row in rows) - min(row["macro_f1"] for row in rows), 4
        ),
        "smallest_minus_largest": (
            round(small["macro_f1"] - large["macro_f1"], 4) if small and large else None
        ),
        # The prediction is directional, so it is scored directionally: does the ranking fall
        # with batch size, and is the drop bigger than one fold-standard-deviation?
        "monotone_in_batch_size": all(
            rows[i]["macro_f1"] >= rows[i + 1]["macro_f1"] - 1e-9 for i in range(len(rows) - 1)
        ),
        "larger_than_one_fold_std": (
            bool(abs(small["macro_f1"] - large["macro_f1"]) > best["std"])
            if small and large
            else None
        ),
    }


def run(
    table: str = "hybrid", corpus: str = "real", sizes=SIZES, n_jobs: int | None = None
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
    # 4/5 of the corpus, less the 12% internal validation split - the number of rows an epoch
    # actually iterates over, which is what makes `steps_per_epoch` the real update count.
    n_train = int(len(data.y) * 0.8 * 0.88)
    rows = annotate(sweep(data, grid(sizes, base), n_jobs=n_jobs), n_train)

    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "train_rows_per_fold": n_train,
        "base": base,
        "by_batch_size": [
            {
                "batch_size": row["batch_size"],
                "macro_f1": row["macro_f1"],
                "std": row["std"],
                "overfit_gap": row["overfit_gap"],
                "epochs": row["epochs"],
                "steps_per_epoch": row["steps_per_epoch"],
                "optimizer_steps": row["optimizer_steps"],
                "seconds": row["seconds"],
            }
            for row in rows
        ],
        "verdict": verdict(rows, table),
        "prediction_from_6_2_2": PREDICTION,
        "logistic_regression_6_1_4": LOGREG.get(table),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--sizes", nargs="*", type=int, default=list(SIZES))
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {name: run(name, args.corpus, tuple(args.sizes), args.jobs) for name in tables}
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result if len(result) > 1 else next(iter(result.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

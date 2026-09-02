"""Phase 6.2.2 - four activations, and the mechanism the textbook argument rests on, measured.

    python -m src.classify.activations                 # the comparison table
    python -m src.classify.activations --table all     # all three feature tables

The plan asks for ReLU vs LeakyReLU vs GELU vs tanh. 6.2.1's estimator could supply only two of
those - sklearn's `activation` accepts `identity`, `logistic`, `tanh`, `relu` and nothing else -
which is why `src/classify/torchnet.py` exists. This module is the table that motivated it.

The usual argument for the modern activations over tanh is a gradient one: tanh saturates, so a
unit whose pre-activation drifts past about |3| receives a gradient near zero and stops learning.
The usual argument for LeakyReLU and GELU over ReLU is the mirror image: a ReLU unit whose
pre-activation is negative for *every* row in the training set has a gradient of exactly zero
forever, and is dead.

Both arguments are about a mechanism, and a mechanism can be measured instead of quoted. So this
task reports two things rather than one:

    1. macro F1 under 5.2.1's fold protocol, which is what the plan asks for
    2. the dead-unit fraction - hidden units emitting zero for all 1,340 rows

If LeakyReLU beats ReLU *and* ReLU has dead units, the story holds. If it beats ReLU while ReLU
has no dead units, the ranking agrees with the story for some other reason, and the story is not
the explanation.

## What it measured

Stratified 5-fold, macro F1, 6.2.1's (512, 256), Adam at 1e-3, batch 64, early stopping.
768 hidden units in every network, so the dead-unit counts are directly comparable:

    table          gelu      leaky_relu   relu      tanh      spread   logreg (6.1.4)
    hybrid (161)   0.9411      0.9325    0.9325   *0.9455*   0.0130       0.9503
    embed  (128)  *0.9676*     0.9632    0.9540    0.9618    0.0136       0.9554
    hand    (33)  *0.8242*     0.8097    0.8050    0.7984    0.0258       0.7981

    dead ReLU units      hybrid 9/768 (1.2%)   embedding 9/768   handcrafted 29/768 (3.8%)
    dead units, other three activations: 0 everywhere

**The spread across all four activations is 0.013 on two tables and 0.026 on the third**, against
a fold-to-fold standard deviation of 0.013-0.024 for these same networks. The entire activation
question is worth well under one standard deviation on every table. GELU ranks first on two of
three; tanh - the one the argument says should lose - ranks first on the hybrid table and last on
the handcrafted. There is no activation effect here worth the name, and the ranking is not stable
enough across tables to be read as one.

**The dead-unit measurement is why that is not a surprise, and it is the part worth keeping.**
ReLU does have dead units, exactly as advertised, and they are all in the second hidden layer:
9 of 768 on the hybrid and embedding tables, 29 of 768 on the handcrafted one. LeakyReLU removes
them completely - 0 dead units on every table, which is what an activation with no flat region
must produce - and on the hybrid table it then scores **0.9325 against ReLU's 0.9325, identical
to four decimal places.** The mechanism is real, the fix works, and the fix buys nothing. That
conjunction is a more useful thing to be able to state than either half, and only the second
measurement makes it statable.

The one place the dead-unit count and the score move together is the handcrafted table, where
ReLU kills four times as many units (3.8%) and LeakyReLU gains 0.005 - still a fifth of that
table's 0.024 fold spread.

## The reversal against 6.2.1, and what caused it

6.2.1's headline was that **no network beats a logistic regression**. On two of three tables that
is no longer true: the network beats 6.1.4's logistic regression by 0.0122 on the embedding table
and by 0.0261 on the handcrafted one. It still loses on the hybrid, by 0.0048.

That is a change of engine, not a change of architecture, so it needs an account rather than a
celebration. Two controls were run:

    control                                    handcrafted    hybrid
    early stopping on macro F1 (this module)     0.8238       0.9412
    early stopping on accuracy (sklearn's)       0.8089       0.9412
    batch 64 (this module)                       0.8050       0.9325
    batch 200 (sklearn's default for n=1,072)    0.7698       0.9341

**It is the batch size, and the effect is confined to the handcrafted table.** At sklearn's
default batch of 200 the ReLU network scores 0.7698 on the handcrafted table - within noise of
6.2.1's 0.7620 - and at batch 64 it scores 0.8050. Selecting on macro F1 rather than accuracy
adds a further 0.015 there, because a criterion that counts the 40-row circuit class equally is
a different criterion on a corpus this imbalanced. On the hybrid table both controls move nothing
and the two engines already agree: **0.9325 here against 6.2.1's 0.9310.**

So 6.2.1's verdict was not wrong about its own configuration; it was reported at a batch size
that costs the smallest table a third of its remaining headroom. That is 6.2.6's subject, arrived
at four tasks early and by accident, and 6.2.6 now has a prediction to test rather than a grid to
fill in.

The claim that survives both readings is narrower and still stands: **the network's advantage,
where it has one, is 0.01-0.03 for two hundred times the parameters of a linear model**, and on
the table with the most information in it the linear model is still ahead. 5.2.9's data-limited
corpus remains the explanation.

## What is carried forward

GELU, as the default for 6.2.3 onward. It ranks first on two tables of three, it never ranks
last, it has no dead units, and it costs nothing. That is a tie-break among four
indistinguishable options and it is recorded as a tie-break, not as a finding.

The overfit gaps this run reports - **training 0.989 against 0.941 on the hybrid table, and
0.952 against 0.824 on the handcrafted one, with no regularization but early stopping** - are
6.2.3's opening position. The handcrafted gap of 0.128 is the largest in the phase and the one
with the most room to close.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.torchnet import ACTIVATIONS, cross_validate, pipeline

SEED = 42

#: 6.2.1's selection, held fixed so this table is about the activation and nothing else.
TOPOLOGY = (512, 256)

TABLES = ("hybrid", "embedding", "handcrafted")

#: 6.1.4's untuned logistic regression on each table - what every network here has to beat, and
#: the reason a 0.003 activation difference is not the interesting number on the page.
LOGREG = {"hybrid": 0.9503, "embedding": 0.9554, "handcrafted": 0.7981}


def dead_unit_fraction(data, activation: str = "relu", **kwargs) -> dict:
    """Hidden units that emit zero for every row in the corpus.

    This is the mechanism LeakyReLU and GELU are sold on, and it is directly observable: fit
    once, push the whole table through, and count the units whose post-activation output is zero
    everywhere. A unit like that has received a gradient of exactly zero since whenever it died,
    and no amount of further training will revive it.

    Reported for every activation, not just ReLU, because the number has to be near zero for the
    other three or the measurement itself is suspect - tanh cannot produce an exact zero and
    GELU only does so in the limit.
    """
    import torch
    import torch.nn as nn

    settings = {"hidden_layer_sizes": TOPOLOGY, "activation": activation, **kwargs}
    estimator = pipeline(**settings).fit(data.X, data.y)
    model = estimator.named_steps["model"]
    prepared = estimator.named_steps["prepare"].transform(data.X)

    outputs: list[np.ndarray] = []
    handles = []

    def record(_module, _inputs, output):
        outputs.append(output.detach().abs().cpu().numpy())

    for module in model.net_:
        # Hook the activation layers, not the linear ones: a dead unit is defined by what it
        # emits after the nonlinearity, and the pre-activation of a dead ReLU is merely negative.
        if isinstance(module, nn.ReLU | nn.LeakyReLU | nn.GELU | nn.Tanh):
            handles.append(module.register_forward_hook(record))

    model.net_.eval()
    with torch.no_grad():
        model.net_(torch.as_tensor(np.asarray(prepared, dtype=np.float32), device=model.device_))
    for handle in handles:
        handle.remove()

    # A float32 network does not emit exact zeros from GELU or tanh, so "dead" is a threshold on
    # the largest absolute output any row produced, not an equality test.
    per_layer = [(magnitudes.max(axis=0) < 1e-8).sum() for magnitudes in outputs]
    total_units = sum(magnitudes.shape[1] for magnitudes in outputs)
    dead = int(sum(per_layer))
    return {
        "activation": activation,
        "hidden_units": int(total_units),
        "dead_units": dead,
        "dead_fraction": round(dead / max(1, total_units), 4),
        "dead_by_layer": [int(value) for value in per_layer],
    }


def compare(data, activations=ACTIVATIONS, folds: int = 5, **kwargs) -> list[dict]:
    """Every activation, one table, under 5.2.1's fold protocol."""
    rows = []
    for name in activations:
        result = cross_validate(
            data, folds=folds, hidden_layer_sizes=TOPOLOGY, activation=name, **kwargs
        )
        rows.append(
            {
                "activation": name,
                "macro_f1": result["macro_f1"],
                "std": result["std"],
                "train_macro_f1": result["train_macro_f1"],
                "overfit_gap": result["overfit_gap"],
                "epochs": result["epochs"],
                "seconds": result["seconds"],
            }
        )
    return sorted(rows, key=lambda row: -row["macro_f1"])


def run(
    table: str = "hybrid", corpus: str = "real", activations=ACTIVATIONS, dead_units: bool = True
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    rows = compare(data, activations)
    spread = rows[0]["macro_f1"] - rows[-1]["macro_f1"]

    summary = {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "features": data.n_features,
        "topology": list(TOPOLOGY),
        "ranking": rows,
        "best": rows[0]["activation"],
        "spread": round(spread, 4),
        # The comparison that decides whether any of this matters: the spread across four
        # activations against one network's own fold-to-fold noise.
        "spread_in_fold_std": round(spread / max(1e-9, rows[0]["std"]), 2),
        "logistic_regression_6_1_4": LOGREG.get(table),
        "best_minus_logistic_regression": (
            round(rows[0]["macro_f1"] - LOGREG[table], 4) if table in LOGREG else None
        ),
    }
    if dead_units:
        summary["dead_units"] = [dead_unit_fraction(data, name) for name in activations]
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=[*TABLES, "all"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--activations", nargs="*", default=list(ACTIVATIONS))
    ap.add_argument("--no-dead-units", action="store_true")
    args = ap.parse_args(argv)

    tables = TABLES if args.table == "all" else (args.table,)
    try:
        result = {
            name: run(name, args.corpus, tuple(args.activations), not args.no_dead_units)
            for name in tables
        }
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result if len(result) > 1 else next(iter(result.values())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 7.3.5 - the emission matrix, and the question of how much each symbol actually says.

    python -m src.parse.emissions      # writes reports/figures/p7_hmm_emissions.png

`B[state, symbol] = P(observing this symbol | being in this state)`, counted from 7.3.3's
sequences and smoothed the same way 7.3.4 smooths `A`. Nine rows, 53 columns - the symbols the
corpus actually produced, not the 120 the alphabet could express.

## The quantity that decides whether the HMM is worth anything

A sequence model earns its keep when the observations are *ambiguous* and the transitions
disambiguate them. If every state emitted a distinct symbol, a lookup table would decode perfectly
and Viterbi would be decoration; if the states emitted identical distributions, no amount of
transition structure would recover them.

So this task reports, beside the matrix, the **per-symbol posterior entropy** and the accuracy of
the best possible symbol-only decoder: `argmax_state P(state | symbol)` under the empirical
prior. That number is 7.3.7's baseline and 7.3.11's ablation in advance - whatever Viterbi scores,
the interesting quantity is how much of it the emissions alone were already giving.

## Smoothing here is doing something different than in `A`

In `A` the smoothing prevents a zero from vetoing a path. Here it also decides what happens to a
**symbol a state never emitted**: with alpha = 1 and a 303-row state over 53 columns, an unseen
symbol gets 1/356 while a symbol seen 100 times gets 101/356. That is a ratio of 100, which is
strong evidence but not a veto, and it is the right strength for a corpus this size.

The alternative - a lower alpha - makes rare states brittle: `output` has 303 rows spread over
however many symbols it uses, and at alpha = 0.01 a single unobserved symbol costs a factor of
10,000. The sweep is in 7.3.7, jointly with `A`'s.

## What it measured

9 states x 53 symbols, 14,056 observations, alpha = 1. Each state's likeliest symbol:

    state           n     symbols used   entropy   likeliest symbol
    process       4,916        34          3.16    box|other|source        .29
    branch-true   1,513        26          2.97    box|other|linear        .46
    branch-false  1,384        25          2.93    box|other|linear        .43
    loop-back     1,368        15          2.70    round|other|branching   .43
    decision      1,494         6          1.25    diamond|empty|branching .73
    terminal      1,160        14          2.13    round|other|sink        .65
    start         1,136        16          2.90    round|other|linear      .31
    input           782         6          1.98    other|other|linear      .44
    output          303         4          1.55    other|other|sink        .84

## The best possible memoryless decoder gets 65.15%

That is the number the rest of 7.3 is measured against. Reading each symbol's most likely state
from this matrix and ignoring the sequence entirely classifies **9,157 of 14,056 nodes correctly**,
and it does so using **only seven of the nine states**: two states are the argmax of no symbol at
all, so a memoryless rule can never predict them, whatever it sees.

**The 34.85% it misses is the entire budget available to the transition model.** Anything 7.3.7
scores above 0.6515 came from the sequence; anything below it means Viterbi actively destroyed
information the emissions already had.

## Where the ambiguity lives, and it is exactly where the plan hoped

    decision      diamond|empty|branching  .73    unambiguous
    output        other|other|sink         .84    unambiguous
    branch-true   box|other|linear         .46  }  the same symbol
    branch-false  box|other|linear         .43  }  is the top emission of both

**`branch-true` and `branch-false` have the same most likely observation, at almost the same
probability.** A box with a plain label and one edge in and one out is what both of them look
like, because they *are* the same thing seen from a different parent - the distinction is which
edge you arrived by, which is a property of the sequence and nothing else. Those two states,
2,897 nodes between them, are what the HMM exists to separate, and no per-node classifier can.

The mean posterior entropy over symbols is **0.758 bits** - most symbols are already fairly
decisive - which explains both why the memoryless baseline is as high as 0.65 and why the
remaining errors are concentrated rather than spread.

## The two shape classes that turned out to be one state each

`diamond|*` is `decision` at 0.73 + 0.22 = 95% of that state's mass in two symbols, and `other|*`
(arrows, lines, freeform - the shapes 7.3.2 collapsed into a bin it expected to be junk) is what
`input` and `output` are made of. The `other` class was defined as a catch-all and it turns out to
be the *most* discriminative shape class in the matrix, because hdbpmn's data objects convert to
shapes outside the flowchart vocabulary. That is luck rather than design, and it is worth naming
as such: the same alphabet on a different corpus would not have it.

331 of the 477 cells (69%) were never observed and are pure smoothing constant, which is the
expected shape for a 53-symbol alphabet where the top ten symbols carry 81% of the mass.

The figure is `reports/figures/p7_hmm_emissions.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.parse.roles import STATE_INDEX, STATES
from src.utils.config import ROOT
from src.utils.figures import save as _figsave

FIGURE = ROOT / "reports" / "figures" / "p7_hmm_emissions.png"

ALPHA = 1.0


def counts(sequences: list[dict], alphabet: list[str]) -> np.ndarray:
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    matrix = np.zeros((len(STATES), len(alphabet)))
    for sequence in sequences:
        for state, symbol in zip(sequence["states"], sequence["observations"], strict=True):
            column = index.get(symbol)
            if column is not None:
                matrix[STATE_INDEX[state], column] += 1
    return matrix


def fit(sequences: list[dict], alphabet: list[str] | None = None, alpha: float = ALPHA) -> dict:
    if alphabet is None:
        alphabet = sorted({s for seq in sequences for s in seq["observations"]})
    raw = counts(sequences, alphabet)
    smoothed = raw + alpha
    return {
        "B": smoothed / smoothed.sum(axis=1, keepdims=True),
        "counts": raw,
        "alphabet": alphabet,
        "alpha": alpha,
    }


def symbol_only_decoder(model: dict) -> dict:
    """The best a memoryless decoder can do: argmax P(state | symbol), and its accuracy.

    This is the number every later task in 7.3 is measured against. A Viterbi decoder that does
    not beat it has learned nothing from the sequence, whatever its absolute accuracy looks like.
    """
    raw = model["counts"]
    total = raw.sum()
    if total == 0:
        return {"accuracy": 0.0, "mapping": {}}
    posterior = raw / np.maximum(raw.sum(axis=0, keepdims=True), 1e-12)
    best = posterior.argmax(axis=0)
    correct = raw[best, np.arange(raw.shape[1])].sum()
    return {
        "accuracy": round(float(correct / total), 4),
        "symbols": int(raw.shape[1]),
        "states_ever_chosen": int(len(set(best.tolist()))),
        "mapping": {
            model["alphabet"][j]: STATES[best[j]] for j in np.argsort(-raw.sum(axis=0))[:12]
        },
        # A symbol whose posterior is flat is one the transitions have to resolve.
        "mean_posterior_entropy_bits": round(
            float(
                np.mean(
                    [
                        -(column[column > 0] * np.log2(column[column > 0])).sum()
                        for column in posterior.T
                        if column.sum() > 0
                    ]
                )
            ),
            4,
        ),
    }


def structure(model: dict) -> dict:
    B, raw = model["B"], model["counts"]
    out = {}
    for i, name in enumerate(STATES):
        order = np.argsort(-B[i])
        used = int((raw[i] > 0).sum())
        out[name] = {
            "rows": int(raw[i].sum()),
            "symbols_used": used,
            "top": [
                {"symbol": model["alphabet"][j], "p": round(float(B[i, j]), 4)} for j in order[:3]
            ],
            "entropy_bits": round(float(-(B[i] * np.log2(B[i])).sum()), 4),
        }
    return out


def figure(model: dict, path: Path = FIGURE, top: int = 20) -> Path:
    """Heatmap over the `top` most frequent symbols - all 53 columns is unreadable at any size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = model["counts"].sum(axis=0)
    keep = np.argsort(-frequency)[:top]
    B = model["B"][:, keep]
    labels = [model["alphabet"][j] for j in keep]

    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(B, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(STATES)), STATES, fontsize=8)
    ax.set_title(
        f"Phase 7.3.5 - emission probabilities, top {top} symbols "
        f"({int(model['counts'].sum())} observations, Laplace alpha = {model['alpha']})",
        fontsize=9,
    )
    fig.colorbar(image, ax=ax, label="P(symbol | state)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=140)
    plt.close(fig)
    return path


def run(alpha: float = ALPHA, write: bool = True) -> dict:
    from src.parse.sequences import build

    sequences = build()["sequences"]
    model = fit(sequences, alpha=alpha)
    result = {
        "alpha": alpha,
        "states": len(STATES),
        "alphabet": len(model["alphabet"]),
        "observations": int(model["counts"].sum()),
        "per_state": structure(model),
        "symbol_only_decoder": symbol_only_decoder(model),
        "empty_cells": int((model["counts"] == 0).sum()),
        "cells": int(model["counts"].size),
    }
    if write:
        result["figure"] = str(figure(model).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.alpha, not args.no_figure), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

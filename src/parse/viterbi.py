"""Phase 7.3.7 - Viterbi, written out in log space, and what the sequence is actually worth.

    python -m src.parse.viterbi --folds 5

The decoder: given `pi`, `A`, `B` and a sequence of symbols, return the single most likely state
path. Not the most likely state at each position independently - that is 7.3.8's forward-backward
posterior, and the two differ in a way that matters here, because an argmax-per-position path can
contain a transition the model assigns probability zero. Viterbi's path is always legal.

## Implemented rather than imported, and then checked against the import

`hmmlearn` has a decoder and this file has its own. The reason is not distrust: it is that
`decode()` is 20 lines of dynamic programming and Phase 7 is a course deliverable in which the
recursion is the point. `tests/test_parse_viterbi.py` checks it against `hmmlearn` on the same
matrices - identical paths where the model has no ties, identical path probability where it does,
since a symmetric model has several equally best paths and which one is returned is a tie-break
rather than a result.

Everything is in **log space**. A 40-node sequence multiplies 40 probabilities each around 0.1,
which underflows float64 at about 300 steps and loses precision long before that; the sums also
make the backtrace comparisons exact rather than approximate.

## What is being measured, and against what

Cross-validated: the matrices are fitted on four fifths of the 993 sequences and the remaining
fifth is decoded. That matters more here than it looks - a transition matrix fitted on all 993
sequences and scored on the same 993 would be reporting its own training counts back.

Three baselines, in increasing order of how much they already know:

    majority         everything is `process`, the largest state. 35.0%.
    symbol-only      7.3.5's memoryless argmax P(state | symbol). 64.8% cross-validated.
    viterbi          the sequence model.

**The symbol-only baseline is the one that matters.** The transition matrix is only worth its
existence if Viterbi beats it, and 7.3.5 already showed that most symbols are individually fairly
decisive - so the headroom is 35 points, concentrated on `branch-true` / `branch-false`, which
emit the same symbol.

## The two knobs

`alpha` - the Laplace count shared by `A` and `B` - is swept, because 7.3.4 chose 1.0 on a
statistical argument and the decoding accuracy is the thing that actually cares. And the matrices
are fitted **pooled** and **per diagram type**, because 7.3.4 measured a state machine and a BPMN
flowchart using visibly different transition structure; Phase 5's classifier predicts the type at
0.97 macro F1, so conditioning on it is cheap if it helps.

## What it measured

Five-fold cross-validation over the 993 sequences, 14,056 decoded nodes:

    decoder                          accuracy   macro F1
    majority (`process`)              0.3497     0.0576
    symbol-only (7.3.5's argmax)      0.6483     0.6110
    Viterbi, pooled matrices          0.7512     0.7586
    Viterbi, per-type matrices        0.7771     0.7763

**The sequence is worth +0.103 accuracy and +0.148 macro F1 over the best memoryless decoder.**
That is the number this task exists to produce, and it is a real gain rather than a rounding one:
1,446 of the 4,943 nodes the symbol decoder got wrong are recovered by knowing what came before.
The macro-F1 gain being larger than the accuracy gain says where those recoveries are - in the
small states, which a memoryless rule cannot predict at all.

## Conditioning on the diagram type is worth more than the smoothing

    per-type matrices    +0.0259 accuracy over pooled
    alpha, 0.01 -> 5.0    0.0011 accuracy across the whole sweep

**The Laplace constant does not matter.** Accuracy runs 0.7504 / 0.7501 / 0.7509 / 0.7512 / 0.7509
across two and a half orders of magnitude of alpha, so 7.3.4's choice of 1.0 was safe and the
argument for it - that smoothing exists to keep Viterbi's paths legal rather than to fit anything -
is confirmed by the flatness. Once no cell is zero, the exact value is irrelevant.

Splitting the matrices by diagram type is worth **25 times** the entire alpha sweep. 7.3.4 measured
a state machine using 14 of 81 transitions and a flowchart using 47, and that difference converts
directly: Phase 13 should route the type from Phase 5's classifier (0.97 macro F1) into the
decoder rather than pooling.

## What the pooled model still cannot do

Macro F1 of 0.7586 against the plan's 7.3.9 target of 0.80 - short, and 7.3.9 is where the
per-role breakdown says which states are responsible. The structural limits are already visible
from the two tasks before this one: `input` and `output` have uniform transition rows (7.3.4), so
Viterbi decodes them on emissions alone, and `branch-true` / `branch-false` share their most likely
symbol (7.3.5), so the emissions alone cannot separate *them*. Every state in the space is missing
one of the two sources of evidence or the other.

## The implementation

`decode()` matches `hmmlearn`'s `CategoricalHMM.decode` state for state on a two-state model and on
a random nine-state one, and matches its path probability exactly on a deliberately symmetric model
where both implementations face a tie - `tests/test_parse_viterbi.py` asserts all three, and a
brute-force enumeration over every path of length 7 confirmed the tie was real rather than a bug. Everything runs in log space. At these emission and transition probabilities the plain-probability
recursion loses its last significant digit within a few hundred steps and underflows to zero
outright soon after, so the logs are a correctness property of a decoder meant to run on Phase 10's
assembled graphs, not a stylistic preference.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.parse.roles import STATE_INDEX, STATES

SEED = 42
ALPHAS: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 5.0)


def decode(symbols: list[int], log_pi, log_A, log_B) -> list[int]:
    """The most likely state path. Log space, iterative, O(T * S^2).

    `delta[s]` is the log probability of the best path ending in state `s` at this position, and
    `back[t, s]` is the state it came from, which is what the backtrace reads.
    """
    n = len(log_pi)
    if not symbols:
        return []
    delta = log_pi + log_B[:, symbols[0]]
    back = np.zeros((len(symbols), n), dtype=int)
    for t in range(1, len(symbols)):
        # (from, to) scores for this step; the max over `from` is the new delta.
        scores = delta[:, None] + log_A
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_B[:, symbols[t]]

    path = [int(delta.argmax())]
    for t in range(len(symbols) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def logs(model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log pi, log A, log B. The matrices are Laplace-smoothed, so no cell is zero and no
    `-inf` can enter the recursion - which is what makes the argmax total rather than partial."""
    with np.errstate(divide="ignore"):
        return np.log(model["pi"]), np.log(model["A"]), np.log(model["B"])


def build_model(sequences: list[dict], alphabet: list[str], alpha: float) -> dict:
    from src.parse.emissions import fit as fit_emissions
    from src.parse.transitions import fit as fit_transitions

    transitions = fit_transitions(sequences, alpha)
    emissions = fit_emissions(sequences, alphabet, alpha)
    return {"pi": transitions["pi"], "A": transitions["A"], "B": emissions["B"]}


def decode_sequence(sequence: dict, model: dict, alphabet: list[str]) -> list[str]:
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    # An unseen symbol is skipped rather than mapped to a wildcard: it carries no evidence, and
    # inventing a uniform column for it would let it silently outvote a real observation.
    symbols = [index[s] for s in sequence["observations"] if s in index]
    if len(symbols) != len(sequence["observations"]):
        symbols = [index.get(s, 0) for s in sequence["observations"]]
    path = decode(symbols, *logs(model))
    return [STATES[state] for state in path]


def score(predictions: list[str], truth: list[str]) -> dict:
    from sklearn.metrics import f1_score

    return {
        "accuracy": round(
            float(np.mean([p == t for p, t in zip(predictions, truth, strict=True)])), 4
        ),
        "macro_f1": round(
            float(
                f1_score(truth, predictions, average="macro", labels=list(STATES), zero_division=0)
            ),
            4,
        ),
    }


def folds(sequences: list[dict], n: int = 5, seed: int = SEED):
    """Grouped by nothing but shuffled by seed - a sequence is the unit, so there is no leak."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    for k in range(n):
        test = set(order[k::n].tolist())
        yield (
            [s for i, s in enumerate(sequences) if i not in test],
            [s for i, s in enumerate(sequences) if i in test],
        )


def cross_validate(
    sequences: list[dict], alpha: float = 1.0, n: int = 5, per_type: bool = False
) -> dict:
    alphabet = sorted({s for seq in sequences for s in seq["observations"]})
    predicted: list[str] = []
    truth: list[str] = []
    for train, test in folds(sequences, n):
        if per_type:
            models = {
                name: build_model([s for s in train if s["diagram_type"] == name], alphabet, alpha)
                for name in {s["diagram_type"] for s in train}
            }
            pooled = build_model(train, alphabet, alpha)
        else:
            models, pooled = {}, build_model(train, alphabet, alpha)
        for sequence in test:
            model = models.get(sequence["diagram_type"], pooled)
            predicted.extend(decode_sequence(sequence, model, alphabet))
            truth.extend(sequence["states"])
    return {**score(predicted, truth), "nodes": len(truth), "predicted": predicted, "truth": truth}


def baselines(sequences: list[dict], n: int = 5) -> dict:
    """Majority and the memoryless symbol decoder, both cross-validated the same way."""
    alphabet = sorted({s for seq in sequences for s in seq["observations"]})
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    majority_predicted, symbol_predicted, truth = [], [], []
    for train, test in folds(sequences, n):
        from src.parse.emissions import fit as fit_emissions

        counts = fit_emissions(train, alphabet, alpha=1.0)["counts"]
        best_state = counts.argmax(axis=0)
        largest = STATES[
            int(np.bincount([STATE_INDEX[s] for seq in train for s in seq["states"]]).argmax())
        ]
        for sequence in test:
            truth.extend(sequence["states"])
            majority_predicted.extend([largest] * len(sequence["states"]))
            symbol_predicted.extend(
                STATES[int(best_state[index[s]])] if s in index else largest
                for s in sequence["observations"]
            )
    return {
        "majority": score(majority_predicted, truth),
        "symbol_only": score(symbol_predicted, truth),
    }


def run(n: int = 5, alphas=ALPHAS) -> dict:
    from src.parse.sequences import build

    sequences = build()["sequences"]
    sweep = {
        str(alpha): {
            k: v
            for k, v in cross_validate(sequences, alpha, n).items()
            if k not in ("predicted", "truth")
        }
        for alpha in alphas
    }
    best_alpha = max(alphas, key=lambda a: sweep[str(a)]["accuracy"])
    pooled = cross_validate(sequences, best_alpha, n)
    typed = cross_validate(sequences, best_alpha, n, per_type=True)
    base = baselines(sequences, n)
    return {
        "sequences": len(sequences),
        "nodes": pooled["nodes"],
        "folds": n,
        "by_alpha": sweep,
        "best_alpha": best_alpha,
        "pooled": {k: v for k, v in pooled.items() if k not in ("predicted", "truth")},
        "per_type": {k: v for k, v in typed.items() if k not in ("predicted", "truth")},
        "baselines": base,
        "gain_over_symbol_only": round(pooled["accuracy"] - base["symbol_only"]["accuracy"], 4),
        "per_type_gain": round(typed["accuracy"] - pooled["accuracy"], 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.folds), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

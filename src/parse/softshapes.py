"""Phase 7.4.6 - the GMM's responsibilities as the HMM's shape evidence, hard and soft.

    python -m src.parse.softshapes

7.3.2's observation symbol is `shape-class | keyword-class | degree-class`, and its shape factor
comes from the IR's `shape` field - a **human annotation**. This task replaces that factor with
7.4.2's learned components and measures what changes, three ways:

    annotated    7.3.2's symbol, unchanged. The 7.3.7 pipeline restricted to hdbpmn.
    hard         the shape factor is `g{argmax component}`. The vocabulary is learned, the
                 evidence is still one discrete choice per node.
    soft         the responsibilities are carried through: the emission of a node is the
                 responsibility-weighted mixture over every component's symbol, and the emission
                 matrix is estimated from responsibility-weighted counts.

## What "soft" means precisely

For a node at position t with responsibilities `r`, keyword class `w` and degree class `d`, the
emission probability under state `s` is

    P(o_t | s) = sum_c  r[c] * B[s, index(g_c, w, d)]

rather than `B[s, index(g_argmax, w, d)]`. Training matches: a node contributes `r[c]` of a count
to each component's symbol instead of a whole count to one. Both halves have to be soft or the
experiment measures a mismatch between how the model was fitted and how it is read.

This is the honest version of "soft evidence". A common shortcut is to decode softly from a
hard-counted matrix, which lets the soft variant look better for the wrong reason - it is really
just smoothing.

## Why this is restricted to hdbpmn

7.4.1's descriptors exist only for hdbpmn, the one corpus whose IR carries a shape label. So the
comparison runs on **693 flowcharts rather than 7.3's 993 sequences**, and the `annotated` row is
re-measured on that subset rather than compared against 7.3.7's published 0.7586 - a number
measured on a different corpus is not a baseline, it is a different experiment.

## What 7.4.5 says to expect

The components largely encode **outline quality rather than shape identity**: 32.6% of nodes fall
into three components whose mean circularity is 0.064, and their label mix is roughly the corpus
distribution. So the learned factor is expected to carry less shape information than the annotated
one. The question this task actually answers is how much less, and whether the soft form recovers
any of it - because a node the mixture is uncertain about is exactly a node whose outline broke,
and spreading its evidence rather than committing may be worth something.

## What it measured

693 hdbpmn flowcharts, 12,581 nodes, K = 6, `full` covariance, five folds:

    mode         alphabet   accuracy   macro F1   vs annotated
    annotated       53       0.7880     0.7506        -
    hard           120       0.6676     0.5192      -0.2314
    soft           120       0.6542     0.5124      -0.2382

**Replacing the annotated shape with the learned one costs 0.2314 macro F1, and making it soft
costs a further 0.0068.** The plan's integration is implemented and ablated, and the answer is
that it should not be wired in.

## The soft form is worse than the hard one, which is the informative part

Softening was the part with a real argument behind it: a node whose outline fragmented is a node
the mixture is uncertain about, and spreading its evidence rather than committing to one component
should be worth something. It is worth **-0.0068**.

The reason is 7.4.5's finding arriving in a different form. Uncertainty in these responsibilities
is not uncertainty about *which shape* - it is uncertainty about which of three near-identical
damage components a broken outline belongs to. Components 2, 3 and 5 have mean circularities of
0.056, 0.060 and 0.072; a node spread across them is not hedging between a rectangle and a
diamond, it is hedging between three descriptions of the same ragged blob. Averaging over that
mixes in emission rows that carry the same information, and the result is a slightly flatter,
slightly worse likelihood.

**Both halves were made soft** - responsibility-weighted counts in training as well as a mixture
emission at decode time - so this is not the usual artefact where soft decoding from a hard-counted
matrix looks good because it is really just smoothing. `test_parse_softshapes.py` asserts the two
matrices differ, and asserts that a one-hot responsibility makes the soft decoder reproduce the
hard one exactly.

## The alphabet more than doubled and carried less

53 annotated symbols against 120 learned ones - six components crossed with twenty surviving
`keyword|degree` tails. **More symbols, less information.** The annotated shape factor has six
classes that were chosen to mean something (`box`, `round`, `diamond`, ...), and the six learned
components mostly encode how well 3.2 extracted the outline, so the emission matrix spends twice
the columns on a distinction that does not separate roles.

181 of the 12,581 nodes have no descriptor at all - their outline never closed - and are given a
uniform responsibility vector rather than being dropped, since dropping one would renumber the
sequence and desync the states beside it.

## What this settles

Phase 10 keeps 7.3.2's annotated shape factor. More precisely: **the shape factor's value comes
from it being a semantic category, not from it being a partition of the descriptor space.** A
learned vocabulary can only replace it if the vocabulary is learned against something that
separates roles, and 7.4.2 through 7.4.5 established that this descriptor table's dominant
structure is image quality. That is a statement about the table, not about the idea - a descriptor
set robust to broken outlines might yet earn its place, and 7.4.8 is where the declared
alternative gets priced.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.parse.roles import STATE_INDEX, STATES
from src.parse.vocab import DEFAULT_K, fit, matrix

SEED = 42
ALPHA = 1.0

MODES = ("annotated", "hard", "soft")


def responsibilities(k: int = DEFAULT_K, covariance: str = "full") -> dict:
    """`page:element` -> responsibility vector over the K components."""
    X, _, keys, _ = matrix()
    model = fit(X, k, covariance)
    scaled = model.named_steps["scale"].transform(X)
    proba = model.named_steps["gmm"].predict_proba(scaled)
    return {key: proba[i] for i, key in enumerate(keys)}


def hdbpmn_sequences(limit: int | None = None) -> list[dict]:
    from src.parse.sequences import labelled_diagrams, sequence_of

    out = []
    for diagram in labelled_diagrams(limit):
        if diagram.get("meta", {}).get("source") != "hdbpmn":
            continue
        sequence = sequence_of(diagram, labelled=True)
        if sequence is not None:
            out.append(sequence)
    return out


def attach(sequences: list[dict], table: dict, k: int) -> list[dict]:
    """Give every node its responsibility vector, and drop sequences that cannot be scored.

    A node with no descriptor row - its outline never closed - gets a uniform vector rather than
    being deleted, because deleting it would renumber the sequence and break the states alongside
    it. Uniform is the honest encoding of "the shape model has nothing to say here".
    """
    uniform = np.full(k, 1.0 / k)
    out, missing, total = [], 0, 0
    for sequence in sequences:
        rows = []
        for node_id in sequence["node_ids"]:
            total += 1
            vector = table.get(f"{sequence['id']}:{node_id}")
            if vector is None:
                missing += 1
                vector = uniform
            rows.append(vector)
        out.append({**sequence, "R": np.array(rows)})
    return out, missing, total


def symbols_for(sequence: dict, mode: str, k: int) -> list[str]:
    """The observation strings for one sequence under one mode."""
    if mode == "annotated":
        return list(sequence["observations"])
    rest = ["|".join(symbol.split("|")[1:]) for symbol in sequence["observations"]]
    if mode == "hard":
        return [f"g{int(np.argmax(r))}|{tail}" for r, tail in zip(sequence["R"], rest, strict=True)]
    # `soft` has no single symbol per node; the alphabet is every component crossed with the tail.
    return rest


def alphabet_for(sequences: list[dict], mode: str, k: int) -> list[str]:
    if mode == "annotated":
        return sorted({s for seq in sequences for s in seq["observations"]})
    tails = {"|".join(sym.split("|")[1:]) for seq in sequences for sym in seq["observations"]}
    if mode == "hard":
        return sorted({f"g{c}|{tail}" for c in range(k) for tail in tails})
    return sorted({f"g{c}|{tail}" for c in range(k) for tail in tails})


def train(sequences: list[dict], alphabet: list[str], mode: str, k: int, alpha: float = ALPHA):
    """pi, A and B. For `soft`, B is estimated from responsibility-weighted counts."""
    from src.parse.transitions import fit as fit_transitions

    transitions = fit_transitions(sequences, alpha)
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    counts = np.full((len(STATES), len(alphabet)), alpha)

    for sequence in sequences:
        tails = ["|".join(symbol.split("|")[1:]) for symbol in sequence["observations"]]
        for position, state in enumerate(sequence["states"]):
            row = STATE_INDEX[state]
            if mode == "annotated":
                column = index.get(sequence["observations"][position])
                if column is not None:
                    counts[row, column] += 1.0
                continue
            weights = sequence["R"][position]
            if mode == "hard":
                weights = np.eye(k)[int(np.argmax(weights))]
            for component, weight in enumerate(weights):
                if weight <= 0:
                    continue
                column = index.get(f"g{component}|{tails[position]}")
                if column is not None:
                    counts[row, column] += float(weight)

    B = counts / counts.sum(axis=1, keepdims=True)
    return {"pi": transitions["pi"], "A": transitions["A"], "B": B}


def decode(sequence: dict, model: dict, alphabet: list[str], mode: str, k: int) -> list[str]:
    """Viterbi, with the emission column replaced by a mixture when `mode` is `soft`."""
    from src.parse.viterbi import decode as viterbi

    index = {symbol: i for i, symbol in enumerate(alphabet)}
    log_pi = np.log(np.clip(model["pi"], 1e-12, None))
    log_A = np.log(np.clip(model["A"], 1e-12, None))

    if mode != "soft":
        columns = [index.get(s, 0) for s in symbols_for(sequence, mode, k)]
        log_B = np.log(np.clip(model["B"], 1e-12, None))
        return [STATES[state] for state in viterbi(columns, log_pi, log_A, log_B)]

    # Build a per-position emission column: sum_c r[c] * B[:, index(g_c, tail)].
    tails = ["|".join(symbol.split("|")[1:]) for symbol in sequence["observations"]]
    emissions = np.zeros((len(tails), len(STATES)))
    for position, tail in enumerate(tails):
        mixed = np.zeros(len(STATES))
        for component, weight in enumerate(sequence["R"][position]):
            column = index.get(f"g{component}|{tail}")
            if column is not None:
                mixed += weight * model["B"][:, column]
        emissions[position] = mixed
    log_emissions = np.log(np.clip(emissions, 1e-12, None))

    # A local Viterbi, because the shared one indexes `log_B` by symbol rather than position.
    trellis = np.full((len(tails), len(STATES)), -np.inf)
    back = np.zeros((len(tails), len(STATES)), dtype=int)
    trellis[0] = log_pi + log_emissions[0]
    for position in range(1, len(tails)):
        scores = trellis[position - 1][:, None] + log_A
        back[position] = scores.argmax(axis=0)
        trellis[position] = scores.max(axis=0) + log_emissions[position]

    path = [int(trellis[-1].argmax())]
    for position in range(len(tails) - 1, 0, -1):
        path.append(int(back[position, path[-1]]))
    return [STATES[state] for state in reversed(path)]


def cross_validate(sequences: list[dict], mode: str, k: int, n: int = 5) -> dict:
    from src.parse.viterbi import folds, score

    alphabet = alphabet_for(sequences, mode, k)
    predicted, truth = [], []
    for train_set, test_set in folds(sequences, n):
        model = train(train_set, alphabet, mode, k)
        for sequence in test_set:
            predicted.extend(decode(sequence, model, alphabet, mode, k))
            truth.extend(sequence["states"])
    return {
        "mode": mode,
        "alphabet": len(alphabet),
        "nodes": len(truth),
        **score(predicted, truth),
    }


def run(k: int = DEFAULT_K, covariance: str = "full", n: int = 5, limit: int | None = None) -> dict:
    table = responsibilities(k, covariance)
    sequences, missing, total = attach(hdbpmn_sequences(limit), table, k)
    rows = {mode: cross_validate(sequences, mode, k, n) for mode in MODES}

    annotated = rows["annotated"]["macro_f1"]
    return {
        "corpus": "hdbpmn",
        "sequences": len(sequences),
        "nodes": total,
        "nodes_without_a_descriptor": missing,
        "k": k,
        "covariance": covariance,
        "by_mode": rows,
        "hard_minus_annotated": round(rows["hard"]["macro_f1"] - annotated, 4),
        "soft_minus_annotated": round(rows["soft"]["macro_f1"] - annotated, 4),
        "soft_minus_hard": round(rows["soft"]["macro_f1"] - rows["hard"]["macro_f1"], 4),
        "best_mode": max(rows, key=lambda m: rows[m]["macro_f1"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--covariance", default="full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.k, args.covariance, args.folds, args.limit), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

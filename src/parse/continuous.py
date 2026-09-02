"""Phase 7.3.12 - a Gaussian-emission HMM on raw geometry, against the discrete one.

    python -m src.parse.continuous

7.3.2 quantized every node into one of 53 symbols. This task asks what that quantization cost, by
replacing the categorical emission with a **Gaussian per state** over the raw measurements the
symbols were derived from:

    width, height        the node's bbox, normalised by the page diagonal
    aspect               width / height, log-scaled - a diamond and a box differ here
    area                 relative to the page
    in_degree, out_degree the counts themselves rather than 7.3.2's four-way class
    text_length          characters in the label, rather than a keyword class
    x, y                 centre position, normalised

Nine continuous dimensions against 53 categories. The comparison is not obviously one-sided and
that is why it is worth running:

    the discrete model   cannot express "in-degree 7 is more extreme than in-degree 3" - both are
                         `branching`. It also cannot interpolate: a symbol seen twice is estimated
                         from two observations and nothing else informs it.
    the continuous model gets the ordering and the interpolation for free, and pays for it with a
                         distributional assumption that is certainly false. `out_degree` is a
                         count with a spike at 1; `aspect` is bimodal across shape families. A
                         Gaussian is the wrong density for most of these columns.

## Covariance type is the real hyperparameter

`full` gives each state a 9x9 covariance - 45 free parameters per state, 405 in total, against
303 rows for the smallest state. `diag` gives it 9. Both are run, along with `tied`, because this
is the same bias-variance question 7.4.2 asks of the GMM and the two tasks should answer it on
the same corpus.

## What is held fixed

The transitions. `A` and `pi` are initialised from 7.3.4's supervised counts and re-estimated by
the same EM that fits the Gaussians, so the only thing that changes between this task and 7.3.7
is the emission model. Anything else would confound the comparison the task exists to make.

## What it measured

993 sequences, 14,056 nodes, five folds, transitions seeded from 7.3.4 and re-estimated:

    covariance   accuracy   macro F1   parameters per state   folds failed
    full          0.3118     0.2579            54                  0
    tied          0.3003     0.2455             9                  0
    diag          0.2726     0.1963            18                  0

    7.3.7, the same folds with discrete emissions:   0.7512 / 0.7586

**The quantization cost nothing; it bought 0.50 macro F1.** Replacing 53 categorical symbols with
nine Gaussians loses **0.5007 macro F1** and 0.44 accuracy, which is the largest gap measured
anywhere in 7.3 and larger than the entire contribution of the sequence model (7.3.11's +0.147).
The comparison was set up to be fair - same folds, same states, same transition matrix, only the
emission model changed - and it is not close.

## Why, and it is the columns rather than the model

Two of the nine features carry most of the discrete symbol's information and neither survives
being treated as Gaussian:

    out_degree    53.9% of nodes have exactly 1 and 22.0% have exactly 0, mean 1.078 and
                  standard deviation 0.808. A Gaussian fitted to that spends real density on
                  negative out-degrees, and the gap between `linear` and `branching` - which
                  7.3.5 showed is what identifies a decision - is 1.0, only 1.2 standard
                  deviations wide in the density the model is forced to use.
    text_length   zero for 19.5% of nodes and long-tailed above it. The same problem: a spike
                  at a boundary value with a tail on one side only.

7.3.2 collapsed exactly these into four-way and five-way classes, and the collapse was doing more
work than it looked like: **a bin boundary is a decision the discrete model gets to make and the
Gaussian does not.** The continuous model has strictly more information - it knows in-degree 7
beats in-degree 3, which 7.3.2 explicitly threw away - and it cannot use it, because the density
it is forced to spend that information through is the wrong shape for a count.

## The covariance ordering is the one 7.4.2 also found

`full` wins at 0.2579 and `diag` is worst at 0.1963, so modelling the correlation between the
columns is worth **+0.062 macro F1** even here. `width`, `height` and `area` are three views of
one measurement and a diagonal covariance has to pretend they are independent - the same
mechanism 7.2.4 priced for Naive Bayes and 7.4.2 priced for the shape mixture. That three
different models in one phase agree about it is worth more than any of them saying it alone.

**No fold failed at any covariance type**, including `full` at 54 free parameters per state
against 303 rows for the smallest. The docstring's worry about singular covariances did not
materialise, because `reg_covar` handles it and because the states that are small are also the
ones with the least varied geometry.

## What this settles for Phase 10

Keep the discrete emissions. The continuous variant is implemented, exercised and measured, and
its number is recorded here so the question is not reopened - **the quantization in 7.3.2 is not
a lossy compromise to be improved on later, it is the thing that makes the emission model work.**
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.parse.roles import STATE_INDEX, STATES
from src.parse.viterbi import folds, score

FEATURES: tuple[str, ...] = (
    "width",
    "height",
    "log_aspect",
    "area",
    "in_degree",
    "out_degree",
    "text_length",
    "x",
    "y",
)

COVARIANCES: tuple[str, ...] = ("diag", "full", "tied")
SEED = 42


def features(diagram: dict) -> dict[str, np.ndarray]:
    """One row of raw geometry per node id, normalised by the page's own extent."""
    nodes = {node["id"]: node for node in diagram["nodes"]}
    boxes = np.array(
        [node.get("bbox") or [0.0, 0.0, 1.0, 1.0] for node in nodes.values()], dtype=float
    )
    if not len(boxes):
        return {}
    extent = max(float(np.ptp(boxes[:, 0]) + boxes[:, 2].max()), 1.0)

    incoming = dict.fromkeys(nodes, 0)
    outgoing = dict.fromkeys(nodes, 0)
    for edge in diagram["edges"]:
        if edge.get("src") in outgoing:
            outgoing[edge["src"]] += 1
        if edge.get("dst") in incoming:
            incoming[edge["dst"]] += 1

    rows = {}
    for node_id, node in nodes.items():
        x, y, w, h = node.get("bbox") or [0.0, 0.0, 1.0, 1.0]
        w, h = max(w, 1e-6), max(h, 1e-6)
        rows[node_id] = np.array(
            [
                w / extent,
                h / extent,
                np.log(w / h),
                (w * h) / (extent**2),
                float(incoming[node_id]),
                float(outgoing[node_id]),
                float(len(node.get("text", "") or "")),
                x / extent,
                y / extent,
            ],
            dtype=float,
        )
    return rows


def dataset(limit: int | None = None) -> list[dict]:
    """Sequences carrying a feature matrix instead of symbols."""
    from src.parse.sequences import labelled_diagrams, sequence_of

    out = []
    for diagram in labelled_diagrams(limit):
        sequence = sequence_of(diagram, labelled=True)
        if sequence is None:
            continue
        rows = features(diagram)
        matrix = np.array([rows[node_id] for node_id in sequence["node_ids"]])
        out.append({**sequence, "X": matrix})
    return out


def fit_decode(train: list[dict], test: list[dict], covariance: str, seed: int = SEED):
    """A GaussianHMM seeded with 7.3.4's transitions, then EM, then Viterbi on the test rows."""
    from hmmlearn import hmm

    from src.parse.transitions import fit as fit_transitions

    supervised = fit_transitions(train)
    machine = hmm.GaussianHMM(
        n_components=len(STATES),
        covariance_type=covariance,
        n_iter=50,
        random_state=seed,
        # Only the covariances are left to hmmlearn. `m` would discard the class means seeded
        # below in favour of k-means, which is the very thing this task is not doing, and `st`
        # would discard 7.3.4's transitions and confound the comparison the task exists to make.
        init_params="c",
        params="stmc",
    )
    machine.startprob_ = supervised["pi"].copy()
    machine.transmat_ = supervised["A"].copy()

    stacked = np.vstack([s["X"] for s in train])
    lengths = [len(s["X"]) for s in train]
    # Means are seeded from the labelled rows rather than k-means: the states have names here,
    # and starting EM at the class means is the closest continuous analogue of 7.3.5's counting.
    labels = np.array([STATE_INDEX[state] for s in train for state in s["states"]])
    machine.means_ = np.array(
        [
            stacked[labels == i].mean(axis=0) if (labels == i).any() else stacked.mean(axis=0)
            for i in range(len(STATES))
        ]
    )
    machine.fit(stacked, lengths)

    predicted, truth = [], []
    for sequence in test:
        path = machine.predict(sequence["X"])
        predicted.extend(STATES[state] for state in path)
        truth.extend(sequence["states"])
    return predicted, truth, machine


def run(n: int = 5, covariances=COVARIANCES, limit: int | None = None) -> dict:
    sequences = dataset(limit)
    rows = {}
    for covariance in covariances:
        predicted, truth = [], []
        failures = 0
        for train, test in folds(sequences, n):
            try:
                p, t, _ = fit_decode(train, test, covariance)
            except (ValueError, np.linalg.LinAlgError):
                # A `full` covariance on a 300-row state is genuinely singular sometimes; the
                # fold is recorded as failed rather than silently retried with a different type.
                failures += 1
                continue
            predicted.extend(p)
            truth.extend(t)
        rows[covariance] = {
            **(score(predicted, truth) if truth else {"accuracy": 0.0, "macro_f1": 0.0}),
            "folds_failed": failures,
            "parameters_per_state": {
                "diag": len(FEATURES) * 2,
                "full": len(FEATURES) + len(FEATURES) * (len(FEATURES) + 1) // 2,
                "tied": len(FEATURES),
            }[covariance],
        }
    best = max(rows, key=lambda name: rows[name]["macro_f1"])
    return {
        "sequences": len(sequences),
        "nodes": sum(len(s["states"]) for s in sequences),
        "features": list(FEATURES),
        "by_covariance": rows,
        "best_covariance": best,
        "best_macro_f1": rows[best]["macro_f1"],
        "discrete_7_3_7": {"accuracy": 0.7512, "macro_f1": 0.7586},
        "continuous_minus_discrete": round(rows[best]["macro_f1"] - 0.7586, 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--covariances", nargs="*", default=list(COVARIANCES))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.folds, tuple(args.covariances), args.limit), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

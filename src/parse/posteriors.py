"""Phase 7.3.8 - forward-backward, per-node role confidence, and where it is honest.

    python -m src.parse.posteriors      # per-node posteriors, calibration, and the IR field

Viterbi returns one path and no doubt. This task returns `P(state_t = s | all observations)` for
every node - the forward-backward posterior - which is what Phase 10's confidence propagation and
Phase 13's confidence gate consume, and what 7.3.10 uses to decide which nodes are safe to repair.

## The two decoders disagree, and the disagreement is the point

    Viterbi              the most likely *path*. Always legal - every transition it uses has
                         non-zero probability - and it can contain a node whose own marginal
                         posterior prefers a different state.
    posterior argmax     the most likely state at each position, independently. Never worse
                         per node, and it can produce a path the transition matrix says is
                         impossible - a `decision` followed by a `terminal` where `A` has a
                         near-zero cell.

Both are computed here and their disagreement rate is reported, because "which decoder" is a real
choice Phase 10 has to make and it should be made on a measured number rather than on habit.

## Scaling, and why this file uses logs like 7.3.7 does

The forward recursion multiplies a probability per step, so alpha underflows on any sequence
longer than a few dozen nodes. The standard fixes are per-step rescaling or working in logs;
`logsumexp` is used here for the same reason 7.3.7 uses logs - it keeps one representation across
the phase, and the numbers stay comparable between the two decoders without a conversion.

## Calibration is the deliverable, not the posterior

A confidence is only useful if it means something. 7.2.4 found Naive Bayes reporting 0.975 mean
confidence at 0.479 accuracy - a number that looked like a probability and was not one - and the
gate in Phase 13.5 would have passed everything through. So this task reports the posterior's
**reliability curve**: bucket the nodes by confidence, and check whether the accuracy inside each
bucket matches it.

## What it measured

Five-fold, 14,056 nodes, alpha = 1:

    decoder            accuracy   macro F1
    Viterbi             0.7512     0.7586
    posterior argmax    0.7548     0.7625

**The two decoders disagree on 7.3% of nodes**, and the per-node decoder is better by 0.0036 - as
it must be, since it optimises exactly the quantity being scored. The gap is small enough that the
choice should be made on the other criterion: Viterbi's path is guaranteed legal under `A`, the
posterior path is not, and Phase 10's validators are written against legal paths. **Phase 10 should
take Viterbi's roles and this task's confidences**, which is what `annotate()` writes.

## The confidence is well calibrated, which is not what the phase's history suggested

    band          nodes   share   mean confidence   accuracy    gap
    0.00-0.50     3,057   21.8%       0.378          0.438    -0.061
    0.50-0.70     2,784   19.8%       0.599          0.651    -0.051
    0.70-0.90     3,571   25.4%       0.800          0.806    -0.007
    0.90-0.99     3,158   22.5%       0.958          0.964    -0.005
    0.99-1.00     1,486   10.6%       0.994          1.000    -0.006

**Every band's accuracy is at or above its stated confidence** - mean confidence 0.7244 against
0.7512 accuracy, an overconfidence of **-0.027**, and a Brier score of 0.144. The model is very
slightly *under*-confident, and the miscalibration is concentrated in the low bands where being
pessimistic is harmless.

Set beside 7.2.4, which found Naive Bayes claiming 0.975 confidence at 0.479 accuracy, that is a
0.52 swing in honesty from one model to the next in the same phase. The difference is structural
rather than lucky: NB multiplies 14 correlated likelihoods and its sharpness is the product of
that double-counting, while a forward-backward marginal is a normalised sum over paths - the
normalisation is over the same evidence set, so nothing is counted twice.

## What that buys Phase 13's confidence gate

    nodes above 0.9        33.0%   accuracy 0.9752
    nodes below 0.5        21.8%   accuracy 0.4383

A gate at 0.9 accepts a third of the nodes and is right on 97.5% of them; the fifth of nodes below
0.5 are right on 44% and are exactly the population worth asking a user about. **The gate has a
usable operating point**, which is the practical deliverable here and which the 7.2.5 prior could
not have supplied at any threshold.

The 1,486 nodes above 0.99 are correct on **all 1,486** - a role the model is that sure of
has not been wrong once in this corpus.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.special import logsumexp

from src.parse.roles import STATES
from src.parse.viterbi import build_model, decode, folds, logs, score

BANDS: tuple[float, ...] = (0.0, 0.5, 0.7, 0.9, 0.99, 1.01)


def forward_backward(symbols: list[int], log_pi, log_A, log_B) -> np.ndarray:
    """Posterior marginals, one row per position. Log space throughout, exponentiated at the end.

    alpha[t, s] = log P(o_1..o_t, state_t = s)
    beta[t, s]  = log P(o_t+1..o_T | state_t = s)
    gamma       = normalise(alpha + beta) over states, per position.
    """
    T, S = len(symbols), len(log_pi)
    if T == 0:
        return np.zeros((0, S))

    alpha = np.zeros((T, S))
    alpha[0] = log_pi + log_B[:, symbols[0]]
    for t in range(1, T):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_A, axis=0) + log_B[:, symbols[t]]

    beta = np.zeros((T, S))
    for t in range(T - 2, -1, -1):
        beta[t] = logsumexp(log_A + log_B[:, symbols[t + 1]] + beta[t + 1], axis=1)

    gamma = alpha + beta
    return np.exp(gamma - logsumexp(gamma, axis=1, keepdims=True))


def log_likelihood(symbols: list[int], log_pi, log_A, log_B) -> float:
    """log P(observations) - the forward pass's own normaliser, useful as a sanity check."""
    T = len(symbols)
    if T == 0:
        return 0.0
    alpha = log_pi + log_B[:, symbols[0]]
    for t in range(1, T):
        alpha = logsumexp(alpha[:, None] + log_A, axis=0) + log_B[:, symbols[t]]
    return float(logsumexp(alpha))


def annotate(diagram: dict, sequence: dict, model: dict, alphabet: list[str]) -> dict:
    """The IR-facing form: `semantic_role` and `role_confidence` written onto each node.

    The role is Viterbi's - a legal path is what Phase 10's validators expect - while the
    confidence is the forward-backward marginal of the state Viterbi chose. Reporting the
    marginal of a *different* state than the one recorded would be the subtle version of the
    Naive Bayes mistake 7.2.4 caught.
    """
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    symbols = [index.get(s, 0) for s in sequence["observations"]]
    parameters = logs(model)
    path = decode(symbols, *parameters)
    gamma = forward_backward(symbols, *parameters)

    by_id = {node["id"]: node for node in diagram["nodes"]}
    for position, node_id in enumerate(sequence["node_ids"]):
        node = by_id.get(node_id)
        if node is None:
            continue
        node["semantic_role"] = STATES[path[position]]
        node["confidence"] = round(float(gamma[position, path[position]]), 4)
        node.setdefault("attrs", {})["role_posterior"] = {
            STATES[s]: round(float(value), 4)
            for s, value in enumerate(gamma[position])
            if value >= 0.01
        }
    return diagram


def reliability(confidences: np.ndarray, correct: np.ndarray) -> dict:
    out = {}
    for low, high in zip(BANDS, BANDS[1:], strict=False):
        mask = (confidences >= low) & (confidences < high)
        if not mask.any():
            continue
        out[f"{low}-{high}"] = {
            "nodes": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "mean_confidence": round(float(confidences[mask].mean()), 4),
            "accuracy": round(float(correct[mask].mean()), 4),
            "gap": round(float(confidences[mask].mean() - correct[mask].mean()), 4),
        }
    return out


def run(n: int = 5, alpha: float = 1.0) -> dict:
    from src.parse.sequences import build

    sequences = build()["sequences"]
    alphabet = sorted({s for seq in sequences for s in seq["observations"]})
    index = {symbol: i for i, symbol in enumerate(alphabet)}

    viterbi_predicted, posterior_predicted, truth = [], [], []
    confidences, disagreements = [], 0
    for train, test in folds(sequences, n):
        model = build_model(train, alphabet, alpha)
        parameters = logs(model)
        for sequence in test:
            symbols = [index.get(s, 0) for s in sequence["observations"]]
            path = decode(symbols, *parameters)
            gamma = forward_backward(symbols, *parameters)
            best = gamma.argmax(axis=1)
            viterbi_predicted.extend(STATES[s] for s in path)
            posterior_predicted.extend(STATES[s] for s in best)
            confidences.extend(gamma[np.arange(len(path)), path].tolist())
            disagreements += int((np.array(path) != best).sum())
            truth.extend(sequence["states"])

    confidence = np.array(confidences)
    correct = np.array([p == t for p, t in zip(viterbi_predicted, truth, strict=True)], dtype=float)
    return {
        "nodes": len(truth),
        "viterbi": score(viterbi_predicted, truth),
        "posterior_argmax": score(posterior_predicted, truth),
        "decoders_disagree_on": round(disagreements / len(truth), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "overconfidence": round(float(confidence.mean() - correct.mean()), 4),
        "brier": round(float(np.mean((confidence - correct) ** 2)), 4),
        "reliability": reliability(confidence, correct),
        "share_above_0_9": round(float((confidence > 0.9).mean()), 4),
        "accuracy_above_0_9": round(float(correct[confidence > 0.9].mean()), 4),
        "accuracy_below_0_5": round(
            float(correct[confidence < 0.5].mean()) if (confidence < 0.5).any() else 0.0, 4
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.folds, args.alpha), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 7.3.6 - Baum-Welch on the unlabelled pages, and the label-switching problem it creates.

    python -m src.parse.baumwelch      # writes reports/figures/p7_baumwelch.png

7.3.4 and 7.3.5 estimated `A` and `B` by counting labelled sequences. This task estimates them by
**EM on sequences with no labels at all** - 3,000 didi pages whose IR carries no `semantic_role` -
and compares the result to the supervised matrices.

## What EM can and cannot recover, stated before the numbers

Baum-Welch maximises the likelihood of the observations. It has no notion of what a state *means*,
so the best it can do is find nine clusters of observation-and-context that explain the data. Two
consequences follow and both are measured here:

    label switching   the state EM calls #3 has no reason to be 7.3.1's `process`. Any comparison
                      to the supervised model has to first *align* the two state spaces, and the
                      alignment is a maximum-weight matching problem, not a guess. `align()` uses
                      the Hungarian algorithm on the co-occurrence matrix over the labelled
                      sequences, which is the only defensible way to ask "is EM's #3 our process?"
    local optima      the likelihood surface is not convex and EM finds a local maximum determined
                      by the initialisation. So three initialisations are run - random, uniform,
                      and **the supervised matrices themselves** - and the third is the one that
                      answers the interesting question: does EM, started from the right answer,
                      stay there or drift away from it?

## The initialisation from the supervised model is the experiment

If EM started at the supervised matrices and improved the likelihood while *keeping* the state
semantics, unlabelled data would be adding information to a supervised model - the semi-supervised
result the plan's "compare to supervised" line is fishing for. If instead the likelihood rises
while agreement with the supervised labels falls, that is the classic and much more common
outcome: **the observations are explained better by a different partition than the one humans
annotated**, and the states drift into it.

## Which data EM sees

The 3,000 didi sequences, and *not* the 993 labelled ones. Mixing them would make "agreement with
the supervised labels" partly a memory of having seen them. The comparison is then genuinely
out-of-corpus: matrices learned on one dataset's pages, scored against another's annotations.

didi's pages are small - a median of 3 nodes against hdbpmn's 18 - which is itself a limit on what
EM can learn: a 3-node sequence contains two transitions, and most of the evidence for a 9x9
transition matrix has to come from having a great many of them rather than long ones.

## What it measured

EM on 1,971 usable didi sequences (5,571 observations; 1,029 of the 3,000 pages hold fewer than
two nodes and cannot be a sequence), 9 states, 53 symbols, tol 1e-4, cap 100 iterations:

    init          iterations   log likelihood   per observation   agreement   states kept
    random           100          -10,510.75        -1.887          0.2901         0
    supervised       100          -10,531.55        -1.890          0.3009         4
    uniform            3          -11,847.51        -2.127          0.3432         0

## The experiment the module was built for has a clean negative answer

Started **at** the supervised matrices, EM improved the likelihood and then walked away from the
annotation: agreement with the human labels ends at **0.3009**, and only **4 of the 9 states are
still matched to themselves** after the Hungarian alignment. 7.3.7 decodes the same labelled
sequences at 0.7512 accuracy with the supervised matrices untouched, so EM on unlabelled data cost
**0.45 of role-labelling accuracy** while making the observations more likely.

That is the textbook outcome and it is worth stating without hedging: **maximum likelihood and
the annotation disagree about what the states should be.** The partition that best explains
didi's symbol sequences is not the partition a human calls start/process/decision, and nothing in
the EM objective knows that it should be.

## The random initialisation wins on likelihood, which settles the other question

Random reaches **-10,510.75** against the supervised start's -10,531.55. The supervised matrices
are not even a good *optimisation* starting point - they are in a slightly worse basin than a
random draw finds - and they retain none of their semantics (0 states matched to themselves).
So the semi-supervised reading is unavailable in both directions: the labels do not help EM, and
EM does not help the labels.

## The uniform initialisation is a degenerate fixed point, not a fast convergence

Three iterations to "convergence" at a likelihood 1,337 nats worse than the other two. A perfectly
uniform `A`, `B` and `pi` is symmetric under permutation of the states, and Baum-Welch cannot
break a symmetry it starts in - every state stays identical to every other, the posteriors are
flat, and the monitor sees no change and stops. Its 0.3432 "agreement" is the alignment matching
whatever the decoder emits against the largest true states, not a model that found anything.

**This is the only run of the three whose termination should not be read as convergence**, and it
is included precisely because a convergence flag that says `true` on a model that learned nothing
is the failure this task should be able to catch.

## What this means for 7.3

Use the supervised matrices. didi's 3,000 unlabelled pages are the largest single source in the
corpus and they contribute nothing here, for a reason visible in 7.3.3's summary: a median didi
page is 3 nodes, so the unlabelled set has 5,571 observations against the labelled set's 14,056 -
it is not only unlabelled, it is *smaller*, and it is smaller in exactly the dimension (sequence
length) that carries transition evidence.

The figure is `reports/figures/p7_baumwelch.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.parse.roles import STATE_INDEX, STATES
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p7_baumwelch.png"

SEED = 42
INITS: tuple[str, ...] = ("supervised", "random", "uniform")
MAX_ITER = 100
TOLERANCE = 1e-4


def encode(sequences: list[dict], alphabet: list[str]):
    """(concatenated symbol indices, lengths) - hmmlearn's multi-sequence input convention."""
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    flat: list[int] = []
    lengths: list[int] = []
    for sequence in sequences:
        symbols = [index[s] for s in sequence["observations"] if s in index]
        if len(symbols) < 2:
            continue
        flat.extend(symbols)
        lengths.append(len(symbols))
    return np.array(flat).reshape(-1, 1), lengths


def model(init: str, alphabet: list[str], supervised: dict | None = None, seed: int = SEED):
    """A `CategoricalHMM` with `A`, `B` and `pi` seeded the requested way.

    `init_params=""` and `params="ste"`: nothing is re-initialised by hmmlearn and everything is
    re-estimated by EM. The default would silently overwrite the supervised start.
    """
    from hmmlearn import hmm

    machine = hmm.CategoricalHMM(
        n_components=len(STATES),
        n_features=len(alphabet),
        n_iter=MAX_ITER,
        tol=TOLERANCE,
        random_state=seed,
        init_params="",
        params="ste",
        implementation="log",
    )
    rng = np.random.default_rng(seed)
    if init == "supervised":
        if supervised is None:
            raise ValueError("supervised init needs the 7.3.4/7.3.5 matrices")
        machine.startprob_ = supervised["pi"].copy()
        machine.transmat_ = supervised["A"].copy()
        machine.emissionprob_ = supervised["B"].copy()
    elif init == "uniform":
        machine.startprob_ = np.full(len(STATES), 1 / len(STATES))
        machine.transmat_ = np.full((len(STATES), len(STATES)), 1 / len(STATES))
        machine.emissionprob_ = np.full((len(STATES), len(alphabet)), 1 / len(alphabet))
    else:

        def rows(shape):
            values = rng.random(shape) + 0.1
            return values / values.sum(axis=-1, keepdims=True)

        machine.startprob_ = rows(len(STATES))
        machine.transmat_ = rows((len(STATES), len(STATES)))
        machine.emissionprob_ = rows((len(STATES), len(alphabet)))
    return machine


def align(machine, labelled: list[dict], alphabet: list[str]) -> tuple[dict, float]:
    """Match EM's anonymous states to 7.3.1's by maximum-weight assignment.

    Without this every comparison is meaningless: EM's states are a permutation of whatever it
    found, and reading state #3 as `process` because it is third would be arithmetic on labels.
    """
    from scipy.optimize import linear_sum_assignment

    index = {symbol: i for i, symbol in enumerate(alphabet)}
    table = np.zeros((len(STATES), len(STATES)))
    total = 0
    for sequence in labelled:
        symbols = [index[s] for s in sequence["observations"] if s in index]
        if len(symbols) < 2:
            continue
        decoded = machine.predict(np.array(symbols).reshape(-1, 1))
        for true_state, found in zip(sequence["states"], decoded, strict=True):
            table[STATE_INDEX[true_state], found] += 1
            total += 1
    rows, columns = linear_sum_assignment(-table)
    mapping = {int(column): STATES[row] for row, column in zip(rows, columns, strict=True)}
    agreement = float(table[rows, columns].sum() / total) if total else 0.0
    return mapping, round(agreement, 4)


def fit(init: str, unlabelled, lengths, alphabet, supervised=None, seed: int = SEED) -> dict:
    machine = model(init, alphabet, supervised, seed)
    machine.fit(unlabelled, lengths)
    history = [float(value) for value in machine.monitor_.history]
    return {
        "init": init,
        "machine": machine,
        "iterations": int(machine.monitor_.iter),
        "converged": bool(machine.monitor_.converged),
        "log_likelihood": round(float(history[-1]), 2) if history else None,
        "history": history,
    }


def figure(results: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["curve"]
        ax.plot(range(1, len(history) + 1), history, marker="o", ms=2.5, label=result["init"])
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log likelihood (3,000 unlabelled sequences)")
    ax.set_title(
        "Phase 7.3.6 - Baum-Welch convergence from three initialisations",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run(inits=INITS, write: bool = True, limit: int | None = None) -> dict:
    from src.parse.emissions import fit as fit_emissions
    from src.parse.sequences import build
    from src.parse.transitions import fit as fit_transitions

    data = build(limit=limit)
    labelled, unlabelled_sequences = data["sequences"], data["unlabelled"]

    emission = fit_emissions(labelled)
    alphabet = emission["alphabet"]
    supervised = {
        "A": fit_transitions(labelled)["A"],
        "pi": fit_transitions(labelled)["pi"],
        "B": emission["B"],
    }
    observations, lengths = encode(unlabelled_sequences, alphabet)

    baseline_mapping = {i: STATES[i] for i in range(len(STATES))}
    rows = []
    curves = []
    for init in inits:
        result = fit(init, observations, lengths, alphabet, supervised)
        mapping, agreement = align(result["machine"], labelled, alphabet)
        rows.append(
            {
                "init": init,
                "iterations": result["iterations"],
                "converged": result["converged"],
                "log_likelihood": result["log_likelihood"],
                "log_likelihood_per_observation": round(
                    result["log_likelihood"] / max(1, len(observations)), 4
                ),
                "agreement_with_supervised_labels": agreement,
                "identity_mapping": mapping == baseline_mapping,
                "states_matched_to_themselves": sum(
                    1 for column, name in mapping.items() if STATES[column] == name
                ),
            }
        )
        curves.append({"init": init, "curve": result["history"]})

    result = {
        "unlabelled_sequences": len(lengths),
        "unlabelled_observations": int(len(observations)),
        "labelled_sequences": len(labelled),
        "alphabet": len(alphabet),
        "runs": rows,
        "best_likelihood": max(rows, key=lambda r: r["log_likelihood"])["init"],
        "best_agreement": max(rows, key=lambda r: r["agreement_with_supervised_labels"])["init"],
    }
    if write:
        result["figure"] = str(figure(curves).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inits", nargs="*", default=list(INITS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.inits, not args.no_figure, args.limit), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

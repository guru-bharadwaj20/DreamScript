"""Phase 7.3.4 - the transition matrix, the smoothing that decides its zeros, and a figure.

    python -m src.parse.transitions        # writes reports/figures/p7_hmm_transitions.png

The supervised half of the HMM: count how often role `i` is followed by role `j` in 7.3.3's 993
sequences, normalise each row, and that is `A`. The task is arithmetic; the content is in the
three decisions around it.

## Which transitions are counted

Not all of them. 7.3.3 recorded `component_breaks` - the 2,524 positions where the DFS jumped
between disconnected pieces of a page, 19.3% of all adjacent pairs - and those are **excluded**.
A break is an artefact of concatenating components into one sequence; counting it would teach the
model that a `terminal` is followed by a `start` because that is what happens when one BPMN pool
ends and the next begins.

The initial distribution `pi` counts the first state of each sequence *and* the first state after
every break, because both are genuinely "the beginning of a flow with nothing before it".

## Laplace smoothing, and what it is actually protecting against

`(count + alpha) / (row total + alpha * n_states)`. With 9 states and alpha = 1 a row with 300
observations gives an unseen transition probability 1/309 - about 0.3%, three orders of magnitude
below a common transition and still non-zero.

The reason it must be non-zero is Viterbi, not statistics: a zero in `A` is a **hard constraint**,
and one zero cell can make the entire correct path have probability zero, at which point the
decoder returns whatever nonsense still has mass. Smoothing converts "never seen" into "very
unlikely", which is what the evidence actually supports - 993 diagrams cannot establish that a
transition is impossible.

Alpha is swept in 7.3.7 against decoding accuracy rather than fixed here, but 1.0 is the default
and the matrix printed below is the alpha = 1 one.

## Why `pi` gets a different treatment

There are only 993 sequences, so `pi` is estimated from 993 + 2,524 observations spread over 9
states, and it is much sparser than any row of `A`. 7.2.5 built `page_prior()` for exactly this
slot - a per-page class distribution from Naive Bayes - but that prior is over *diagram types*,
not over roles, and the two are different alphabets. What it can do, and what 7.3.7 tests, is
select **which** transition matrix to use: a state machine and a BPMN flowchart have visibly
different transition structure, and `by_type()` returns the per-type matrices for that.

## What it measured

10,539 transitions over 993 sequences, alpha = 1, component breaks excluded. Each state's three
likeliest successors:

    from            n      ->  top three                                              self-loop
    start        1,075         process .640   decision .210   loop-back .093            .015
    process      3,238         process .407   decision .228   terminal  .168            .407
    decision     1,473         branch-true .795  branch-false .148  loop-back .036      .001
    branch-true  1,488         process .340   terminal .186   decision .180             .086
    branch-false 1,058         process .305   branch-false .253  terminal .159          .253
    loop-back    1,207         process .341   loop-back .322  start .095                .322
    terminal     1,000         branch-false .517  start .138  process .136              .050
    input            0         (uniform - nothing was ever observed leaving this state)
    output           0         (uniform)

## Two of the nine states have no outgoing transitions at all

**`input` and `output` are never followed by anything.** Not rarely - never, in 14,056
observations. The reason is structural and it is a finding about the corpus rather than a bug:
every node mapped to those states is a BPMN data object, and hdbpmn's converted IR gives data
objects no flow edges. All 782 `input` and 303 `output` nodes are isolated single-node components,
so the position after each of them is always a component break.

Their rows in `A` are therefore pure smoothing - uniform at 1/9 - which means Viterbi can enter
those states only through the emission model and the initial distribution, and the sequence model
contributes nothing to identifying them. **7.3.9 should be read with that in mind: whatever those
two states score is a per-node classification result wearing an HMM's clothes.**

## The DFS bias is visible exactly where 7.3.3 predicted it

`decision -> branch-true` is **0.795** against `decision -> branch-false` at 0.148. In the graph
those two are equally real: a decision has a true branch and a false branch. In a DFS
linearisation the first branch is entered immediately and the second is reached much later, after
the whole subtree of the first has been walked - so the adjacency is systematically true-heavy.
That 5.4:1 ratio is a property of the ordering, not of how people draw decisions, and it is the
clearest measurable consequence of the graph-to-sequence conversion.

The compensating fact is that `branch-false` has the second-highest self-loop in the matrix
(0.253): once the second branch is finally reached, its own subtree follows, and every node in it
arrives from a `branch-false` parent.

## Excluding the component breaks was worth more than expected

    transitions dropped        2,524   (19.3%)
    largest cell change        0.333
    mean cell change           0.032

**One cell moved by a third of a probability unit.** The design note called the breaks noise; this
says they would have been *structured* noise, concentrated in a few cells - hdbpmn pools end in a
`terminal` and the next pool starts with a `start`, so the discarded pairs are far from uniform.
Had they been left in, `terminal -> start` would have been the matrix's confident lie.

## The two diagram types do not share a transition structure

    type             transitions   empty cells   entropy (bits)
    flowchart            9,365       34 of 81        2.19
    state_machine        1,174       67 of 81        2.47

**A state machine uses 14 of the 81 possible transitions and a flowchart uses 47.** They are
different processes, and the pooled matrix is a weighted average dominated 8:1 by the flowcharts.
`by_type()` returns them separately and 7.3.7 measures whether conditioning on the diagram type -
which Phase 5's classifier already predicts at 0.97 macro F1 - is worth anything at decode time.

Overall the matrix is sparse (42% of cells never observed) and low-entropy (2.26 bits against the
3.17 of a uniform row), which is the property that makes Viterbi worth running: the sequence
carries real constraint.

The figure is `reports/figures/p7_hmm_transitions.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.parse.roles import STATE_INDEX, STATES
from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p7_hmm_transitions.png"

#: Laplace count added to every cell. See the docstring: this is a Viterbi safety property more
#: than a statistical one.
ALPHA = 1.0


def counts(sequences: list[dict], drop_breaks: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Raw transition counts and initial-state counts, before any smoothing."""
    n = len(STATES)
    transition = np.zeros((n, n))
    initial = np.zeros(n)
    for sequence in sequences:
        states = sequence["states"]
        breaks = set(sequence.get("component_breaks", ())) if drop_breaks else set()
        if not states:
            continue
        initial[STATE_INDEX[states[0]]] += 1
        for position in range(1, len(states)):
            if position in breaks:
                # The flow restarts here rather than continuing; it is a first state, not a
                # successor of the previous one.
                initial[STATE_INDEX[states[position]]] += 1
                continue
            transition[STATE_INDEX[states[position - 1]], STATE_INDEX[states[position]]] += 1
    return transition, initial


def normalise(matrix: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Laplace-smoothed row normalisation. A row with no counts becomes uniform, not `nan`."""
    smoothed = matrix + alpha
    return smoothed / smoothed.sum(axis=-1, keepdims=True)


def fit(sequences: list[dict], alpha: float = ALPHA, drop_breaks: bool = True) -> dict:
    transition, initial = counts(sequences, drop_breaks)
    return {
        "A": normalise(transition, alpha),
        "pi": normalise(initial, alpha),
        "counts": transition,
        "initial_counts": initial,
        "alpha": alpha,
        "sequences": len(sequences),
    }


def by_type(sequences: list[dict], alpha: float = ALPHA) -> dict[str, dict]:
    """One matrix per diagram type. 7.3.7 measures whether using them beats one pooled matrix."""
    types = sorted({s["diagram_type"] for s in sequences})
    return {name: fit([s for s in sequences if s["diagram_type"] == name], alpha) for name in types}


def structure(model: dict) -> dict:
    """The readable summary: what each state goes to, and where the matrix is empty."""
    A, raw = model["A"], model["counts"]
    out = {}
    for i, name in enumerate(STATES):
        order = np.argsort(-A[i])
        out[name] = {
            "observed_from": int(raw[i].sum()),
            "top": [
                {"to": STATES[j], "p": round(float(A[i, j]), 4), "count": int(raw[i, j])}
                for j in order[:3]
            ],
            "self_loop": round(float(A[i, i]), 4),
            "unseen_targets": int((raw[i] == 0).sum()),
        }
    return {
        "per_state": out,
        "empty_cells": int((raw == 0).sum()),
        "cells": int(raw.size),
        "empty_share": round(float((raw == 0).mean()), 4),
        "transitions_counted": int(raw.sum()),
        "entropy_bits": round(float(-(model["A"] * np.log2(model["A"])).sum(axis=1).mean()), 4),
    }


def figure(model: dict, path: Path = FIGURE, title: str = "") -> Path:
    """A heatmap of `A`. Log colour, because the useful cells span three orders of magnitude."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    A = model["A"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(A, cmap="viridis", norm=LogNorm(vmin=max(A.min(), 1e-4), vmax=A.max()))
    ax.set_xticks(range(len(STATES)), STATES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(STATES)), STATES, fontsize=8)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            ax.text(
                j,
                i,
                f"{A[i, j]:.02f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if A[i, j] < 0.25 else "black",
            )
    ax.set_title(
        title
        or "Phase 7.3.4 - supervised transition matrix, Laplace alpha = 1 "
        "(993 sequences, component breaks excluded)",
        fontsize=9,
    )
    fig.colorbar(image, ax=ax, label="P(to | from), log scale")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run(alpha: float = ALPHA, write: bool = True) -> dict:
    from src.parse.sequences import build

    sequences = build()["sequences"]
    model = fit(sequences, alpha)
    unfiltered = fit(sequences, alpha, drop_breaks=False)

    result = {
        "alpha": alpha,
        "sequences": len(sequences),
        "structure": structure(model),
        "initial_distribution": {
            name: round(float(model["pi"][i]), 4) for i, name in enumerate(STATES)
        },
        # What excluding the component breaks was worth, as the difference it makes to the matrix.
        "breaks_excluded": {
            "transitions_dropped": int(unfiltered["counts"].sum() - model["counts"].sum()),
            "max_cell_change": round(float(np.abs(unfiltered["A"] - model["A"]).max()), 4),
            "mean_cell_change": round(float(np.abs(unfiltered["A"] - model["A"]).mean()), 4),
        },
        "per_type": {name: structure(sub) for name, sub in by_type(sequences, alpha).items()},
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

"""Headline S4: HMM semantic-role macro-F1.

    python -m src.parse.s4

Target macro F1 >= 0.80. 7.3.9 reported 0.7763 and read the shortfall as structural - "what would
raise it is an arrival-edge factor in the observation alphabet". This module does not add that
factor, and the reason is worth stating before the result.

## The fix that was refused

`derive_states` assigns `branch-true`, `branch-false` and `loop-back` *from* the arrival edge and
the sibling rank (7.3.1's precedence list, rules 1 and 2). Putting an arrival-edge factor into the
observation alphabet therefore hands the model the label function for three of the nine states,
and a macro F1 raised that way measures the model's ability to re-run a deterministic rule it was
given. 7.3.1 said as much: the exercise stays real "because `derive_states` uses what the HMM
never sees - arrival edge, sibling order, DFS stack". That route is available, it clears the
target easily, and it is not taken here.

## The bug that was found instead

**7.3.7's decoder runs Viterbi straight through the component breaks 7.3.4 excluded from
training.** `transitions.counts` treats a break as a new initial state - "the flow restarts here
rather than continuing" - because a BPMN page is several pools and `terminal -> start` across two
of them "would have become the matrix's confident lie". `viterbi.decode_sequence` then decodes
each page as one unbroken chain, applying the transition matrix across the very boundaries it was
trained to ignore. 555 of 993 sequences contain at least one, 2,524 in total.

Decoding each connected component as its own chain - same matrices, same folds, same alphabet, no
new information - is worth **+0.0244 macro F1** (mean over five seeds, minimum +0.0238), and the
per-class attribution shows it is a mechanism rather than a wiggle:

    state       one-node segments    whole-seq F1   per-component F1
    output          303 / 303  100%      0.8305          0.9934
    input           782 / 782  100%      0.9629          0.9968

**Those two states carry 91% of the gain**, which is precisely what 7.3.4 predicts: `input` and
`output` are never followed by anything, every one of them is an isolated single-node component,
their transition rows are pure smoothing, and the row says Viterbi "can reach those states only
through the emissions and the initial distribution". Per-component decoding is what lets it. The
old decoder was forcing them into transitions its own matrix says do not exist. `loop-back` is
the one state that gets slightly worse (-0.0132) and that is reported rather than netted away.

The break positions are computed by `sequences.components` from `nodes` and `edges` alone, by
union-find, before `derive_states` is ever called - so this uses graph structure available at
inference and no label information.

## The second change

The emission matrix is 69% unobserved cells over a symbol that is already a `shape|keyword|degree`
product, so the atomic Laplace estimate is interpolated with the product of its three per-factor
marginals. Worth a further **+0.0013**. The weight is chosen by cross-validation *inside each
training fold*, never on the evaluation rows, because at this margin selection-on-test is the
difference between a result and an artefact.

Two levers that did not work are recorded so they are not retried: a second-order transition model
is worth +0.0002 under nested CV (and its inner folds pick lambda anywhere from 0.0 to 0.6, which
is what noise looks like), and dropping 7.3.11's inert text-keyword factor is worth -0.0005.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.parse.roles import STATES
from src.parse.viterbi import folds
from src.utils.config import ROOT

TARGET_MACRO_F1 = 0.80
REPORT = ROOT / "reports" / "s4_role_labelling.json"
LAMBDAS = (0.6, 0.7, 0.8, 0.9, 1.0)
SEEDS = (42, 43, 44, 45, 46)
STATE_INDEX = {state: i for i, state in enumerate(STATES)}


def segments(sequence: dict) -> list[tuple[list[str], list[str]]]:
    """Split one sequence at its component breaks.

    A break is where the traversal jumps to a different connected component. 7.3.4 excludes the
    transition into each one from training; this is the decode-side half of that decision.
    """
    states = sequence.get("states") or [""] * len(sequence["observations"])
    breaks = set(sequence.get("component_breaks", ()))
    out: list[tuple[list[str], list[str]]] = []
    current: tuple[list[str], list[str]] = ([], [])
    for position, (state, observation) in enumerate(
        zip(states, sequence["observations"], strict=False)
    ):
        if position in breaks and current[0]:
            out.append(current)
            current = ([], [])
        current[0].append(state)
        current[1].append(observation)
    if current[0]:
        out.append(current)
    return out


def emissions(train: list[dict], alphabet: list[str], lam: float, alpha: float = 1.0) -> np.ndarray:
    """Atomic Laplace estimate interpolated with the product of the three factor marginals."""
    parts = [symbol.split("|") for symbol in alphabet]
    levels = [sorted({part[i] for part in parts}) for i in range(3)]
    maps = [{value: i for i, value in enumerate(levels[i])} for i in range(3)]
    index = {symbol: i for i, symbol in enumerate(alphabet)}

    full = np.zeros((len(STATES), len(alphabet)))
    factors = [np.zeros((len(STATES), len(level))) for level in levels]
    for sequence in train:
        for state, observation in zip(sequence["states"], sequence["observations"], strict=False):
            if observation not in index:
                continue
            row = STATE_INDEX[state]
            full[row, index[observation]] += 1
            for i, value in enumerate(observation.split("|")):
                factors[i][row, maps[i][value]] += 1

    atomic = (full + alpha) / (full + alpha).sum(axis=1, keepdims=True)
    product = np.ones_like(atomic)
    for i, counts in enumerate(factors):
        marginal = (counts + alpha) / (counts + alpha).sum(axis=1, keepdims=True)
        product *= marginal[:, np.array([maps[i][part[i]] for part in parts])]
    product /= product.sum(axis=1, keepdims=True)

    blended = lam * atomic + (1.0 - lam) * product
    return blended / blended.sum(axis=1, keepdims=True)


def build(train: list[dict], alphabet: list[str], lam: float, alpha: float = 1.0) -> dict:
    from src.parse.transitions import fit as fit_transitions

    transitions = fit_transitions(train, alpha)
    return {"pi": transitions["pi"], "A": transitions["A"], "B": emissions(train, alphabet, lam)}


def fit_per_type(train: list[dict], alphabet: list[str], lam: float) -> tuple[dict, dict]:
    """7.3.7 found per-type matrices worth +0.0259; the pooled model is the fallback."""
    models = {
        name: build([s for s in train if s["diagram_type"] == name], alphabet, lam)
        for name in {s["diagram_type"] for s in train}
    }
    return models, build(train, alphabet, lam)


def decode_split(test: list[dict], models: dict, pooled: dict, index: dict) -> tuple[list, list]:
    """Decode each component of each sequence as its own chain."""
    from src.parse.viterbi import decode

    predicted: list[str] = []
    truth: list[str] = []
    for sequence in test:
        model = models.get(sequence["diagram_type"], pooled)
        with np.errstate(divide="ignore"):
            log_pi, log_a, log_b = np.log(model["pi"]), np.log(model["A"]), np.log(model["B"])
        for states, observations in segments(sequence):
            symbols = [index.get(o, 0) for o in observations]
            predicted.extend(STATES[state] for state in decode(symbols, log_pi, log_a, log_b))
            truth.extend(states)
    return predicted, truth


# `folds` was defined here and in `viterbi`, with identical bodies and different defaults - so
# the two could have been re-tuned apart while still producing "the same" split. One definition,
# in the module whose docstring explains why a sequence is the unit and there is no leak.
# `folds` is imported above. It was defined here and in `viterbi` with identical bodies and
# different defaults, so the two could have been re-tuned apart while still producing "the same"
# split. One definition, in the module whose docstring explains why a sequence is the unit.


def choose_lambda(train: list[dict], alphabet: list[str], index: dict, seed: int) -> float:
    """Inner cross-validation over the training fold only - the evaluation rows are never seen."""
    from src.parse.viterbi import score

    best, best_f1 = LAMBDAS[0], -1.0
    for lam in LAMBDAS:
        predicted, truth = [], []
        for inner_train, inner_test in folds(train, 3, seed + 100):
            models, pooled = fit_per_type(inner_train, alphabet, lam)
            p, t = decode_split(inner_test, models, pooled, index)
            predicted.extend(p)
            truth.extend(t)
        f1 = score(predicted, truth)["macro_f1"]
        if f1 > best_f1:
            best, best_f1 = lam, f1
    return best


def evaluate(sequences: list[dict], seed: int = 42, n: int = 5) -> dict:
    from src.parse.viterbi import score

    alphabet = sorted({s for seq in sequences for s in seq["observations"]})
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    predicted, truth, chosen = [], [], []
    for train, test in folds(sequences, n, seed):
        lam = choose_lambda(train, alphabet, index, seed)
        chosen.append(lam)
        models, pooled = fit_per_type(train, alphabet, lam)
        p, t = decode_split(test, models, pooled, index)
        predicted.extend(p)
        truth.extend(t)
    return {**score(predicted, truth), "lambdas": chosen, "nodes": len(truth)}


def run(seeds=SEEDS) -> dict:

    from src.parse.sequences import load

    sequences = load()["sequences"]
    repeats = [evaluate(sequences, seed) for seed in seeds]
    macro = np.array([r["macro_f1"] for r in repeats])
    accuracy = np.array([r["accuracy"] for r in repeats])
    return {
        "criterion": "S4",
        "target_macro_f1": TARGET_MACRO_F1,
        "protocol": (
            "5-fold over 993 sequences x len(seeds) seeds; per-component Viterbi; "
            "per-type matrices; emission interpolation weight by inner 3-fold CV"
        ),
        "published_7_3_9": 0.7763,
        "sequences": len(sequences),
        "nodes": repeats[0]["nodes"],
        "macro_f1": round(float(macro.mean()), 4),
        "macro_f1_std": round(float(macro.std()), 4),
        "minimum_repeat_macro_f1": round(float(macro.min()), 4),
        "accuracy": round(float(accuracy.mean()), 4),
        "seeds_clearing_target": int((macro >= TARGET_MACRO_F1).sum()),
        "seeds": len(seeds),
        "passes": bool(macro.mean() >= TARGET_MACRO_F1),
        "repeats": [
            {"seed": int(s), "macro_f1": round(r["macro_f1"], 4), "lambdas": r["lambdas"]}
            for s, r in zip(seeds, repeats, strict=False)
        ],
    }


def write_report(result: dict, path: Path = REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args(argv)
    result = run(tuple(args.seeds))
    write_report(result)
    print(json.dumps(result, indent=2))
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    sys.exit(main())

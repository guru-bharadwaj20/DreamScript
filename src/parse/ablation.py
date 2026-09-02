"""Phase 7.3.11 - what the HMM is worth, against the baseline that needs no sequence at all.

    python -m src.parse.ablation

The plan asks for *pipeline accuracy with the HMM removed (bag-of-shapes baseline)*, and for the
contribution to be quantified. Removing a component is only informative against a stated
alternative, so this task runs four decoders of increasing knowledge on identical folds:

    bag of shapes      the majority role per shape class. Six shape classes, one role each,
                       nothing else. This is the plan's named baseline and it is what the
                       pipeline would do with no 7.3 at all.
    + text and degree  the same idea over the full 7.3.2 symbol - still memoryless, but the
                       observation is the whole triple. This isolates *the alphabet* from
                       *the sequence*.
    HMM (pooled)       7.3.7's decoder.
    HMM (per type)     conditioned on the diagram type.

The gap between the first two is what 7.3.2's feature engineering bought. The gap between the
second and the third is what the Markov assumption bought. Reporting only the first-to-last
difference would credit the sequence model with both.

## The factor ablation

The same idea applied inside the alphabet: rebuild the symbol with one factor removed and rerun
the full HMM. Three runs - no shape, no keyword, no degree - and the drop names what each factor
contributes *in the presence of the others*, which is not what its marginal distribution suggests.
7.3.2 predicted the keyword factor would be nearly inert on this corpus (69% of nodes carry
`other`), and this is where that prediction is either confirmed or refuted.

## What it measured

Identical five folds, 14,056 nodes:

    decoder                     accuracy   macro F1   step
    bag of shapes                0.5023     0.3228      -
    full symbol, memoryless      0.6484     0.6113   +0.2885
    HMM, pooled                  0.7512     0.7586   +0.1473
    HMM, per diagram type        0.7771     0.7763   +0.0177
                                                     -------
    total over the baseline                          +0.4535

**Removing the HMM costs 0.454 macro F1** against the plan's named baseline - the bag of shapes
labels roles at 0.3228, and shape alone cannot do better because six shape classes cannot address
nine states.

## But two thirds of that gain is the alphabet, not the sequence

    alphabet beyond shape   +0.2885     (7.3.2's text and degree factors)
    the Markov assumption   +0.1473     (7.3.4's transition matrix)
    the diagram type        +0.0177     (per-type matrices)

The single largest contribution in 7.3 is **not the HMM** - it is adding in/out degree and a text
keyword to the observation. A memoryless lookup table over the full triple already reaches 0.6113,
which is 63% of the distance from the baseline to the final model, and it needs no transition
matrix, no Viterbi and no training beyond counting.

That is the honest form of the ablation the plan asked for. The sequence model is worth a
substantial +0.147 and it is the second-largest term, not the first; a report that quoted only
"+0.45 for the HMM" would be crediting the transition matrix with 7.3.2's feature engineering.

## The factor ablation confirms 7.3.2's own prediction against itself

Full symbol per-type HMM 0.7763. Rebuild the symbol with one factor blanked:

    factor removed   macro F1   cost
    shape             0.5028    0.2735
    degree            0.5541    0.2222
    keyword           0.7772   -0.0009

**Dropping the text keyword makes the model very slightly better.** 7.3.2 argued the keyword class
was the one factor that could separate `start` from `terminal`, then measured that 69% of nodes
carry `other` and warned the factor was probably inert on this corpus. It is worse than inert: it
adds columns to the emission matrix that split counts without carrying evidence, and removing it
concentrates the same mass into fewer, better-estimated cells.

That is a real result about *this corpus*, not about the idea. hdbpmn labels its gateways with
message names and fa_bresler labels states with single letters, so neither dataset writes the
words the factor looks for. On a corpus of hand-drawn flowcharts that say "start" and "is x > 0?"
- which is what Phase 9's OCR will be reading - the factor would be doing what it was designed to
do. It is kept, with its measured contribution of zero recorded here, because removing it would
bake this corpus's labelling conventions into the alphabet.

Shape and degree are both load-bearing and roughly equally so, which is the tidiest fact in the
task: what a node looks like and how it is wired matter about the same amount, and neither alone
gets past 0.56.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.parse.roles import STATES
from src.parse.viterbi import build_model, decode_sequence, folds, score

FACTORS: tuple[str, ...] = ("shape", "keyword", "degree")


def rebuild(sequences: list[dict], drop: str | None = None) -> list[dict]:
    """The same sequences with one factor of every symbol removed.

    The symbol keeps its three-part shape with the dropped factor replaced by a constant, rather
    than becoming a two-part string: it keeps the alphabet comparable and makes the ablation a
    loss of information rather than a change of representation.
    """
    out = []
    for sequence in sequences:
        symbols = []
        for symbol in sequence["observations"]:
            parts = symbol.split("|")
            if drop is not None:
                parts[FACTORS.index(drop)] = "-"
            symbols.append("|".join(parts))
        out.append({**sequence, "observations": symbols})
    return out


def majority_map(sequences: list[dict], key) -> dict:
    """The most common state for each value of `key(symbol)` - a lookup table, no sequence."""
    counts: dict[str, dict[str, int]] = {}
    for sequence in sequences:
        for state, symbol in zip(sequence["states"], sequence["observations"], strict=True):
            bucket = counts.setdefault(key(symbol), {})
            bucket[state] = bucket.get(state, 0) + 1
    return {value: max(bucket, key=bucket.get) for value, bucket in counts.items()}


def memoryless(sequences: list[dict], n: int, key) -> dict:
    predicted, truth = [], []
    for train, test in folds(sequences, n):
        table = majority_map(train, key)
        fallback = majority_map(train, lambda _: "all")["all"]
        for sequence in test:
            predicted.extend(table.get(key(s), fallback) for s in sequence["observations"])
            truth.extend(sequence["states"])
    return score(predicted, truth)


def hmm(sequences: list[dict], n: int, alpha: float = 1.0, per_type: bool = False) -> dict:
    alphabet = sorted({s for seq in sequences for s in seq["observations"]})
    predicted, truth = [], []
    for train, test in folds(sequences, n):
        pooled = build_model(train, alphabet, alpha)
        models = (
            {
                name: build_model([s for s in train if s["diagram_type"] == name], alphabet, alpha)
                for name in {s["diagram_type"] for s in train}
            }
            if per_type
            else {}
        )
        for sequence in test:
            model = models.get(sequence["diagram_type"], pooled)
            predicted.extend(decode_sequence(sequence, model, alphabet))
            truth.extend(sequence["states"])
    return score(predicted, truth)


def run(n: int = 5) -> dict:
    from src.parse.sequences import build

    sequences = build()["sequences"]

    ladder = {
        "bag_of_shapes": memoryless(sequences, n, lambda s: s.split("|")[0]),
        "full_symbol_memoryless": memoryless(sequences, n, lambda s: s),
        "hmm_pooled": hmm(sequences, n),
        "hmm_per_type": hmm(sequences, n, per_type=True),
    }
    factors = {
        f"without_{name}": hmm(rebuild(sequences, name), n, per_type=True) for name in FACTORS
    }
    full = ladder["hmm_per_type"]["macro_f1"]
    return {
        "nodes": sum(len(s["states"]) for s in sequences),
        "ladder": ladder,
        "contributions": {
            "alphabet_beyond_shape": round(
                ladder["full_symbol_memoryless"]["macro_f1"] - ladder["bag_of_shapes"]["macro_f1"],
                4,
            ),
            "sequence_model": round(
                ladder["hmm_pooled"]["macro_f1"] - ladder["full_symbol_memoryless"]["macro_f1"], 4
            ),
            "diagram_type": round(
                ladder["hmm_per_type"]["macro_f1"] - ladder["hmm_pooled"]["macro_f1"], 4
            ),
            "total": round(
                ladder["hmm_per_type"]["macro_f1"] - ladder["bag_of_shapes"]["macro_f1"], 4
            ),
        },
        "factor_ablation": factors,
        "factor_cost": {
            name.replace("without_", ""): round(full - result["macro_f1"], 4)
            for name, result in factors.items()
        },
        "states": len(STATES),
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

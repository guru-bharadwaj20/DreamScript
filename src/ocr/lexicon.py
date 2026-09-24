"""Phase 9.3.4 - constraining the decode with a diagram vocabulary, and the control that prices it.

    python -m src.ocr.lexicon
    python -m src.ocr.lexicon --model finetune --beam 12

A CTC posterior is a per-column distribution over the alphabet. Greedy decoding reads the argmax
of each column independently, which is free and which routinely produces strings no diagram has
ever contained - `subrnit`, `deciston`, `q6` where the alphabet stops at `q9`. A lexicon says
which strings are possible, and this task measures what saying so is worth.

## The four arms

    greedy        9.3.2's decode, unchanged. The baseline every other row is read against.
    beam          prefix beam search over the same posterior, no lexicon. **This is the first
                  control.** A beam finds a higher-probability path than the argmax path, so any
                  improvement it produces is the decoder's and not the vocabulary's.
    domain        the same beam, with a vocabulary built from the *training* split's strings.
    generic       the same beam, with a vocabulary of the same size drawn from common English
                  words that are **not** in the domain vocabulary. **This is the second control**,
                  and it is the one that matters.

`domain - beam` is what a vocabulary is worth. `domain - generic` is what a *diagram* vocabulary
is worth over any word list of the same size. Reporting only the first would credit the domain
lexicon for an effect that is partly just "words are more likely than non-words", and 8.7, 9.1.5,
9.1.7 and 9.2.7 each found a headline shrink or reverse once its matched control was added.

## The vocabulary is built from training text only

Every string in the training split, split into words, plus the whole strings. **Not the test
split**, obviously, but also not the full corpus: a lexicon containing the exact test strings
turns the task into retrieval and would score near zero CER while measuring nothing. The share of
test words that the training vocabulary covers is reported as `oov_rate`, because it is the
ceiling on what any lexicon can do here - a word the vocabulary does not contain cannot be
recovered by constraining to it, and can only be made worse.

## How the constraint is applied

Prefix beam search over the posterior, and at the end each surviving beam's words are snapped to
their nearest vocabulary entry within an edit-distance budget that scales with word length. A
hard trie constraint during the beam - refusing to extend a prefix that no vocabulary word
starts with - is the stronger form and it is deliberately not used: it cannot represent a word
the vocabulary lacks, so on a corpus with an OOV rate of any size it converts a partially correct
read into a confidently wrong one. Snapping with a budget degrades to the unconstrained answer
instead, which is the behaviour Phase 12 needs from a component whose output becomes an
identifier.

## What it measured

2,811 validation crops, vocabulary 4,149 entries from the training split, OOV rate **0.0281**.

    arm         CER      WER    exact   mean-crop CER
    greedy   0.6972   0.9420   0.1021          0.6440
    beam     0.6839   0.9438   0.1046          0.6340
    domain   0.6865   0.8907   0.1185          0.6398
    generic  0.7171   0.9911   0.0928          0.6668

    beam    - greedy     +0.0133 CER   the decoder alone
    domain  - beam       -0.0026 CER   the vocabulary, on characters
    generic - domain     -0.0306 CER   what *this* vocabulary is worth over any word list
    domain  - greedy     +0.0107 CER   the plan's comparison, end to end

**Against the 9.3.2 decode the plan asks about, CER improves by 0.0107 - and most of that is the
beam, not the lexicon.** Read one line further and the lexicon *costs* 0.0026 CER against the
unconstrained beam. Without the `beam` control the honest +0.0107 would have been reported as the
vocabulary's contribution, and it is mostly not.

**The lexicon's real effect is on words, and it is large.** WER falls **0.9438 -> 0.8907** and
exact match rises **0.1046 -> 0.1185, a 13% relative gain**, at the same time as CER gets very
slightly worse. That is not a contradiction, it is what snapping *is*: replacing `subrnit` with
`submit` fixes a word and removes one character error, while replacing it with `summit` costs two
characters and fixes nothing. **A lexicon trades characters for words** - it converts
near-misses into either exact hits or confident misses - and on a corpus whose consumer is Phase
12, which needs an identifier rather than a nearly-correct string, the word-level column is the
one that matters. The recommendation is therefore to keep it despite the CER row.

**The `generic` control is unambiguous and it is what makes this a measurement.** A word list of
**exactly the same size** drawn from IAM prose and disjoint from the diagram vocabulary makes
everything worse - CER 0.7171, WER 0.9911, exact match 0.0928, all worse than not snapping at all
- so the domain vocabulary is worth **0.0306 CER and 0.1004 WER over an arbitrary one**. Snapping
to the wrong dictionary is actively harmful, which also rules out the deflationary reading that
these gains are just "words beat non-words".

**OOV is not the limitation.** Only **2.81%** of validation words are absent from the training
vocabulary, so the ceiling a lexicon could reach here is high and the modest gains are not
explained by missing entries. What limits it is upstream: at CER 0.68 the beam's output is often
too far from any real word for a length-scaled budget to reach, and the budget is deliberately not
widened, because a wider one would start snapping unrelated words together. **This measurement
would look completely different on a 0.15-CER recogniser, and that is a statement about 9.3.2
rather than about lexicons.**
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.ocr.metrics import RUNS, edit_distance, normalise, score
from src.utils.config import ROOT

BEAM = 12
BLANK = 0

#: A word may be snapped to a vocabulary entry this far away, as a fraction of its length,
#: capped by `MAX_SNAP`. Length-scaled because one wrong character in `q0` is half the word and
#: one wrong character in `evaluate the application` is nothing.
SNAP_RATIO = 0.34
MAX_SNAP = 3

#: Words shorter than this are never snapped. `a`, `b`, `0`, `1` are 40% of the edge labels and
#: every one of them is within edit distance 1 of every other, so snapping them is a coin flip.
MIN_SNAP_LENGTH = 3

GENERIC_WORDS = ROOT / "data" / "processed" / "generic_words.json"

_WORD = re.compile(r"[a-z0-9_]+")


# ------------------------------------------------------------------------------------------
# vocabularies
# ------------------------------------------------------------------------------------------


def build_vocabulary(texts) -> Counter:
    """Words and whole strings from the training split, with their frequencies."""
    vocabulary: Counter = Counter()
    for text in texts:
        clean = normalise(text)
        vocabulary.update(_WORD.findall(clean))
        if clean:
            vocabulary[clean] += 1
    return vocabulary


def generic_vocabulary(size: int, exclude: set[str], seed: int = 42) -> Counter:
    """A word list of the same size that has nothing to do with diagrams.

    Drawn from the IAM training transcriptions - ordinary English prose, already in this
    project, and guaranteed disjoint in subject matter from BPMN task names. Words that also
    appear in the domain vocabulary are removed, so the two lists share nothing and the only
    thing being varied between the arms is relevance.
    """
    import pandas as pd

    from src.ocr.crnn import IAM

    texts = pd.read_parquet(IAM / "train.parquet", columns=["text"])["text"].tolist()
    counts: Counter = Counter()
    for text in texts:
        counts.update(_WORD.findall(normalise(text)))
    candidates = [w for w, _ in counts.most_common() if w not in exclude]
    rng = np.random.default_rng(seed)
    if len(candidates) > size:
        chosen = rng.choice(len(candidates), size=size, replace=False)
        candidates = [candidates[i] for i in sorted(chosen)]
    return Counter({w: counts[w] for w in candidates})


def oov_rate(vocabulary: Counter, texts) -> float:
    """Share of test words the vocabulary does not contain - the ceiling on any lexicon here."""
    total = missing = 0
    for text in texts:
        for word in _WORD.findall(normalise(text)):
            total += 1
            missing += word not in vocabulary
    return round(missing / total, 4) if total else float("nan")


# ------------------------------------------------------------------------------------------
# decoding
# ------------------------------------------------------------------------------------------


def prefix_beam(logprobs: np.ndarray, chars: str, beam: int = BEAM, prune: float = -12.0) -> str:
    """CTC prefix beam search: the standard two-probability recursion, blank and non-blank.

    Each prefix carries `(p_blank, p_non_blank)` because CTC's collapse rule depends on whether
    the path reaching a prefix ended in a blank - `aa` and `a-a` collapse differently, and a
    single probability per prefix cannot express that. Columns whose best character is far below
    the blank are still expanded; only characters below `prune` in a column are skipped, which is
    a speed measure and not an approximation of the ranking.
    """
    neg_inf = -math.inf

    def logsum(a, b):
        if a == neg_inf:
            return b
        if b == neg_inf:
            return a
        top = max(a, b)
        return top + math.log(math.exp(a - top) + math.exp(b - top))

    beams: dict[str, tuple[float, float]] = {"": (0.0, neg_inf)}
    for column in logprobs:
        candidates = [i for i in range(len(column)) if column[i] > prune] or [int(column.argmax())]
        nxt: dict[str, list[float]] = defaultdict(lambda: [neg_inf, neg_inf])
        for prefix, (p_blank, p_non_blank) in beams.items():
            total = logsum(p_blank, p_non_blank)
            entry = nxt[prefix]
            entry[0] = logsum(entry[0], total + column[BLANK])
            last = prefix[-1] if prefix else None
            for index in candidates:
                if index == BLANK:
                    continue
                char = chars[index - 1]
                probability = column[index]
                if char == last:
                    # extending with the same character: only a blank-ended path grows the
                    # prefix; a non-blank-ended one is the same character repeated and collapses.
                    entry[1] = logsum(entry[1], p_non_blank + probability)
                    grown = nxt[prefix + char]
                    grown[1] = logsum(grown[1], p_blank + probability)
                else:
                    grown = nxt[prefix + char]
                    grown[1] = logsum(grown[1], total + probability)
        beams = dict(sorted(nxt.items(), key=lambda kv: -logsum(kv[1][0], kv[1][1]))[:beam])
        beams = {k: (v[0], v[1]) for k, v in beams.items()}
    return max(beams, key=lambda k: logsum(*beams[k])) if beams else ""


def snap_all(texts, vocabulary: Counter) -> list[str]:
    """`snap` over many strings, resolving each distinct word once.

    The beam's outputs repeat heavily - `insurer` and `claim` appear on hundreds of pages - and
    a snap is a scan over the whole vocabulary, so without this the arm costs a quadratic number
    of edit distances for no additional information.
    """
    cache: dict[str, str] = {}
    out = []
    for text in texts:
        whole = normalise(text)
        words = _WORD.findall(whole)
        if not words:
            out.append(whole)
            continue
        replaced = []
        for word in words:
            if word not in cache:
                cache[word] = _nearest(word, vocabulary)
            replaced.append(cache[word])
        out.append(" ".join(replaced))
    return out


def _nearest(word: str, vocabulary: Counter) -> str:
    """The vocabulary entry `word` snaps to, or `word` itself when nothing is close enough."""
    if word in vocabulary or len(word) < MIN_SNAP_LENGTH:
        return word
    budget = min(MAX_SNAP, max(1, int(round(SNAP_RATIO * len(word)))))
    best, best_key = word, (budget + 1, 0)
    for candidate in vocabulary:
        if abs(len(candidate) - len(word)) > budget:
            continue
        distance = edit_distance(word, candidate)
        key = (distance, -vocabulary[candidate])
        if distance <= budget and key < best_key:
            best, best_key = candidate, key
    return best


def snap(text: str, vocabulary: Counter) -> str:
    """Replace each word with its nearest vocabulary entry inside the budget, or leave it alone.

    Ties are broken by frequency in the vocabulary, so a word equidistant from `claim` and
    `claims` becomes whichever the corpus writes more often. That is a real modelling choice and
    not a tiebreak of convenience: it is a unigram prior, the weakest useful language model.
    """
    if not vocabulary:
        return text
    return snap_all([text], vocabulary)[0]


# ------------------------------------------------------------------------------------------
# the grid
# ------------------------------------------------------------------------------------------


def posteriors(model, files, device, batch: int = 32):
    """Log-softmax columns for every crop, computed once and reused by all four arms."""
    import torch

    from src.ocr.crnn import Batches

    batches = Batches(files, [""] * len(files), {}, batch=batch, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _, _, _ in batches:
            out.extend(model(x.to(device)).log_softmax(-1).cpu().numpy())
    return out


def run(model_name: str = "finetune", beam: int = BEAM, limit: int | None = None) -> dict:
    from src.ocr.crnn import diagram_split, greedy_decode, load

    model, chars, device = load(model_name)
    train_files, train_texts, _ = diagram_split("train")
    val_files, val_texts, val_frame = diagram_split("val")
    if limit:
        val_files, val_texts = val_files[:limit], val_texts[:limit]
        val_frame = val_frame.head(limit)

    domain = build_vocabulary(train_texts)
    generic = generic_vocabulary(len(domain), set(domain))
    columns = posteriors(model, val_files, device)

    greedy = [greedy_decode(c, chars) for c in columns]
    beamed = [prefix_beam(c, chars, beam) for c in columns]
    arms = {
        "greedy": greedy,
        "beam": beamed,
        "domain": snap_all(beamed, domain),
        "generic": snap_all(beamed, generic),
    }

    RUNS.mkdir(parents=True, exist_ok=True)
    scores = {}
    for name, predictions in arms.items():
        scores[name] = score(val_texts, predictions)
        (RUNS / f"lexicon_{name}.json").write_text(
            json.dumps(
                {
                    "model": f"lexicon_{name}",
                    "split": "val",
                    "files": val_frame["file"].tolist(),
                    "predictions": predictions,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "model": model_name,
        "beam": beam,
        "vocabulary_size": len(domain),
        "generic_size": len(generic),
        "oov_rate": oov_rate(domain, val_texts),
        "arms": scores,
        # The three differences the two controls exist to separate.
        "beam_gain": round(scores["greedy"]["cer"] - scores["beam"]["cer"], 4),
        "lexicon_gain": round(scores["beam"]["cer"] - scores["domain"]["cer"], 4),
        "domain_over_generic": round(scores["generic"]["cer"] - scores["domain"]["cer"], 4),
        "total_gain": round(scores["greedy"]["cer"] - scores["domain"]["cer"], 4),
        "best": min(scores, key=lambda k: scores[k]["cer"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="finetune")
    ap.add_argument("--beam", type=int, default=BEAM)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=RUNS / "lexicon.json")
    args = ap.parse_args(argv)

    result = run(args.model, args.beam, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

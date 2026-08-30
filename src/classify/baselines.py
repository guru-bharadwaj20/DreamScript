"""Phase 5.1.4 - the baselines every other number in Phase 5 has to be read against.

    python -m src.classify.baselines

Three of them, because they fail in different directions and each one calibrates a different
metric:

    majority     always the largest class
    stratified   samples the training class distribution at random
    uniform      picks one of the five classes at random

## Why a baseline table is not a formality here

The real corpus is 45% flowchart and 45% wireframe, with 40 circuits, 50 ER diagrams and 50
state machines making up the rest. On a corpus like that **accuracy is close to useless as a
headline**: the majority baseline scores 0.448 without looking at a single feature, and any
model that quietly ignores the three small classes still scores about 0.9 while being useless
for three fifths of the diagram types this project claims to support.

Macro F1 is what Phase 5 selects and reports on, and the baselines are what make it legible.
Five folds, and the two random baselines averaged over five seeds as well:

    baseline      accuracy   macro F1
    majority       0.4478     0.1237
    stratified     0.4179     0.2207
    uniform        0.1784     0.1331

    5.1.1 logistic regression      -    0.7890
    5.1.2 kNN                      -    0.7635
    5.1.3 decision tree            -    0.7539

The gap between `majority` on accuracy (0.4478) and `majority` on macro F1 (0.1237) is the whole
argument in one row. **A metric that a constant predictor scores 0.448 on is not measuring what
this project cares about**, and every headline in Phase 5 is therefore macro F1.

Against the right baseline the three models are worth **0.53 to 0.57 macro F1 over stratified
guessing** - a real result on five classes, three of which have fewer than 50 examples. Against
accuracy the same models would have looked like a modest improvement on doing nothing.

On the synthetic corpus, where the five classes are balanced 600 each, all three baselines
collapse to about 0.20 accuracy and majority scores 0.0667 macro F1. That corpus is an easier
setting in every way, which is another reason 5's `data` module makes the real photographs the
headline.

`stratified` is the more informative of the random baselines because it knows the class
prior - it is the score to beat for "has this model learned anything beyond how common each
type is". `uniform` is lower still and is included because it is the floor for a five-class
problem where the classes were balanced, which is what the synthetic corpus is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.classify.data import Dataset, load

SEED = 42

STRATEGIES = ("most_frequent", "stratified", "uniform")

#: Reported names, because "most_frequent" is sklearn's word for it and "majority" is everyone
#: else's.
NAMES = {"most_frequent": "majority", "stratified": "stratified", "uniform": "uniform"}


def estimator(strategy: str = "most_frequent") -> DummyClassifier:
    return DummyClassifier(strategy=strategy, random_state=SEED)


def evaluate(dataset: Dataset, folds: int = 5, n_jobs: int | None = None, repeats: int = 5) -> dict:
    """Cross-validated accuracy and macro F1 for each baseline.

    The random baselines are averaged over `repeats` seeds as well as folds: a stratified guess
    has real variance, and quoting one draw of it would be quoting noise.
    """
    out = {}
    for strategy in STRATEGIES:
        accuracy, macro = [], []
        rounds = repeats if strategy != "most_frequent" else 1
        for seed in range(rounds):
            scores = cross_validate(
                DummyClassifier(strategy=strategy, random_state=SEED + seed),
                dataset.X,
                dataset.y,
                cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
                scoring=("accuracy", "f1_macro"),
                n_jobs=n_jobs if n_jobs is not None else -1,
            )
            accuracy.append(float(scores["test_accuracy"].mean()))
            macro.append(float(scores["test_f1_macro"].mean()))
        out[NAMES[strategy]] = {
            "accuracy": round(float(np.mean(accuracy)), 4),
            "macro_f1": round(float(np.mean(macro)), 4),
            "accuracy_std": round(float(np.std(accuracy)), 4),
            "macro_f1_std": round(float(np.std(macro)), 4),
            "seeds": rounds,
        }
    return out


def run(corpus: str = "real", folds: int = 5, n_jobs: int | None = None) -> dict:
    dataset = load(corpus)
    counts = dataset.class_counts()
    largest = max(counts.values()) / sum(counts.values())
    return {
        "corpus": corpus,
        "rows": int(len(dataset.y)),
        "classes": counts,
        "majority_class_share": round(float(largest), 4),
        "workers": n_jobs if n_jobs is not None else os.cpu_count(),
        "baselines": evaluate(dataset, folds, n_jobs),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, args.folds, args.jobs), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

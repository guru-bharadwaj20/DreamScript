"""Headline S1: diagram-type accuracy on held-out scribes.

    python -m src.classify.s1

S1 is deliberately not the Phase 5 handcrafted logistic-regression result. That model was an
interpretable baseline, but its 0.789 macro F1 made it a poor production router. The deployed
candidate uses the selected CLIP ViT-B/32 embedding (PCA-reduced to 128 dimensions) and the
one-vs-one RBF SVM selected in Phase 6.3. Every score below is out-of-fold: a known scribe is
never present in both the training and evaluation side of a fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classify.cv import SEEDS, splitter
from src.classify.data import Dataset
from src.utils.config import ROOT

TARGET_ACCURACY = 0.92
REPORT = ROOT / "reports" / "s1_heldout_scribes.json"


def estimator():
    """The Phase 6.3 production candidate, not a model selected on S1 folds."""
    from src.classify.multiclass import pipeline

    return pipeline("ovo", "rbf", "embedding", n_jobs=1)


def evaluate(data: Dataset, seeds=SEEDS, folds: int = 5, estimator_factory=estimator) -> dict:
    """Score an embedding classifier with every known writer held out by group.

    Rows without a published writer id retain the dataset contract's unique ``row:`` group;
    they cannot be falsely grouped with another writer. The report records their count so the
    limitation is visible rather than silently treating all 1,340 rows as writer-identified.
    """
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score

    repeats = []
    known = np.array([str(group).startswith("scribe:") for group in data.groups])
    for seed in seeds:
        predicted = np.empty(len(data.y), dtype=object)
        fold_rows = []
        for fold, (train, test) in enumerate(splitter(data, "grouped", seed, folds)):
            train_scribes = set(data.groups[train][known[train]])
            test_scribes = set(data.groups[test][known[test]])
            overlap = train_scribes & test_scribes
            if overlap:
                raise AssertionError(f"scribe leaked across fold {fold}: {sorted(overlap)!r}")
            model = clone(estimator_factory())
            model.fit(data.X[train], data.y[train])
            predicted[test] = model.predict(data.X[test])
            fold_rows.append(
                {
                    "fold": fold,
                    "rows": int(len(test)),
                    "held_out_known_scribes": int(len(test_scribes)),
                }
            )
        repeats.append(
            {
                "seed": int(seed),
                "accuracy": float(accuracy_score(data.y, predicted)),
                "macro_f1": float(f1_score(data.y, predicted, average="macro", zero_division=0)),
                "folds": fold_rows,
            }
        )

    accuracies = np.array([row["accuracy"] for row in repeats])
    macro_f1s = np.array([row["macro_f1"] for row in repeats])
    return {
        "criterion": "S1",
        "target_accuracy": TARGET_ACCURACY,
        "protocol": "repeated GroupKFold by scribe_id; 5 folds x 3 seeded group permutations",
        "model": "CLIP ViT-B/32 grayscale embedding (128 PCA components) + OvO RBF SVM",
        "rows": int(len(data.y)),
        "known_scribe_rows": int(known.sum()),
        "known_scribes": int(len(set(data.groups[known].tolist()))),
        "unknown_scribe_rows": int((~known).sum()),
        "accuracy": round(float(accuracies.mean()), 4),
        "accuracy_std": round(float(accuracies.std()), 4),
        "macro_f1": round(float(macro_f1s.mean()), 4),
        "macro_f1_std": round(float(macro_f1s.std()), 4),
        "minimum_repeat_accuracy": round(float(accuracies.min()), 4),
        "passes": bool(accuracies.mean() >= TARGET_ACCURACY),
        "repeats": [
            {
                **row,
                "accuracy": round(row["accuracy"], 4),
                "macro_f1": round(row["macro_f1"], 4),
            }
            for row in repeats
        ],
    }


def run() -> dict:
    from src.embed.hybrid import dataset

    return evaluate(dataset("embedding", "real"))


def write_report(result: dict, path: Path = REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        result = run()
        write_report(result)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    sys.exit(main())

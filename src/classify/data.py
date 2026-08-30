"""Phase 5 - the dataset contract every classifier and every evaluation shares.

    from src.classify.data import load

    dataset = load()                     # real photographs, the headline corpus
    dataset.X, dataset.y, dataset.groups

One place decides what a Phase 5 experiment is trained on, because the alternative is nine
modules each making a slightly different choice and a set of numbers that cannot be compared.
Four decisions are made here, and each of them is a finding from Phase 4 rather than a
preference.

## 1. The headline corpus is the real photographs, not the pooled table

4.2.7 measured a linear model separating synthetic pages from real ones at **AUC 0.953** on the
same 34 features. Training on the pool would let a classifier answer "which corpus is this"
instead of "which diagram type is this", and every accuracy afterwards would be partly that
easier question. So `load()` returns the 1,340 real rows; `load(corpus="synthetic")` returns
the 3,000 generated ones, and the two are never mixed unless a caller asks for `"all"`.

The price is a hard, imbalanced problem: **600 flowcharts, 600 wireframes, 50 ER diagrams, 50
state machines, 40 circuits**. The minority classes are 3% of the corpus each, which is the
whole reason 5.2.5 plots precision-recall curves and 5.2.3 reports macro F1 rather than
accuracy alone.

## 2. Cross-validation, not the manifest split

The manifest's own split is unusable for this task, and the number says so plainly: the
training side contains **zero wireframes**, because 1.2 split each source by its own native
convention and every sketch2code page landed in validation. A five-class model cannot be
trained on four classes. Phase 5 therefore evaluates by cross-validation over the whole real
corpus - 5.2.1 stratified, 5.2.2 grouped by scribe - and the manifest split is left for the
deep models in Phase 9, which need a fixed held-out set rather than folds.

## 3. Grouping is honest about where it applies

5.2.2's grouped CV asks whether a model recognises diagram types or handwriting, and it can
only ask that where the corpus records who held the pen. **600 of the 1,340 real rows have no
`scribe_id`** - sketch2code ships no writer identity - so those rows are each given their own
group. That is the conservative reading: a row with an unknown writer is never assumed to share
a writer with anything else. It also means grouped CV constrains the flowchart and chaos rows
(107 and 63 writers) and does nothing for wireframes, and any conclusion drawn from it has to
say so.

## 4. The leak is a switch, not a decision

4.1.5 established that `global_aspect` is a property of the camera that tracks the source
dataset, and 4.2.6 found it **ranks first of 60 by mutual information**. It is kept in the
default feature set, because removing it silently would make Phase 5's numbers look worse
without making them more honest - and `load(drop_leaky=True)` removes it, so every model in 5.1
is measured both ways and the difference is reported rather than argued about.

    python -m src.classify.data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.config import ROOT

#: Features that are properties of the photograph rather than of the diagram. 4.1.5 measured
#: the aspect ratio separating sketch2code (median 0.822) from hdBPMN (1.361), and every
#: sketch2code page is a wireframe.
LEAKY_FEATURES = ("global_aspect",)

#: 4.2.5 dropped this one at |r| = 0.962 against `node_count`. Kept out of the default matrix so
#: that 5.1's models see the pruned table the phase actually recommends.
PRUNED_FEATURES = ("layout_node_density",)

TABLE = ROOT / "data" / "features" / "handcrafted.parquet"


@dataclass
class Dataset:
    """A feature matrix with everything an evaluation needs to be honest about it."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    ids: np.ndarray
    feature_names: list[str]
    corpus: str
    sources: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.y.tolist()))

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def class_counts(self) -> dict[str, int]:
        names, counts = np.unique(self.y, return_counts=True)
        return {str(name): int(count) for name, count in zip(names, counts, strict=True)}

    def summary(self) -> dict:
        known = np.array([str(g).startswith("scribe:") for g in self.groups])
        return {
            "corpus": self.corpus,
            "rows": int(len(self.y)),
            "features": self.n_features,
            "classes": self.class_counts(),
            "groups": int(len(set(self.groups.tolist()))),
            "rows_with_a_known_scribe": int(known.sum()),
            "minority_share": round(float(min(self.class_counts().values()) / len(self.y)), 4),
        }


def feature_columns(
    all_names: list[str], *, drop_leaky: bool = False, drop_pruned: bool = True
) -> list[str]:
    """The feature columns a Phase 5 model sees, in the frozen 4.2.1 order."""
    excluded: set[str] = set()
    if drop_leaky:
        excluded |= set(LEAKY_FEATURES)
    if drop_pruned:
        excluded |= set(PRUNED_FEATURES)
    return [name for name in all_names if name not in excluded]


def _groups_from(frame) -> np.ndarray:
    """One group per writer, and one group per row where the writer is unknown.

    Never a single "unknown" group: that would put 600 wireframes in one fold and make grouped
    CV report a wireframe recall of zero for reasons that have nothing to do with handwriting.
    """
    out = []
    for row_id, scribe in zip(frame["id"], frame["scribe_id"], strict=True):
        if scribe is None or (isinstance(scribe, float) and np.isnan(scribe)) or scribe == "":
            out.append(f"row:{row_id}")
        else:
            out.append(f"scribe:{scribe}")
    return np.array(out, dtype=object)


def load(
    corpus: str = "real",
    *,
    table: Path = TABLE,
    drop_leaky: bool = False,
    drop_pruned: bool = True,
    frame=None,
) -> Dataset:
    """The Phase 5 dataset. `corpus` is one of real | synthetic | all."""
    import pandas as pd

    from src.features.extractor import FEATURE_NAMES

    if frame is None:
        if not Path(table).is_file():
            raise FileNotFoundError(
                f"no feature table at {table}; run python -m src.features.build"
            )
        frame = pd.read_parquet(table)

    if corpus == "real":
        frame = frame[~frame["synthetic"].astype(bool)]
    elif corpus == "synthetic":
        frame = frame[frame["synthetic"].astype(bool)]
    elif corpus != "all":
        raise ValueError(f"corpus must be real, synthetic or all; got {corpus!r}")
    frame = frame.reset_index(drop=True)

    names = feature_columns(list(FEATURE_NAMES), drop_leaky=drop_leaky, drop_pruned=drop_pruned)
    return Dataset(
        X=frame[names].to_numpy(float),
        y=frame["diagram_type"].to_numpy(dtype=object),
        groups=_groups_from(frame),
        ids=frame["id"].to_numpy(dtype=object),
        feature_names=names,
        corpus=corpus,
        sources=frame["source"].to_numpy(dtype=object),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--drop-leaky", action="store_true")
    args = ap.parse_args(argv)

    try:
        dataset = load(args.corpus, drop_leaky=args.drop_leaky)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(dataset.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

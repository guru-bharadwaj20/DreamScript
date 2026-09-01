"""Phase 6.1.4 - handcrafted plus learned, and whether the two together beat either alone.

    python -m src.embed.hybrid          # the three-way comparison
    from src.embed.hybrid import dataset

This is what 6.1 was for. Phase 4 built 33 features by describing a diagram on purpose; 6.1.1
took 512 from a network that has never seen one. The three tables are scored against each other
under one protocol:

    handcrafted   4.2.2's table, exactly as Phase 5 used it
    embedding     6.1.3's 128 PCA columns
    hybrid        both, concatenated

There are only three outcomes and each means something different. If the hybrid wins, the two
representations see different things and Phase 6's models should get both. If the embedding
alone wins, the handcrafted geometry was an expensive detour. If handcrafted alone wins, a
frozen backbone does not transfer to line drawings and 6.2's MLP should be fitted on the
features Phase 4 already has.

## Joined by id, never by position

`cache.aligned` returns embedding rows in the handcrafted table's own id order and raises on a
missing id. The two tables are built by different code from different files, and the failure
mode of concatenating them by position is a model that trains on one page's geometry beside
another page's pixels - which would not error, would score plausibly, and would be wrong.

## The output is a Phase 5 `Dataset`

`dataset()` returns the same structure `src.classify.data.load` returns, so 5.2.1's
cross-validation harness, 5.2.3's report and 5.2.8's significance test all work on an embedding
table unchanged. Reusing that machinery is what makes the numbers below comparable with Phase 5
rather than merely similar to it.

## What it measured

Repeated stratified 5-fold x 3, one untuned multinomial logistic regression for all three
tables, 1,340 real photographs:

    table         features   accuracy   macro F1
    embedding        128      0.9888     0.9554 +- 0.0053
    hybrid           161      0.9876     0.9503 +- 0.0058
    handcrafted       33      0.9468     0.7981 +- 0.0104

**The hybrid does not win.** It scores 0.0051 *below* the embedding alone, and that difference
is the same size as the seed-to-seed spread - which is exactly the situation 5.2.8 was built for,
so the means are not left to speak for themselves:

    comparison                    a right / b right   discordant   p
    embedding vs handcrafted           65 / 2             67       < 0.000001
    handcrafted vs hybrid               2 / 63            65       < 0.000001
    embedding vs hybrid                 2 / 0              2         0.50

**The embedding and the hybrid disagree on two pages out of 1,340.** That is the finding, and it
is far stronger than the 0.0051: adding all thirty-three handcrafted features to the embedding
changes two decisions and neither of them significantly. Against the handcrafted table both
learned representations win on 63-65 discordant pages against 2, at a p-value that leaves
nothing to argue about.

## What this means, stated plainly

Of the three outcomes this task was set up to distinguish, **the second one happened: the
embedding alone wins, and Phase 4's geometry is redundant given it.** Not wrong - the
handcrafted table beats every Phase 5 baseline and 0.7981 is a real result - but it carries
almost no information the CLIP embedding does not already carry, on this corpus. Nine feature
families, four sub-phases, and a 512-number vector from a network that has never seen a diagram
subsumes them.

Three qualifications keep that honest:

**It is one corpus, and a confounded one.** 6.1.1 measured that source fixes the label for 1,200
of these 1,340 rows. It also showed the embedding's advantage survives inside the chaos corpus
where provenance is constant, so this is not purely a provenance result - but the handcrafted
features were built to be robust to capture conditions and the embedding was not, and only
Phase 14's ablation on crossed sources can price that difference properly.

**Redundant is not useless.** 5.3.2 and 5.3.3 read a decision tree and a coefficient table as
sentences about diagrams; no one reads PCA component 47. The handcrafted features remain the
interpretable half of this project and Phase 10 parses graphs with the primitives behind them,
not with an embedding.

**Phase 6 keeps the hybrid table anyway.** It costs 33 extra columns, it is statistically
indistinguishable from the embedding here, and 6.2's MLP and 6.3's kernels are not this logistic
regression - a representation that adds nothing to a linear model can still matter to an RBF
kernel. What this task establishes is that the *linear* case is settled, and that any later
claim for the handcrafted features has to be made against 0.9554 rather than against 0.7981.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classify.data import Dataset
from src.classify.data import load as load_handcrafted

KINDS = ("handcrafted", "embedding", "hybrid")


def dataset(kind: str = "hybrid", corpus: str = "real", *, reduced: bool = True) -> Dataset:
    """One of the three feature tables, as a Phase 5 `Dataset` over the same rows and labels."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}; got {kind!r}")

    base = load_handcrafted(corpus)
    if kind == "handcrafted":
        return base

    from src.embed.cache import aligned

    if reduced:
        from src.embed.cache import load as load_cache
        from src.embed.reduce import REDUCED

        if not REDUCED.is_file():
            raise FileNotFoundError(
                f"no reduced embeddings at {REDUCED}; run python -m src.embed.reduce"
            )
        _, frame, _ = load_cache()
        matrix = np.load(REDUCED)
        position = dict(zip(frame["id"].tolist(), frame["row"].tolist(), strict=True))
        missing = [identifier for identifier in base.ids if identifier not in position]
        if missing:
            raise KeyError(
                f"{len(missing)} of {len(base.ids)} ids are not in the embedding cache "
                f"(first: {missing[0]!r}); rebuild it"
            )
        embeddings = matrix[[position[identifier] for identifier in base.ids]]
        width = embeddings.shape[1]
        names = [f"pca_{i:03d}" for i in range(width)]
    else:
        embeddings, _ = aligned(base.ids)
        names = [f"emb_{i:03d}" for i in range(embeddings.shape[1])]

    if kind == "embedding":
        matrix, feature_names = embeddings.astype(float), names
    else:
        matrix = np.hstack([base.X, embeddings.astype(float)])
        feature_names = list(base.feature_names) + names

    return Dataset(
        X=matrix,
        y=base.y,
        groups=base.groups,
        ids=base.ids,
        feature_names=feature_names,
        corpus=f"{corpus}:{kind}",
        sources=base.sources,
    )


def score(data: Dataset, seeds=(42, 43, 44), folds: int = 5) -> dict:
    """Repeated stratified CV with a logistic regression, the way 5.2.1 scores.

    Deliberately *not* 5.1.1's tuned `BEST_PARAMS`: those hyperparameters were selected on the
    handcrafted table, and reusing them on a 128-column embedding would hand one of the three
    candidates a tuning advantage the other two never got. An untuned multinomial fit is the
    same model for all three, which is the only way this comparison is about the features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    accuracy, macro = [], []
    for seed in seeds:
        pipeline = Pipeline(
            [
                ("prepare", feature_scaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=seed)),
            ]
        )
        predicted = cross_val_predict(
            pipeline,
            data.X,
            data.y,
            cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
            n_jobs=-1,
        )
        accuracy.append(accuracy_score(data.y, predicted))
        macro.append(f1_score(data.y, predicted, average="macro", zero_division=0))

    return {
        "kind": data.corpus,
        "features": data.n_features,
        "rows": int(len(data.y)),
        "accuracy": round(float(np.mean(accuracy)), 4),
        "macro_f1": round(float(np.mean(macro)), 4),
        "macro_f1_std": round(float(np.std(macro)), 4),
        "macro_f1_by_seed": [round(float(value), 4) for value in macro],
    }


def predictions(data: Dataset, seed: int = 42, folds: int = 5) -> np.ndarray:
    """Out-of-fold predictions under one fixed partition, for 5.2.8's McNemar test.

    The three tables share `y`, so an identical seed gives an identical partition and every
    model is compared on the same pages - which is the only condition under which McNemar means
    anything, and the reason 5.2.1 made out-of-fold predictions the shared artefact.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    pipeline = Pipeline(
        [
            ("prepare", feature_scaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=seed)),
        ]
    )
    return cross_val_predict(
        pipeline,
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        n_jobs=-1,
    )


def significance(tables: dict, seed: int = 42) -> list[dict]:
    """McNemar for every pair of feature tables, on identical rows under one partition.

    A 0.005 difference in macro F1 with a seed-to-seed spread of the same size is not a result,
    and 5.2.8 already built the test that says so. Reusing it here rather than restating the
    means is the difference between "the hybrid scored lower" and "the hybrid is worse".
    """
    import itertools

    from src.classify.significance import mcnemar

    predicted = {kind: predictions(table, seed) for kind, table in tables.items()}
    truth = next(iter(tables.values())).y

    rows = []
    for first, second in itertools.combinations(sorted(predicted), 2):
        correct_a = predicted[first] == truth
        correct_b = predicted[second] == truth
        result = mcnemar(correct_a, correct_b)
        rows.append(
            {
                "a": first,
                "b": second,
                "accuracy_a": round(float(correct_a.mean()), 4),
                "accuracy_b": round(float(correct_b.mean()), 4),
                "a_right_b_wrong": result["b"],
                "b_right_a_wrong": result["c"],
                "discordant": result["discordant"],
                "p_value": round(result["p_value"], 6),
                "significant_at_05": bool(result["p_value"] < 0.05),
            }
        )
    return rows


def run(corpus: str = "real", kinds=KINDS, reduced: bool = True) -> dict:
    tables = {kind: dataset(kind, corpus, reduced=reduced) for kind in kinds}
    scores = {kind: score(table) for kind, table in tables.items()}
    best = max(scores.values(), key=lambda row: row["macro_f1"])

    ordered = sorted(scores.values(), key=lambda row: -row["macro_f1"])
    return {
        "corpus": corpus,
        "reduced": reduced,
        "scores": scores,
        "ranking": [row["kind"] for row in ordered],
        "best": best["kind"],
        "mcnemar": significance(tables),
        "hybrid_over_handcrafted": (
            round(scores["hybrid"]["macro_f1"] - scores["handcrafted"]["macro_f1"], 4)
            if {"hybrid", "handcrafted"} <= set(scores)
            else None
        ),
        "hybrid_over_embedding": (
            round(scores["hybrid"]["macro_f1"] - scores["embedding"]["macro_f1"], 4)
            if {"hybrid", "embedding"} <= set(scores)
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--full-width", action="store_true", help="512-d embeddings, not 6.1.3's PCA")
    args = ap.parse_args(argv)

    try:
        print(json.dumps(run(args.corpus, tuple(args.kinds), not args.full_width), indent=2))
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

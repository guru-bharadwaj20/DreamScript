"""Phase 6.2.1 - how deep and how wide, measured rather than guessed.

    python -m src.classify.mlp --search              # the topology sweep
    from src.classify.mlp import best_estimator      # what 6.2.2 onwards refits

The plan asks for 1-3 hidden layers at widths {64, 128, 256, 512}. That is 28 topologies, and
the reason to sweep them rather than pick one is that the right answer depends on a ratio this
corpus makes uncomfortable: **1,072 training rows against a first layer that, at width 512 on a
161-column hybrid table, already has 82,000 parameters.** A network with seventy parameters per
training example is not obviously going to work, and 5.2.9 measured this corpus as still
data-limited. So the sweep is expected to select something small, and if it does not, that is
the finding.

Everything here is scored under Phase 5's protocol - stratified 5-fold, macro F1, the scaler
refit inside every fold - so an MLP's number can be put beside 5.1.1's 0.7898 and 6.1.4's hybrid
table without an argument about how it was measured.

## Why `MLPClassifier` and not torch, for this one task

The topology search is 28 fits of a small dense network on a few thousand rows: it is a CPU
problem across 32 cores, not a GPU one, and sklearn's implementation gets the cross-validation,
the early stopping and the pipeline integration for free. 6.2.4's optimizer comparison is where
a hand-written training loop earns its place, because that task needs the loss at every epoch
and sklearn only exposes the final curve.

## What it measured

Sixteen topologies per table, stratified 5-fold, macro F1, 32 cores, 15 s a sweep:

    table          features   best topology      parameters   macro F1   logreg (6.1.4)
    embedding        128      (256, 256, 256)      165,893     0.9484        0.9554
    hybrid           161      (512, 256)           215,557     0.9310        0.9503
    handcrafted       33      (256, 256)            75,781     0.7620        0.7981

**The prediction in this docstring was wrong, and that is the first finding.** The sweep was
expected to select something small - 1,072 training rows against a 512-wide first layer is
seventy parameters per example, and 5.2.9 measured this corpus as data-limited. It selected the
opposite. Width 512 takes the top two places on the hybrid table, 64 takes the bottom, and the
ordering is close to monotone in width on all three tables. The best hybrid network carries
**215,557 parameters for 1,340 rows - 161 parameters per training example** - and beats every
smaller one. Early stopping on a 12% internal validation split is doing the regularizing that
the parameter count says should be impossible, which is the ordinary modern result and still
worth measuring rather than assuming.

Depth is the one place small wins: two hidden layers beat three on the hybrid table
(0.9310 against 0.9133) and one layer is worst everywhere. Three layers win only on the
embedding table, by 0.0035 against a fold-to-fold spread of 0.017 - which is to say not at all.

**The second finding is the one that matters: none of these networks beats a logistic
regression.** The MLP loses on all three tables - by 0.036 on handcrafted, 0.019 on hybrid,
0.007 on embedding - and it loses with two hundred times the parameters. 6.1.4 got 0.9554 out of
a linear model on the embedding; the best network here gets 0.9484 on the same columns. Nothing
in Unit 2 has yet earned its complexity on this corpus, and 5.2.9's learning curve says why:
the corpus is still on the steep part, so extra capacity has nothing to spend itself on.

That verdict is provisional in one specific way. `alpha` is left at its default here because
regularization is 6.2.3's task, and an under-regularized network on 161 parameters per row is
exactly the configuration that penalty is for. **6.2.3 is where the MLP either closes the 0.019
or confirms it**, and the number it has to beat is 0.9554 rather than 0.7898.

Finally, the embedding table beats the hybrid for the network too (0.9484 against 0.9310), more
decisively than it did for 6.1.4's linear model. Thirty-three handcrafted columns among 128
embedding ones cost a network more than they cost a logistic regression - they are 20% of the
input width and, on 6.1.4's evidence, close to none of the information.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.features.scaling import feature_scaler

SEED = 42

WIDTHS = (64, 128, 256, 512)


#: 1, 2 and 3 hidden layers at each width, plus the two tapering shapes that a fixed-width grid
#: cannot express and that are the usual answer for a small corpus.
def topologies(widths=WIDTHS) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for width in widths:
        shapes.append((width,))
        shapes.append((width, width))
        shapes.append((width, width, width))
    shapes += [(256, 128), (512, 256), (256, 128, 64), (512, 256, 128)]
    return shapes


class StringSafeMLP(MLPClassifier):
    """`MLPClassifier` that accepts the string label arrays this project uses everywhere.

    With `early_stopping=True` sklearn holds out an internal validation split during `fit` and
    scores it through a path that calls `np.isnan(y_pred)` on the **predicted labels**. When
    those labels are strings - as every `Dataset` in this repository has carried since Phase 5 -
    that raises `TypeError`, and casting the dtype does not help because the predictions are
    still text. Encoding to integers for the duration of the fit does, and decoding on the way
    out keeps the estimator's public behaviour identical: `predict` returns the original labels,
    so 5.2.1's harness, 5.2.3's report and `scoring="f1_macro"` all need no special case.
    """

    def fit(self, X, y):  # sklearn's parameter name
        from sklearn.preprocessing import LabelEncoder

        self.encoder_ = LabelEncoder().fit(np.asarray(y).astype(str))
        super().fit(X, self.encoder_.transform(np.asarray(y).astype(str)))
        # `classes_` is part of the estimator's contract - `predict_proba`'s columns are read
        # through it by 5.2.1 - so it has to hold the labels the caller passed in, not the
        # integers this class used internally.
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        encoded = np.asarray(super().predict(X))
        if not hasattr(self, "encoder_"):
            return encoded
        # `super().predict` argmaxes over the network's outputs and maps through `classes_`,
        # which this class has already replaced with the string labels - so a value that is
        # already a label is passed straight back rather than decoded twice.
        if encoded.dtype.kind in "US" or encoded.dtype == object:
            return encoded
        return self.encoder_.inverse_transform(encoded.astype(int))


def pipeline(**kwargs) -> Pipeline:
    """4.2.3's imputation and 4.2.4's scaling, then the network. Every fold refits all three.

    Scaling is not optional for a network the way it was optional for 5.1.3's tree: an unscaled
    `node_count` of 40 beside a `global_ink_coverage` of 0.04 saturates the first layer's
    activations and the gradient through that unit is zero for the rest of training.
    """
    settings = {
        "random_state": SEED,
        "max_iter": 600,
        "early_stopping": True,
        "n_iter_no_change": 15,
        "validation_fraction": 0.12,
        **kwargs,
    }
    return Pipeline([("prepare", feature_scaler()), ("model", StringSafeMLP(**settings))])


#: Selected by the sweep below on the hybrid table (0.9310 macro F1), which 6.1.4 keeps for
#: exactly this reason. `alpha` is still the sklearn default: 6.2.3 is the task that sets it,
#: and it is the one with a real chance of closing the 0.019 gap to 6.1.4's logistic regression.
BEST_PARAMS: dict = {"hidden_layer_sizes": (512, 256), "alpha": 0.0001}


def best_estimator(**kwargs) -> Pipeline:
    return pipeline(**{**BEST_PARAMS, **kwargs})


def search(data, widths=WIDTHS, n_jobs: int | None = None, folds: int = 5) -> dict:
    """Every topology, scored by macro F1 under 5.2.1's fold protocol."""
    shapes = topologies(widths)
    search = GridSearchCV(
        pipeline(),
        {"model__hidden_layer_sizes": shapes},
        scoring="f1_macro",
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs if n_jobs is not None else -1,
        refit=False,
    )
    started = time.perf_counter()
    search.fit(data.X, data.y)
    elapsed = time.perf_counter() - started

    order = np.argsort(-search.cv_results_["mean_test_score"])
    rows = [
        {
            "hidden_layer_sizes": list(
                search.cv_results_["params"][i]["model__hidden_layer_sizes"]
            ),
            "layers": len(search.cv_results_["params"][i]["model__hidden_layer_sizes"]),
            "parameters": parameter_count(
                data.n_features,
                search.cv_results_["params"][i]["model__hidden_layer_sizes"],
                len(set(data.y.tolist())),
            ),
            "macro_f1": round(float(search.cv_results_["mean_test_score"][i]), 4),
            "std": round(float(search.cv_results_["std_test_score"][i]), 4),
        }
        for i in order
    ]
    by_depth: dict[int, float] = {}
    for row in rows:
        by_depth[row["layers"]] = max(by_depth.get(row["layers"], 0.0), row["macro_f1"])

    return {
        "rows": int(len(data.y)),
        "features": data.n_features,
        "candidates": len(shapes),
        "seconds": round(elapsed, 1),
        "workers": n_jobs if n_jobs is not None else os.cpu_count(),
        "best": rows[0],
        "top5": rows[:5],
        "worst": rows[-1],
        "best_by_depth": {str(k): v for k, v in sorted(by_depth.items())},
        "all": rows,
    }


def parameter_count(n_features: int, hidden, n_classes: int) -> int:
    """Weights plus biases, the number the viva will ask for.

    A dense layer from `a` to `b` units is `a*b` weights and `b` biases. The count matters here
    because it is the quantity the corpus size has to be compared against, and 6.2.7 derives the
    gradient that every one of these parameters receives.
    """
    sizes = [n_features, *list(hidden), n_classes]
    return int(sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1)))


def run(table: str = "hybrid", corpus: str = "real", widths=WIDTHS, n_jobs: int | None = None):
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    result = search(data, widths, n_jobs)
    result["table"] = table
    result["corpus"] = corpus
    result["rows_per_parameter"] = round(len(data.y) / max(1, result["best"]["parameters"]), 5)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=["handcrafted", "embedding", "hybrid"])
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--widths", nargs="*", type=int, default=list(WIDTHS))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.table, args.corpus, tuple(args.widths), args.jobs)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if not args.search:
        result = {k: v for k, v in result.items() if k != "all"}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 7.1.8 - what the ensembles cost in latency and memory, against what they bought.

    python -m src.classify.cost --table hybrid

Every task in 7.1 reported a macro F1 and none reported a price. This one is the price list, and
it exists because Phase 16 puts this classifier behind a live camera demo: a model that adds
0.004 macro F1 and 300 ms of latency is a worse product than the one it replaced, and no accuracy
table can say so.

## What is measured, and what each number is worth

    fit_seconds        one fit on the full corpus. Matters for Phase 15's retraining cadence and
                       nothing else - it is paid offline.
    predict_ms_batch   milliseconds per page when 1,340 pages are scored at once. The number a
                       throughput calculation wants.
    predict_ms_single  milliseconds for **one** page, measured one page at a time. This is the
                       number Phase 16 actually experiences, and it is not
                       `predict_ms_batch` - per-call overhead dominates at batch size 1, and for
                       some models the two differ by an order of magnitude.
    model_bytes        the pickled estimator. Phase 15 versions these artifacts and Phase 16
                       loads them into a web process.

**Both latency numbers are reported because they disagree**, and reporting only the batch figure
is the standard way to make an ensemble look cheaper than it is in a request handler.

## What this cannot measure

Feature extraction. Every number here starts from a prepared row of the feature table, and 6.1.2
measured the CLIP embedding at 38 pages/s end-to-end - about 26 ms a page, before any classifier
runs. For the embedding-based models that is the dominant cost and it is identical across all of
them, so it is excluded here and stated rather than folded in silently.

## What it measured

Hybrid table, 1,340 pages, 200 single-page repeats per model, ranked by the latency Phase 16 will
actually see:

    model            fit s    batch ms/page   single ms/page   p95 ms   size KB   latency x
    logreg            0.03       0.0034           0.34          0.57        14      1.0
    linear_svm        0.04       0.0125           0.39          0.81       533      1.2
    rbf_svm           0.04       0.0381           0.42          0.73       548      1.3
    tree              0.16       0.0027           0.61          0.78        13      1.8
    lightgbm          0.99       0.0129           0.71          1.03       472      2.1
    mlp               8.30       0.0105           1.11          1.28       903      3.3
    bagged_trees      2.37       0.0194           2.01          3.60       155      6.0
    random_forest     2.29       0.0186           8.13         16.64      4450     24.3
    stack            34.84       0.0551          13.20         19.07      5899     39.4
    adaboost          5.05       0.0584          18.50         27.96       126     55.2

## The headline is that the ensembles are not, mostly, worth their price

**6.3.7's RBF SVM is the third cheapest model in the table and the most accurate one in the
project** (0.9718 macro F1, 0.42 ms a page, 548 KB). Everything 7.1 built is slower and less
accurate:

    rbf_svm         0.9718 macro F1     0.42 ms
    adaboost        0.9287              18.50 ms      -0.043 F1, 44x the latency
    random_forest   0.8799               8.13 ms      -0.092 F1, 19x
    bagged_trees    0.8644               2.01 ms      -0.107 F1,  4.8x
    stack           0.9723              13.20 ms      +0.0005 F1, 31x

**AdaBoost is the most expensive model in the table and the second-least accurate of the four.**
Its 126 KB pickle is small - it stores shallow trees - and that is exactly why it is slow: 500
trees, each a Python-level `predict` call on one row, is 500 dispatches for one page. The stack is
the other clear loss: +0.0005 macro F1 for 31x the latency, 408x the disk, and a 35-second fit.
7.1.7 already documented that its accuracy gain is inside noise; this is the second half of the
same verdict.

## Batch and single latency disagree by up to 437x

    random_forest   0.0186 ms batched   ->   8.13 ms single    (437x)
    adaboost        0.0584              ->  18.50              (317x)
    stack           0.0551              ->  13.20              (240x)
    tree            0.0027              ->   0.61              (224x)
    rbf_svm         0.0381              ->   0.42              ( 11x)

Every model is dominated by per-call overhead at batch size 1, and the ensembles are dominated
*most*, because their overhead is per member rather than per model. **Quoting the batch column
would have made the forest look 2.0x cheaper than the RBF SVM when it is in fact 19x dearer** in
the setting Phase 16 runs in. That inversion is the reason both columns are here.

The p95 column carries the same warning further: the forest's 95th-percentile page costs 16.6 ms
against a median 8.1, so its tail is twice its typical case, while the RBF SVM's tail (0.73 ms)
stays under a millisecond.

## What all of this is measured against

6.1.2's CLIP feature extraction is **26 ms a page**, and every number above is downstream of it.
Seven of the ten models cost under 4% of that; the classifier is not the bottleneck for any of
them, and even AdaBoost's 18.5 ms only doubles the page cost rather than changing its order of
magnitude. **On latency alone there is no wrong answer here.** The reason to prefer the RBF SVM is
that it is simultaneously the most accurate, nearly the cheapest, and 8x smaller on disk than the
forest - not that its rivals would blow a budget.

Fit time is the one place the ensembles are genuinely expensive: the stack's 34.8 s is 1,202x the
logistic regression's, which matters for Phase 15's retraining cadence and nowhere else.

## What is not in these numbers

Feature extraction, image decoding and preprocessing - `predict_ms_*` starts from a prepared row.
7.2.6 measures the whole chain end to end, and it is there that the 26 ms above turns out to be
the number that decides which path is fast.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import time

import numpy as np

from src.classify.activations import TABLES

SEED = 42

#: One model per family in the project so far, cheapest first. `logreg` is the reference: every
#: cost below is also reported as a multiple of it.
MODELS = (
    "logreg",
    "tree",
    "linear_svm",
    "rbf_svm",
    "bagged_trees",
    "random_forest",
    "adaboost",
    "lightgbm",
    "mlp",
    "stack",
)

REFERENCE = "logreg"

#: Repeats for the single-page timing. One call is dominated by whatever the interpreter was
#: doing a microsecond earlier; the median of many is a number that reproduces.
SINGLE_REPEATS = 200


def estimator(name: str, table: str = "hybrid"):
    """One model per family, at the settings its own task selected."""
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        inner = LogisticRegression(max_iter=3000, random_state=SEED)
    elif name == "tree":
        from sklearn.tree import DecisionTreeClassifier

        from src.classify.tree import BEST_PARAMS

        inner = DecisionTreeClassifier(random_state=SEED, **BEST_PARAMS)
    elif name in ("linear_svm", "rbf_svm"):
        from src.classify.multiclass import base_estimator

        inner = base_estimator("linear" if name == "linear_svm" else "rbf", table)
    elif name == "bagged_trees":
        from src.classify.bagging import pipeline as bag

        return bag("tree_tuned", n_estimators=25)
    elif name == "random_forest":
        from src.classify.forest import pipeline as rf

        return rf(n_estimators=300, max_features="sqrt", max_depth=None)
    elif name == "adaboost":
        from src.classify.adaboost import pipeline as ada

        return ada(depth=1, n_estimators=200, learning_rate=1.0)
    elif name == "lightgbm":
        from src.classify.boosting import pipeline as boost

        return boost(
            "lightgbm",
            learning_rate=0.1,
            max_depth=3,
            n_estimators=100,
            subsample=1.0,
            colsample=1.0,
        )
    elif name == "mlp":
        from src.classify.regularize import ACTIVATION, best_settings
        from src.classify.torchnet import TorchMLP

        inner = TorchMLP(
            hidden_layer_sizes=(512, 256),
            activation=ACTIVATION,
            batch_size=16,
            device="cpu",
            **best_settings(table),
        )
    elif name == "stack":
        from src.classify.stacking import stack

        return stack(table)
    else:
        raise ValueError(f"model must be one of {list(MODELS)}; got {name!r}")

    return Pipeline([("prepare", feature_scaler()), ("model", inner)])


def pickled_bytes(fitted) -> int:
    """Size of the serialised estimator - what Phase 15 versions and Phase 16 loads."""
    buffer = io.BytesIO()
    pickle.dump(fitted, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return buffer.tell()


def measure(data, name: str, table: str = "hybrid", repeats: int = SINGLE_REPEATS) -> dict:
    """Fit once, then time a batch prediction and a single-page prediction separately."""
    fitted_estimator = estimator(name, table)

    started = time.perf_counter()
    fitted = fitted_estimator.fit(data.X, data.y)
    fit_seconds = time.perf_counter() - started

    # A warm-up call: the first prediction of several of these models allocates buffers, and
    # timing it would report the allocator rather than the model.
    fitted.predict(data.X[:8])

    started = time.perf_counter()
    fitted.predict(data.X)
    batch_seconds = time.perf_counter() - started

    rng = np.random.default_rng(SEED)
    rows = rng.integers(0, len(data.y), size=repeats)
    singles = []
    for row in rows:
        one = data.X[row : row + 1]
        started = time.perf_counter()
        fitted.predict(one)
        singles.append(time.perf_counter() - started)

    return {
        "model": name,
        "fit_seconds": round(fit_seconds, 3),
        "predict_ms_batch": round(batch_seconds / len(data.y) * 1000, 4),
        "predict_ms_single": round(float(np.median(singles)) * 1000, 3),
        "predict_ms_single_p95": round(float(np.percentile(singles, 95)) * 1000, 3),
        "model_bytes": pickled_bytes(fitted),
    }


def relative(rows: list[dict], reference: str = REFERENCE) -> list[dict]:
    """Every cost as a multiple of the cheapest model's, which is how a table like this is read."""
    base = next((row for row in rows if row["model"] == reference), None)
    if base is None:
        return rows
    out = []
    for row in rows:
        out.append(
            {
                **row,
                "fit_x": round(row["fit_seconds"] / max(base["fit_seconds"], 1e-9), 1),
                "single_latency_x": round(
                    row["predict_ms_single"] / max(base["predict_ms_single"], 1e-9), 1
                ),
                "size_x": round(row["model_bytes"] / max(base["model_bytes"], 1), 1),
            }
        )
    return out


def to_markdown(rows: list[dict], accuracy: dict | None = None) -> str:
    header = (
        "| model | macro F1 | fit s | batch ms/page | single ms/page | p95 ms | size KB | "
        "latency x |"
    )
    divider = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, divider]
    for row in rows:
        score = (accuracy or {}).get(row["model"])
        lines.append(
            f"| `{row['model']}` | {score if score is not None else '-'} | "
            f"{row['fit_seconds']:.2f} | {row['predict_ms_batch']:.4f} | "
            f"{row['predict_ms_single']:.2f} | {row['predict_ms_single_p95']:.2f} | "
            f"{row['model_bytes'] / 1024:.0f} | {row.get('single_latency_x', '-')} |"
        )
    return "\n".join(lines)


def run(
    table: str = "hybrid",
    corpus: str = "real",
    models=MODELS,
    repeats: int = SINGLE_REPEATS,
    accuracy: dict | None = None,
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    rows = relative([measure(data, name, table, repeats) for name in models])
    rows.sort(key=lambda row: row["predict_ms_single"])

    batch = {row["model"]: row["predict_ms_batch"] for row in rows}
    single = {row["model"]: row["predict_ms_single"] for row in rows}
    return {
        "table": table,
        "corpus": corpus,
        "rows": int(len(data.y)),
        "repeats": repeats,
        "costs": rows,
        "cheapest": rows[0]["model"],
        "dearest": rows[-1]["model"],
        "single_latency_spread_x": round(
            rows[-1]["predict_ms_single"] / max(rows[0]["predict_ms_single"], 1e-9), 1
        ),
        # The reason both latencies are reported: they rank the models differently.
        "batch_to_single_ratio": {
            name: round(single[name] / max(batch[name], 1e-9), 1) for name in single
        },
        "feature_extraction_ms_per_page_6_1_2": 26.0,
        "markdown": to_markdown(rows, accuracy),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=list(TABLES))
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--repeats", type=int, default=SINGLE_REPEATS)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.table, args.corpus, tuple(args.models), args.repeats)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(result["markdown"] if args.markdown else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

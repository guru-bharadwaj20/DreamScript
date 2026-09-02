"""Phase 6.3.7 - which feature set wins per model, and the interaction Phase 6 has been implying.

    python -m src.classify.crosstable

The plan asks for a cross-table: feature set against model, which wins where. This is the last
task of Phase 6 and the one that can settle a question none of the preceding sixteen could
answer on its own, because each of them held one axis fixed.

Every earlier task reported its numbers per table and every one of them noticed the same thing
from a different angle:

    6.1.4  the embedding beats the handcrafted table for a *linear* model, by 0.157
    6.2.6  the network's best table is the hybrid, and its worst is the handcrafted
    6.3.1  the linear SVM beats logistic regression on the learned tables and *loses* on the
           handcrafted one, by 0.033
    6.3.3  the RBF gains +0.065 on the handcrafted table and +0.005 on the embedding
    6.3.4  the multiclass scheme is worth 0.047 on the handcrafted table and 0.001 on the learned

**Each of those is one row or one column of the same table.** Stated together they amount to a
claim about an *interaction* - that the value of model capacity depends on the feature set, and
specifically that capacity is worth a great deal on the 33-column geometric table and almost
nothing on the learned ones. This task builds the full grid and checks whether that claim
survives being made explicit.

## The design

Six models x three feature tables, one protocol, 5.2.1's folds. The models span the whole phase
rather than only 6.3's: two linear baselines, the network, and the three kernels, because a
cross-table containing only SVMs could not show that the interaction is about capacity rather
than about kernels.

Cells are compared **within a column** (which model wins on this table) and **within a row**
(which table wins for this model), and the interaction is quantified as the difference between
the two spreads. A pure main effect would give the same model ranking on every table.

## What it measured

Six models x three feature tables, stratified 5-fold, macro F1, each model at the settings its
own task selected:

    model          hybrid (161)   embedding (128)   handcrafted (33)   best table
    logreg           0.9570         *0.9627*           0.7873          embedding
    linear_svm      *0.9628*         0.9558            0.7637          hybrid
    mlp             *0.9595*         0.9468            0.7777          hybrid
    poly_svm        *0.9690*         0.9661            0.7935          hybrid
    rbf_svm          0.9690         *0.9742*           0.8310          embedding
    ovo_rbf          0.9658         *0.9742*           0.8227          embedding

    model choice is worth   0.0120         0.0274            0.0673

## The headline: the feature set matters far more than the model

    changing the feature table, for a fixed model      0.143 - 0.199
    changing the model, for a fixed feature table      0.012 - 0.067

**Across every row and every column, the choice of feature table is worth between three and
sixteen times the choice of model.** The worst model on the best table (0.9468) beats the best
model on the worst table (0.8310) by 0.116. Seventeen tasks of Phase 6 have tuned architectures,
activations, optimizers, schedules, batch sizes, kernels and multiclass schemes, and the largest
single decision in the whole phase was made in 6.1.1 when a frozen CLIP backbone replaced
thirty-three hand-built geometric features.

## The interaction is real, and it is the one the earlier tasks kept implying

**Model choice is worth 0.0673 on the handcrafted table and 0.0120 on the hybrid - 5.6 times as
much. The interaction term is 0.0553.** Capacity buys a great deal where the representation is
poor and almost nothing where it is good, which is exactly what 6.3.3 saw as the RBF's +0.065 on
the handcrafted table against +0.005 on the embedding, and what 6.3.4 saw as the multiclass scheme
being worth 0.047 there and 0.001 on the learned tables. Six separate observations across the
phase were one interaction seen from six angles, and the grid confirms it as a main statement.

The hybrid table is the flattest of the three: six models spanning a logistic regression and a
228,357-parameter network land inside 0.012 of each other. **On a good enough representation, the
model stops mattering.**

## Ranking stability

    embedding vs handcrafted    rho = 0.928
    hybrid    vs embedding      rho = 0.691
    hybrid    vs handcrafted    rho = 0.667

The model ordering is nearly identical between the embedding and handcrafted tables and only
moderately preserved on the hybrid - which is the flatness above showing up again, since a
0.012 spread orders six models mostly by noise. `rbf_svm` and `ovo_rbf` take the top two places on
both the embedding and handcrafted tables; on the hybrid, `poly_svm` edges ahead and the ordering
below it is not meaningful.

## What wins

**`rbf_svm` on the embedding table, 0.9742** - the best cell in Phase 6, and the same
configuration 6.3.3 selected. It is an SVM with an RBF kernel on 128 PCA components of a frozen
CLIP embedding, and it beats the best network in the phase (0.9595) by 0.015 with a fraction of
the parameters and no training loop.

Two qualifications carry forward unchanged. **6.1.1's source confound is not resolved by any of
this**: provenance fixes the label for 1,200 of 1,340 rows, and every number in this table
inherits that. And **the handcrafted table remains the interpretable one** - 5.3.2 and 5.3.3 read
a tree and a coefficient table as statements about diagrams, and nobody reads PCA component 47 -
so its 0.83 is not simply a worse version of 0.97.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.classify.activations import TABLES

SEED = 42

#: The whole phase, not only 6.3. `logreg` is 6.1.4's baseline and `mlp` is 6.2's best network,
#: so the table can distinguish "kernels like this feature set" from "capacity likes it".
MODELS = ("logreg", "linear_svm", "mlp", "poly_svm", "rbf_svm", "ovo_rbf")


def estimator(model: str, table: str = "hybrid"):
    """One model at the settings its own task selected, wrapped in 4.2.3/4.2.4's preparation."""
    from sklearn.pipeline import Pipeline

    from src.features.scaling import feature_scaler

    if model == "logreg":
        from sklearn.linear_model import LogisticRegression

        inner = LogisticRegression(max_iter=3000, random_state=SEED)
    elif model == "linear_svm":
        from sklearn.svm import LinearSVC

        from src.classify.svm import BEST_PARAMS

        inner = LinearSVC(
            C=BEST_PARAMS["C"],
            loss=BEST_PARAMS["loss"],
            dual="auto",
            max_iter=200_000,
            random_state=SEED,
        )
    elif model == "mlp":
        from src.classify.multiclass import base_estimator  # noqa: F401  (kept symmetric)
        from src.classify.regularize import ACTIVATION, best_settings
        from src.classify.torchnet import TorchMLP

        inner = TorchMLP(
            hidden_layer_sizes=(512, 256),
            activation=ACTIVATION,
            batch_size=16,
            device="cpu",
            **best_settings(table),
        )
    elif model in ("poly_svm", "rbf_svm"):
        from src.classify.multiclass import base_estimator

        inner = base_estimator("poly" if model == "poly_svm" else "rbf", table)
    elif model == "ovo_rbf":
        from sklearn.multiclass import OneVsOneClassifier

        from src.classify.multiclass import base_estimator

        inner = OneVsOneClassifier(base_estimator("rbf", table))
    else:
        raise ValueError(f"model must be one of {list(MODELS)}; got {model!r}")

    return Pipeline([("prepare", feature_scaler()), ("model", inner)])


def cell(data, model: str, table: str, folds: int = 5, n_jobs: int | None = None) -> float:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # The network is fitted sequentially: 6.2.2 measured 32 CUDA-free workers around a torch fit
    # wedging the pool, and `sweep` exists precisely to avoid nesting that inside a CV loop.
    workers = 1 if model == "mlp" else (n_jobs if n_jobs is not None else -1)
    predicted = cross_val_predict(
        estimator(model, table),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=workers,
    )
    return round(float(f1_score(data.y, predicted, average="macro", zero_division=0)), 4)


def build(models=MODELS, tables=TABLES, corpus: str = "real", n_jobs: int | None = None) -> dict:
    """The full grid, as model -> table -> macro F1."""
    from src.embed.hybrid import dataset

    loaded = {table: dataset(table, corpus) for table in tables}
    return {
        model: {table: cell(loaded[table], model, table, n_jobs=n_jobs) for table in tables}
        for model in models
    }


def winners(grid: dict, tables=TABLES) -> dict:
    """Which model wins each table, and which table wins for each model."""
    best_model_per_table = {
        table: max(grid, key=lambda model: grid[model][table]) for table in tables
    }
    best_table_per_model = {
        model: max(tables, key=lambda table: grid[model][table]) for model in grid
    }
    return {
        "best_model_per_table": best_model_per_table,
        "best_table_per_model": best_table_per_model,
        "best_overall": max(
            ((model, table) for model in grid for table in tables),
            key=lambda pair: grid[pair[0]][pair[1]],
        ),
    }


def spreads(grid: dict, tables=TABLES) -> dict:
    """How much the model choice is worth on each table, and the table choice for each model.

    The interaction this task exists to test is the *difference between the model spreads*: if
    capacity is worth much more on one feature set than another, these numbers separate.
    """
    model_spread = {
        table: round(max(grid[m][table] for m in grid) - min(grid[m][table] for m in grid), 4)
        for table in tables
    }
    table_spread = {
        model: round(max(grid[model][t] for t in tables) - min(grid[model][t] for t in tables), 4)
        for model in grid
    }
    return {
        "model_choice_worth_on_table": model_spread,
        "table_choice_worth_for_model": table_spread,
        "interaction": round(max(model_spread.values()) - min(model_spread.values()), 4),
    }


def ranking_stability(grid: dict, tables=TABLES) -> dict:
    """Does the model ranking change between feature tables?

    This is the qualitative form of the interaction, and the more convincing one: a pure main
    effect means the same ordering on every table, whatever the spreads happen to be. Reported as
    Spearman correlation between the per-table orderings.
    """
    from scipy import stats

    orders = {table: [grid[model][table] for model in grid] for table in tables}
    pairs = {}
    names = list(tables)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = stats.spearmanr(orders[names[i]], orders[names[j]]).statistic
            pairs[f"{names[i]} vs {names[j]}"] = round(float(rho), 4)
    return {
        "spearman_between_tables": pairs,
        "ranking_per_table": {
            table: sorted(grid, key=lambda model: -grid[model][table]) for table in tables
        },
    }


def to_markdown(grid: dict, tables=TABLES) -> str:
    header = "| model | " + " | ".join(tables) + " | best |"
    divider = "| :--- | " + " | ".join("---:" for _ in tables) + " | :--- |"
    lines = [header, divider]
    for model in grid:
        row = grid[model]
        best = max(tables, key=lambda table: row[table])
        cells = " | ".join((f"**{row[t]:.4f}**" if t == best else f"{row[t]:.4f}") for t in tables)
        lines.append(f"| `{model}` | {cells} | {best} |")
    return "\n".join(lines)


def run(corpus: str = "real", models=MODELS, tables=TABLES, n_jobs: int | None = None) -> dict:
    grid = build(models, tables, corpus, n_jobs)
    return {
        "corpus": corpus,
        "grid": grid,
        "winners": winners(grid, tables),
        "spreads": spreads(grid, tables),
        "ranking_stability": ranking_stability(grid, tables),
        "markdown": to_markdown(grid, tables),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.corpus, tuple(args.models), TABLES, args.jobs)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    if args.markdown:
        print(result["markdown"])
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

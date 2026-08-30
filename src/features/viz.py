"""Phase 4.2.7 - PCA, t-SNE and UMAP of the feature space, coloured by diagram type.

    python -m src.features.viz          # writes reports/figures/p4_feature_space.png

Three projections of the same 34-dimensional table, side by side, because each answers a
different question and the differences between them are the finding:

* **PCA** is linear, deterministic and reversible. If the classes separate here, a linear model
  in 5.1 can find them, and the explained-variance ratio says how much of the table's variation
  the picture is showing.
* **t-SNE** preserves local neighbourhoods and destroys global distances. Clusters are real;
  the space between them is not, and the size of a cluster means nothing.
* **UMAP** keeps more global structure than t-SNE and is faster, at the cost of a projection
  that is harder to reason about.

All three are fitted on the **scaled** matrix from 4.2.4 - unscaled, every projection is a plot
of `node_count` against `edge_count`, since those two columns carry variance in the hundreds
while `global_ink_coverage` carries 1e-4.

## What the picture is not

A projection is not a classifier and separation in it is not accuracy. Two classes that overlap
in two dimensions can be perfectly separable in 34. **Nothing here is evidence for Phase 5's
numbers.**

## What the figure shows, and the number the eye could not give

Over 3,000 sampled rows:

* **PCA holds 45% of the variance in two components** (0.329 and 0.119) and the five types are
  not linearly separable in them: one dense lobe with every colour in it, plus a detached
  cluster on the right. A linear model working in two dimensions would fail here, which is not
  the same as a linear model failing in 34.
* **t-SNE and UMAP both find many small, largely single-colour clusters.** State machines and ER
  diagrams form their own groups; flowcharts and wireframes each break into several. Local
  structure exists and it follows the labels, which is the encouraging half of the figure.

The fourth panel is the check the module exists for, and **it does not pass cleanly**. By eye
the real and synthetic rows overlap in the main lobe. But a logistic model asked to predict
`synthetic` from the same 34 features scores **AUC 0.992 on this sample, and 0.953 over the full
4,340-row table**, in five-fold cross-validation: the two corpora are almost perfectly
separable, and the overlap in the picture is an artefact of projecting 34 dimensions into 2.

That has a direct consequence for Phase 5, so it is stated here rather than discovered there:
**a model trained on the synthetic corpus is not expected to transfer to photographs**, and a
model trained on the pool will find "which corpus is this" a much easier problem than "which
diagram type is this". The `synthetic` flag in the table exists so that the two can be trained
and evaluated separately, and 14's ablations should include a synthetic-to-real transfer number.

For scale: a linear model on this table predicts *diagram type* at 0.937 accuracy under the same
naive five-fold split - a number that is not to be quoted, because it pools the two corpora,
ignores the scribe-disjoint splits from 1.2 and includes the `global_aspect` leak that 4.2.6
ranked first. It is recorded only to make the point that the corpus is the easier question.

    python -m src.features.viz --sample 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.utils.config import ROOT

FIGURE = ROOT / "reports" / "figures" / "p4_feature_space.png"

#: Colours per diagram type. Fixed so this figure and every later one agree.
COLOURS = {
    "flowchart": "#3b6ea5",
    "wireframe": "#c1663d",
    "state_machine": "#4b8b3b",
    "er_diagram": "#8a5fa8",
    "circuit": "#a63d54",
}

#: t-SNE and UMAP are O(n^2)-ish and this is a figure, not an experiment.
DEFAULT_SAMPLE = 3000
SEED = 42


def projections(matrix: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    """PCA, t-SNE and UMAP embeddings of one scaled matrix."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out: dict[str, np.ndarray] = {}
    pca = PCA(n_components=2, random_state=seed)
    out["pca"] = pca.fit_transform(matrix)
    out["_pca_explained"] = pca.explained_variance_ratio_

    perplexity = min(30.0, max(5.0, (len(matrix) - 1) / 3))
    out["tsne"] = TSNE(
        n_components=2, random_state=seed, perplexity=perplexity, init="pca"
    ).fit_transform(matrix)

    try:
        import umap

        out["umap"] = umap.UMAP(n_components=2, random_state=seed).fit_transform(matrix)
    except (ImportError, TypeError) as error:  # pragma: no cover - depends on the environment
        out["_umap_error"] = str(error)
    return out


def figure(embeddings: dict, labels, synthetic, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [key for key in ("pca", "tsne", "umap") if key in embeddings]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.6 * (len(panels) + 1), 4.4))
    labels = np.asarray(labels)

    for ax, key in zip(axes[: len(panels)], panels, strict=True):
        points = embeddings[key]
        for kind in sorted(set(labels.tolist())):
            mask = labels == kind
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=6,
                alpha=0.55,
                linewidths=0,
                label=kind,
                color=COLOURS.get(kind, "#777777"),
            )
        title = key.upper()
        if key == "pca" and "_pca_explained" in embeddings:
            share = embeddings["_pca_explained"]
            title += f" ({100 * share.sum():.0f}% of variance)"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=7, markerscale=1.6, loc="best", frameon=False)

    check = axes[-1]
    synthetic = np.asarray(synthetic, bool)
    for flag, colour, name in ((False, "#333333", "real"), (True, "#bbbbbb", "synthetic")):
        mask = synthetic == flag
        check.scatter(
            embeddings["pca"][mask, 0],
            embeddings["pca"][mask, 1],
            s=6,
            alpha=0.55,
            linewidths=0,
            color=colour,
            label=name,
        )
    check.set_title("PCA, coloured by corpus\n(two clusters here would be bad news)", fontsize=10)
    check.set_xticks([])
    check.set_yticks([])
    check.legend(fontsize=7, markerscale=1.6, frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def corpus_separability(table, seed: int = SEED) -> float:
    """Cross-validated AUC of a linear model asked to tell synthetic pages from real ones.

    The eye cannot read this off a PCA panel - overlapping ink in two dimensions says nothing
    about 34 - so the panel gets a number beside it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    from src.features.extractor import FEATURE_NAMES
    from src.features.scaling import feature_scaler

    labels = table["synthetic"].to_numpy(int)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    pipeline = Pipeline(
        [
            ("prepare", feature_scaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    matrix = table[list(FEATURE_NAMES)].to_numpy(float)
    return float(cross_val_score(pipeline, matrix, labels, cv=5, scoring="roc_auc").mean())


def run(table=None, sample: int = DEFAULT_SAMPLE, seed: int = SEED) -> dict:
    import pandas as pd

    from src.features.build import OUT
    from src.features.extractor import FEATURE_NAMES
    from src.features.scaling import feature_scaler

    if table is None:
        if not OUT.is_file():
            return {"rows": 0}
        table = pd.read_parquet(OUT)

    if sample and len(table) > sample:
        table = table.sample(sample, random_state=seed)

    matrix = table[list(FEATURE_NAMES)].to_numpy(float)
    scaled = feature_scaler().fit_transform(matrix)
    embeddings = projections(scaled, seed)
    path = figure(embeddings, table["diagram_type"].to_numpy(), table["synthetic"].to_numpy())

    return {
        "rows": int(len(table)),
        "columns_after_scaling": int(scaled.shape[1]),
        "corpus_separability_auc": round(corpus_separability(table, seed), 4),
        "pca_explained_variance": [round(float(v), 4) for v in embeddings["_pca_explained"]],
        "projections": [key for key in ("pca", "tsne", "umap") if key in embeddings],
        "umap_error": embeddings.get("_umap_error"),
        "figure": str(path.relative_to(ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    args = ap.parse_args(argv)

    table = None
    if args.table:
        import pandas as pd

        table = pd.read_parquet(args.table)
    result = run(table, args.sample)
    if not result["rows"]:
        print("no feature table; run python -m src.features.build first", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

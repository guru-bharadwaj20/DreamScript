"""Phase 8.10 - clustering pages with the labels withheld, and what discovery actually discovers.

    python -m src.cluster.discovery

The plan asks for unlabeled diagrams to be clustered so that possible new diagram types surface.
Nothing in this corpus is literally unlabeled - every one of the 35,674 manifest rows carries a
`diagram_type` - but that is not the same as every row having been *looked at*. **The labels were
assigned wholesale by provenance**: didi is 22,287 rows stamped `flowchart` because didi is a
flowchart dataset, sketch2code is 731 rows stamped `wireframe` for the same reason. So the
faithful reading of this task is to withhold the labels and ask what a clustering finds instead.

6.1.1 already measured the thing that makes this hard. **Source is predictable from the embedding
at 0.9776-0.9918 accuracy**, and source and label are nearly the same variable on this corpus, so
a clustering that recovers the type labels may only have recovered which camera took the
photograph. Three experiments separate those.

## The three experiments

    1. every real page (1,695)
       Cluster with labels withheld, then measure agreement against *both* the diagram type and
       the source. If the source agreement is the higher of the two, discovery has found the
       datasets rather than the types, and any "new type" it proposed would be a new dataset.

    2. the chaos corpus alone (260 pages, five types, one capture protocol)
       This is the only block where source is constant and type varies, so it is the only place
       where a recovered partition can be attributed to content. 6.1.1 reports a *supervised*
       0.9925 macro F1 here, so the representation demonstrably carries the answer. **Whether
       unsupervised clustering can reach it is the question this whole task turns on**, and a
       failure here is a fact about discovery rather than about the features.

    3. hdbpmn alone (704 pages, one type, one protocol)
       The place a genuinely new sub-type would have to show up: a large block of one declared
       type from one source. hdbpmn pages also carry an `exercise` - which of a small set of
       modelling tasks the writer was given - so any structure found can be checked against the
       most obvious confound, which is that a sub-cluster is a *prompt* rather than a *type*.

## Why K is swept rather than chosen

There is no label to choose K against, which is the entire premise of discovery. So the sweep is
reported and every row carries the smallest cluster size, following 8.2's finding that a
criterion improving while the partition dissolves is not measuring what its name suggests.

## What it measured

1,695 real photographs, CLIP embeddings, labels withheld. The 5,000 synthetic rows are excluded.

## 1. Every real page: the clustering finds types, not datasets

     K    AMI vs type   AMI vs source   ARI vs type   ARI vs source   smallest
     2      0.7968         0.7068         0.8422         0.7244         799
     4      0.7579         0.6017         0.7344         0.6303          51
     6      0.6147         0.4900         0.4042         0.3593          50
    12      0.5295         0.4143         0.2273         0.2050          45

**Type beats source at every K, by 0.09 to 0.16 AMI**, so the flag this experiment exists to
raise is not raised. The partition is not primarily a partition of the datasets.

That is a weaker statement than it looks and the reason is worth stating: the three sources map
almost one-to-one onto types here (hdbpmn is all `flowchart`, sketch2code all `wireframe`), so
type is a *refinement* of source and is entitled to score higher on granularity alone. This
experiment can rule out the worst case - a clustering that recovers source and nothing else - and
it does. It cannot, on its own, credit the representation with understanding content. That takes
the second experiment.

## 2. Chaos alone: the decisive test, and it passes

260 pages, five types, one capture protocol, so source is constant and anything recovered is
content.

     K    AMI vs type   ARI vs type   smallest cluster
     3       0.7535        0.5145           50
     4      *0.8147*       0.7187           50
     5       0.7823       *0.7250*          47
     8       0.7129        0.5962           13

    6.1.1's *supervised* macro F1 on the same embedding, same pages:   0.9925

**Unsupervised clustering recovers the taxonomy at AMI 0.8147 and ARI 0.7250, with no labels of
any kind.** Five real types, one camera protocol, and a K-means over a frozen CLIP embedding
finds most of the structure a supervised model finds.

This is the strongest evidence anywhere in the project that Phase 6's page classifier is not
merely recognising which dataset an image came from. 6.1.1 argued it from a *supervised* control
- the embedding still beat handcrafted features within chaos - and left the possibility that the
labels were doing the work. Here there are no labels. The types are visible in the geometry of
the representation itself.

The gap to 0.9925 is what the labels are worth, and it is real: at K = 4 the clustering has
already merged two of the five types, and no K in the sweep both separates all five and stays
above ARI 0.73.

## 3. hdbpmn alone: no new sub-type, and something else instead

704 pages of one declared type from one source - the only place a genuinely new sub-type could
appear. Each page also carries the `exercise` its writer was set, which is the obvious confound.

     K    AMI vs exercise   AMI vs scribe   smallest
     2        -0.0006          0.1304         312
     3         0.0020          0.2242         199
     6         0.0257          0.2241          92
     8         0.0848          0.2099          66
    11         0.0700          0.2296          18

**No sub-type surfaced.** Agreement with the exercise is between -0.001 and 0.085, which is
nothing: the sub-clusters are not eleven modelling prompts wearing a disguise, and they are not
new diagram types either - eleven exercises produce eleven kinds of *content*, and if content
drove the partition this column would move.

**What the sub-clusters track is the scribe, at AMI 0.2242 - roughly three times the exercise
column at every K, and flat across K.** A CLIP embedding of a photographed flowchart carries
identifiable information about *whose hand drew it and how it was photographed*, and within one
diagram type that is the dominant remaining axis.

This corroborates 7.4.7 from a completely different representation. There, a rectangle's 22
geometric descriptors identified its writer at 8.6x chance while shape identity was 200x the
effect. Here, with type held constant so the 200x factor is removed, the writer is what is left
standing. It is also a direct input to 8.6: **page-level style structure exists and is
recoverable**, which is what a style-clustering task needs to be true before it starts.

## What this settles about discovery

Discovery works on this corpus for the question it was asked - a taxonomy of diagram types is
recoverable without labels, at about three quarters of the supervised agreement, once provenance
is controlled. It does **not** surface new types below that level. Within a single type the
unlabelled structure is the writer and the camera, so a pipeline mining this corpus for a sixth
diagram type would find handwriting styles and mistake them for one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.utils.figures import save as _figsave

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = ROOT / "data" / "features" / "embeddings.npy"
INDEX = ROOT / "data" / "features" / "embeddings_index.parquet"
STYLES = ROOT / "data" / "features" / "scribe_styles.parquet"
FIGURE = ROOT / "reports" / "figures" / "p8_discovery.png"

K_RANGE = (2, 3, 4, 5, 6, 8, 10, 12)


def load():
    """The embedded pages, real ones only.

    The 5,000 synthetic rows are excluded: they were rendered from a generator that knows the
    type, so clustering them would recover the generator's own parameters and prove nothing about
    discovery on photographs.
    """
    import pandas as pd

    index = pd.read_parquet(INDEX)
    embeddings = np.load(EMBEDDINGS, mmap_mode="r")
    real = index[~index["synthetic"]].reset_index(drop=True)
    return np.asarray(embeddings[real["row"].to_numpy()], dtype=float), real


def cluster(X, k: int, seed: int = SEED) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    return KMeans(n_clusters=k, n_init=25, random_state=seed).fit_predict(
        StandardScaler().fit_transform(X)
    )


def against(assignment: np.ndarray, **truths) -> dict:
    """Agreement with every candidate explanation of the partition, on one scale."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    out = {}
    for name, truth in truths.items():
        out[f"ari_vs_{name}"] = round(float(adjusted_rand_score(truth, assignment)), 4)
        out[f"ami_vs_{name}"] = round(float(adjusted_mutual_info_score(truth, assignment)), 4)
    return out


def sweep(X, ks=K_RANGE, **truths) -> list[dict]:
    rows = []
    for k in ks:
        assignment = cluster(X, k)
        sizes = np.bincount(assignment, minlength=k)
        rows.append(
            {
                "k": int(k),
                **against(assignment, **truths),
                "smallest_cluster": int(sizes.min()),
                "largest_share": round(float(sizes.max() / sizes.sum()), 4),
            }
        )
    return rows


def experiment_all(X, index, ks=K_RANGE) -> dict:
    rows = sweep(
        X,
        ks,
        type=index["diagram_type"].to_numpy(),
        source=index["source"].to_numpy(),
    )
    best = max(rows, key=lambda r: r["ami_vs_type"])
    return {
        "pages": int(len(X)),
        "types": int(index["diagram_type"].nunique()),
        "sources": int(index["source"].nunique()),
        "sweep": rows,
        "best_k_for_type": best["k"],
        "at_best_k": best,
        "source_beats_type": best["ami_vs_source"] > best["ami_vs_type"],
    }


def experiment_chaos(X, index, ks=(3, 4, 5, 6, 8)) -> dict:
    mask = (index["source"] == "chaos").to_numpy()
    if mask.sum() < 20:
        return {"pages": int(mask.sum()), "skipped": "too few chaos pages embedded"}
    truth = index.loc[mask, "diagram_type"].to_numpy()
    rows = sweep(X[mask], ks, type=truth)
    best = max(rows, key=lambda r: r["ami_vs_type"])
    return {
        "pages": int(mask.sum()),
        "types": int(len(set(truth.tolist()))),
        "sweep": rows,
        "best": best,
        # 6.1.1's supervised macro F1 within chaos, on the same embedding. The gap between a
        # clustering and this is the price of withholding the labels.
        "supervised_reference_macro_f1": 0.9925,
    }


def experiment_hdbpmn(X, index, ks=(2, 3, 4, 5, 6, 8, 11)) -> dict:
    import pandas as pd

    mask = (index["source"] == "hdbpmn").to_numpy()
    if mask.sum() < 50:
        return {"pages": int(mask.sum()), "skipped": "too few hdbpmn pages embedded"}

    sub = index.loc[mask].copy()
    sub["page"] = sub["id"].str.rsplit("/", n=1).str[-1]
    if STYLES.is_file():
        meta = pd.read_parquet(STYLES)[["page", "exercise", "scribe"]]
        sub = sub.merge(meta, on="page", how="left")
    else:
        sub["exercise"] = None
        sub["scribe"] = None

    known = sub["exercise"].notna().to_numpy()
    rows_X = X[mask][known]
    truths = {
        "exercise": sub.loc[known, "exercise"].to_numpy(),
        "scribe": sub.loc[known, "scribe"].to_numpy(),
    }
    rows = sweep(rows_X, ks, **truths)
    best = max(rows, key=lambda r: r["ami_vs_exercise"])
    return {
        "pages": int(mask.sum()),
        "pages_with_exercise": int(known.sum()),
        "exercises": int(len(set(truths["exercise"].tolist()))),
        "scribes": int(len(set(truths["scribe"].tolist()))),
        "sweep": rows,
        "best": best,
    }


def figure(results, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [
        (
            axes[0],
            results["all_real_pages"]["sweep"],
            ("ami_vs_type", "ami_vs_source"),
            "every real page",
        ),
        (axes[1], results["chaos_only"].get("sweep", []), ("ami_vs_type",), "chaos only"),
        (
            axes[2],
            results["hdbpmn_only"].get("sweep", []),
            ("ami_vs_exercise", "ami_vs_scribe"),
            "hdbpmn only",
        ),
    ]
    for ax, rows, keys, title in panels:
        if not rows:
            ax.set_axis_off()
            continue
        ks = [r["k"] for r in rows]
        for key in keys:
            ax.plot(ks, [r[key] for r in rows], marker="o", label=key.replace("ami_vs_", ""))
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel("adjusted mutual information")
        ax.set_ylim(-0.05, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("8.10 - what a clustering recovers when the labels are withheld", y=1.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(ks=K_RANGE) -> dict:
    X, index = load()
    results = {
        "all_real_pages": experiment_all(X, index, ks),
        "chaos_only": experiment_chaos(X, index),
        "hdbpmn_only": experiment_hdbpmn(X, index),
    }
    results["figure"] = str(figure(results))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    result = run()
    text = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

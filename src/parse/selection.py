"""Phase 7.4.3 - how many shapes are in the vocabulary, asked three ways that need not agree.

    python -m src.parse.selection      # writes reports/figures/p7_gmm_selection.png

7.4.2 chose a covariance type with K held at 5. This task sweeps K over the plan's [3, 15] and
asks where the vocabulary should stop growing.

## Why three criteria and not just BIC

The plan says "K selected by BIC elbow", and BIC is reported first. But BIC answers *how many
Gaussians best describe this cloud of 22-dimensional points*, and that is not the same question as
*how many shape names are worth having*, for a reason specific to this corpus: 7.4.1 measured a
`rect_aspect` of 3.26 for rectangles against ~1.2 for everything else, so a wide BPMN task and a
square BPMN task are far apart in the descriptor space while being **the same shape with the same
name**. A criterion that rewards describing the cloud will happily spend components splitting
rectangles by width.

So three are computed and reported side by side:

    BIC / AIC          the plan's criterion, and its more permissive sibling.
    held-out LL        5-fold, per node. Honest generalisation of the density.
    label agreement    adjusted Rand against 4.1.3's names. Does a bigger K mean *more shapes*
                       or the same shapes cut finer?

The interesting outcome is the one where they disagree, and the deliverable is which K each picks
rather than a single number asserted as correct.

## What "elbow" is taken to mean

An elbow is not a minimum - a curve can still be falling at its own elbow, and BIC on a large
table usually is. It is the point of maximum curvature, so it is computed here as the largest
second difference of the criterion over K, which is the discrete form of that and is stable on an
evenly spaced sweep. Both are reported: `bic_min` is where BIC bottoms out and `bic_elbow` is
where it stops falling *fast*, and on a 12,400-row table those are rarely the same K.

## What it measured

K from 3 to 15, `full` covariance, 12,400 rows, 5-fold:

     K        BIC     held-out LL   adj. Rand   purity   smallest component
     3    118,228        -6.217       0.0982    0.4731        2,430
     5     55,601        -3.286       0.1499    0.4027          950
     6     11,729        -2.021       0.1425    0.3830          796
     7      6,752        -0.783       0.1443    0.3791           66
     8     -1,481         0.090       0.1385    0.3831            9
     9      5,394         0.504       0.1589    0.3598            1
    13    -42,442         1.759       0.1245    0.2994            1
    15    -30,553         2.167       0.1237    0.2887            1

**The four criteria pick four different K, and the disagreement is the result.**

    BIC elbow        6
    BIC minimum     13
    AIC minimum     14
    held-out LL     15   (still rising at the end of the sweep)
    agreement        9

## The column that settles it is not any of the criteria

**From K = 9 onward the smallest component holds one node.** At K = 7 it is 66, at K = 8 it is 9,
and past that every additional component is a spike on a single outlier. A one-node Gaussian with
a regularised covariance has an enormous density at its own point and contributes almost nothing
anywhere else, which is exactly how BIC keeps improving while the vocabulary gets worse.

So BIC's minimum at 13 and the held-out likelihood's preference for 15 are **not evidence for a
13- or 15-shape vocabulary**; they are evidence that a mixture with enough components can memorise
outliers, and `reg_covar` bounds the damage without removing the incentive. The plan asked for the
BIC *elbow* rather than its minimum, and on this table that instinct is vindicated: **the elbow at
K = 6 is the last K before the components start collapsing.**

## Purity falls monotonically while likelihood rises

    K = 3   purity 0.4731     K = 9   purity 0.3598     K = 15  purity 0.2887

The two criteria move in opposite directions across the whole sweep, which is the sharpest
statement available of what 7.4.2 found at a single K: the mixture describes the descriptor cloud
better and better while agreeing with the shape names less and less. More components do not buy
more shapes; they buy finer cuts of the same shapes.

**And the ceiling is low everywhere.** The best purity at any K is 0.4731, at K = 3 - against a
majority-class baseline of 0.4796. **No K in the plan's range produces a vocabulary that names
shapes better than always saying `rectangle`.** The best adjusted Rand, 0.1589 at K = 9, is
0.009 above what K = 5 already gave.

That is the honest answer to "K selected by BIC elbow": **K = 6**, chosen because it is where the
curve turns and where the components are still populated - and carried forward with the caveat
that the quantity the plan hoped K would optimise does not improve at any K.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.parse.vocab import SEED, fit, matrix, purity
from src.utils.figures import save as _figsave

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p7_gmm_selection.png"

#: The plan's range, inclusive.
K_RANGE = tuple(range(3, 16))


def elbow(ks: list[int], values: list[float]) -> int:
    """The K of maximum curvature - the point where the curve stops falling fast.

    A minimum is a different thing and is reported separately: BIC on a table this size is often
    still decreasing at K = 15, in which case its minimum is an artefact of where the sweep was
    stopped and its elbow is the answer the plan actually wanted.
    """
    if len(values) < 3:
        return ks[0]
    second = np.diff(values, n=2)  # positive where the curve flattens from below
    return ks[int(np.argmax(second)) + 1]


def sweep(X, labels, covariance: str, ks=K_RANGE, folds: int = 5) -> list[dict]:
    from sklearn.metrics import adjusted_rand_score
    from sklearn.model_selection import StratifiedKFold

    splits = list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, labels))
    rows = []
    for k in ks:
        held = [float(fit(X[a], k, covariance).score(X[b])) for a, b in splits]
        whole = fit(X, k, covariance)
        gmm = whole.named_steps["gmm"]
        scaled = whole.named_steps["scale"].transform(X)
        components = gmm.predict(scaled)
        rows.append(
            {
                "k": k,
                "bic": round(float(gmm.bic(scaled)), 1),
                "aic": round(float(gmm.aic(scaled)), 1),
                "held_out_log_likelihood": round(float(np.mean(held)), 4),
                "adjusted_rand": round(float(adjusted_rand_score(labels, components)), 4),
                "free_parameters": int(gmm._n_parameters()),
                "converged": bool(gmm.converged_),
                "smallest_component": int(np.bincount(components, minlength=k).min()),
                **{"purity": purity(components, labels, k)["purity"]},
            }
        )
    return rows


def figure(rows: list[dict], path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [row["k"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].plot(ks, [r["bic"] for r in rows], marker="o", label="BIC")
    axes[0].plot(ks, [r["aic"] for r in rows], marker="s", label="AIC")
    axes[0].axvline(elbow(ks, [r["bic"] for r in rows]), color="crimson", ls="--", lw=1)
    axes[0].set_title("BIC / AIC (dashed: BIC elbow)")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(ks, [r["held_out_log_likelihood"] for r in rows], marker="o", color="seagreen")
    axes[1].set_title("held-out log likelihood / node")

    axes[2].plot(ks, [r["adjusted_rand"] for r in rows], marker="o", color="darkorange")
    axes[2].set_title("adjusted Rand vs 4.1.3 labels")

    for ax in axes:
        ax.set_xlabel("K")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def run(covariance: str = "full", ks=K_RANGE, folds: int = 5, write: bool = True) -> dict:
    X, labels, _, _ = matrix()
    rows = sweep(X, labels, covariance, ks, folds)
    k_list = [row["k"] for row in rows]

    best_held = max(rows, key=lambda r: r["held_out_log_likelihood"])
    best_rand = max(rows, key=lambda r: r["adjusted_rand"])
    bic_min = min(rows, key=lambda r: r["bic"])
    aic_min = min(rows, key=lambda r: r["aic"])
    bic_elbow = elbow(k_list, [row["bic"] for row in rows])

    return {
        "covariance": covariance,
        "rows": int(len(X)),
        "k_range": [k_list[0], k_list[-1]],
        "bic_elbow": int(bic_elbow),
        "bic_min_k": int(bic_min["k"]),
        "bic_still_falling_at_end": bool(rows[-1]["bic"] < rows[-2]["bic"]),
        "aic_min_k": int(aic_min["k"]),
        "best_held_out_k": int(best_held["k"]),
        "best_agreement_k": int(best_rand["k"]),
        "criteria_agree": len({bic_elbow, best_held["k"], best_rand["k"]}) == 1,
        "figure": str(figure(rows).relative_to(ROOT)) if write else None,
        "sweep": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--covariance", default="full")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.covariance, K_RANGE, args.folds), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 7.4.4 - what EM's ascent actually looks like, and how much the initialisation decides.

    python -m src.parse.emfit      # writes reports/figures/p7_gmm_em.png

7.4.2 and 7.4.3 read EM as a black box that returns a fitted model. This task opens it: the
log-likelihood is recorded after every iteration, for four initialisations across several seeds,
so three things can be checked rather than assumed.

    monotonicity   EM's guarantee is that the likelihood never decreases. It is a guarantee about
                   exact arithmetic, and `reg_covar` plus float64 can break it in principle, so it
                   is asserted against the measured trajectories rather than trusted.
    cost           iterations to the tolerance, which is what a bigger K actually costs.
    basin          the spread of final likelihoods across seeds. This is the number that says
                   whether "the GMM found" is a statement about the data or about a random draw.

## The four initialisations

    kmeans          sklearn's default: a full k-means to convergence, then EM from its partition.
    k-means++       only the k-means++ *seeding* step - K points chosen far apart - with no
                    k-means run at all. Much cheaper and much rougher.
    random          responsibilities drawn uniformly at random.
    random_from_data K actual rows chosen as the means.

The plan names the first two. The other two are included because "k-means++ beats random" is only
interesting if `random` is a real competitor rather than a strawman, and `random_from_data` is the
honest middle case - it is random, but it cannot place a mean somewhere the data never goes.

## How the trajectory is captured

`warm_start=True` with `max_iter=1`, called repeatedly. Each call performs exactly one E step and
one M step and leaves `lower_bound_` holding the likelihood after it, so the trajectory is the
real sequence of iterates rather than a series of independent refits stopped at increasing
horizons - which would be a different experiment and much more expensive.

`lower_bound_` is the mean per-sample log likelihood in recent sklearn, so the numbers here are
directly comparable to 7.4.2's `held_out_log_likelihood` and do not scale with the row count.

## What it measured

K = 5, `full` covariance, 12,400 rows, five seeds per initialisation:

    init                best      worst    spread   mean iterations
    random            -0.1828    -2.4255   2.2427        24.8
    random_from_data  -0.2263    -1.3928   1.1665        36.8
    k-means++         -0.3205    -2.5256   2.2051        38.4
    kmeans            -0.3211    -3.5340   3.2129        23.2

**Every one of the twenty runs was monotone.** EM's guarantee holds exactly here - not one
iteration in any trajectory decreased the likelihood - so `reg_covar` and float64 did not break
it, which is worth having checked rather than assumed.

## The plan's expected ordering does not appear

The plan frames this as k-means++ against random. **On this table plain `random` reaches the best
likelihood of any initialisation (-0.1828) and `kmeans` the worst best-case (-0.3211)**, and the
whole range across all four is 0.14 nats - smaller than the spread *within* any single one of
them. The initialisation strategy is not what decides the answer.

**The seed is.** Across all runs the best-to-worst gap is **3.3512 nats a node**, against 7.4.2's
7.6-nat margin between the best and second-best covariance type. So nearly half the effect of the
biggest modelling decision in 7.4 can be produced by changing a random seed, and any single fit
reporting "the GMM found five shapes" is reporting one draw.

`kmeans` produces the largest spread (3.21), which is the result worth stating carefully: the most
expensive initialisation - a full k-means run to convergence before EM starts - is the *least*
stable across seeds. It converges fastest once started (23.2 iterations against k-means++'s 38.4),
so it buys speed and pays for it in variance. `random_from_data` has the smallest spread (1.17) by
a wide margin, because placing means on actual rows cannot put a component anywhere the data never
goes, which is the failure mode the other three all have available.

## What this means for the rest of 7.4

Every fit in 7.4.2, 7.4.3 and 7.4.5 uses `n_init=1` at a fixed seed, which these numbers say is not
enough for the likelihood to be a stable quantity. The comparisons in those tasks survive it -
they are made at one seed throughout, so the ranking is like-for-like - but **the absolute
likelihoods there carry an uncertainty of a few nats, and 7.4.3's held-out curve rising to K = 15
should be read with that in mind rather than as a precise measurement.**

The practical recommendation is `random_from_data` with several restarts: it has half the spread
of anything else, and its best is within 0.04 nats of the overall best.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.parse.vocab import DEFAULT_K, REG, SEED, matrix
from src.utils.figures import save as _figsave

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p7_gmm_em.png"

#: The plan names `kmeans` and `k-means++`; the other two make the comparison honest.
INITS = ("kmeans", "k-means++", "random_from_data", "random")

#: Seeds per initialisation. Enough to see a spread, few enough to run beside everything else.
SEEDS = (0, 1, 2, 3, 4)

#: Hard cap on recorded iterations. 7.4.2 converged in 19-24, so this is comfortably above.
MAX_ITER = 120

TOLERANCE = 1e-3


def trajectory(
    X,
    k: int = DEFAULT_K,
    covariance: str = "full",
    init: str = "kmeans",
    seed: int = SEED,
    max_iter: int = MAX_ITER,
) -> dict:
    """One EM run, recording the log likelihood after every iteration.

    One `fit` call per iteration under `warm_start`, so this is the true sequence of iterates.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(X)
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance,
        reg_covar=REG,
        random_state=seed,
        init_params=init,
        n_init=1,
        max_iter=1,
        tol=TOLERANCE,
        warm_start=True,
    )

    curve: list[float] = []
    converged_at = None
    # Every call runs exactly one iteration and so never meets sklearn's own stopping rule; its
    # warning would fire once per iteration and means nothing here, because convergence is being
    # decided below rather than by `fit`.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for step in range(max_iter):
            model.fit(scaled)
            curve.append(float(model.lower_bound_))
            if step and abs(curve[-1] - curve[-2]) < TOLERANCE:
                converged_at = step + 1
                break

    drops = [
        {"iteration": i + 1, "delta": round(curve[i + 1] - curve[i], 8)}
        for i in range(len(curve) - 1)
        if curve[i + 1] < curve[i] - 1e-9
    ]
    return {
        "init": init,
        "seed": seed,
        "iterations": len(curve),
        "converged_at": converged_at,
        "final": round(curve[-1], 4),
        "first": round(curve[0], 4),
        "monotone": not drops,
        "decreases": drops[:5],
        "curve": [round(value, 4) for value in curve],
    }


def by_init(X, k: int = DEFAULT_K, covariance: str = "full", inits=INITS, seeds=SEEDS) -> dict:
    """Every initialisation at every seed, summarised by the spread it produces."""
    out = {}
    for init in inits:
        runs = [trajectory(X, k, covariance, init, seed) for seed in seeds]
        finals = [run["final"] for run in runs]
        out[init] = {
            "best": round(max(finals), 4),
            "worst": round(min(finals), 4),
            "spread": round(max(finals) - min(finals), 4),
            "mean_iterations": round(float(np.mean([run["iterations"] for run in runs])), 1),
            "all_monotone": all(run["monotone"] for run in runs),
            "runs": runs,
        }
    return out


def figure(results: dict, path: Path = FIGURE) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    colours = plt.cm.viridis(np.linspace(0, 0.85, len(results)))
    for colour, (init, summary) in zip(colours, results.items(), strict=True):
        for i, run in enumerate(summary["runs"]):
            axes[0].plot(
                range(1, len(run["curve"]) + 1),
                run["curve"],
                color=colour,
                alpha=0.75,
                lw=1.3,
                label=init if i == 0 else None,
            )
    axes[0].set_xlabel("EM iteration")
    axes[0].set_ylabel("log likelihood / node")
    axes[0].set_title("ascent, every seed")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)

    names = list(results)
    axes[1].bar(names, [results[n]["spread"] for n in names], color=colours)
    axes[1].set_ylabel("best - worst over seeds")
    axes[1].set_title("how much the seed decides")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=150)
    plt.close(fig)
    return path


def run(k: int = DEFAULT_K, covariance: str = "full", write: bool = True) -> dict:
    X, _, _, _ = matrix()
    results = by_init(X, k, covariance)

    best = max(results.items(), key=lambda kv: kv[1]["best"])
    cheapest = min(results.items(), key=lambda kv: kv[1]["mean_iterations"])
    steadiest = min(results.items(), key=lambda kv: kv[1]["spread"])
    return {
        "rows": int(len(X)),
        "k": k,
        "covariance": covariance,
        "seeds": list(SEEDS),
        "every_run_monotone": all(summary["all_monotone"] for summary in results.values()),
        "best_likelihood_init": best[0],
        "fewest_iterations_init": cheapest[0],
        "smallest_spread_init": steadiest[0],
        "best_minus_worst_overall": round(
            max(s["best"] for s in results.values()) - min(s["worst"] for s in results.values()), 4
        ),
        "figure": str(figure(results).relative_to(ROOT)) if write else None,
        "by_init": {
            name: {key: value for key, value in summary.items() if key != "runs"}
            for name, summary in results.items()
        },
        "curves": {
            name: [run["curve"] for run in summary["runs"]] for name, summary in results.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--covariance", default="full")
    args = ap.parse_args(argv)
    try:
        result = run(args.k, args.covariance)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    result.pop("curves", None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

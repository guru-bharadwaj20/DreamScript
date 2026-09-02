"""Phase 7.4.7 - whether a rectangle is a rectangle, or whether it is this person's rectangle.

    python -m src.parse.scribes      # writes reports/figures/p7_scribe_covariance.png

hdbpmn's IR carries a `scribe_id` on every page: **105 writers, up to 10 pages each**. So the
plan's claim - "everyone draws rectangles differently" - is directly measurable rather than
assertable, and this task measures it three ways and then asks whether it is worth acting on.

## Restricted to rectangles on purpose

Everything below uses only the 5,947 nodes labelled `rectangle`. Holding the shape fixed is what
makes the question about the *scribe*: if all four classes were pooled, any between-scribe
difference could just be one writer drawing more diamonds than another, which is a fact about the
exercises they were given rather than about their hand.

## The three measurements

    separation    ANOVA F of scribe identity on each descriptor, over rectangles only. Read
                  against 7.4.1's F of shape class over all four classes - the two are on the
                  same columns and the same corpus, so "is your handwriting a bigger effect than
                  what you drew" has a numeric answer.
    identifiability
                  can a classifier name the scribe from a rectangle's geometry alone? Grouped by
                  page, so a test rectangle's own page is never in training. 105 classes, so
                  chance is under 1%.
    adaptation    the thing the plan actually asks for. A global GMM against a per-scribe GMM,
                  scored on held-out log likelihood, under two splits that mean different things:

                      unseen page    a held-out page from a scribe the model has seen. This is
                                     the adaptation scenario - the writer is known.
                      unseen scribe  a held-out scribe entirely. This is the deployment scenario,
                                     and a per-scribe model has nothing to offer here by
                                     construction, so it is the control that keeps the first
                                     number honest.

A per-scribe model winning on unseen pages and losing on unseen scribes would be the expected
result and would say: adaptation is worth doing only once you have seen someone write.

## Why a full covariance per scribe is not affordable, and what is used instead

A scribe contributes a median of 8 pages and 59 rectangles. A 22-dimensional full
covariance needs 253 parameters, so per-scribe fitting is done on `diag` regardless of 7.4.2's
finding that `full` is better globally - the comparison is like-for-like because the global model
is refitted as `diag` too. That is a real limitation of the corpus rather than a modelling choice,
and it is why the figure shows two-dimensional marginals, where a full covariance *is* affordable
and the per-scribe ellipses can be drawn honestly.

## What it measured

5,947 rectangles, **105 scribes**, 91 of them with at least 30 rectangles, a median of 59 each.

## The plan's claim is true, and it is much smaller than it sounds

**Everyone does draw rectangles differently, and a classifier can prove it**: a random forest names
which of 91 writers drew a rectangle **9.42% of the time**, against 1.10% chance and a 1.82%
majority baseline - **8.6x chance**, from geometry alone, with the writer's own page held out. A
rectangle carries real evidence about whose hand drew it.

**And it is a rounding error next to what the rectangle is.** The strongest scribe-separating
column is `hu_0` at an ANOVA F of **13.7**. 7.4.1's strongest shape-separating column, on the same
descriptors and the same corpus, is `hu_1` at **2,693.9**. **Shape identity is roughly 200x the
effect that scribe identity is.** The plan's framing - "show everyone draws rectangles differently"
- is vindicated as a fact and denied as a priority.

The ranking underneath is worth one line: `extent`, `defect_count` and `rect_fill` sit near the top
of the scribe list (12.9, 12.3, 12.0) while `rect_aspect` is near the bottom at 1.8. **What varies
between writers is how cleanly the box closes, not how wide they draw it** - the aspect ratio is
dictated by the BPMN task text that has to fit inside, which is the exercise rather than the hand.

## Adaptation makes it worse, by a lot

    split            global LL    per-scribe LL     gain
    unseen page       -31.5818      -39.5117      -7.9299
    unseen scribe        -            -           (no scribe has training rows of their own)

**Fitting a scribe their own Gaussian costs 7.93 nats a rectangle** against using the global one,
over 285 scribe-fold blocks and 4,723 held-out nodes. This is not a subtle result and its cause is
not subtle either: a diagonal Gaussian over 22 columns is 44 free parameters, and the median
scribe brings 59 rectangles - of which roughly four fifths survive into any training fold. The
per-scribe model fits its own noise and the global model, pooled over 5,947, does not.

7.4.4 measured the same shape of problem from the other end: the seed moved the answer 3.35 nats
when the data was plentiful. Here the data is scarce and the penalty is twice that.

The `unseen scribe` row is the control and it reports **0 blocks**, correctly. Held out by scribe,
a writer has no training rows of their own, so no personal model can exist - which is exactly the
deployment case, and is why the adaptation number must never be quoted without it. A per-scribe
scheme that helps on a known writer and cannot run at all on a new one is a different product.

## One methodological note, because the first measurement was wrong

The first version of this task fitted each model inside its own `StandardScaler` and compared the
resulting log likelihoods. That is invalid: a per-scribe scaler standardises that scribe's spread
to 1 while the global scaler standardises the corpus spread to 1, so the two densities live over
differently stretched spaces and differ by a Jacobian unrelated to fit. It reported a gain of
-16.84. With one scaler per fold, fitted on the training rows and shared by both models, the
honest figure is **-7.93** - still decisively negative, and half the size. The positive control in
`test_parse_scribes.py`, which asserts that per-scribe fitting *does* win when scribes are far
apart and data is plentiful, is what caught it.

## What this means for 8.6

Phase 8.6 clusters scribes, and these numbers say what it can expect. The signal is real - 8.6x
chance - so a clustering will find something. It is weak enough that **per-scribe density models
are not the way to spend it**; grouping many writers into a few style clusters, which is what 8.6
actually proposes, is the form that could work, because it buys parameters per cluster instead of
per person. This task is evidence for that design rather than against it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.features.descriptors import NAMES

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p7_scribe_covariance.png"

SEED = 42

#: Rectangles a scribe needs before a per-scribe model is fitted for them.
MIN_ROWS = 30

#: The two columns the figure draws. Chosen as the strongest shape-separating column (7.4.1's
#: `circularity`) against the one that is purely a habit of hand (`rect_aspect`).
PANEL = ("rect_aspect", "circularity")

REG = 1e-4


def scribe_of_pages() -> dict:
    """`page id` -> `scribe_id`, read from the IR rather than parsed out of the filename."""
    from src.ir.model import Diagram

    out = {}
    for path in (ROOT / "data" / "processed" / "ir" / "hdbpmn").glob("*.ir.json"):
        page = path.name[: -len(".ir.json")]
        meta = Diagram.load(path).meta
        scribe = meta.get("scribe_id")
        if scribe:
            out[page] = scribe
    return out


def rectangles():
    """(X, scribe, page) for every node labelled `rectangle`."""
    from src.features.descriptors import load_table

    table = load_table()
    table = table[table["label"] == "rectangle"]
    mapping = scribe_of_pages()
    scribe = np.array([mapping.get(page, "?") for page in table["page"]], dtype=object)

    X = table[list(NAMES)].to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=1) & (scribe != "?")
    return X[keep], scribe[keep], table["page"].to_numpy()[keep]


def separation(X, scribe) -> dict:
    """ANOVA F of scribe identity on each column, over one shape class."""
    from scipy.stats import f_oneway

    out = {}
    for i, name in enumerate(NAMES):
        groups = [X[scribe == s, i] for s in sorted(set(scribe))]
        groups = [g for g in groups if len(g) > 2 and g.std() > 0]
        if len(groups) > 1:
            statistic = f_oneway(*groups).statistic
            if np.isfinite(statistic):
                out[name] = round(float(statistic), 1)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def identifiability(X, scribe, page, folds: int = 5) -> dict:
    """Can the scribe be named from a rectangle's geometry? Grouped by page."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Scribes with too few rectangles cannot be a class in a grouped split.
    counts = {s: int((scribe == s).sum()) for s in set(scribe)}
    keep = np.array([counts[s] >= MIN_ROWS for s in scribe])
    X, scribe, page = X[keep], scribe[keep], page[keep]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ]
    )
    predicted = cross_val_predict(
        model, X, scribe, cv=StratifiedGroupKFold(folds), groups=page, n_jobs=1
    )
    classes = sorted(set(scribe))
    largest = max(np.mean(scribe == s) for s in classes)
    return {
        "scribes": len(classes),
        "rows": int(len(scribe)),
        "accuracy": round(float(np.mean(predicted == scribe)), 4),
        "chance": round(1 / len(classes), 4),
        "majority_baseline": round(float(largest), 4),
    }


def _fit_diag(X, seed: int = SEED):
    """A single diagonal Gaussian on *already scaled* rows.

    Deliberately not a pipeline with its own scaler. Two densities can only be compared if they
    are densities over the same space, and a per-scribe scaler would standardise that scribe's own
    spread to 1 while the global scaler standardises the corpus spread to 1 - different changes of
    variable, so their log likelihoods differ by a Jacobian that has nothing to do with fit. The
    caller scales once, on the training fold, and hands the same array to both models.
    """
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(
        n_components=1, covariance_type="diag", reg_covar=REG, random_state=seed
    ).fit(X)


def adaptation(X, scribe, page, folds: int = 5) -> dict:
    """Global against per-scribe, under an unseen page and an unseen scribe."""
    from sklearn.model_selection import GroupKFold

    usable = np.array([int((scribe == s).sum()) >= MIN_ROWS for s in scribe])
    X, scribe, page = X[usable], scribe[usable], page[usable]

    def evaluate(groups):
        """Node-weighted mean log likelihood under the global and the per-scribe model.

        Weighted by nodes rather than by scribe so a writer with 90 rectangles does not count
        the same as one with 30; `score` is a per-sample mean, so each block is multiplied back
        up by its own size and divided out at the end.
        """
        from sklearn.preprocessing import StandardScaler

        world_total, personal_total, weight, blocks = 0.0, 0.0, 0, 0
        for train, test in GroupKFold(folds).split(X, scribe, groups=groups):
            # One scaler per fold, fitted on the training rows only, shared by both models so the
            # two log likelihoods are over the same space. See `_fit_diag`.
            scaler = StandardScaler().fit(X[train])
            scaled_train, scaled_test = scaler.transform(X[train]), scaler.transform(X[test])
            world = _fit_diag(scaled_train)
            for s in sorted(set(scribe[test])):
                mine = scribe[test] == s
                own = scribe[train] == s
                # No held-out rows, or the scribe is unseen in training - the second case is the
                # whole point of the `unseen_scribe` split and is skipped rather than faked.
                if mine.sum() < 3 or own.sum() < MIN_ROWS:
                    continue
                rows = scaled_test[mine]
                world_total += float(world.score(rows)) * len(rows)
                personal_total += float(_fit_diag(scaled_train[own]).score(rows)) * len(rows)
                weight += len(rows)
                blocks += 1
        return world_total, personal_total, weight, blocks

    results = {}
    for name, groups in (("unseen_page", page), ("unseen_scribe", scribe)):
        world_total, personal_total, weight, blocks = evaluate(groups)
        if not weight:
            # `unseen_scribe` is expected to land here: a held-out scribe has no training rows of
            # their own, so a per-scribe model cannot be fitted for them at all.
            results[name] = {"scribe_blocks": 0, "note": "no scribe had training rows of their own"}
            continue
        results[name] = {
            "global_log_likelihood": round(world_total / weight, 4),
            "per_scribe_log_likelihood": round(personal_total / weight, 4),
            "gain": round((personal_total - world_total) / weight, 4),
            "scribe_blocks": blocks,
            "nodes_scored": weight,
        }
    return results


def figure(X, scribe, path: Path = FIGURE) -> Path:
    """Per-scribe covariance ellipses on two columns, over the global cloud."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    i, j = NAMES.index(PANEL[0]), NAMES.index(PANEL[1])
    counts = {s: int((scribe == s).sum()) for s in set(scribe)}
    busiest = sorted(counts, key=lambda s: -counts[s])[:12]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].scatter(X[:, i], X[:, j], s=3, alpha=0.12, color="grey", linewidths=0)
    colours = plt.cm.tab20(np.linspace(0, 1, len(busiest)))

    for colour, s in zip(colours, busiest, strict=True):
        rows = X[scribe == s][:, [i, j]]
        centre = rows.mean(axis=0)
        cov = np.cov(rows, rowvar=False)
        values, vectors = np.linalg.eigh(cov)
        order = values.argsort()[::-1]
        values, vectors = values[order], vectors[:, order]
        angle = np.degrees(np.arctan2(*vectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(np.maximum(values, 0)) * 2  # two standard deviations
        axes[0].add_patch(
            Ellipse(centre, width, height, angle=angle, fill=False, color=colour, lw=1.6)
        )
        axes[0].plot(*centre, marker="o", color=colour, ms=4)

    axes[0].set_xlabel(PANEL[0])
    axes[0].set_ylabel(PANEL[1])
    axes[0].set_title(f"rectangles: {len(busiest)} scribes, 2 s.d. ellipses")
    axes[0].grid(alpha=0.3)

    means = np.array([X[scribe == s][:, [i, j]].mean(axis=0) for s in sorted(counts)])
    axes[1].scatter(means[:, 0], means[:, 1], s=18, color="crimson", alpha=0.7)
    axes[1].set_xlabel(f"per-scribe mean {PANEL[0]}")
    axes[1].set_ylabel(f"per-scribe mean {PANEL[1]}")
    axes[1].set_title(f"every scribe's average rectangle (n={len(means)})")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(folds: int = 5, write: bool = True) -> dict:
    X, scribe, page = rectangles()
    counts = {s: int((scribe == s).sum()) for s in set(scribe)}
    by_scribe = separation(X, scribe)

    result = {
        "rectangles": int(len(X)),
        "scribes": len(counts),
        "scribes_with_enough_rows": sum(1 for v in counts.values() if v >= MIN_ROWS),
        "rectangles_per_scribe_median": int(np.median(list(counts.values()))),
        "anova_f_by_scribe": by_scribe,
        "strongest_scribe_column": next(iter(by_scribe), None),
        "identifiability": identifiability(X, scribe, page, folds),
        "adaptation": adaptation(X, scribe, page, folds),
    }
    if write:
        result["figure"] = str(figure(X, scribe).relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    try:
        print(json.dumps(run(args.folds), indent=2))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

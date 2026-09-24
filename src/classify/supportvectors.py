"""Phase 6.3.5 - which pages become support vectors, and whether they are the ambiguous ones.

    python -m src.classify.supportvectors     # writes reports/figures/p6_support_vectors.png

The plan says: count, margin width, which images become support vectors - **expect ambiguous
ones**. That last clause is a hypothesis, and this task is built to be able to reject it.

A support vector is a training row lying on or inside the margin. The received account is that
these are the hard, borderline, ambiguous examples, and that a gallery of them will look like a
gallery of bad photographs. There is a competing account that this corpus makes very live:
**a support vector might just be a member of a small class.** 6.3.1 already saw the shape of it -
39 of 40 circuit pages were support vectors, against 50 of 600 wireframes - and the reason is
structural rather than about difficulty. A class with 40 examples spread across a region has
almost no interior; nearly every one of its members is near some boundary, ambiguous or not.

So the two accounts are separated rather than assumed:

    ambiguity account    support vectors should be the rows the model gets wrong, or is
                         least confident about, controlling for class
    class-size account   support-vector share should track class size and little else

Both are measurable on the same fit, and the answer decides whether the gallery the plan asks for
is a gallery of hard pages or a gallery of rare ones.

## Margin width

Reported as `2 / ||w||` for the linear model, which is the textbook quantity and is only defined
for a linear kernel - in an RBF feature space `w` is never formed and `coef_` does not exist. For
a kernel model `margin_width` therefore returns **None** rather than a plausible-looking
substitute, because putting two different quantities in one column is worse than leaving it
empty.

## What it measured

Linear SVC at 6.3.1's C = 0.1 on the hybrid table, fitted on all 1,340 rows.
**353 support vectors, 26.3% of the corpus.** Margin width `2/||w||` over the ten one-vs-one
problems: mean 6.42, range 2.59 to 10.26.

    class            rows    support vectors    share
    circuit            40          36           0.900
    er_diagram         50          31           0.620
    state_machine      50          25           0.500
    flowchart         600         134           0.223
    wireframe         600         127           0.212

    correlation between support-vector share and log class size:  -0.894

**Both accounts are true, and the measurement is able to say exactly how.**

**The class-size account is confirmed and it is strong.** The correlation between a class's
support-vector share and its (log) size is **-0.894**. 90% of the 40-row circuit class sits on
the margin against 21% of the 600-row wireframe class - a factor of four - and the ordering
follows class size exactly, with no exceptions. A small class has almost no interior, and that is
most of what the support-vector count is measuring.

**The ambiguity account is also confirmed, in a much sharper form than expected:**

    error rate among support vectors        0.0312
    error rate among non-support vectors    0.0000
    all 11 out-of-fold errors are support vectors                     11 of 11
    but support vectors classified correctly                         342 of 353

**Every single page the model gets wrong is a support vector, and 97% of support vectors are
classified correctly.** Being a support vector is **necessary but nowhere near sufficient** for
being an error - and that is not a restatement of the class-size effect, because it holds
*within* every class:

    class            error rate | SV     error rate | not SV
    circuit             0.1667                0.0000
    er_diagram          0.1290                0.0000
    flowchart           0.0075                0.0000
    state_machine       0.0000                0.0000
    wireframe           0.0000                0.0000

The non-support-vector column is zero everywhere. **The 987 pages outside the margin are
classified perfectly, in every class**, which is exactly what the geometry says must happen - a
point outside the margin of a fitted model is correctly classified by that model by definition -
and it is worth reporting because it puts a precise bound on what the gallery can show.

## What that means for the plan's expectation

The plan says *expect ambiguous ones*. Read as **"the hard pages are all support vectors"** it is
correct without exception. Read as **"the support vectors are the hard pages"** it is wrong by a
factor of thirty: 353 pages are support vectors, 11 are hard.

So the gallery is drawn as a **contrast** - support vectors above, non-support-vectors of the
same class below - rather than as a grid of support vectors alone. A grid of 353 hand-drawn
diagrams looks difficult whatever it contains, and without the matched row underneath a reader
would confirm the ambiguity hypothesis from a picture that cannot test it. With the contrast in
place, the honest description of the top row is *the outer 26% of each class*, which for circuits
is very nearly the whole class and for wireframes is a genuinely messier-looking selection.

## For the phases that follow

**The 987 non-support-vectors are, on this model, free.** Every one of them is correct and none
is near a boundary, which makes them the population Phase 14's ablations can safely subsample and
the one Phase 9's detector need not be tuned for. The 11 errors are all inside the margin, and **10 of the
11 are circuit or er_diagram pages** (6 and 4; the eleventh is a flowchart) - the same hard core
5.3.4 measured at 30 pages across three weaker models, narrowed here to 11 by a better one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.classify.activations import TABLES
from src.utils.figures import save as _figsave

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "reports" / "figures" / "p6_support_vectors.png"
MANIFEST = ROOT / "data" / "processed" / "manifest.parquet"

SEED = 42


def fit_svc(data, kernel: str = "linear", table: str = "hybrid"):
    """One SVC on all rows, at 6.3.1-6.3.4's settings, keeping the scaler for the margin maths."""
    from sklearn.pipeline import Pipeline

    from src.classify.multiclass import base_estimator
    from src.features.scaling import feature_scaler

    estimator = Pipeline([("prepare", feature_scaler()), ("model", base_estimator(kernel, table))])
    return estimator.fit(data.X, data.y)


def margin_width(fitted) -> dict | None:
    """`2 / ||w||` per one-vs-one problem - defined only for the linear kernel.

    Returns None for a kernel model rather than a plausible-looking number: `coef_` does not
    exist when the feature map is implicit, and inventing a substitute would put two different
    quantities in one column.
    """
    model = fitted.named_steps["model"]
    if getattr(model, "kernel", None) != "linear":
        return None
    norms = np.linalg.norm(model.coef_, axis=1)
    widths = 2.0 / norms
    return {
        "problems": int(len(widths)),
        "mean_margin_width": round(float(widths.mean()), 4),
        "min_margin_width": round(float(widths.min()), 4),
        "max_margin_width": round(float(widths.max()), 4),
    }


def support_flags(fitted, data) -> np.ndarray:
    """Boolean per row: is this training page a support vector?"""
    model = fitted.named_steps["model"]
    flags = np.zeros(len(data.y), dtype=bool)
    flags[model.support_] = True
    return flags


def by_class(fitted, data) -> dict:
    """Support-vector share per class - the class-size account's evidence."""
    flags = support_flags(fitted, data)
    counts = data.class_counts()
    return {
        name: {
            "rows": counts[name],
            "support_vectors": int(flags[data.y == name].sum()),
            "share": round(float(flags[data.y == name].mean()), 4),
        }
        for name in sorted(counts)
    }


def ambiguity(fitted, data, folds: int = 5) -> dict:
    """Are support vectors the rows the model gets wrong - once class size is controlled for?

    The control is the point. Support vectors are concentrated in small classes and errors are
    too, so a raw correlation between "is a support vector" and "is misclassified" would be
    largely an artefact of both tracking class size. The comparison is therefore made **within
    each class**, and the pooled number is reported beside it so the difference is visible.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    from src.classify.multiclass import base_estimator
    from src.features.scaling import feature_scaler

    model = fitted.named_steps["model"]
    predicted = cross_val_predict(
        Pipeline([("prepare", feature_scaler()), ("model", base_estimator(model.kernel))]),
        data.X,
        data.y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    wrong = predicted != data.y
    flags = support_flags(fitted, data)

    within = {}
    for name in sorted(data.class_counts()):
        mask = data.y == name
        sv, non_sv = mask & flags, mask & ~flags
        within[name] = {
            "error_rate_if_support_vector": (
                round(float(wrong[sv].mean()), 4) if sv.any() else None
            ),
            "error_rate_if_not": round(float(wrong[non_sv].mean()), 4) if non_sv.any() else None,
            "support_vectors": int(sv.sum()),
            "others": int(non_sv.sum()),
        }

    return {
        "pooled_error_rate_if_support_vector": round(float(wrong[flags].mean()), 4),
        "pooled_error_rate_if_not": round(float(wrong[~flags].mean()), 4),
        "within_class": within,
        # Every misclassified row that is not a support vector is a row the ambiguity account
        # does not explain, and every support vector the model gets right is the same problem
        # from the other side.
        "errors_that_are_support_vectors": int((wrong & flags).sum()),
        "errors_total": int(wrong.sum()),
        "support_vectors_classified_correctly": int((flags & ~wrong).sum()),
        "support_vectors_total": int(flags.sum()),
    }


def gallery(fitted, data, path: Path = FIGURE, per_class: int = 4) -> Path | None:
    """The plan's gallery: support-vector pages beside non-support-vector pages of the same class.

    The contrast is the whole point. A grid of support vectors alone always looks like a grid of
    messy diagrams, because every grid of hand-drawn diagrams does; putting a matched row of
    non-support-vectors underneath is what lets a reader judge whether the top row is actually
    harder, and it is the layout the ambiguity hypothesis has to survive.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = image_paths(data.ids)
    if not paths:
        return None

    flags = support_flags(fitted, data)
    classes = sorted(data.class_counts())
    fig, axes = plt.subplots(
        2 * len(classes), per_class, figsize=(2.1 * per_class, 2.3 * len(classes) * 2)
    )
    axes = np.atleast_2d(axes)

    for index, name in enumerate(classes):
        for offset, (want, label) in enumerate(((True, "SV"), (False, "not SV"))):
            row = 2 * index + offset
            mask = (data.y == name) & (flags == want)
            chosen = [i for i in np.flatnonzero(mask) if data.ids[i] in paths][:per_class]
            for column in range(per_class):
                ax = axes[row, column]
                ax.set_xticks([])
                ax.set_yticks([])
                if column < len(chosen):
                    ax.imshow(_thumbnail(paths[data.ids[chosen[column]]]), cmap="gray")
                if column == 0:
                    ax.set_ylabel(
                        f"{name}\n{label}",
                        fontsize=6.5,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=28,
                    )

    fig.suptitle(
        "Phase 6.3.5 - support vectors (top row of each pair) against non-support vectors "
        "of the same class",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    _figsave(fig, path, dpi=140)
    plt.close(fig)
    return path


def image_paths(ids) -> dict:
    """id -> on-disk image path, from 1.3's processed manifest. Empty when it is not present."""
    if not MANIFEST.is_file():
        return {}
    import pandas as pd

    frame = pd.read_parquet(MANIFEST, columns=["id", "path"])
    wanted = set(map(str, ids))
    found = {}
    for identifier, relative in zip(frame["id"], frame["path"], strict=True):
        if str(identifier) in wanted:
            candidate = ROOT / str(relative)
            if candidate.is_file():
                found[identifier] = candidate
    return found


def _thumbnail(path: Path, size: int = 220):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((size, size), dtype=np.uint8)
    height, width = image.shape
    scale = size / max(height, width)
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))


def run(
    table: str = "hybrid", corpus: str = "real", kernel: str = "linear", write: bool = True
) -> dict:
    from src.embed.hybrid import dataset

    data = dataset(table, corpus)
    fitted = fit_svc(data, kernel, table)
    flags = support_flags(fitted, data)

    shares = by_class(fitted, data)
    sizes = np.array([shares[name]["rows"] for name in shares], dtype=float)
    fractions = np.array([shares[name]["share"] for name in shares], dtype=float)

    summary = {
        "table": table,
        "corpus": corpus,
        "kernel": kernel,
        "rows": int(len(data.y)),
        "support_vectors": int(flags.sum()),
        "support_vector_share": round(float(flags.mean()), 4),
        "by_class": shares,
        # The class-size account in one number: if share is a function of class size, this is
        # strongly negative and there is little left for ambiguity to explain.
        "correlation_share_vs_class_size": round(
            float(np.corrcoef(np.log(sizes), fractions)[0, 1]), 4
        ),
        "margin": margin_width(fitted),
        "ambiguity": ambiguity(fitted, data),
    }
    if write:
        path = gallery(fitted, data)
        summary["figure"] = str(path.relative_to(ROOT)) if path else None
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="hybrid", choices=list(TABLES))
    ap.add_argument("--corpus", default="real", choices=["real", "synthetic", "all"])
    ap.add_argument("--kernel", default="linear", choices=["linear", "rbf", "poly"])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    try:
        result = run(args.table, args.corpus, args.kernel, not args.no_figure)
    except (FileNotFoundError, KeyError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
